import Foundation

/// THE WALK, the way the measurements say it should be done.
///
/// The crawl gates the first embed - every root is walked to completion before anything is indexed -
/// so on a large library it is minutes of a launch during which nothing happens. Measured on a real
/// set of roots (2.66M files across five roots, four local and one external drive): 127s.
///
/// Two things make it faster, and only one of them is the one you would guess.
///
/// GETATTRLISTBULK(2) returns many directory entries WITH their attributes - name, type, mtime,
/// size - in a single syscall, where FileManager's enumerator pays a `resourceValues` round trip per
/// entry. That is worth ~1.5x on local storage and nothing at all on an external drive.
///
/// CONCURRENCY is the real lever, and it is not the one that was easiest to reach for: parallelising
/// the EXISTING enumerator measured 1.0x on the external drive and 1.6x locally, because its
/// per-entry cost does not overlap. A pool over directories reading in bulk measured 7.9x locally
/// and 3.2x on the external drive.
///
///   Documents, 63k files, warm         external drive, 2.4M files
///     enumerator (was)   1.25s           enumerator (was)   115.7s
///     fts(3)             0.83s           fts(3)             114.8s
///     bulk, 1 thread     0.86s           bulk, 1 thread     111.9s
///     bulk, 8 workers    0.16s           bulk, 8 workers     36.5s
///     bulk, 16 workers   0.17s           bulk, 16 workers    30.8s
///
/// EIGHT WORKERS, not sixteen, and the reason is in the kernel column rather than the clock. APFS
/// takes a global lock inside readdir (Szorc, 2018), so past a point the pool buys wall-clock with
/// kernel CPU: at 16 workers the external drive gained 18% of the clock for 84% more system time
/// (97.7s -> 179.5s), and locally system time tripled for nothing. That CPU is not free - it is
/// taken from the embedding pipeline this walk exists to feed.
///
/// Published guidance disagrees with this, and it is worth saying why. The most careful public
/// benchmark of macOS directory reads (Tempelmann, Find Any File) concludes "always fts" on APFS and
/// later that enumeratorAtURL is faster still. Those tests are name-only scans on macOS 10.13-10.14;
/// this one needs mtime and size, which is precisely what the bulk call returns for free, and six
/// years of APFS have moved the numbers. Measured here on macOS 26, bulk beats both.
enum BulkDirWalker {
    /// One directory entry, as the kernel handed it over.
    struct Entry {
        var name: String
        var isDir: Bool
        var isSymlink: Bool
        var size: Int
        var mtime: Double
        /// Which volume the entry lives on. Comes back in the same syscall as everything else, so
        /// staying inside one volume costs nothing - asking stat(2) per directory instead measured
        /// 2.1x against 7.9x on a real root, i.e. it gave away most of the speedup.
        var device: dev_t
    }

    /// 128 KB. Larger buffers stop helping once a directory fits in one call, and this is the size
    /// the fastest published macOS implementation settled on for the same reason.
    private static let bufferBytes = 128 * 1024

    /// Read one directory in as few syscalls as the kernel allows. Returns nil when the directory
    /// cannot be opened at all (permissions, or it vanished mid-walk), which the caller treats as
    /// empty rather than as an error - the same way the enumerator's errorHandler did.
    static func list(_ dir: String, into buf: inout [UInt8]) -> [Entry]? {
        let fd = open(dir, O_RDONLY | O_DIRECTORY, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

        var attrs = attrlist()
        attrs.bitmapcount = u_short(ATTR_BIT_MAP_COUNT)
        attrs.commonattr = attrgroup_t(UInt32(ATTR_CMN_RETURNED_ATTRS) | UInt32(ATTR_CMN_NAME)
                                       | UInt32(ATTR_CMN_DEVID) | UInt32(ATTR_CMN_OBJTYPE)
                                       | UInt32(ATTR_CMN_MODTIME))
        attrs.fileattr = attrgroup_t(UInt32(ATTR_FILE_DATALENGTH))

        var out: [Entry] = []
        while true {
            let got: Int = buf.withUnsafeMutableBytes { raw -> Int in
                Int(getattrlistbulk(fd, &attrs, raw.baseAddress, raw.count, 0))
            }
            if got <= 0 { break }   // 0 = done; negative = error, and a partial listing is all we get
            buf.withUnsafeBytes { raw in
                var base = raw.baseAddress!
                for _ in 0 ..< got {
                    let entryStart = base
                    let entryLen = Int(entryStart.loadUnaligned(as: UInt32.self))
                    var p = entryStart.advanced(by: 4)

                    // ATTR_CMN_RETURNED_ATTRS comes first when requested and says which of the rest
                    // actually follow, so nothing below is parsed on faith.
                    let returned = p.loadUnaligned(as: attribute_set_t.self)
                    p = p.advanced(by: MemoryLayout<attribute_set_t>.size)

                    var name = ""
                    var isDir = false
                    var isSymlink = false
                    var mtime: Double = 0
                    var size = 0
                    var device: dev_t = 0

                    if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_NAME)) != 0 {
                        let ref = p.loadUnaligned(as: attrreference_t.self)
                        name = String(cString: p.advanced(by: Int(ref.attr_dataoffset))
                            .assumingMemoryBound(to: CChar.self))
                        p = p.advanced(by: MemoryLayout<attrreference_t>.size)
                    }
                    // Attributes arrive in bitmap order, so DEVID is parsed between NAME and
                    // OBJTYPE - not where it was requested in the mask.
                    if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_DEVID)) != 0 {
                        device = p.loadUnaligned(as: dev_t.self)
                        p = p.advanced(by: MemoryLayout<dev_t>.size)
                    }
                    if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_OBJTYPE)) != 0 {
                        let t = p.loadUnaligned(as: fsobj_type_t.self)
                        isDir = t == fsobj_type_t(VDIR.rawValue)
                        isSymlink = t == fsobj_type_t(VLNK.rawValue)
                        p = p.advanced(by: MemoryLayout<fsobj_type_t>.size)
                    }
                    if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_MODTIME)) != 0 {
                        let ts = p.loadUnaligned(as: timespec.self)
                        mtime = Double(ts.tv_sec) + Double(ts.tv_nsec) / 1e9
                        p = p.advanced(by: MemoryLayout<timespec>.size)
                    }
                    if returned.fileattr & attrgroup_t(UInt32(ATTR_FILE_DATALENGTH)) != 0 {
                        size = Int(p.loadUnaligned(as: off_t.self))
                    }

                    if !name.isEmpty, name != ".", name != ".." {
                        out.append(Entry(name: name, isDir: isDir, isSymlink: isSymlink,
                                         size: size, mtime: mtime, device: device))
                    }
                    base = entryStart.advanced(by: entryLen)
                }
            }
        }
        return out
    }

    /// Shared state for the pool. A class because every worker mutates it under one lock; the lock
    /// is taken once per DIRECTORY, never per file, so 2.4M files cost ~200k acquisitions.
    private final class Shared: @unchecked Sendable {
        let lock = NSCondition()
        var stack: [String]
        var pending: Int          // directories claimed but not finished
        var stopped = false
        init(_ roots: [String]) { stack = roots; pending = roots.count }
    }

    /// Eight. See the note above: past this the pool buys clock with kernel CPU, and that CPU
    /// belongs to the embedding pipeline. OMNI_CRAWL_WORKERS overrides, for measuring.
    static var workerCount: Int {
        if let n = ProcessInfo.processInfo.environment["OMNI_CRAWL_WORKERS"].flatMap(Int.init), n > 0 { return n }
        return min(8, max(2, ProcessInfo.processInfo.activeProcessorCount))
    }

    /// Walk `roots`, calling `onEntry(directoryPath, entry)` for every entry that survives
    /// `shouldDescend` (directories) - the caller decides what to keep and what to recurse into.
    ///
    /// `onEntry` is called SERIALLY, under the pool's lock, one directory's worth at a time: the
    /// callers here append to arrays, and making them thread-safe individually would be both slower
    /// and easier to get wrong than holding the lock for a batch.
    /// `keep` runs IN THE WORKER, in parallel, and is where the caller's per-file policy belongs -
    /// extension matching, ignore rules, building whatever the caller wants. `deliver` is called
    /// under the lock with one directory's worth of survivors.
    ///
    /// The split is the whole point. Filtering inside `deliver` looks equivalent and is not: it puts
    /// the most expensive per-file work back under one lock, and the pool goes from 7.9x to 2.1x -
    /// measured, after building it that way first.
    static func walk<T>(roots: [String],
                        shouldContinue: () -> Bool,
                        shouldDescend: (String, Entry?) -> Bool,
                        keep: (String, Entry) -> T?,
                        deliver: ([T]) -> Void) {
        let live = roots.filter { shouldDescend($0, nil) }
        guard !live.isEmpty else { return }
        let shared = Shared(live)
        let workers = min(workerCount, max(1, live.count * 4))

        // Every worker needs its own read buffer; sharing one would serialise the syscall.
        DispatchQueue.concurrentPerform(iterations: workers) { _ in
            var buf = [UInt8](repeating: 0, count: bufferBytes)
            while true {
                shared.lock.lock()
                while shared.stack.isEmpty && !shared.stopped && shared.pending > 0 { shared.lock.wait() }
                if shared.stopped || (shared.stack.isEmpty && shared.pending <= 0) { shared.lock.unlock(); break }
                let dir = shared.stack.removeLast()
                shared.lock.unlock()

                // Cancellation is checked per directory, not per file: a directory is one syscall
                // batch, so this is already a fine granularity, and per-file would put a call to the
                // caller's closure in the inner loop.
                if !shouldContinue() {
                    shared.lock.lock(); shared.stopped = true; shared.lock.broadcast(); shared.lock.unlock()
                    break
                }

                let entries = list(dir, into: &buf) ?? []
                var subdirs: [String] = []
                var kept: [T] = []
                kept.reserveCapacity(entries.count)
                for e in entries {
                    if e.isDir && !e.isSymlink {
                        let child = dir + "/" + e.name
                        if shouldDescend(child, e) { subdirs.append(child) }
                    } else if let k = keep(dir, e) {
                        kept.append(k)
                    }
                }

                shared.lock.lock()
                if !kept.isEmpty { deliver(kept) }
                shared.stack.append(contentsOf: subdirs)
                // This directory is finished; its surviving children are new work.
                shared.pending += subdirs.count - 1
                shared.lock.broadcast()
                shared.lock.unlock()
            }
        }
    }
}
