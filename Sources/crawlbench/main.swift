import Foundation
import OmniKit

// HOW FAST CAN THE WALK BE? The crawl gates the first embed - every root is walked to completion
// before anything is indexed - so this prices the walk itself, three ways, on real directories:
//
//   enumerator   what ships: FileManager.enumerator + one resourceValues read per entry.
//   bulk         getattrlistbulk(2): one syscall returns MANY entries WITH their attributes
//                (name, type, mtime, size), so a directory costs a handful of syscalls instead of
//                one readdir plus one stat per file.
//   bulk-par     the same, with subdirectories walked by a pool of workers.
//
// usage: crawlbench <root>...            parity + speed of the two SHIPPING engines
//        crawlbench --arms <root>...     every candidate, which is how the design was chosen
let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: crawlbench [--arms] <root>..."); exit(2) }
let allArms = args.contains("--arms")
let roots = args.dropFirst().filter { !$0.hasPrefix("--") }.map { URL(fileURLWithPath: $0) }

struct Entry { var path: String; var isDir: Bool; var size: Int; var mtime: Double }

// MARK: - getattrlistbulk

/// One directory's worth of entries, attributes included, in as few syscalls as the kernel allows.
func bulkList(_ dir: String) -> [Entry] {
    let fd = open(dir, O_RDONLY | O_DIRECTORY, 0)
    guard fd >= 0 else { return [] }
    defer { close(fd) }

    var attrs = attrlist()
    attrs.bitmapcount = u_short(ATTR_BIT_MAP_COUNT)
    attrs.commonattr = attrgroup_t(UInt32(ATTR_CMN_RETURNED_ATTRS) | UInt32(ATTR_CMN_NAME) | UInt32(ATTR_CMN_OBJTYPE) | UInt32(ATTR_CMN_MODTIME))
    attrs.fileattr = attrgroup_t(UInt32(ATTR_FILE_DATALENGTH))

    let cap = 512 * 1024
    var buf = [UInt8](repeating: 0, count: cap)
    var out: [Entry] = []

    while true {
        let got: Int = buf.withUnsafeMutableBytes { raw -> Int in
            Int(getattrlistbulk(fd, &attrs, raw.baseAddress, cap, 0))
        }
        if got <= 0 { break }
        buf.withUnsafeBytes { raw in
            var base = raw.baseAddress!
            for _ in 0 ..< got {
                let entryStart = base
                let len = entryStart.loadUnaligned(as: UInt32.self)
                var p = entryStart.advanced(by: 4)

                // ATTR_CMN_RETURNED_ATTRS always comes first when asked for: it says which of the
                // rest actually follow, so nothing is parsed on faith.
                let returned = p.loadUnaligned(as: attribute_set_t.self)
                p = p.advanced(by: MemoryLayout<attribute_set_t>.size)

                var name = ""
                var isDir = false
                var mtime: Double = 0
                var size = 0

                if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_NAME)) != 0 {
                    let ref = p.loadUnaligned(as: attrreference_t.self)
                    let strP = p.advanced(by: Int(ref.attr_dataoffset)).assumingMemoryBound(to: CChar.self)
                    name = String(cString: strP)
                    p = p.advanced(by: MemoryLayout<attrreference_t>.size)
                }
                if returned.commonattr & attrgroup_t(UInt32(ATTR_CMN_OBJTYPE)) != 0 {
                    isDir = p.loadUnaligned(as: fsobj_type_t.self) == fsobj_type_t(VDIR.rawValue)
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
                    out.append(Entry(path: dir + "/" + name, isDir: isDir, size: size, mtime: mtime))
                }
                base = entryStart.advanced(by: Int(len))
            }
        }
    }
    return out
}

/// PRUNED like the shipping crawler: an ignored directory is not descended into, and only files of
/// an enabled kind that survive the ignore rules are counted. Without this the bulk arms walked 3x
/// the entries (1.1M against 362k on one root) and the comparison flattered the enumerator.
/// The enumerator arm runs with .skipsHiddenFiles, so the bulk arms must too or they walk a
/// different tree (762k entries against 362k on one root) and the comparison is meaningless.
func isHidden(_ name: String) -> Bool { name.hasPrefix(".") }
func keepDir(_ path: String, _ ignore: OmniIgnore) -> Bool {
    !isHidden((path as NSString).lastPathComponent) && !ignore.isIgnored(path, isDir: true)
}
func keepFile(_ path: String, _ ignore: OmniIgnore, _ kinds: Set<FileKind>) -> Bool {
    guard !isHidden((path as NSString).lastPathComponent) else { return false }
    guard let k = FileExtractor.kind(for: URL(fileURLWithPath: path)), kinds.contains(k) else { return false }
    return !ignore.isIgnored(path, isDir: false)
}

func bulkWalk(_ root: String, _ ignore: OmniIgnore, _ kinds: Set<FileKind>, onFile: (Entry) -> Void) {
    var stack = [root]
    while let dir = stack.popLast() {
        for e in bulkList(dir) {
            if e.isDir {
                if keepDir(e.path, ignore) { stack.append(e.path) }
            } else if keepFile(e.path, ignore, kinds) {
                onFile(e)
            }
        }
    }
}

/// The same walk with directories handed to a pool. Directory reads are latency-bound, so the
/// question is whether overlapping them beats one thread saturating the queue.
final class WalkState: @unchecked Sendable {
    let lock = NSCondition()
    var stack: [String]
    var pending: Int
    var files = 0
    var done = false
    init(root: String) { stack = [root]; pending = 1 }
}

func bulkWalkParallel(_ root: String, workers: Int, _ ignore: OmniIgnore, _ kinds: Set<FileKind>) -> Int {
    let st = WalkState(root: root)
    let lock = st.lock

    DispatchQueue.concurrentPerform(iterations: workers) { _ in
        var local = 0
        while true {
            lock.lock()
            while st.stack.isEmpty && !st.done && st.pending > 0 { lock.wait() }
            if st.stack.isEmpty && (st.done || st.pending == 0) { lock.unlock(); break }
            let dir = st.stack.removeLast()
            lock.unlock()

            var subdirs: [String] = []
            var n = 0
            for e in bulkList(dir) {
                if e.isDir {
                    if keepDir(e.path, ignore) { subdirs.append(e.path) }
                } else if keepFile(e.path, ignore, kinds) {
                    n += 1
                }
            }
            local += n

            lock.lock()
            st.stack.append(contentsOf: subdirs)
            st.pending += subdirs.count - 1   // this one is finished; its children are new work
            if st.pending <= 0 && st.stack.isEmpty { st.done = true }
            lock.broadcast()
            lock.unlock()
        }
        lock.lock(); st.files += local; lock.broadcast(); lock.unlock()
    }
    return st.files
}


// MARK: - fts(3), which the Find Any File author measures as the fastest on APFS

/// fts_read gives names AND stat in one walk, and knows not to cross volumes. The comparison that
/// matters is against enumeratorAtURL, which the same author later found faster still on local
/// disks - the reason this bench exists rather than trusting either claim.
func ftsWalk(_ root: String, _ ignore: OmniIgnore, _ kinds: Set<FileKind>) -> Int {
    var n = 0
    root.withCString { rootC in
        let argv = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: 2)
        argv[0] = UnsafeMutablePointer(mutating: rootC)
        argv[1] = nil
        defer { argv.deallocate() }
        guard let fts = fts_open(argv, FTS_PHYSICAL | FTS_NOCHDIR, nil) else { return }
        defer { fts_close(fts) }
        while let ent = fts_read(fts) {
            let e = ent.pointee
            let path = String(cString: e.fts_path)
            switch Int32(e.fts_info) {
            case FTS_D:
                if !keepDir(path, ignore) { fts_set(fts, ent, FTS_SKIP) }
            case FTS_F:
                if keepFile(path, ignore, kinds) { n += 1 }   // fts_statp already carries size/mtime
            default:
                break
            }
        }
    }
    return n
}

/// The SHIPPING enumerator, but with top-level subtrees handed to workers. If parallelism is the
/// whole story, this is the cheap answer - no new syscall layer, no new parsing, same semantics.
func enumeratorWalkParallel(_ root: String, workers: Int, _ ignore: OmniIgnore, _ kinds: Set<FileKind>) -> Int {
    let fm = FileManager.default
    var subtrees: [URL] = []
    var topFiles = 0
    let rootURL = URL(fileURLWithPath: root)
    if let items = try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: [.isDirectoryKey],
                                               options: [.skipsHiddenFiles]) {
        for u in items {
            if (try? u.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                if keepDir(u.path, ignore) { subtrees.append(u) }
            } else if keepFile(u.path, ignore, kinds) {
                topFiles += 1
            }
        }
    }
    let counter = WalkState(root: root)
    counter.files = topFiles
    DispatchQueue.concurrentPerform(iterations: min(workers, max(1, subtrees.count))) { w in
        var local = 0
        var i = w
        while i < subtrees.count {
            FileCrawler(roots: [subtrees[i]], ignore: ignore, enabledKinds: kinds)
                .walk(shouldContinue: { true }) { _ in local += 1 }
            i += workers
        }
        counter.lock.lock(); counter.files += local; counter.lock.unlock()
    }
    return counter.files
}

/// Kernel vs user time, because a parallel walk on APFS can burn its gains inside a global lock
/// (Szorc, 2018: readdir takes one). A wall-clock win with runaway sys time is not a real win.
func cpuTimes() -> (user: Double, sys: Double) {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return (Double(u.ru_utime.tv_sec) + Double(u.ru_utime.tv_usec) / 1e6,
            Double(u.ru_stime.tv_sec) + Double(u.ru_stime.tv_usec) / 1e6)
}

// MARK: - arms

// THE USER'S OWN RULES, not the defaults. Both arms take the same OmniIgnore object, so the
// comparison stays fair either way - but the absolute numbers only mean something if the walk
// prunes what the app actually prunes. Directory rules matter most: an ignored directory is never
// descended into, which is where most of the saving comes from.
var settings = IndexSettings(enabledKinds: Set(FileKind.allCases))
let ignoreURL = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/Omni/.omniignore")
if let text = try? String(contentsOf: ignoreURL, encoding: .utf8) {
    settings.ignore = OmniIgnore(text: text)
    print("using .omniignore (\(text.split(separator: "\n").filter { !$0.hasPrefix("#") && !$0.isEmpty }.count) rules)")
} else {
    print("no .omniignore found; using defaults")
}
let cores = ProcessInfo.processInfo.activeProcessorCount

// PARITY FIRST, THEN SPEED. A faster walk that returns a different set of files is not a faster
// walk - the paths are the index's keys, and a difference means re-embedding or silently dropping.
for root in roots {
    print("\n\(root.path)")
    _ = bulkWalkParallel(root.path, workers: 8, settings.ignore, settings.enabledKinds)   // warm-up

    if allArms {
        // THE EVIDENCE. Kept runnable rather than written down, because the conclusion is specific
        // to this machine and this macOS: published benchmarks favour fts and enumeratorAtURL, and
        // parallelising the enumerator - the cheap-looking answer - measured 1.0x on an external
        // drive. Anyone doubting the design can re-run this instead of trusting the comment.
        func timed(_ label: String, _ body: () -> Int) {
            let c0 = cpuTimes(); let t = Date(); let n = body(); let s = -t.timeIntervalSinceNow; let c1 = cpuTimes()
            print(String(format: "  %-24@ %8d files %7.2fs  user %5.1fs sys %5.1fs", label as NSString, n, s,
                         c1.user - c0.user, c1.sys - c0.sys))
        }
        timed("enumerator 1T") {
            var n = 0
            setenv("OMNI_CRAWLER", "legacy", 1)
            FileCrawler(roots: [root], ignore: settings.ignore, enabledKinds: settings.enabledKinds).walk { _ in n += 1 }
            return n
        }
        timed("fts(3) 1T") { ftsWalk(root.path, settings.ignore, settings.enabledKinds) }
        timed("bulk 1T") {
            var n = 0
            bulkWalk(root.path, settings.ignore, settings.enabledKinds) { _ in n += 1 }
            return n
        }
        for w in [4, 8, 16] {
            timed("bulk + \(w) workers") { bulkWalkParallel(root.path, workers: w, settings.ignore, settings.enabledKinds) }
        }
        for w in [8, 16] {
            timed("enumerator + \(w) workers") { enumeratorWalkParallel(root.path, workers: w, settings.ignore, settings.enabledKinds) }
        }
    }

    func run(legacy: Bool) -> (files: [String: CrawledFile], secs: Double, user: Double, sys: Double) {
        setenv("OMNI_CRAWLER", legacy ? "legacy" : "bulk", 1)
        var out: [String: CrawledFile] = [:]
        let c0 = cpuTimes()
        let t = Date()
        FileCrawler(roots: [root], ignore: settings.ignore, enabledKinds: settings.enabledKinds)
            .walk { out[$0.path] = $0 }
        let secs = -t.timeIntervalSinceNow
        let c1 = cpuTimes()
        return (out, secs, c1.user - c0.user, c1.sys - c0.sys)
    }

    let old = run(legacy: true)
    let new = run(legacy: false)

    // The REAL predicate the indexer passes is `queue.sync { cancelled }` - a serial-queue hop, now
    // called from eight workers once per directory. If that throttles the pool, the pool is a lie.
    // In a CLASS, not a local var: top-level code here is main-actor isolated, and reading a
    // main-actor variable from the walker's worker threads traps in Swift 6 - which is a property
    // of this bench, not of the walker (Indexer is a plain class and unaffected).
    final class CancelProbe: @unchecked Sendable {
        let q = DispatchQueue(label: "cancel-probe")
        var flag = false
        var count = 0
        func check() -> Bool { q.sync { !flag } }
    }
    let probe = CancelProbe()
    setenv("OMNI_CRAWLER", "bulk", 1)
    let t3 = Date()
    FileCrawler(roots: [root], ignore: settings.ignore, enabledKinds: settings.enabledKinds)
        .walk(shouldContinue: { probe.check() }) { _ in probe.count += 1 }
    let s3 = -t3.timeIntervalSinceNow
    print(String(format: "  bulk + real cancel check %8d files  %7.2fs   (vs %.2fs with a free check)",
                 probe.count, s3, new.secs))
    print(String(format: "  enumerator (was)  %8d files  %7.2fs  user %5.1fs sys %5.1fs", old.files.count, old.secs, old.user, old.sys))
    print(String(format: "  bulk walker (now) %8d files  %7.2fs  user %5.1fs sys %5.1fs   %.1fx",
                 new.files.count, new.secs, new.user, new.sys, old.secs / max(new.secs, 0.001)))

    let onlyOld = Set(old.files.keys).subtracting(new.files.keys)
    let onlyNew = Set(new.files.keys).subtracting(old.files.keys)
    var badSize = 0, badTime = 0
    var examples: [String] = []
    for (p, o) in old.files {
        guard let n = new.files[p] else { continue }
        if o.size != n.size { badSize += 1 }
        if abs(o.modified - n.modified) > 0.001 { badTime += 1 }
        if o.size != n.size || abs(o.modified - n.modified) > 0.001, examples.count < 5 {
            // A live directory changes UNDER the two walks, so a difference here is only a bug if
            // the file did not actually change. Re-stat it now and say which story the numbers tell.
            var st = stat()
            let liveSize = stat(p, &st) == 0 ? Int(st.st_size) : -1
            let liveTime = stat(p, &st) == 0 ? Double(st.st_mtimespec.tv_sec) : -1
            examples.append(String(format: "    %@\n      enumerator size=%d mtime=%.1f\n      bulk       size=%d mtime=%.1f\n      now        size=%d mtime=%.1f",
                                   (p as NSString).lastPathComponent, o.size, o.modified, n.size, n.modified, liveSize, liveTime))
        }
    }
    if onlyOld.isEmpty && onlyNew.isEmpty && badSize == 0 && badTime == 0 {
        print("  PARITY: identical (paths, sizes, mtimes)")
    } else {
        print("  PARITY FAILED  missed \(onlyOld.count)  added \(onlyNew.count)  size \(badSize)  mtime \(badTime)")
        for p in onlyOld.sorted().prefix(4) { print("    only in enumerator: \(p)") }
        for p in onlyNew.sorted().prefix(4) { print("    only in bulk:       \(p)") }
        for e in examples { print(e) }
    }
}
