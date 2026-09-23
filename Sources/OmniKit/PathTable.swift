import Foundation

/// The resident path table, shaped the way v5 stores paths: directories once, file names once, and
/// a directory id per file.
///
/// It replaces two collections that held every path as a Swift String - `idPath: [String]` (id ->
/// path) and `pathID: [String: Int32]` (path -> id). Measured on a 2,738,897-file index: 393 MB of
/// path text held in 549 MB of String objects, plus the two collections' own storage, for a set of
/// paths that is 89 MB of names over 36 MB of directories. That is the same duplication the
/// `dirs`/`files` split removed from the database, still present in memory.
///
/// WHAT IT PRESERVES EXACTLY
///
/// Lookup is by Swift's CANONICAL string equality, not bytes, because the store depends on it:
/// `storedSpellingLocked` maps an NFC spelling from a watcher event onto the NFD bytes an older
/// build wrote, and a byte-keyed table would call that file new and index it twice. So entries are
/// hashed with Swift's own String hash (which is consistent with canonical equality by the Hashable
/// contract), and a hash match is confirmed by bytes when both sides are ASCII - where the two
/// equalities coincide - and by String comparison otherwise.
///
/// Appending a path that is already a key does not deduplicate: it adds a second id and re-points
/// the key at it, exactly what `idPath.append(p); pathID[p] = id` did. Only a sidecar install can
/// do that (internPath looks up first), and only from a table this one wrote.
///
/// WHAT IT OFFERS THAT THE OLD PAIR COULD NOT
///
/// Byte access without building a String, and a directory id per file. A folder filter becomes one
/// decision per DIRECTORY: a name cannot contain "/", so a path lies under `folder/` exactly when
/// its directory part does. On the index above that is 279,097 decisions instead of 2,738,897
/// String prefix tests.
struct PathTable: Sendable {
    // Directories: the path up to and including its last "/", interned. "" for a path with none.
    private var dirBlob: [UInt8] = []
    private var dirEnd: [Int] = []           // dir d spans dirBlob[dirStart(d) ..< dirEnd[d]]
    private var dirHash: [UInt64] = []
    private var dirIndex: [Int32] = []       // open addressing, id + 1, 0 = empty

    // Files: the rest of the path, and which directory it sits in.
    private var nameBlob: [UInt8] = []
    private var nameEnd: [Int] = []
    private var fileDir: [Int32] = []
    private var fileHash: [Int] = []         // Swift's canonical String hash of the full path
    private var index: [Int32] = []          // open addressing over fileHash, id + 1, 0 = empty

    /// Number of ids, including any a duplicated append left unreachable by key. `idPath.count`.
    private(set) var count = 0
    /// Number of distinct keys. `pathID.count`.
    private(set) var keyCount = 0

    init() {}

    var isEmpty: Bool { count == 0 }
    var indices: Range<Int> { 0 ..< count }
    var dirCount: Int { dirEnd.count }

    // MARK: - Reading

    @inline(__always) private func dirStart(_ d: Int) -> Int { d == 0 ? 0 : dirEnd[d - 1] }
    @inline(__always) private func nameStart(_ i: Int) -> Int { i == 0 ? 0 : nameEnd[i - 1] }

    /// The path for id `i`, as a new String. One allocation; use the byte accessors in loops.
    subscript(i: Int) -> String {
        let d = Int(fileDir[i])
        let ds = dirStart(d), de = dirEnd[d], ns = nameStart(i), ne = nameEnd[i]
        let n = (de - ds) + (ne - ns)
        return String(unsafeUninitializedCapacity: n) { buf in
            dirBlob.withUnsafeBufferPointer { db in
                _ = UnsafeMutableBufferPointer(rebasing: buf[0 ..< de - ds])
                    .initialize(from: UnsafeBufferPointer(rebasing: db[ds ..< de]))
            }
            nameBlob.withUnsafeBufferPointer { nb in
                _ = UnsafeMutableBufferPointer(rebasing: buf[de - ds ..< n])
                    .initialize(from: UnsafeBufferPointer(rebasing: nb[ns ..< ne]))
            }
            return n
        }
    }

    @inline(__always) func dirID(_ i: Int) -> Int32 { fileDir[i] }

    /// A directory's bytes, including its trailing "/".
    @inline(__always) func withDirBytes<R>(_ d: Int, _ body: (UnsafeBufferPointer<UInt8>) -> R) -> R {
        let s = dirStart(d), e = dirEnd[d]
        return dirBlob.withUnsafeBufferPointer { body(UnsafeBufferPointer(rebasing: $0[s ..< e])) }
    }

    /// A file's name bytes: the path after its last "/".
    @inline(__always) func withNameBytes<R>(_ i: Int, _ body: (UnsafeBufferPointer<UInt8>) -> R) -> R {
        let s = nameStart(i), e = nameEnd[i]
        return nameBlob.withUnsafeBufferPointer { body(UnsafeBufferPointer(rebasing: $0[s ..< e])) }
    }

    /// Byte length of the full path.
    @inline(__always) func byteCount(_ i: Int) -> Int {
        let d = Int(fileDir[i])
        return (dirEnd[d] - dirStart(d)) + (nameEnd[i] - nameStart(i))
    }

    /// The canonical hash of path `i` - equal for any two paths that compare equal as Strings.
    @inline(__always) func hashOf(_ i: Int) -> Int { fileHash[i] }

    /// Does path `i` equal `s` under Swift's String equality? No allocation unless the hashes agree
    /// and one side is not ASCII.
    func equals(_ i: Int, _ s: String, hash: Int) -> Bool {
        guard fileHash[i] == hash else { return false }
        var s = s
        let fast: Bool? = s.withUTF8 { q -> Bool? in
            guard q.count == byteCount(i) else { return allASCII(q) && pathIsASCII(i) ? false : nil }
            let d = Int(fileDir[i])
            let dl = dirEnd[d] - dirStart(d)
            let dirSame = dl == 0 || withDirBytes(d) { memcmp($0.baseAddress, q.baseAddress, dl) == 0 }
            let nameSame = withNameBytes(i) { nb in
                nb.isEmpty || memcmp(nb.baseAddress, q.baseAddress! + dl, nb.count) == 0
            }
            if dirSame && nameSame { return true }
            return allASCII(q) && pathIsASCII(i) ? false : nil
        }
        if let fast { return fast }
        return self[i] == s
    }

    private func pathIsASCII(_ i: Int) -> Bool {
        withDirBytes(Int(fileDir[i])) { allASCII($0) } && withNameBytes(i) { allASCII($0) }
    }

    /// `self[i] < self[j]` under Swift's String ordering, without building either String when both
    /// are ASCII - there the ordering is plain byte order. A sort comparator runs this n log n times,
    /// so building two Strings per call turned a 15 ms listing into 81 ms at 100k files.
    func less(_ i: Int, _ j: Int) -> Bool {
        if let r = asciiCompare(i, j) { return r < 0 }
        return self[i] < self[j]
    }

    /// Byte comparison of two ASCII paths: <0, 0, >0. nil when either is not ASCII.
    private func asciiCompare(_ i: Int, _ j: Int) -> Int? {
        let di = Int(fileDir[i]), dj = Int(fileDir[j])
        return dirBlob.withUnsafeBufferPointer { db -> Int? in
            nameBlob.withUnsafeBufferPointer { nb -> Int? in
                let a0 = UnsafeBufferPointer(rebasing: db[dirStart(di) ..< dirEnd[di]])
                let a1 = UnsafeBufferPointer(rebasing: nb[nameStart(i) ..< nameEnd[i]])
                let b0 = UnsafeBufferPointer(rebasing: db[dirStart(dj) ..< dirEnd[dj]])
                let b1 = UnsafeBufferPointer(rebasing: nb[nameStart(j) ..< nameEnd[j]])
                let la = a0.count + a1.count, lb = b0.count + b1.count
                @inline(__always) func at(_ x0: UnsafeBufferPointer<UInt8>, _ x1: UnsafeBufferPointer<UInt8>,
                                          _ k: Int) -> UInt8 { k < x0.count ? x0[k] : x1[k - x0.count] }
                var k = 0
                while k < la && k < lb {
                    let x = at(a0, a1, k), y = at(b0, b1, k)
                    if x >= 0x80 || y >= 0x80 { return nil }
                    if x != y { return Int(x) - Int(y) }
                    k += 1
                }
                // A common prefix: the shorter sorts first - provided neither tail is non-ASCII.
                for q in k ..< la where at(a0, a1, q) >= 0x80 { return nil }
                for q in k ..< lb where at(b0, b1, q) >= 0x80 { return nil }
                return la - lb
            }
        }
    }

    // MARK: - Lookup

    /// The id a path maps to, by canonical equality. `pathID[p]`.
    func id(_ p: String) -> Int32? {
        guard !index.isEmpty else { return nil }
        let h = p.hashValue
        let mask = index.count - 1
        var b = Int(UInt(bitPattern: h) & UInt(mask))
        while true {
            let e = index[b]
            if e == 0 { return nil }
            if equals(Int(e - 1), p, hash: h) { return e - 1 }
            b = (b + 1) & mask
        }
    }

    /// The spelling stored for the key `p` matches, which may differ from `p` in normalization.
    func storedSpelling(_ p: String) -> String? { id(p).map { self[Int($0)] } }

    /// Look up, or append if absent. `internPath` without the side tables.
    mutating func intern(_ p: String) -> (id: Int32, isNew: Bool) {
        if let e = id(p) { return (e, false) }
        return (append(p), true)
    }

    // MARK: - Writing

    /// Add an id for `p` and point its key at it. Does NOT deduplicate - see the type comment.
    @discardableResult
    mutating func append(_ p: String) -> Int32 {
        let h = p.hashValue
        var p = p
        p.withUTF8 { q in
            // Split after the last "/": that is the directory part, the rest is the name.
            var cut = q.count
            while cut > 0, q[cut - 1] != UInt8(ascii: "/") { cut -= 1 }
            let d = internDir(UnsafeBufferPointer(rebasing: q[0 ..< cut]))
            fileDir.append(d)
            nameBlob.append(contentsOf: UnsafeBufferPointer(rebasing: q[cut ..< q.count]))
            nameEnd.append(nameBlob.count)
        }
        let id = Int32(count)
        fileHash.append(h)
        count += 1
        if (keyCount + 1) * 10 > index.count * 7 { growIndex() }
        // Re-point an existing equal key, or claim the first empty bucket.
        let mask = index.count - 1
        var b = Int(UInt(bitPattern: h) & UInt(mask))
        while true {
            let e = index[b]
            if e == 0 { index[b] = id + 1; keyCount += 1; break }
            if equals(Int(e - 1), p, hash: h) { index[b] = id + 1; break }
            b = (b + 1) & mask
        }
        return id
    }

    private mutating func growIndex() {
        var n = Swift.max(16, index.count * 2)
        while (keyCount + 1) * 10 > n * 7 { n *= 2 }
        var fresh = [Int32](repeating: 0, count: n)
        let mask = n - 1
        // Each bucket's CURRENT target, so a key re-pointed by a duplicate append stays re-pointed.
        for e in index where e != 0 {
            var b = Int(UInt(bitPattern: fileHash[Int(e - 1)]) & UInt(mask))
            while fresh[b] != 0 { b = (b + 1) & mask }
            fresh[b] = e
        }
        index = fresh
    }

    private mutating func internDir(_ q: UnsafeBufferPointer<UInt8>) -> Int32 {
        let h = Self.fnv(q)
        if (dirEnd.count + 1) * 10 > dirIndex.count * 7 { growDirIndex() }
        let mask = dirIndex.count - 1
        var b = Int(h & UInt64(mask))
        while true {
            let e = dirIndex[b]
            if e == 0 { break }
            let d = Int(e - 1)
            if dirHash[d] == h, dirEnd[d] - dirStart(d) == q.count,
               withDirBytes(d, { q.count == 0 || memcmp($0.baseAddress, q.baseAddress, q.count) == 0 }) {
                return Int32(d)
            }
            b = (b + 1) & mask
        }
        let d = Int32(dirEnd.count)
        dirBlob.append(contentsOf: q)
        dirEnd.append(dirBlob.count)
        dirHash.append(h)
        dirIndex[b] = d + 1
        return d
    }

    private mutating func growDirIndex() {
        var n = Swift.max(16, dirIndex.count * 2)
        while (dirEnd.count + 1) * 10 > n * 7 { n *= 2 }
        var fresh = [Int32](repeating: 0, count: n)
        let mask = n - 1
        for d in 0 ..< dirEnd.count {
            var b = Int(dirHash[d] & UInt64(mask))
            while fresh[b] != 0 { b = (b + 1) & mask }
            fresh[b] = Int32(d) + 1
        }
        dirIndex = fresh
    }

    mutating func removeAll(keepingCapacity keep: Bool = false) {
        dirBlob.removeAll(keepingCapacity: keep); dirEnd.removeAll(keepingCapacity: keep)
        dirHash.removeAll(keepingCapacity: keep)
        nameBlob.removeAll(keepingCapacity: keep); nameEnd.removeAll(keepingCapacity: keep)
        fileDir.removeAll(keepingCapacity: keep); fileHash.removeAll(keepingCapacity: keep)
        // The indexes are cleared, not shrunk, when capacity is kept: zeroing is the reset.
        if keep {
            for i in index.indices { index[i] = 0 }
            for i in dirIndex.indices { dirIndex[i] = 0 }
        } else {
            index = []; dirIndex = []
        }
        count = 0; keyCount = 0
    }

    mutating func reserveCapacity(_ files: Int) {
        fileDir.reserveCapacity(files); fileHash.reserveCapacity(files); nameEnd.reserveCapacity(files)
    }

    // MARK: - Folder membership

    /// How "under a folder" is judged. Both exist in the store today and each call site keeps its
    /// own: the search filter and folder delete compare BYTES (the way SQLite's range scan does),
    /// while the folder counts and the folder map use Swift's `String.hasPrefix`, which is
    /// canonical and grapheme-aware.
    enum PrefixSemantics { case bytes, string }

    /// For every id: `path == folder || path` lies under `folder + "/"`, judged by `semantics`.
    /// `==` is canonical String equality in both cases, as it was at every call site.
    ///
    /// ONE DECISION PER DIRECTORY, and that is exact rather than approximate: the folder's closing
    /// "/" must fall inside the directory part, because a name contains no "/". With byte semantics
    /// that settles it. With String semantics one case remains: a file sitting DIRECTLY in the
    /// folder whose name begins with a combining mark, where the full path clusters the mark onto
    /// the "/" and `hasPrefix` says no. Those files - direct children with a non-ASCII first byte -
    /// are judged on their full String; nothing else builds one.
    func filesUnder(_ folder: String, semantics: PrefixSemantics) -> [Bool] {
        var out = [Bool](repeating: false, count: count)
        guard count > 0 else { return out }
        let pfx = folder + "/"
        let pfxBytes = Array(pfx.utf8)
        let pfxASCII = pfxBytes.allSatisfy { $0 < 0x80 }
        var dirMatch = [Bool](repeating: false, count: dirCount)
        var dirExact = [Bool](repeating: false, count: dirCount)   // the dir IS the folder
        for d in 0 ..< dirCount {
            withDirBytes(d) { db in
                let bytePrefix = db.count >= pfxBytes.count
                    && memcmp(db.baseAddress, pfxBytes, pfxBytes.count) == 0
                switch semantics {
                case .bytes:
                    dirMatch[d] = bytePrefix
                case .string:
                    if pfxASCII && allASCII(db) {
                        dirMatch[d] = bytePrefix
                        dirExact[d] = bytePrefix && db.count == pfxBytes.count
                    } else {
                        let ds = String(decoding: db, as: UTF8.self)
                        dirMatch[d] = ds.hasPrefix(pfx)
                        dirExact[d] = dirMatch[d] && ds == pfx
                    }
                }
            }
        }
        for i in 0 ..< count {
            let d = Int(fileDir[i])
            var m = dirMatch[d]
            if semantics == .string, dirExact[d],
               withNameBytes(i, { !$0.isEmpty && $0[0] >= 0x80 }) {
                m = self[i].hasPrefix(pfx)
            }
            out[i] = m
        }
        if let e = id(folder) { out[Int(e)] = true }
        return out
    }

    /// Does the path end in "." + `ext`, ASCII-case-insensitively? `SearchFilter.hasExtensionCI` on
    /// the name bytes, which is the same answer: an extension cannot contain "/", so its dot and
    /// letters all sit in the name.
    func hasExtensionCI(_ i: Int, _ ext: [UInt8]) -> Bool {
        withNameBytes(i) { nb in
            guard nb.count >= ext.count + 1, nb[nb.count - ext.count - 1] == UInt8(ascii: ".") else {
                return false
            }
            let base = nb.count - ext.count
            for k in 0 ..< ext.count {
                var a = nb[base + k]; if a >= 65 && a <= 90 { a += 32 }
                var b = ext[k];       if b >= 65 && b <= 90 { b += 32 }
                if a != b { return false }
            }
            return true
        }
    }

    // MARK: - Sidecar encoding

    /// The row sidecar's string table, byte-identical to what `[String]` produced: UInt32 little-
    /// endian offsets, one per id plus the end, over the concatenated UTF-8. Built from the blobs
    /// directly, with no String per path.
    func sidecarTable() -> (offsets: Data, blob: Data) {
        var offs = [UInt8](); offs.reserveCapacity((count + 1) * 4)
        var blob = [UInt8](); blob.reserveCapacity(textBytes + count * 16)
        for i in 0 ..< count {
            withUnsafeBytes(of: UInt32(truncatingIfNeeded: blob.count).littleEndian) { offs.append(contentsOf: $0) }
            withDirBytes(Int(fileDir[i])) { blob.append(contentsOf: $0) }
            withNameBytes(i) { blob.append(contentsOf: $0) }
        }
        withUnsafeBytes(of: UInt32(truncatingIfNeeded: blob.count).littleEndian) { offs.append(contentsOf: $0) }
        return (Data(offs), Data(blob))
    }

    /// Decode that table. Each String lives only long enough to be hashed and split, so opening an
    /// index no longer holds every path twice at its peak. Appends in order, so duplicated entries
    /// re-point their key exactly as the dictionary assignment they replace did. nil on a
    /// malformed table.
    static func decode(offsets: Data, blob: Data, count n: Int,
                       progress: ((Double) -> Void)? = nil) -> PathTable? {
        guard offsets.count >= (n + 1) * 4 else { return nil }
        var t = PathTable()
        t.reserveCapacity(n)
        var ok = true
        offsets.withUnsafeBytes { op in
            blob.withUnsafeBytes { bp in
                var prev = op.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                for i in 1 ... max(1, n) where n > 0 {
                    let end = op.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)
                    guard end >= prev, Int(end) <= bp.count else { ok = false; return }
                    t.append(String(decoding: UnsafeRawBufferPointer(rebasing: bp[Int(prev) ..< Int(end)]),
                                    as: UTF8.self))
                    prev = end
                    if let progress, i % 131_072 == 0 { progress(Double(i) / Double(n)) }
                }
            }
        }
        return ok ? t : nil
    }

    /// `(self[i] as NSString).pathExtension.lowercased()`, read off the name's bytes.
    ///
    /// The index load asks this once per file - 2.7M times on a large index - and building the
    /// path String to hand to NSString was about half of the row rebuild's 2.2 s. Only a plain
    /// ASCII letters-and-digits extension is decided here; everything else (a leading dot as in
    /// `.bashrc`, a space, non-ASCII, a trailing dot) is answered by NSString itself, so the result
    /// is identical by construction. Checked against NSString on 2,739,258 real paths: 0 mismatches.
    func lowercasedExtension(_ i: Int) -> String {
        let ns = nameStart(i), ne = nameEnd[i]
        let fast: String? = nameBlob.withUnsafeBufferPointer { b in
            var dot = -1
            var j = ne - 1
            while j >= ns { if b[j] == UInt8(ascii: ".") { dot = j; break }; j -= 1 }
            if dot < 0 { return "" }
            if dot == ns || dot == ne - 1 { return nil }
            var out = [UInt8](); out.reserveCapacity(ne - dot - 1)
            for k in (dot + 1) ..< ne {
                let c = b[k]
                if (c >= 48 && c <= 57) || (c >= 97 && c <= 122) { out.append(c) }
                else if c >= 65 && c <= 90 { out.append(c + 32) }
                else { return nil }
            }
            return String(decoding: out, as: UTF8.self)
        }
        return fast ?? (self[i] as NSString).pathExtension.lowercased()
    }

    // MARK: - Accounting

    /// Every byte this table holds, by capacity - which is what is resident.
    var residentBytes: Int {
        dirBlob.capacity + nameBlob.capacity
            + (dirEnd.capacity + nameEnd.capacity + fileHash.capacity) * MemoryLayout<Int>.stride
            + dirHash.capacity * MemoryLayout<UInt64>.stride
            + (dirIndex.capacity + index.capacity + fileDir.capacity) * MemoryLayout<Int32>.stride
    }

    /// Bytes of path text, the part a shorter path would shrink.
    var textBytes: Int { dirBlob.count + nameBlob.count }

    // MARK: - Helpers

    @inline(__always) static func fnv(_ q: UnsafeBufferPointer<UInt8>) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in q { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }
}

@inline(__always) private func allASCII(_ q: UnsafeBufferPointer<UInt8>) -> Bool {
    for b in q where b >= 0x80 { return false }
    return true
}

/// A SearchFilter's PATH clauses compiled against a PathTable, answering per file id from bytes.
///
/// For code that asks about files LAZILY - the reducer tests only the files that could still make
/// the top K, which on a broad filter is a handful - where precomputing a whole [Bool] would cost
/// O(files) per query and a String per file would cost an allocation each. Compiling is
/// O(folders + tag members); each answer is a memcmp on the file's directory and an id lookup.
///
/// Exactly `SearchFilter.acceptsPath`, clause by clause: the folder clause is byte-prefix on the
/// directory part (a folder's closing "/" cannot fall in the name) plus canonical `==` resolved to
/// ids; the extension reads the name bytes; the tag sets are resolved to ids through the table's
/// canonical lookup.
struct CompiledPathFilter {
    private let table: PathTable
    private let folders: [[UInt8]]
    private let folderIDs: Set<Int32>
    private let ext: [UInt8]?
    private let extSlow: String?
    private let allow: Set<Int32>?
    private let deny: Set<Int32>?
    /// False when the filter has no path clause at all, so a caller can skip asking.
    let active: Bool

    init(_ f: SearchFilter, table: PathTable) {
        self.table = table
        folders = f.folderPrefixes.map { Array(($0 + "/").utf8) }
        folderIDs = Set(f.folderPrefixes.compactMap { table.id($0) })
        if let e = f.ext, !e.isEmpty {
            if e.contains("/") { ext = nil; extSlow = e } else { ext = Array(e.utf8); extSlow = nil }
        } else { ext = nil; extSlow = nil }
        allow = f.tagAllow.map { Set($0.compactMap { table.id($0) }) }
        deny = f.tagDeny.map { Set($0.compactMap { table.id($0) }) }
        active = !folders.isEmpty || ext != nil || extSlow != nil || allow != nil || deny != nil
    }

    func accepts(_ i: Int) -> Bool {
        guard active else { return true }
        guard i >= 0, i < table.count else { return false }
        if !folders.isEmpty {
            let d = Int(table.dirID(i))
            let under = table.withDirBytes(d) { db in
                folders.contains { pb in
                    db.count >= pb.count && memcmp(db.baseAddress, pb, pb.count) == 0
                }
            }
            if !under && !folderIDs.contains(Int32(i)) { return false }
        }
        if let e = ext, !table.hasExtensionCI(i, e) { return false }
        if let e = extSlow, !SearchFilter.hasExtensionCI(table[i], e) { return false }
        if let a = allow, !a.contains(Int32(i)) { return false }
        if let d = deny, d.contains(Int32(i)) { return false }
        return true
    }
}
