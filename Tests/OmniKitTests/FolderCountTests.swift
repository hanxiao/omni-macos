import XCTest
@testable import OmniKit

/// The stats tick's per-folder file counts are kept incrementally (indexSummary): a file whose
/// liveness flips is logged and applied at the next ask. They must equal the full pass over every
/// file (`fileCounts(underFolders:)`, which still does it the old way) after any sequence of
/// writes, on the cases the folder decision has: nested and overlapping folders, a folder that is
/// itself a file, names starting with a combining mark or a non-ASCII letter directly in a folder,
/// deletes, a folder delete, a collect, a reopen (the bulk load that settles), a new folder list.
final class FolderCountTests: XCTestCase {
    private static let dim = 32
    private func vec(_ seed: Int) -> [Float] {
        var s = UInt64(seed &* 2_654_435_761 &+ 777)
        var v = [Float](repeating: 0, count: Self.dim)
        for i in 0 ..< Self.dim { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
        let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / n }
    }
    private func put(_ store: VectorStore, _ path: String, _ seed: Int, chunks k: Int = 1) throws {
        try store.replace(path: path, chunks: (0 ..< k).map { j in
            IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: j, snippet: "s\(seed)-\(j)",
                         embedding: vec(seed * 7 + j), chunkKey: String(format: "%016x", seed * 7 + j))
        })
    }
    @discardableResult
    private func check(_ store: VectorStore, _ folders: [String], _ what: String,
                       file: StaticString = #filePath, line: UInt = #line) -> [String: Int] {
        let kept = store.indexSummary(folders: folders).folderCounts
        XCTAssertEqual(kept, store.fileCounts(underFolders: folders), what, file: file, line: line)
        return kept
    }

    // Names the folder decision treats specially, directly inside a folder and deeper.
    private let names = ["a.txt", "\u{301}accent.txt", "日本.txt", "Zeta.md", "e\u{301}.txt"]
    private let dirs = ["/r", "/r/sub", "/r/sub/deep", "/r/sub2", "/other", "/r/sub\u{301}x"]

    func testExactCountsByHand() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("t.sqlite"))
        defer { store.close() }
        let folders = ["/r", "/r/sub", "/r/a.txt"]
        try put(store, "/r/a.txt", 1)                 // a file, and one of the "folders"
        try put(store, "/r/sub/b.txt", 2, chunks: 3)
        try put(store, "/r/\u{301}c.txt", 3)          // combining mark first: not under "/r/"
        try put(store, "/other/d.txt", 4)
        XCTAssertEqual(check(store, folders, "start"), ["/r": 2, "/r/sub": 1, "/r/a.txt": 1])
        try put(store, "/r/sub/new.txt", 5)
        XCTAssertEqual(check(store, folders, "a file added"), ["/r": 3, "/r/sub": 2, "/r/a.txt": 1])
        try put(store, "/r/sub/b.txt", 6, chunks: 1)  // re-indexed: dies and comes back in one write
        XCTAssertEqual(check(store, folders, "a file re-indexed"), ["/r": 3, "/r/sub": 2, "/r/a.txt": 1])
        store.deletePaths(["/r/a.txt"])
        XCTAssertEqual(check(store, folders, "the file-folder deleted"), ["/r": 2, "/r/sub": 2, "/r/a.txt": 0])
        store.deleteUnderFolder("/r/sub")
        XCTAssertEqual(check(store, folders, "a folder deleted"), ["/r": 0, "/r/sub": 0, "/r/a.txt": 0])
    }

    // Both layouts: a row-keyed (v4) index reclaims by a whole-file copy that reloads the store,
    // which renumbers file ids in-session, as a wipe does.
    func testRandomWritesContentKeyed() throws { try random(legacy: false) }
    func testRandomWritesRowKeyed() throws { try random(legacy: true) }

    private func random(legacy: Bool) throws {
        let saved = VectorStore.legacyWriteForTest
        VectorStore.legacyWriteForTest = legacy
        defer { VectorStore.legacyWriteForTest = saved }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("t.sqlite")
        var store = try VectorStore(dbURL: url)
        defer { store.close() }
        var rng = SystemRandomNumberGenerator()
        var folders = ["/r", "/r/sub", "/other", "/r/sub/deep", "/r/sub2/a.txt"]
        var nonZero = 0
        func path() -> String {
            dirs.randomElement(using: &rng)! + "/" + names.randomElement(using: &rng)!
        }
        for step in 0 ..< 500 {
            switch Int.random(in: 0 ..< 21, using: &rng) {
            case 0 ..< 9:   try put(store, path(), Int.random(in: 0 ..< 1000, using: &rng),
                                    chunks: Int.random(in: 1 ... 3, using: &rng))
            case 9 ..< 13:  store.deletePaths([path()])
            case 13:        store.deleteUnderFolder(dirs.randomElement(using: &rng)!)
            case 14:        _ = store.reclaimVectorHolesForTest()
            case 15:        _ = store.search(vec(Int.random(in: 0 ..< 1000, using: &rng)), filter: SearchFilter(), topK: 3)
            case 16:        // the stats tick does not run between every write: let flips pile up
                for _ in 0 ..< 5 { try put(store, path(), Int.random(in: 0 ..< 1000, using: &rng)) }
                store.deletePaths([path(), path()])
            case 17:        // a different folder list: rebuilt, then kept again
                folders = Array(Set(dirs + ["/r/sub2/a.txt"]).shuffled(using: &rng).prefix(Int.random(in: 1 ... 6, using: &rng)))
            case 18:        // a reopen: the bulk load settles the aggregates
                store.close(); store = try VectorStore(dbURL: url)
            case 19:        // a wipe, in-session; rare, so files build up again after it
                if Int.random(in: 0 ..< 4, using: &rng) == 0 { store.wipeChunks() }
            default:        continue   // no write: the next ask must change nothing
            }
            let kept = check(store, folders, "step \(step)")
            if kept != store.fileCounts(underFolders: folders) { return }
            if kept.values.contains(where: { $0 > 0 }) { nonZero += 1 }
        }
        XCTAssertGreaterThan(nonZero, 250, "files under the folders in \(nonZero) checks")
    }

    /// The kept match is `filesUnder` answered one file at a time.
    func testMatchEqualsFilesUnder() {
        var t = PathTable()
        for d in dirs { for n in names { _ = t.append(d + "/" + n) } }
        _ = t.append("/r"); _ = t.append("/r/sub")   // folders that are files too
        for semantics in [PathTable.PrefixSemantics.string, .bytes] {
            for f in dirs + ["/r/a.txt", "/", "/r/su"] {
                var m = PathTable.FolderMatch(f, semantics: semantics)
                t.extend(&m)
                let all = t.filesUnder(f, semantics: semantics)
                for i in 0 ..< t.count {
                    XCTAssertEqual(t.contains(i, m), all[i], "\(f) \(semantics) \(t[i])")
                }
            }
        }
    }
}
