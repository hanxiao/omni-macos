import XCTest
@testable import OmniKit

/// THE KIND A FILE IS: the kind of its first row, in the database (`files.kind`), in the counts the
/// filter menu is built from, and in what a pass is told the index holds. A screenshot is a picture
/// with text rows after it; it was counted in under "image" and out under "text", and a pass read
/// it as a text file.
final class FileKindTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("filekind-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    private func vec(_ seed: Int) -> [Float] {
        // 64 wide: a row file is adopted only at a width the scan's quantizer takes. At 8 this
        // test's "opened from the row file" was an open from the database, and passed.
        var v = [Float](repeating: 0.01, count: 64)
        v[seed % 64] = 1; v[(seed / 8) % 64] += 0.5
        return v
    }

    /// A file's rows: its kinds in order, one row each.
    private func rows(_ path: String, _ kinds: [String], seed: Int) -> [IndexedChunk] {
        kinds.enumerated().map { i, kind in
            IndexedChunk(path: path, modified: 5, size: 100, kind: kind, chunkIndex: i, snippet: "\(path) \(i)",
                         embedding: vec(seed * 8 + i), locator: kind == "text" ? "Text" : "")
        }
    }

    private let shots = ["/k/shot1.png", "/k/shot2.png", "/k/shot3.png"]
    private var files: [(path: String, kinds: [String])] {
        [("/k/a.txt", ["text", "text"]), ("/k/b.txt", ["text"])]
            + shots.map { ($0, ["image", "text", "text"]) }
            + [("/k/talk.mov", ["video", "video", "audio", "text"]),
               ("/k/report.pdf", ["text", "text", "scan"]),
               ("/k/scan.pdf", ["scan", "scan"]),
               ("/k/voice.m4a", ["audio"])]
    }

    private func fill(_ store: VectorStore) throws {
        for (i, f) in files.enumerated() { try store.replace(path: f.path, chunks: rows(f.path, f.kinds, seed: i + 1)) }
    }

    private func check(_ store: VectorStore, without gone: Set<String> = [], _ what: String, line: UInt = #line) {
        let live = files.filter { !gone.contains($0.path) }
        var expected: [String: Int] = [:]
        for f in live { expected[f.kinds[0], default: 0] += 1 }
        XCTAssertEqual(store.aggregatesForTest().kinds, expected, what, line: line)
        XCTAssertEqual(store.aggregatesForTest().live, live.count, what, line: line)
        XCTAssertEqual(store.kinds(), Set(expected.keys), what, line: line)
        let known = store.knownFiles()
        let inDatabase = store.storedFiles(paths: Set(files.map(\.path)))
        for f in files {
            let want = gone.contains(f.path) ? nil : f.kinds[0]
            XCTAssertEqual(known[f.path]?.kind, want, "\(what): \(f.path) as a pass is told", line: line)
            XCTAssertEqual(inDatabase[f.path]?.kind, want, "\(what): \(f.path) in the database", line: line)
        }
    }

    func testAFileIsCountedAsItsFirstRowsKindWhicheverRowGoesLast() throws {
        let url = dir.appendingPathComponent("index.sqlite")
        var store = try VectorStore(dbURL: url)
        try fill(store)
        check(store, "written")

        // Indexed again, and one of them twice: the counts do not move.
        try store.replace(path: shots[0], chunks: rows(shots[0], ["image", "text", "text"], seed: 40))
        try store.replace(path: shots[0], chunks: rows(shots[0], ["image", "text"], seed: 41))
        try store.replace(path: "/k/talk.mov", chunks: rows("/k/talk.mov", ["video", "video", "audio", "text"], seed: 42))
        check(store, "written again")

        // Opened from the row file, and from the database alone.
        store.stampRowSidecarForTest()
        store.close()
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        check(store, "opened from the row file")
        store.close()
        try FileManager.default.removeItem(atPath: url.path + ".rows")
        store = try VectorStore(dbURL: url)
        XCTAssertFalse(store.adoptedRowSidecar)
        check(store, "opened from the database")

        // Deleted: a screenshot leaves the pictures' count and not the text's.
        XCTAssertTrue(store.deletePaths(Set(shots)))
        check(store, without: Set(shots), "screenshots deleted")
        XCTAssertNil(store.aggregatesForTest().kinds["image"])
        XCTAssertEqual(store.fileCount(kind: "text"), 3, "two text files and a PDF")

        XCTAssertTrue(store.deletePaths(["/k/talk.mov", "/k/report.pdf"]))
        check(store, without: Set(shots + ["/k/talk.mov", "/k/report.pdf"]), "a video and a PDF deleted")

        // Written again after it was gone, as another kind of file: counted as what it is now.
        try store.replace(path: shots[1], chunks: rows(shots[1], ["text"], seed: 43))
        XCTAssertEqual(store.aggregatesForTest().kinds, ["text": 3, "scan": 1, "audio": 1])
        XCTAssertEqual(store.knownFiles()[shots[1]]?.kind, "text")
        store.close()
    }

    /// Purging a kind takes the files that are that kind, with every row they have, and no other
    /// file's rows of that kind.
    func testPurgingAKindTakesTheFilesOfThatKindWhole() throws {
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        try fill(store)
        store.deleteKind("text")
        let text: Set<String> = ["/k/a.txt", "/k/b.txt", "/k/report.pdf"]
        check(store, without: text, "text purged")
        XCTAssertEqual(store.storedChunks(path: shots[0])?.map(\.kind), ["image", "text", "text"], "a screenshot's words are the screenshot's")
        store.deleteKind("image")
        check(store, without: text.union(shots), "pictures purged")
        XCTAssertNil(store.storedChunks(path: shots[0]))
    }
}
