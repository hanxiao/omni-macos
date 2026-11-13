import XCTest
import SQLite3
@testable import OmniKit

/// A FILE'S SIZE AND LENGTH ARE ITS FIRST ROW'S: in what a search returns, in the row file, and in
/// the database, however the index was last written or opened. A screenshot is a picture row with
/// its size and text rows with none; the last row used to win in memory, so its size was gone once
/// its words were written and came back when the index opened from the database.
final class FileFactsTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("filefacts-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        VectorStore.lastRowFactsForTest = false
    }

    override func tearDownWithError() throws {
        VectorStore.lastRowFactsForTest = false
        try? FileManager.default.removeItem(at: dir)
    }

    private struct Facts: Equatable, CustomStringConvertible {
        var width = 0, height = 0, duration = 0.0
        var description: String { "\(width)x\(height) \(duration)s" }
    }

    /// One row: its kind and what that row alone carries.
    private typealias RowSpec = (kind: String, facts: Facts)
    private let files: [(path: String, rows: [RowSpec])] = [
        ("/f/shot.heic", [("image", Facts(width: 1920, height: 1080)), ("text", Facts()), ("text", Facts())]),
        ("/f/film.mov", [("video", Facts(width: 1280, height: 720, duration: 12.5)),
                         ("video", Facts(width: 1280, height: 720, duration: 12.5)),
                         ("audio", Facts(duration: 12.5)), ("text", Facts())]),
        ("/f/report.pdf", [("text", Facts()), ("text", Facts()), ("scan", Facts(width: 800, height: 1000))]),
        ("/f/a.txt", [("text", Facts()), ("text", Facts())]),
        ("/f/photo.jpg", [("image", Facts(width: 640, height: 480))]),
        ("/f/voice.m4a", [("audio", Facts(duration: 30))]),
    ]

    private func chunks(_ path: String, _ rows: [RowSpec], seed: Int) -> [IndexedChunk] {
        rows.enumerated().map { i, row in
            // 64 wide: a row file is adopted only at a width the scan's quantizer takes.
            var v = [Float](repeating: 0.01, count: 64)
            v[(seed + i) % 64] = 1; v[(seed / 8 + i * 7) % 64] += 0.5
            return IndexedChunk(path: path, modified: 5, size: 100, kind: row.kind, chunkIndex: i, snippet: "\(path) \(i)",
                                embedding: v, width: row.facts.width, height: row.facts.height,
                                duration: row.facts.duration, locator: row.kind == "text" ? "Text" : "")
        }
    }

    private func fill(_ store: VectorStore) throws {
        for (i, f) in files.enumerated() { try store.replace(path: f.path, chunks: chunks(f.path, f.rows, seed: i * 11 + 1)) }
    }

    /// What a search returns for each file, and what the database holds for it.
    private func read(_ store: VectorStore) -> (search: [String: Facts], database: [String: Facts]) {
        var search: [String: Facts] = [:], database: [String: Facts] = [:]
        for probe in 0 ..< 64 {
            var q = [Float](repeating: 0.05, count: 64)
            q[probe] = 1
            for hit in store.search(q, topK: 50) {
                let facts = Facts(width: hit.width, height: hit.height, duration: hit.duration)
                if let seen = search[hit.path] { XCTAssertEqual(seen, facts, "\(hit.path) answers two ways in one index") }
                search[hit.path] = facts
            }
        }
        for f in files {
            if let first = store.storedChunks(path: f.path)?.first {
                database[f.path] = Facts(width: first.width, height: first.height, duration: first.duration)
            }
        }
        return (search, database)
    }

    private func check(_ store: VectorStore, _ what: String, line: UInt = #line) {
        let got = read(store)
        for f in files {
            XCTAssertEqual(got.search[f.path], f.rows[0].facts, "\(what): \(f.path) in a search", line: line)
            XCTAssertEqual(got.database[f.path], f.rows[0].facts, "\(what): \(f.path) in the database", line: line)
        }
    }

    func testAFilesFactsAreItsFirstRowsHoweverTheIndexWasOpened() throws {
        let url = dir.appendingPathComponent("index.sqlite")
        var store = try VectorStore(dbURL: url)
        try fill(store)
        check(store, "written")

        // Written again with fewer rows, and with its rows the other way round.
        try store.replace(path: files[0].path, chunks: chunks(files[0].path, Array(files[0].rows.prefix(2)), seed: 90))
        try store.replace(path: files[0].path, chunks: chunks(files[0].path, files[0].rows, seed: 91))
        check(store, "written again")

        store.stampRowSidecarForTest()
        store.close()
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertNil(store.fileFactsCorrected, "a row file of this build is not checked against the database")
        check(store, "opened from the row file")
        store.close()
        try FileManager.default.removeItem(atPath: url.path + ".rows")
        store = try VectorStore(dbURL: url)
        XCTAssertFalse(store.adoptedRowSidecar)
        check(store, "opened from the database")
        store.close()
    }

    /// The row file a build before 0.20.3 left: the open corrects it from the database, once.
    func testARowFileHoldingTheLastRowsFactsIsCorrectedAtOpenOnce() throws {
        let url = dir.appendingPathComponent("index.sqlite")
        VectorStore.lastRowFactsForTest = true
        var store = try VectorStore(dbURL: url)
        try fill(store)
        // Written again: the vectors go to their file, without which no row file is written.
        try store.replace(path: files[0].path, chunks: chunks(files[0].path, files[0].rows, seed: 90))
        // NEGATIVE CONTROL: as it was, the screenshot has no size, the film no length, and the
        // PDF has its pictured page's size - while the database holds the first row's for each.
        var old = read(store)
        XCTAssertEqual(old.search["/f/shot.heic"], Facts())
        XCTAssertEqual(old.search["/f/film.mov"], Facts())
        XCTAssertEqual(old.search["/f/report.pdf"], Facts(width: 800, height: 1000))
        XCTAssertEqual(old.database["/f/shot.heic"], Facts(width: 1920, height: 1080))
        XCTAssertEqual(old.database["/f/report.pdf"], Facts())
        store.stampRowSidecarForTest()
        store.close()

        // The old build opening its own row file: still wrong, which is what a user had.
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        old = read(store)
        XCTAssertEqual(old.search["/f/shot.heic"], Facts())
        XCTAssertEqual(old.search["/f/report.pdf"], Facts(width: 800, height: 1000))
        store.close()

        // A file row nothing hangs from, with a size: not a reason to refuse the row file.
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(db, """
            INSERT INTO files(dir_id, name, modified, size, kind, width, height, duration, indexed_at, first_indexed_at)
              SELECT dir_id, 'gone.jpg', modified, size, kind, 640, 480, 0, indexed_at, first_indexed_at FROM files LIMIT 1;
            """, nil, nil, nil), SQLITE_OK)
        sqlite3_close(db)

        VectorStore.lastRowFactsForTest = false
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar, "corrected, not thrown away")
        XCTAssertEqual(store.fileFactsCorrected, 3, "the screenshot, the film and the PDF")
        check(store, "an older row file, opened")

        // Killed before the row file was written again: the older one is still there, and the
        // next open asks again and answers the same.
        let older = try Data(contentsOf: URL(fileURLWithPath: url.path + ".rows"))
        store.close()
        try older.write(to: URL(fileURLWithPath: url.path + ".rows"))
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertEqual(store.fileFactsCorrected, 3)
        check(store, "an older row file, opened a second time")

        // Written again by that open, so the next one asks nothing.
        store.stampRowSidecarForTest()
        store.close()
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertNil(store.fileFactsCorrected)
        check(store, "opened again")
        store.close()
    }

    /// An older row file with nothing wrong in it is asked about once too.
    func testAnOlderRowFileWithNothingToCorrectIsWrittenAgain() throws {
        let url = dir.appendingPathComponent("index.sqlite")
        VectorStore.lastRowFactsForTest = true
        var store = try VectorStore(dbURL: url)
        for f in files where f.rows.count == 1 || f.path == "/f/a.txt" {
            try store.replace(path: f.path, chunks: chunks(f.path, f.rows, seed: 3))
        }
        try store.replace(path: "/f/a.txt", chunks: chunks("/f/a.txt", files[3].rows, seed: 4))
        store.stampRowSidecarForTest()
        store.close()
        VectorStore.lastRowFactsForTest = false
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertEqual(store.fileFactsCorrected, 0)
        store.stampRowSidecarForTest()
        store.close()
        store = try VectorStore(dbURL: url)
        XCTAssertTrue(store.adoptedRowSidecar)
        XCTAssertNil(store.fileFactsCorrected)
        store.close()
    }
}
