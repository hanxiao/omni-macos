import XCTest
import SQLite3
@testable import OmniKit

/// Rebuild correctness for the filename sidecar.
///
/// The channel is a CONTENTLESS FTS5 index, which cannot be cleared with `DELETE FROM`. The
/// original rebuild used exactly that and discarded the error, so each rebuild appended a second
/// copy of the term index while `pathmap` (an ordinary table) was cleared and renumbered. A
/// surviving posting for rowid K then resolved to whatever file now held id K. Measured on a live
/// 2,666,141-file sidecar that had rebuilt once: 90.4% of returned rows were the wrong file.
///
/// These tests pin the two properties that failure violated: a rebuild forgets what it dropped,
/// and a rebuild does not grow the file by a copy each time. The sidecar's schema is unchanged -
/// the fix is in how the tables are cleared, not in how they are shaped.
final class LexicalIndexRebuildTests: XCTestCase {

    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("lexrebuild-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// The sidecar derives its own path from the index URL, the way VectorStore builds it.
    private func makeIndex() -> (LexicalIndex, URL) {
        let indexURL = dir.appendingPathComponent("index.sqlite")
        return (LexicalIndex(indexURL: indexURL), indexURL.appendingPathExtension("names"))
    }

    private func size(_ url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int) ?? 0
    }

    /// A path present only in the FIRST build must not be findable after a rebuild that drops it,
    /// and must never surface a different file in its place.
    func testRebuildForgetsRemovedPaths() throws {
        let (lex, _) = makeIndex()

        let first = ["/Users/me/Documents/quarterly-design-review.pdf",
                     "/Users/me/Documents/budget-2024.xlsx"]
        lex.rebuildIfStale(paths: first, stamp: 1)
        XCTAssertEqual(lex.match("design", limit: 10), [first[0]],
                       "the file that carries the term should be the only match")

        // Same store, later generation, and the "design" file is gone. The replacement list is a
        // different length so ids are reassigned - the exact shape that produced wrong answers.
        let second = ["/Users/me/Documents/holiday-photos.zip",
                      "/Users/me/Documents/budget-2024.xlsx",
                      "/Users/me/Documents/invoice-jina.pdf"]
        lex.rebuildIfStale(paths: second, stamp: 2)

        XCTAssertEqual(lex.match("design", limit: 10), [],
                       "a term from the dropped file must return nothing, not another file's path")
        XCTAssertEqual(lex.match("holiday", limit: 10), [second[0]])
        XCTAssertEqual(lex.match("invoice", limit: 10), [second[2]])
        // Every path the channel returns must still exist in the current set.
        for term in ["budget", "photos", "jina", "xlsx", "pdf"] {
            for hit in lex.match(term, limit: 10) {
                XCTAssertTrue(second.contains(hit), "\(term) returned a path outside the current set: \(hit)")
            }
        }
    }

    /// Rebuilding the identical path set repeatedly must not grow the file. Under the old reset the
    /// term index gained a full copy each pass (984 MB against 487 MB fresh, on the live index).
    func testRepeatedRebuildDoesNotGrowTheFile() throws {
        let (lex, sidecar) = makeIndex()
        // Enough distinct terms that a duplicated term index would be plainly visible in the size.
        let paths = (0..<4000).map { "/Users/me/Documents/report-\($0)-alpha\($0 % 97)-beta\($0 % 89).pdf" }

        lex.rebuildIfStale(paths: paths, stamp: 1)
        let afterFirst = size(sidecar)
        XCTAssertGreaterThan(afterFirst, 0)

        // Four more rebuilds of the same content, each with a fresh stamp so none is skipped.
        for stamp in Int64(2)...5 { lex.rebuildIfStale(paths: paths, stamp: stamp) }
        let afterFive = size(sidecar)

        XCTAssertLessThan(Double(afterFive), Double(afterFirst) * 1.5,
                          "five rebuilds of the same paths grew the sidecar from \(afterFirst) to \(afterFive) bytes")
        // And the content is still right, not merely small.
        XCTAssertEqual(lex.match("alpha3", limit: 50).isEmpty, false)
        for hit in lex.match("alpha3", limit: 50) {
            XCTAssertTrue(hit.contains("alpha3"), "stale posting resolved to \(hit)")
        }
    }

    /// A sidecar the shipped 0.7.2 wrote has pathmap(id, dir_id, name) and a dirs table. Every
    /// lookup here selects m.path, so without a repair that file would answer nothing for good -
    /// CREATE IF NOT EXISTS leaves a wrongly-shaped table alone. Only that shape is touched.
    func testSidecarFrom072IsRepairedInPlace() throws {
        let (_, sidecar) = makeIndex()

        // Build the 0.7.2 file by hand, including a matching stamp so a stamp-only gate would
        // adopt it untouched.
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(sidecar.path, &db), SQLITE_OK)
        for sql in ["CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);",
                    "CREATE VIRTUAL TABLE names USING fts5(name, content='', columnsize=0);",
                    "CREATE TABLE dirs(id INTEGER PRIMARY KEY, path TEXT NOT NULL);",
                    "CREATE TABLE pathmap(id INTEGER PRIMARY KEY, dir_id INTEGER NOT NULL, name TEXT NOT NULL);",
                    "INSERT INTO dirs(id, path) VALUES(1, '/Users/me/Documents');",
                    "INSERT INTO pathmap(id, dir_id, name) VALUES(1, 1, 'legacy-widget.txt');",
                    "INSERT INTO names(rowid, name) VALUES(1, 'legacy widget txt');",
                    "INSERT INTO meta(k,v) VALUES('stamp','9');",
                    "INSERT INTO meta(k,v) VALUES('layout','2');"] {
            XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK, "setup failed: \(sql)")
        }

        sqlite3_close(db)

        // Open it with the shipping code at the SAME stamp the file claims.
        let (lex, _) = makeIndex()
        lex.rebuildIfStale(paths: ["/Users/me/Documents/current-gadget.txt"], stamp: 9)

        XCTAssertEqual(lex.match("gadget", limit: 10), ["/Users/me/Documents/current-gadget.txt"],
                       "a 0.7.2-shaped sidecar must be repaired and rebuilt, not left dead")
        XCTAssertEqual(lex.match("legacy", limit: 10), [],
                       "the stale row from the old file must not survive the repair")

        // The table is back to the shape this code reads, and the stray dirs table is gone.
        var check: OpaquePointer?
        XCTAssertEqual(sqlite3_open(sidecar.path, &check), SQLITE_OK)
        var st: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(check,
            "SELECT count(*) FROM pragma_table_info('pathmap') WHERE name='path';", -1, &st, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_step(st), SQLITE_ROW)
        XCTAssertEqual(sqlite3_column_int(st, 0), 1, "pathmap must carry a path column again")
        sqlite3_finalize(st)

        var st2: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(check,
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='dirs';", -1, &st2, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_step(st2), SQLITE_ROW)
        XCTAssertEqual(sqlite3_column_int(st2, 0), 0, "the stray dirs table must be dropped")
        sqlite3_finalize(st2)
        sqlite3_close(check)
    }

    /// A sidecar written by 0.7.1 or earlier already has the right shape and must be left exactly
    /// alone: no forced rebuild, no file replacement, no migration.
    func testExistingSidecarIsAdoptedUnchanged() throws {
        let (lex, sidecar) = makeIndex()
        lex.rebuildIfStale(paths: ["/Users/me/Documents/a-widget.txt"], stamp: 7)
        let builtSize = size(sidecar)
        let builtMtime = (try? FileManager.default.attributesOfItem(atPath: sidecar.path)[.modificationDate]) as? Date

        // Reopen at the SAME stamp: the sidecar is current, so nothing should run.
        let (again, _) = makeIndex()
        again.rebuildIfStale(paths: ["/Users/me/Documents/a-widget.txt"], stamp: 7)
        XCTAssertEqual(again.match("widget", limit: 10), ["/Users/me/Documents/a-widget.txt"])
        XCTAssertEqual(size(sidecar), builtSize, "an up-to-date sidecar must not be rewritten")
        let afterMtime = (try? FileManager.default.attributesOfItem(atPath: sidecar.path)[.modificationDate]) as? Date
        XCTAssertEqual(builtMtime, afterMtime, "an up-to-date sidecar must not be touched at all")
    }
}
