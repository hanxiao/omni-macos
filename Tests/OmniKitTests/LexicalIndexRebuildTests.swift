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
/// and a rebuild does not grow the file by a copy each time.
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

    /// The path map interns its directory, so every returned path is reassembled from two columns.
    /// `NSString`'s path API cannot be used for that split: it collapses "//", which would turn
    /// every `photos://` asset into an unusable "photos:/..." and a root-level "/foo.txt" into
    /// "//foo.txt". These are the shapes that would break silently.
    func testPathsRoundTripThroughTheInternedDirectory() throws {
        let (lex, _) = makeIndex()
        let paths = [
            "/Users/me/Documents/annual-widget-report.pdf",
            "/rootlevel-gizmo.txt",                                  // dir is "" after the split
            "photos://library/2A9C1F30-ABCD-4E5F/beach-sunset.heic",  // "//" must survive verbatim
            "photos://library/7B3D2E41-1234-4A6B/beach-sunset.heic",  // same name, different asset
            "/Users/me/Documents/subdir/annual-widget-report.pdf",    // same name, different dir
        ]
        lex.rebuildIfStale(paths: paths, stamp: 1)

        XCTAssertEqual(lex.match("gizmo", limit: 10), ["/rootlevel-gizmo.txt"])
        XCTAssertEqual(Set(lex.match("sunset", limit: 10)), Set([paths[2], paths[3]]),
                       "photos:// paths must come back with their double slash intact")
        XCTAssertEqual(Set(lex.match("widget", limit: 10)), Set([paths[0], paths[4]]),
                       "two files sharing a basename must resolve to their own directories")
        // Nothing may be mangled: every path the channel can return must be one we put in.
        for term in ["report", "gizmo", "sunset", "heic", "pdf", "txt", "beach", "annual"] {
            for hit in lex.match(term, limit: 20) {
                XCTAssertTrue(paths.contains(hit), "\(term) produced a path that was never indexed: \(hit)")
            }
        }
    }

    /// `dirs` ids restart at 1 on every rebuild, so a surviving row from the previous build would
    /// make the insert a primary-key conflict and leave rows pointing at the OLD directory - the
    /// same stale-id failure the term index had. A second rebuild with different directories pins it.
    func testRebuildReassignsDirectoryIdsCleanly() throws {
        let (lex, _) = makeIndex()
        lex.rebuildIfStale(paths: ["/first/place/alpha-widget.txt",
                                   "/first/place/beta-widget.txt"], stamp: 1)
        XCTAssertEqual(lex.match("alpha", limit: 10), ["/first/place/alpha-widget.txt"])

        // Entirely different directories, so a reused dir id would surface the old ones.
        let second = ["/second/elsewhere/gamma-widget.txt", "/third/other/delta-widget.txt"]
        lex.rebuildIfStale(paths: second, stamp: 2)
        XCTAssertEqual(lex.match("gamma", limit: 10), ["/second/elsewhere/gamma-widget.txt"])
        XCTAssertEqual(lex.match("delta", limit: 10), ["/third/other/delta-widget.txt"])
        for hit in lex.match("widget", limit: 10) {
            XCTAssertTrue(second.contains(hit), "rebuild resolved a path through a stale directory id: \(hit)")
        }
    }

    /// A sidecar written by an older build carries no layout key, so it must be rebuilt even when
    /// its stamp still matches the store. Without this the fix would never reach existing users.
    func testOlderLayoutIsRebuiltEvenWhenTheStampMatches() throws {
        let (lex, _) = makeIndex()
        lex.rebuildIfStale(paths: ["/Users/me/a-widget.txt"], stamp: 7)
        XCTAssertEqual(lex.match("widget", limit: 10), ["/Users/me/a-widget.txt"])

        // Simulate the pre-fix file: same stamp, no layout key.
        let (stale, sidecar) = makeIndex()
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(sidecar.path, &db), SQLITE_OK)
        sqlite3_exec(db, "DELETE FROM meta WHERE k='layout';", nil, nil, nil)
        sqlite3_close(db)

        // Same stamp as the store: a stamp-only gate would adopt the file untouched.
        stale.rebuildIfStale(paths: ["/Users/me/b-gadget.txt"], stamp: 7)
        XCTAssertEqual(stale.match("widget", limit: 10), [],
                       "an older-layout sidecar must be rebuilt, not adopted")
        XCTAssertEqual(stale.match("gadget", limit: 10), ["/Users/me/b-gadget.txt"])
    }
}
