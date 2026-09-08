import XCTest
import SQLite3
@testable import OmniKit

/// One file must never become two rows because its name is spelled differently.
///
/// Swift's String equality is CANONICAL: "Eigentümer" written NFC (U+00FC) and NFD (u + U+0308)
/// compare equal, hash equal, and are ONE key in pathID / presentPaths / indexedFiles(). SQLite
/// compares BYTES, so the same two spellings are TWO rows. The store reads through both views.
///
/// Both spellings really do occur in one index. Before 0.7.0 the indexer stored file.url.path, and
/// URL(fileURLWithPath:).path decomposes - it returns NFD for an NFC input as readily as for an
/// NFD one - so non-ASCII paths written then are NFD. Since 9ca0817 the crawler's raw string is
/// stored instead, which is the filesystem's own form and usually NFC.
///
/// The damage is not just a duplicate row. replaceMany finds the victim through pathID (canonical,
/// so it matches) and releases that file's vector slots as holes, while the byte-keyed DELETE
/// misses the row entirely - leaving holes recorded over chunk rows that are still live. That is
/// the state coverageAudit() refuses to open on.
final class PathNormalizationTests: XCTestCase {

    /// Deterministic vectors; the indexer only needs something finite and unit-length.
    final class StubEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        var interactiveQueryActive = false
        private func vec(_ text: String) -> [Float] {
            var s = UInt64(bitPattern: Int64(text.hashValue)) | 1
            var v = [Float](repeating: 0, count: 64)
            for i in 0 ..< 64 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17
                v[i] = Float(s % 2048) / 1024 - 1
            }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return n > 0 ? v.map { $0 / n } : v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { vec($0) } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pathnorm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    /// "Eigentümer-Vertragsnummer.pdf", the shape that actually collided on a live index.
    private var nfc: String { "/docs/Eigent\u{00FC}mer-Vertragsnummer.pdf" }
    private var nfd: String { "/docs/Eigentu\u{0308}mer-Vertragsnummer.pdf" }

    private func chunks(_ path: String, _ n: Int, modified: Double) -> [IndexedChunk] {
        (0..<n).map { i in
            var v = [Float](repeating: 0, count: 64)
            v[i % 64] = 1
            return IndexedChunk(path: path, modified: modified, size: 1234, kind: "text",
                                chunkIndex: i, snippet: "chunk \(i)", embedding: v, locator: "Line \(i + 1)")
        }
    }

    /// Raw row count, because indexedFiles() is keyed by Swift String and would fold the two
    /// spellings into one entry - hiding exactly the duplication under test.
    private func fileRowCount(_ dbURL: URL) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbURL.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT count(*) FROM files;", -1, &st, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW ? Int(sqlite3_column_int(st, 0)) : -1
    }

    /// The exact sequence that corrupted the bookkeeping: a row written under the old NFD spelling,
    /// then a watcher-style rewrite of the same file under the on-disk NFC spelling.
    func testRewriteUnderTheOtherNormalizationDoesNotDuplicateTheFile() throws {
        let dbURL = dir.appendingPathComponent("index.sqlite")

        let store = try VectorStore(dbURL: dbURL)
        try store.replace(path: nfd, chunks: chunks(nfd, 3, modified: 1000))
        store.close()

        // A later watcher event carries the filesystem's spelling. Same file, different bytes.
        let store2 = try VectorStore(dbURL: dbURL)
        try store2.replaceMany([(path: nfc, chunks: chunks(nfc, 3, modified: 2000))])
        let audit = store2.coverageAudit()
        store2.close()

        XCTAssertNil(audit, "coverage bookkeeping broken after the rewrite: \(audit ?? "")")
        XCTAssertEqual(fileRowCount(dbURL), 1,
                       "the two spellings of one filename became two rows")

        // And the file is findable under either spelling, with the NEW content.
        let check = try VectorStore(dbURL: dbURL)
        defer { check.close() }
        XCTAssertEqual(check.storedFiles(paths: [nfc])[nfc]?.modified, 2000,
                       "lookup by the on-disk spelling must find the row")
        XCTAssertEqual(check.storedFiles(paths: [nfd])[nfd]?.modified, 2000,
                       "lookup by the stored spelling must find the same row")
    }

    /// Deleting by one spelling must remove the row written under the other, or the file lingers
    /// in the index after it is gone from disk.
    func testDeleteByTheOtherNormalizationRemovesTheRow() throws {
        let dbURL = dir.appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        try store.replace(path: nfd, chunks: chunks(nfd, 2, modified: 1000))
        store.deletePath(nfc)
        let audit = store.coverageAudit()
        store.close()

        XCTAssertNil(audit, "coverage bookkeeping broken after the delete: \(audit ?? "")")
        XCTAssertEqual(fileRowCount(dbURL), 0, "the row survived a delete under the other spelling")
    }

    /// End to end through the path that actually caused this: a real file on disk, a legacy row
    /// written under the decomposed spelling, and a watcher event carrying the filesystem's own
    /// spelling. Indexer.update resolves the file through storedFiles() - the SQL lookup that used
    /// to miss - so this is the exact sequence that produced holes over live rows on a live index.
    func testWatcherUpdateOnALegacyDecomposedRowDoesNotDuplicate() throws {
        let root = dir.appendingPathComponent("root", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        // Create the file with the PRECOMPOSED name, which is what the crawler will read back.
        let onDisk = root.appendingPathComponent("Eigent\u{00FC}mer-Vertrag.txt")
        try "annual statement for the owners association".write(to: onDisk, atomically: true, encoding: .utf8)
        let nfcPath = onDisk.path.precomposedStringWithCanonicalMapping
        let nfdPath = nfcPath.decomposedStringWithCanonicalMapping
        XCTAssertNotEqual(Array(nfcPath.utf8), Array(nfdPath.utf8), "test needs two distinct spellings")

        let dbURL = dir.appendingPathComponent("index.sqlite")
        // Stand in for a row written before 0.7.0, when the stored path came from URL.path (NFD).
        let seed = try VectorStore(dbURL: dbURL)
        try seed.replace(path: nfdPath, chunks: chunks(nfdPath, 2, modified: 1))
        seed.close()
        XCTAssertEqual(fileRowCount(dbURL), 1)

        // Now the watcher fires for that file, carrying the on-disk (precomposed) spelling.
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: StubEmbedder())
        indexer.update(paths: [nfcPath], settings: IndexSettings())
        let audit = store.coverageAudit()
        store.close()

        XCTAssertNil(audit, "coverage bookkeeping broken after the watcher update: \(audit ?? "")")
        XCTAssertEqual(fileRowCount(dbURL), 1,
                       "the watcher re-indexed the file as a second row under the other spelling")
    }

    /// An index with no such collision must behave exactly as before: ASCII paths never take the
    /// resolution branch, and two genuinely different files stay two rows.
    func testAsciiAndDistinctPathsAreUnaffected() throws {
        let dbURL = dir.appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        try store.replace(path: "/docs/budget.txt", chunks: chunks("/docs/budget.txt", 2, modified: 1))
        try store.replace(path: "/docs/design.txt", chunks: chunks("/docs/design.txt", 2, modified: 2))
        // Same basename, different directories: must not collapse.
        try store.replace(path: "/other/budget.txt", chunks: chunks("/other/budget.txt", 1, modified: 3))
        let audit = store.coverageAudit()
        store.close()

        XCTAssertNil(audit, "coverage bookkeeping broken: \(audit ?? "")")
        XCTAssertEqual(fileRowCount(dbURL), 3)

        let check = try VectorStore(dbURL: dbURL)
        defer { check.close() }
        XCTAssertEqual(check.storedFiles(paths: ["/docs/budget.txt"])["/docs/budget.txt"]?.modified, 1)
        XCTAssertEqual(check.storedFiles(paths: ["/other/budget.txt"])["/other/budget.txt"]?.modified, 3)
    }
}
