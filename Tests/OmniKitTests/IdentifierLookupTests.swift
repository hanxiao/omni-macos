import XCTest
import CoreGraphics
@testable import OmniKit

final class IdentifierLookupTests: XCTestCase {
    func testWhatCountsAsAnIdentifier() {
        // No exceptions by shape: a date, a reading or a size is found exactly like an order number.
        for t in ["INV-2041", "K7Q2LM", "OPS-0417", "4815162342", "114-3328-9912", "YN20260815", "M-417", "kv24", "JTMRFREV0HD123456",
                  "2026-03-01", "500ml", "1280x720", "4e-06"] {
            XCTAssertTrue(IDTokens.isIdentifier(t), t)
        }
        // Not a token at all (a '.' or ':' inside), too short, or too few digits with no letter.
        for t in ["21.4C", "101.3kPa", "2026-03-01T00:00:00Z", "v1.2.3", "110-115", "2026", "hello", "abc"] {
            XCTAssertFalse(IDTokens.isIdentifier(t), t)
        }
        XCTAssertEqual(IDTokens.tokens(in: "Invoice INV-2041, see ref/ref_layer0.py and 4.28e-06."), ["inv-2041", "ref_layer0", "28e-06"])
        XCTAssertEqual(IDTokens.tokens(in: "Meeting on 2026-03-01: bring 500ml."), ["2026-03-01", "500ml"])
    }

    /// The sidecars are built with the store, not lazily: the decode workers record identifiers
    /// concurrently, and a lazy first use from two threads built two indexes and freed one under the
    /// thread still using it (EXC_BAD_ACCESS in IDIndex.deinit, a fresh store's first pass).
    func testSidecarsTakeConcurrentFirstUse() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ids-race-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            store.idIndex.record(contentKey: StoreSchema.contentDigestHex("race-\(i)"), chunks: [["inv-\(1000 + i)"]])
            _ = store.enrichLedger
        }
        store.idIndex.flush()
        XCTAssertEqual(store.idIndex.lookup(["inv-1007"]).first?.contentKey, StoreSchema.contentDigestHex("race-7"))
    }

    final class Fixed: Embedder, @unchecked Sendable {
        let dim = 8
        private func v(_ s: String) -> [Float] {
            var out = [Float](repeating: 0, count: 8)
            for (i, b) in s.utf8.enumerated() { out[i % 8] += Float(b % 13) }
            let n = out.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return n > 0 ? out.map { $0 / n } : [1, 0, 0, 0, 0, 0, 0, 0]
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { v(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map(v) }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    func testALookupFindsTheFileAtItsChunkThroughRenameBackfillAndDelete() throws {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent("ids-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let rp = realpath(root.path, nil) { root = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let db = root.deletingLastPathComponent().appendingPathComponent("ids-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        // The code sits in the third chunk of a long note; a twin note has a near-miss code.
        let filler = (0 ..< 200).map { "Line \($0) of the weekly notes about planning and hiring." }.joined(separator: "\n")
        let note = root.appendingPathComponent("notes.txt")
        try (filler + "\nThe booking reference is ZX-90417 for the hotel.\n" + filler).write(to: note, atomically: true, encoding: .utf8)
        try (filler + "\nThe booking reference is ZX-90471 for the car.\n").write(to: root.appendingPathComponent("other.txt"), atomically: true, encoding: .utf8)
        let embedder = Fixed()
        var store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        func pass() {
            let indexer = Indexer(store: store, embedder: embedder)
            let done = expectation(description: "pass")
            indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { done.fulfill() } }
            wait(for: [done], timeout: 300)
        }
        func lookup(_ q: String, filter: SearchFilter = SearchFilter()) -> [SearchHit] {
            store.search(embedder.embedText(q, as: .query), filter: filter, topK: 20, textQuery: q)
        }
        pass()
        XCTAssertNotNil(store.contentKeys(paths: [note.path])[note.path], "content keys by path read nothing (MCP content_key)")
        var hits = lookup("ZX-90417")
        XCTAssertEqual(hits.first?.path, note.path)
        XCTAssertGreaterThan(hits.first?.chunkIndex ?? 0, 0, "opened at the first chunk, not the one holding the code")
        XCTAssertNotEqual(hits.dropFirst().first?.score ?? 0, hits.first?.score ?? 0)
        XCTAssertFalse(lookup("ZX-90417", filter: { var f = SearchFilter(); f.kinds = ["image"]; return f }()).contains { $0.path == note.path }, "the kind filter was ignored")

        // The sidecar lost: the next full pass reads the identifiers back without embedding.
        store.close()
        for ext in ["ids", "ids-wal", "ids-shm"] { try? FileManager.default.removeItem(at: db.appendingPathComponent("index.sqlite.\(ext)")) }
        store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        XCTAssertNotEqual(lookup("ZX-90417").first?.path, note.path, "found without the sidecar (control)")
        pass()
        hits = lookup("ZX-90417")
        XCTAssertEqual(hits.first?.path, note.path, "the backfill did not restore the lookup")

        // Renamed: the content key goes with it.
        let renamed = root.appendingPathComponent("hotel notes.txt")
        try FileManager.default.moveItem(at: note, to: renamed)
        pass()
        XCTAssertEqual(lookup("ZX-90417").first?.path, renamed.path)

        // Deleted: nothing to find; the key resolves to no file, and a sweep drops it.
        try FileManager.default.removeItem(at: renamed)
        pass()
        XCTAssertFalse(lookup("ZX-90417").contains { $0.path == renamed.path })
        store.idIndex.sweep(live: { store.liveContentDigests($0) })
        XCTAssertTrue(store.idIndex.lookup(["zx-90417"]).isEmpty, "a deleted file's identifiers survived the sweep")
    }
}
