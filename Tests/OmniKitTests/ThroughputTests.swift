import XCTest
@testable import OmniKit

/// Measures real end-to-end text indexing throughput at different batch sizes (decode + embed
/// + store), so we can see how much cross-file batching actually helps in the app.
final class ThroughputTests: XCTestCase {
    func env(_ k: String, _ f: String) -> String { ProcessInfo.processInfo.environment[k] ?? f }

    func testTextThroughputByBatch() async throws {
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", "/private/tmp/omni-nano"))
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) else { throw XCTSkip("model absent: \(modelDir.path)") }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("omni-tput-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let N = 600
        let s = ["The quarterly revenue report shows strong cloud growth this year.",
                 "Paris is the capital and most populous city of France on the Seine.",
                 "A classic recipe for moist chocolate cake with cocoa and buttermilk.",
                 "Distributed systems require careful consistency and latency tradeoffs."]
        for i in 0 ..< N {
            try (s[i % s.count] + " Document number \(i) with some extra context.").write(to: root.appendingPathComponent("doc\(i).txt"), atomically: true, encoding: .utf8)
        }

        let engine = try await OmniEngine(modelDir: modelDir)
        func run(_ batch: Int) async throws -> (files: Int, sec: Double) {
            let store = try VectorStore(dbURL: root.appendingPathComponent("idx-\(batch)-\(UUID().uuidString).sqlite"))
            let idx = Indexer(store: store, embedder: engine)
            idx.textBatchSize = batch
            let exp = expectation(description: "done-\(batch)")
            let t0 = Date(); var sec = 0.0
            idx.index(roots: [root], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { sec = Date().timeIntervalSince(t0); exp.fulfill() } }
            await fulfillment(of: [exp], timeout: 900)
            return (store.fileCount, sec)
        }

        let b1 = try await run(1)
        let b48 = try await run(48)
        let r1 = Double(b1.files) / b1.sec, r48 = Double(b48.files) / b48.sec
        print(String(format: "THROUGHPUT batch=1 : %d files in %.2fs = %.0f files/s", b1.files, b1.sec, r1))
        print(String(format: "THROUGHPUT batch=48: %d files in %.2fs = %.0f files/s", b48.files, b48.sec, r48))
        print(String(format: "SPEEDUP end-to-end: %.2fx", r48 / r1))
        XCTAssertEqual(b1.files, N); XCTAssertEqual(b48.files, N)
    }

    func testStoreOnlyThroughput() throws {
        let fm = FileManager.default
        let dbURL = fm.temporaryDirectory.appendingPathComponent("store-only-\(UUID().uuidString).sqlite")
        defer { try? fm.removeItem(at: dbURL) }
        let store = try VectorStore(dbURL: dbURL)
        let dim = 768
        var vec = [Float](repeating: 0, count: dim); for i in 0 ..< dim { vec[i] = Float(i) / 768 }
        let N = 600
        let t0 = Date()
        for i in 0 ..< N {
            let path = "/tmp/doc\(i).txt"
            try store.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "d\(i)", embedding: vec)])
        }
        let sec = Date().timeIntervalSince(t0)
        print(String(format: "STORE-ONLY: %d files in %.3fs = %.3f ms/file = %.0f files/s", N, sec, sec / Double(N) * 1000, Double(N) / sec))
        XCTAssertEqual(store.fileCount, N)
    }

    func testDecodeOnlyThroughput() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("decode-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let N = 600
        let body = "The quarterly revenue report shows strong cloud growth this year. Document with extra context."
        var urls: [URL] = []
        for i in 0 ..< N {
            let u = root.appendingPathComponent("doc\(i).txt")
            try body.write(to: u, atomically: true, encoding: .utf8)
            urls.append(u)
        }
        let t0 = Date()
        for u in urls { _ = try FileExtractor.extract(u) }   // read + text extraction (no chunk/embed/store)
        let sec = Date().timeIntervalSince(t0)
        print(String(format: "DECODE-ONLY: %d files in %.3fs = %.3f ms/file = %.0f files/s", N, sec, sec / Double(N) * 1000, Double(N) / sec))
    }
}