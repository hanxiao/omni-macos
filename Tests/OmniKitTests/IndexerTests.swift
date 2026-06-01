import XCTest
@testable import OmniKit

/// End-to-end: crawl a temp folder, extract + embed text, store, and confirm
/// semantic search returns the right file for a paraphrased query.
final class IndexerTests: XCTestCase {
    func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }

    func testEndToEndSemanticSearch() async throws {
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", "/private/tmp/omni-model"))
        guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("model dir not found: \(modelDir.path)")
        }

        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("omni-itest-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let docs: [String: String] = [
            "france.txt": "Paris is the capital and most populous city of France, on the Seine river.",
            "germany.md": "Berlin is the capital of Germany and its largest city.",
            "dessert.txt": "A classic recipe for moist chocolate cake with cocoa and buttermilk.",
            "swift.swift": "func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float { return 0 }",
        ]
        for (name, body) in docs {
            try body.write(to: root.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }

        let dbURL = root.appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let engine = try await OmniEngine(modelDir: modelDir)
        let indexer = Indexer(store: store, embedder: engine)

        var final = IndexProgress()
        let done = expectation(description: "indexing done")
        let settings = IndexSettings(enabledKinds: [.text])
        indexer.index(roots: [root], settings: settings) { p in if p.done { final = p; done.fulfill() } }
        await fulfillment(of: [done], timeout: 120)

        XCTAssertEqual(store.fileCount, 4, "all four files indexed")
        XCTAssertGreaterThanOrEqual(final.embedded, 4)

        func topPath(_ q: String) -> String {
            let v = engine.embedText(q, as: .query)
            return store.search(v, topK: 1).first.map { URL(fileURLWithPath: $0.path).lastPathComponent } ?? "<none>"
        }

        XCTAssertEqual(topPath("What is the capital of France?"), "france.txt")
        XCTAssertEqual(topPath("which city is the German capital"), "germany.md")
        XCTAssertEqual(topPath("how do I bake a chocolate dessert"), "dessert.txt")
        XCTAssertEqual(topPath("function to compute vector similarity in swift"), "swift.swift")
    }
}
