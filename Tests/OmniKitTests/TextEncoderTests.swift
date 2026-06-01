import XCTest
@testable import OmniKit

/// Numeric parity test against Python reference fixtures.
/// Run via xcodebuild (compiles the Metal shaders SwiftPM CLI cannot):
///   OMNI_MODEL_DIR=<snapshot> xcodebuild test -scheme Omni-Package -destination 'platform=macOS'
final class TextEncoderTests: XCTestCase {
    struct Record: Decodable {
        let text: String
        let query_token_ids: [Int]
        let passage_token_ids: [Int]
        let query_embedding: [Float]
        let passage_embedding: [Float]
    }
    struct Fixtures: Decodable { let records: [Record] }

    func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    func env(_ key: String, _ fallback: String) -> String {
        ProcessInfo.processInfo.environment[key] ?? fallback
    }

    func testTextEmbeddingsMatchReference() async throws {
        let defaultSnap = "/private/tmp/omni-model"
        let modelDir = URL(fileURLWithPath: env("OMNI_MODEL_DIR", defaultSnap))
        guard let fixturesURL = Bundle.module.url(forResource: "text_fixtures", withExtension: "json") else {
            throw XCTSkip("fixtures resource missing")
        }
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("model dir not found: \(modelDir.path)")
        }
        let fx = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: fixturesURL))

        let config = try OmniConfig(modelDir: modelDir)
        let weights = try WeightStore(modelDir: modelDir, loraScale: config.loraScale, keepVision: false)
        let encoder = try await OmniTextEncoder(modelDir: modelDir, weights: weights, config: config)

        var worstQ: Float = 1, worstP: Float = 1
        for r in fx.records {
            XCTAssertEqual(encoder.tokenIds(r.text, .query), r.query_token_ids, "query token ids: \(r.text.prefix(30))")
            XCTAssertEqual(encoder.tokenIds(r.text, .passage), r.passage_token_ids, "passage token ids: \(r.text.prefix(30))")
            let cq = cosine(encoder.encode(r.text, as: .query), r.query_embedding)
            let cp = cosine(encoder.encode(r.text, as: .passage), r.passage_embedding)
            worstQ = min(worstQ, cq); worstP = min(worstP, cp)
            print(String(format: "cosQ=%.5f cosP=%.5f  %@", cq, cp, String(r.text.prefix(40))))
        }
        print(String(format: "WORST cosQ=%.5f cosP=%.5f", worstQ, worstP))
        XCTAssertGreaterThanOrEqual(worstQ, 0.999, "query embedding parity")
        XCTAssertGreaterThanOrEqual(worstP, 0.999, "passage embedding parity")
    }
}
