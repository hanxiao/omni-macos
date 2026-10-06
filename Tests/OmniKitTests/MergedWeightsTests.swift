import XCTest
import MLX
@testable import OmniKit

/// The downloaded weights already carry the retrieval merge (omni-verify exportmerged). An adapter
/// left beside them, from an older Hugging Face download, must not be merged a second time.
final class MergedWeightsTests: XCTestCase {
    private func makeModel(_ dir: URL, merged: Bool) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir.appendingPathComponent("adapters/retrieval"), withIntermediateDirectories: true)
        let w = MLXArray(converting: [1, 2, 3, 4], [2, 2]).asType(.bfloat16)
        try save(arrays: ["language_model.layers.0.mlp.up_proj.weight": w],
                 metadata: merged ? ["format": "mlx", "omni": "retrieval-lora-merged"] : ["format": "mlx"],
                 url: dir.appendingPathComponent("model.safetensors"))
        // delta = B @ A = [[1, 1], [1, 1]]
        try save(arrays: ["base_model.model.language_model.layers.0.mlp.up_proj.lora_A.weight": MLXArray(converting: [1, 1], [1, 2]),
                          "base_model.model.language_model.layers.0.mlp.up_proj.lora_B.weight": MLXArray(converting: [1, 1], [2, 1])],
                 url: dir.appendingPathComponent("adapters/retrieval/adapter_model.safetensors"))
    }

    private func weight(_ merged: Bool) throws -> [Float] {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("merged-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try makeModel(dir, merged: merged)
        let store = try WeightStore(modelDir: dir)
        return store["language_model.layers.0.mlp.up_proj.weight"].asType(.float32).asArray(Float.self)
    }

    func testMergedWeightsIgnoreAStrayAdapter() throws {
        XCTAssertEqual(try weight(true), [1, 2, 3, 4])
    }

    func testUnmergedWeightsTakeTheAdapter() throws {
        XCTAssertEqual(try weight(false), [2, 3, 4, 5])
    }
}
