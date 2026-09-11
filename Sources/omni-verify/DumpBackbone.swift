import Foundation
import MLX
import OmniKit

/// Export the LoRA-merged text backbone so another runtime can be built from EXACTLY the weights
/// this app runs. Re-deriving the merge elsewhere is the obvious way to get a wrong answer that
/// looks like a numerics bug, so the merge is done once, here, by the shipping code path.
///
///   omni-verify dumpbackbone <modelDir> <out.safetensors>
///
/// `OMNI_BACKBONE_DTYPE=fp16` gives the dtype the Neural Engine needs.
enum DumpBackbone {
    static func run(modelDir: URL, out: URL) throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let store = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale,
                                    keepVision: false, keepAudio: false)
        var picked: [String: MLXArray] = [:]
        for (k, v) in store.weights where k.hasPrefix("language_model.") {
            picked[k] = v
        }
        try MLX.save(arrays: picked, url: out)
        let bytes = picked.values.reduce(0) { $0 + $1.nbytes }
        let dt = picked["language_model.layers.0.self_attn.q_proj.weight"]?.dtype
        print("""
            wrote \(picked.count) tensors, \(String(format: "%.2f", Double(bytes) / 1e9)) GB, \
            dtype \(dt.map { "\($0)" } ?? "?")
            layers=\(cfg.text.numLayers) hidden=\(cfg.text.hiddenSize) heads=\(cfg.text.numHeads) \
            kv=\(cfg.text.numKVHeads) headDim=\(cfg.text.headDim) eps=\(cfg.text.rmsNormEps) \
            theta=\(cfg.text.ropeTheta) causal=\(cfg.text.isCausal)
            """)
    }
}
