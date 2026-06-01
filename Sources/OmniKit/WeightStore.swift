import Foundation
import MLX

/// Loads jina-embeddings-v5-omni-small-mlx weights and merges the retrieval LoRA
/// adapter at load time, mirroring `utils.JinaMultiTaskModel` + `sanitize`:
///   - `language_model.*` upcast bf16 -> fp32 (the reference does this for fidelity)
///   - retrieval LoRA merged in place: W += (alpha/r) * (B @ A)
///   - vision/merger kept in their stored dtype; audio dropped (not used here)
public struct WeightStore {
    public private(set) var weights: [String: MLXArray]

    public subscript(_ key: String) -> MLXArray { weights[key]! }
    public func has(_ key: String) -> Bool { weights[key] != nil }

    /// - Parameters:
    ///   - modelDir: directory with model.safetensors and adapters/retrieval/adapter_model.safetensors
    ///   - loraScale: alpha / r (retrieval = 1.0)
    ///   - keepVision: also keep vision_tower.* / merger.* (image path); audio always dropped
    public init(modelDir: URL, loraScale: Float = 1.0, keepVision: Bool = true) throws {
        var w = try loadArrays(url: modelDir.appendingPathComponent("model.safetensors"))

        // Drop modalities we do not run.
        for key in Array(w.keys) {
            if key.contains("position_ids")
                || key.hasPrefix("audio_tower.")
                || key.hasPrefix("audio_projector.")
                || (!keepVision && (key.hasPrefix("vision_tower.") || key.hasPrefix("merger.")))
            {
                w.removeValue(forKey: key)
            }
        }

        // Upcast the language backbone to fp32 (matches reference sanitize()).
        for key in Array(w.keys) where key.hasPrefix("language_model.") {
            if w[key]!.dtype != .float32 {
                w[key] = w[key]!.asType(.float32)
            }
        }

        // Merge the retrieval LoRA adapter into the backbone.
        let adapterURL = modelDir
            .appendingPathComponent("adapters/retrieval/adapter_model.safetensors")
        if FileManager.default.fileExists(atPath: adapterURL.path) {
            let adapter = try loadArrays(url: adapterURL)
            for (key, aArr) in adapter where key.contains("lora_A") {
                let baseKey = key
                    .replacingOccurrences(of: "base_model.model.", with: "")
                    .replacingOccurrences(of: ".lora_A.weight", with: ".weight")
                let bKey = key.replacingOccurrences(of: "lora_A", with: "lora_B")
                guard let bArr = adapter[bKey], let base = w[baseKey] else { continue }
                let a = aArr.asType(.float32)        // [r, in]
                let b = bArr.asType(.float32)         // [out, r]
                let delta = matmul(b, a)              // [out, in]
                w[baseKey] = base + (loraScale * delta)
            }
        }

        // Force-evaluate so the merge cost is paid once, not on first encode.
        eval(Array(w.values))
        self.weights = w
    }
}
