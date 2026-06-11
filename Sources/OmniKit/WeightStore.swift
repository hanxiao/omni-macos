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
    public init(modelDir: URL, loraScale: Float = 1.0, keepVision: Bool = true, keepAudio: Bool = true) throws {
        var w = try loadArrays(url: modelDir.appendingPathComponent("model.safetensors"))

        // Drop modalities we do not run.
        for key in Array(w.keys) {
            if key.contains("position_ids")
                || (!keepAudio && (key.hasPrefix("audio_tower.") || key.hasPrefix("audio_projector.")))
                || (!keepVision && (key.hasPrefix("vision_tower.") || key.hasPrefix("merger.")))
            {
                w.removeValue(forKey: key)
            }
        }

        let bf16Backbone = ProcessInfo.processInfo.environment["OMNI_BACKBONE_BF16"] != "0"

        // Load the retrieval LoRA adapter and compute which backbone weights it modifies, so the
        // fp32 round-trip is paid only where it matters.
        let adapterURL = modelDir
            .appendingPathComponent("adapters/retrieval/adapter_model.safetensors")
        let adapter = FileManager.default.fileExists(atPath: adapterURL.path)
            ? try loadArrays(url: adapterURL) : [:]
        var loraTargets = Set<String>()
        for key in adapter.keys where key.contains("lora_A") {
            loraTargets.insert(key
                .replacingOccurrences(of: "base_model.model.", with: "")
                .replacingOccurrences(of: ".lora_A.weight", with: ".weight"))
        }

        // Upcast to fp32 for the merge (reference sanitize() upcasts the whole backbone). In the
        // default bf16 path we only need fp32 on the LoRA-target linears - every other weight is
        // bf16->fp32->bf16, which is the identity, so upcasting the whole backbone (incl. the large
        // embed_tokens) is pure transient memory + time. The exact path (OMNI_BACKBONE_BF16=0, used
        // by the fp32 parity fixtures) still upcasts everything so the result stays fp32.
        for key in Array(w.keys) where key.hasPrefix("language_model.") {
            let needsFP32 = !bf16Backbone || loraTargets.contains(key)
            if needsFP32 && w[key]!.dtype != .float32 {
                w[key] = w[key]!.asType(.float32)
            }
        }

        // Merge the retrieval LoRA adapter into the backbone (in fp32).
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

        // Cast the fp32-merged weights back to bf16. In the default path only the LoRA targets were
        // upcast, so only they need casting back; non-target weights are already bf16. The result is
        // byte-identical to upcasting + casting the whole backbone, at a fraction of the load memory.
        if bf16Backbone {
            for key in loraTargets where w[key] != nil {
                w[key] = w[key]!.asType(.bfloat16)
            }
        }

        // Force-evaluate EVERY loaded tensor before any forward runs. Upstream norm (mlx-lm
        // loads with lazy=False and evals all parameters; mlx-swift-lm ends loadWeights with
        // eval(model)): MLX's lazy Load buffers are recycled-never-zeroed MTLBuffers filled by
        // pread on background thread pools, and a GPU consumer racing those reads sees garbage
        // (ml-explore/mlx#3329 is the crash-flavored sibling). Measured here: with only the
        // language backbone force-evaluated, 4 of 12 cold processes had persistent media-tower
        // corruption (2-37% per-embed NaN rates); the towers were exactly the tensors left to
        // materialize lazily mid-flight. Launch pays the tower read it previously deferred to
        // the first media embed; loadValidated's probes and recoverMediaPath() remain as the
        // behavioral backstops.
        eval(Array(w.values))
        self.weights = w
    }
}
