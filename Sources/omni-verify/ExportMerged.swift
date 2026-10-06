import Foundation
import MLX
import OmniKit

/// omni-verify exportmerged <modelDir> <outDir>
///
/// Writes the weights exactly as WeightStore holds them after load (retrieval LoRA merged, backbone
/// bf16) as one model.safetensors with the runtime's config and tokenizer, then reloads that
/// directory through WeightStore and checks every tensor is bit-identical to the merge it came from.
/// A directory without adapters/ loads as-is, so the app reads the result with no merge at all.
func exportMergedRun(_ src: URL, _ out: URL) throws -> Int32 {
    let cfg = try OmniConfig(modelDir: src)
    let merged = try WeightStore(modelDir: src, loraScale: cfg.loraScale, keepVision: true, keepAudio: true)
    let fm = FileManager.default
    try fm.createDirectory(at: out, withIntermediateDirectories: true)
    let target = out.appendingPathComponent("model.safetensors")
    try? fm.removeItem(at: target)
    try save(arrays: merged.weights, metadata: ["format": "mlx", "omni": "retrieval-lora-merged"], url: target)
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let dst = out.appendingPathComponent(f)
        try? fm.removeItem(at: dst)
        try fm.copyItem(at: src.appendingPathComponent(f).resolvingSymlinksInPath(), to: dst)   // HF snapshots are symlinks
    }
    let reloaded = try WeightStore(modelDir: out, loraScale: cfg.loraScale, keepVision: true, keepAudio: true)
    var bad = 0
    for (k, a) in merged.weights {
        guard let b = reloaded.weights[k] else { print("missing \(k)"); bad += 1; continue }
        if a.dtype != b.dtype || a.shape != b.shape || !arrayEqual(a, b).item(Bool.self) {
            print("differs \(k) \(a.dtype) \(b.dtype) \(a.shape) \(b.shape)"); bad += 1
        }
    }
    if reloaded.weights.count != merged.weights.count { print("key count \(reloaded.weights.count) vs \(merged.weights.count)"); bad += 1 }
    let bytes = (try? fm.attributesOfItem(atPath: target.path)[.size] as? Int64) ?? 0
    print("exportmerged: \(merged.weights.count) tensors, \(bytes) bytes, \(bad == 0 ? "bit-identical on reload" : "\(bad) MISMATCHES")")
    return bad == 0 ? 0 : 1
}

/// omni-verify fetchmodel <nano|small> <outDir>: run the app's downloader into a scratch directory.
func fetchModelRun(_ variant: String, _ out: URL) async throws -> Int32 {
    guard let v = ModelVariant(rawValue: variant) else { print("unknown variant \(variant)"); return 2 }
    let t0 = Date()
    try await ModelDownloader().download(variant: v, to: out) { p in
        if p.received == p.total { print("\(p.file) \(p.total / 1_000_000) MB") }
    }
    print(String(format: "fetchmodel: %@ done in %.1f s", variant, Date().timeIntervalSince(t0)))
    return 0
}
