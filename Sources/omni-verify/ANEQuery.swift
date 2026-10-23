import CoreGraphics
import CoreML
import Foundation
import ImageIO
import MLX
import OmniKit

/// The app's Neural Engine text backend measured: latency by bucket, and parity with the GPU path
/// and an fp32 encoder over the eval's queries and the bed's passages (index.md, "The Neural
/// Engine, measured again"). `anetext` runs a package built by Tools/ane/build_*_text.py;
/// `aneprogram` the program as the app builds it on the Mac from its own weights (ANEProgram).
@available(macOS 15.0, *)
enum ANEQuery {
    static func evalQueries() -> [String] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var out: [String] = []
        for f in ["Tools/eval/bed.json", "Tools/eval/queries-generated.json"] {
            guard let d = try? Data(contentsOf: root.appendingPathComponent(f)),
                  let j = try? JSONSerialization.jsonObject(with: d) else { continue }
            let list = (j as? [String: Any])?["queries"] as? [[String: Any]] ?? (j as? [[String: Any]]) ?? []
            out += list.compactMap { $0["q"] as? String }
        }
        return out
    }

    /// The app's Neural Engine backend (OmniKit.ANETextBackend) on a built package:
    /// compile and load time, latency by bucket, and parity with the GPU path and an fp32 encoder
    /// for the eval's queries and the bed's passages.
    ///   omni-verify anetext <nanoModelDir> <nano_text.mlpackage> <docs dir>
    static func textBackend(modelDir: URL, package: URL, docs: URL) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        let saved = ["OMNI_BACKBONE_DTYPE", "OMNI_BF16_COMPUTE", "OMNI_COMPUTE_DTYPE"].map { ($0, ProcessInfo.processInfo.environment[$0]) }
        setenv("OMNI_BACKBONE_DTYPE", "fp32", 1); setenv("OMNI_BF16_COMPUTE", "0", 1); setenv("OMNI_COMPUTE_DTYPE", "fp32", 1)
        let wsRef = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let ref = try await OmniTextEncoder(modelDir: modelDir, weights: wsRef, config: cfg)
        for (k, v) in saved { if let v { setenv(k, v, 1) } else { unsetenv(k) } }
        var t0 = Date()
        // A compiled program at a fixed place, as the app keeps it: ANETEXT_COMPILED=<dir>.
        var compiled = try await MLModel.compileModel(at: package)
        if let keep = ProcessInfo.processInfo.environment["ANETEXT_COMPILED"] {
            let dst = URL(fileURLWithPath: keep)
            if !FileManager.default.fileExists(atPath: dst.path) { try FileManager.default.moveItem(at: compiled, to: dst) }
            compiled = dst
        }
        print(String(format: "compile %.1f s", -t0.timeIntervalSinceNow))
        t0 = Date()
        let lengths = ProcessInfo.processInfo.environment["ANETEXT_LENGTHS"].map { $0.split(separator: ",").compactMap { Int($0) } }
            ?? [32, 64, 128, 256, 512, 1024, 2048]
        let ane = try ANETextBackend(compiled: compiled, config: cfg, weights: ws, lengths: lengths)
        print(String(format: "load %.1f s: ", -t0.timeIntervalSinceNow) + ane.loadSeconds.sorted { $0.key < $1.key }.map { String(format: "L%d %.1fs", $0.key, $0.value) }.joined(separator: " "))
        var texts: [String] = []
        let it = FileManager.default.enumerator(at: docs, includingPropertiesForKeys: nil)
        while let u = it?.nextObject() as? URL, texts.count < 150 {
            guard ["txt", "md"].contains(u.pathExtension.lowercased()), let t = try? String(contentsOf: u, encoding: .utf8) else { continue }
            var i = t.startIndex
            while i < t.endIndex, texts.count < 150 {
                let j = t.index(i, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
                if t.distance(from: i, to: j) > 200 { texts.append(String(t[i ..< j])) }
                i = j
            }
        }
        runTextBackend(ane: ane, enc: enc, ref: ref, lengths: lengths, queries: evalQueries(), passages: texts)
    }

    static func runTextBackend(ane: ANETextBackend, enc: OmniTextEncoder, ref: OmniTextEncoder, lengths: [Int],
                               queries: [String], passages: [String]) {
        var rng = SystemRandomNumberGenerator()
        for L in lengths {
            let ids = (0 ..< L).map { _ in Int.random(in: 1000 ..< 100_000, using: &rng) }
            var t = Date(); _ = ane.encode(ids)
            let first = -t.timeIntervalSinceNow * 1000
            var ms: [Double] = []
            for _ in 0 ..< 10 { t = Date(); _ = ane.encode(ids); ms.append(-t.timeIntervalSinceNow * 1000) }
            ms.sort()
            print(String(format: "L%-5d first %7.1f ms  then p50 %6.2f ms  => %6.0f tok/s", L, first, ms[5], Double(L) / ms[5] * 1000))
        }
        func cos(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
        }
        for (label, texts, type) in [("queries", queries, OmniInputType.query), ("passages", passages, .passage)] {
            var a: [Float] = [], g: [Float] = [], lens: [Int] = []
            for text in texts {
                let ids = enc.tokenIds(text, type)
                guard let v = ane.encode(ids) else { continue }
                let r = ref.encodeTokenBatch([ids])[0]
                a.append(cos(v, r)); g.append(cos(enc.encodeTokenBatch([ids])[0], r)); lens.append(ids.count)
            }
            let sa = a.sorted(), sg = g.sorted()
            print(String(format: "%@ n=%d tokens median %d max %d | ANE vs fp32 min %.5f median %.5f | GPU vs fp32 min %.5f median %.5f",
                         label as NSString, a.count, lens.sorted()[lens.count / 2], lens.max() ?? 0, sa.first ?? .nan, sa[sa.count / 2], sg.first ?? .nan, sg[sg.count / 2]))
        }
    }

    /// The program as the app builds it, from the app's own weights (OmniKit.ANEProgram), then the
    /// same checks as `anetext`: omni-verify aneprogram <nanoModelDir> <cache dir> <docs dir>
    static func onDevice(modelDir: URL, cache: URL, docs: URL) async throws {
        let cfg = try OmniConfig(modelDir: modelDir)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let enc = try await OmniTextEncoder(modelDir: modelDir, weights: ws, config: cfg)
        let saved = ["OMNI_BACKBONE_DTYPE", "OMNI_BF16_COMPUTE", "OMNI_COMPUTE_DTYPE"].map { ($0, ProcessInfo.processInfo.environment[$0]) }
        setenv("OMNI_BACKBONE_DTYPE", "fp32", 1); setenv("OMNI_BF16_COMPUTE", "0", 1); setenv("OMNI_COMPUTE_DTYPE", "fp32", 1)
        let wsRef = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let ref = try await OmniTextEncoder(modelDir: modelDir, weights: wsRef, config: cfg)
        for (k, v) in saved { if let v { setenv(k, v, 1) } else { unsetenv(k) } }
        var t0 = Date()
        guard let name = OmniEngine.aneProgramName(cfg), let compiled = ANEProgram.compiled(name, weights: ws, cache: cache) else {
            print("ANEProgram: no program (see the perf log)"); return
        }
        print(String(format: "program ready in %.1f s at %@", -t0.timeIntervalSinceNow, compiled.path))
        t0 = Date()
        let lengths = [32, 64, 128, 256, 512, 1024, 2048]
        let ane = try ANETextBackend(compiled: compiled, config: cfg, weights: ws, lengths: lengths)
        print(String(format: "load %.1f s", -t0.timeIntervalSinceNow))
        var texts: [String] = []
        let it = FileManager.default.enumerator(at: docs, includingPropertiesForKeys: nil)
        while let u = it?.nextObject() as? URL, texts.count < 150 {
            guard ["txt", "md"].contains(u.pathExtension.lowercased()), let t = try? String(contentsOf: u, encoding: .utf8) else { continue }
            var i = t.startIndex
            while i < t.endIndex, texts.count < 150 {
                let j = t.index(i, offsetBy: 1800, limitedBy: t.endIndex) ?? t.endIndex
                if t.distance(from: i, to: j) > 200 { texts.append(String(t[i ..< j])) }
                i = j
            }
        }
        runTextBackend(ane: ane, enc: enc, ref: ref, lengths: lengths, queries: evalQueries(), passages: texts)
    }
}
