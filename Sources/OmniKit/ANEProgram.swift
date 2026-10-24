import CoreML
import CryptoKit
import Foundation
import MLX

/// A NEURAL ENGINE PROGRAM, BUILT ON THE USER'S MAC FROM THE WEIGHTS ALREADY THERE. The app ships the
/// program's graph (Resources/ANE/<name>.mlmodel, ~2 MB) and a recipe (<name>.recipe.json), never its
/// weights: the weight file is written here from the WeightStore - each blob a tensor as it is, times
/// 1/sqrt(head_dim), or rotated (Tools/ane/extract_recipe.py) - compiled once, and kept beside the
/// models. It is used only if the file written is byte for byte the one built and measured: its
/// SHA-256 must be the recipe's, or this returns nil and the GPU does the work.
@available(macOS 15.0, *)
public enum ANEProgram {
    struct Recipe: Decodable {
        struct Blob: Decodable {
            let tensor: String, op: String, bytes: Int
            var heads: Int?, scaled: Bool?, transpose: Bool?, value: Double?, count: Int?
        }
        let format: Int, weightBytes: Int, headDim: Int, blobs: [Blob], sha256: String
    }

    /// The compiled program for `name` ("nano_text"), from `cache` when it is there and current,
    /// else written, checked and compiled into it. Slow the first time (the compile and the Neural
    /// Engine's own first load are about 40 s for Nano); call off the main actor.
    public static func compiled(_ name: String, weights: WeightStore, cache: URL) -> URL? {
        guard let graphURL = Bundle.module.url(forResource: name, withExtension: "mlmodel", subdirectory: "ANE"),
              let recipeURL = Bundle.module.url(forResource: name + ".recipe", withExtension: "json", subdirectory: "ANE"),
              let recipe = try? JSONDecoder().decode(Recipe.self, from: Data(contentsOf: recipeURL)), recipe.format == 2 else { return nil }
        let fm = FileManager.default
        let dir = cache.appendingPathComponent(name, isDirectory: true)
        let stamp = dir.appendingPathComponent("weights.sha256")
        let model = dir.appendingPathComponent(name + ".mlmodelc", isDirectory: true)
        if (try? String(contentsOf: stamp, encoding: .utf8)) == recipe.sha256, fm.fileExists(atPath: model.path) { return model }
        try? fm.removeItem(at: dir)
        let work = cache.appendingPathComponent(name + ".building-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: work) }
        let pkg = work.appendingPathComponent(name + ".mlpackage", isDirectory: true)
        let data = pkg.appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)
        do {
            try fm.createDirectory(at: data.appendingPathComponent("weights", isDirectory: true), withIntermediateDirectories: true)
            try fm.copyItem(at: graphURL, to: data.appendingPathComponent("model.mlmodel"))
            try manifest.write(to: pkg.appendingPathComponent("Manifest.json"), atomically: true, encoding: .utf8)
            let digest = try writeWeights(recipe, weights: weights, to: data.appendingPathComponent("weights/weight.bin"))
            guard digest == recipe.sha256 else {
                omniPerfLog("ane \(name): the weights written are not the ones built (\(digest.prefix(12)) vs \(recipe.sha256.prefix(12))): GPU")
                return nil
            }
            let built = try awaitCompile(pkg)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try fm.moveItem(at: built, to: model)
            try recipe.sha256.write(to: stamp, atomically: true, encoding: .utf8)
            return model
        } catch {
            omniPerfLog("ane \(name): build failed: \(error)")
            try? fm.removeItem(at: dir)
            return nil
        }
    }

    /// MILBlob v2: a 64-byte header (count, version), then per blob a 64-byte record (0xDEADBEEF,
    /// dtype 1 = fp16, size, data offset) and the data at the next 64-byte boundary. Returns the
    /// file's SHA-256, hex.
    /// The weight file `name` would get from these weights, written to `url`, and whether it is the
    /// one built and measured (its SHA-256 is the recipe's). No compile: the test of the writer.
    public static func writeWeightsChecked(_ name: String, weights: WeightStore, to url: URL) throws -> Bool {
        guard let recipeURL = Bundle.module.url(forResource: name + ".recipe", withExtension: "json", subdirectory: "ANE") else { return false }
        let recipe = try JSONDecoder().decode(Recipe.self, from: Data(contentsOf: recipeURL))
        return try writeWeights(recipe, weights: weights, to: url) == recipe.sha256
    }

    static func writeWeights(_ r: Recipe, weights: WeightStore, to url: URL) throws -> String {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        var sha = SHA256()
        var written = 0
        func put(_ d: Data) throws { try h.write(contentsOf: d); sha.update(data: d); written += d.count }
        func record(_ words: [UInt64], halves: [UInt32]) -> Data {
            var d = Data(count: 64)
            d.withUnsafeMutableBytes { p in
                for (i, v) in halves.enumerated() { p.storeBytes(of: v.littleEndian, toByteOffset: i * 4, as: UInt32.self) }
                for (i, v) in words.enumerated() { p.storeBytes(of: v.littleEndian, toByteOffset: 8 + i * 8, as: UInt64.self) }
            }
            return d
        }
        try put(record([], halves: [UInt32(r.blobs.count), 2]))
        for b in r.blobs {
            let values: [UInt16]
            if let c = constant(b, headDim: r.headDim) {
                values = c
            } else {
                guard let t = weights.weights[b.tensor] else { throw OmniError.store("ane recipe: no tensor \(b.tensor)") }
                values = transformed(t, b, headDim: r.headDim)
            }
            guard values.count * 2 == b.bytes else { throw OmniError.store("ane recipe: \(b.tensor) is \(values.count * 2) bytes, not \(b.bytes)") }
            let dataAt = written + 64
            try put(record([UInt64(b.bytes), UInt64(dataAt)], halves: [0xDEADBEEF, 1]))
            try put(values.withUnsafeBufferPointer { Data(buffer: $0) })
            let pad = (64 - written % 64) % 64
            if pad > 0 { try put(Data(count: pad)) }
        }
        guard written == r.weightBytes else { throw OmniError.store("ane recipe: wrote \(written) bytes, expected \(r.weightBytes)") }
        return sha.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// A tensor as the program holds it, fp16 row-major: as it is (`id`), times 1/sqrt(head_dim)
    /// (`q`), or a per-head norm weight tiled over the heads (`tile`, scaled for q) and permuted as
    /// the rotated channels see it (`tileP`). Computed in fp32 and rounded once, as numpy does in
    /// Tools/ane/extract_recipe.py; the SHA-256 check proves it.
    static func transformed(_ t: MLXArray, _ b: Recipe.Blob, headDim: Int) -> [UInt16] {
        let f = t.asType(.float32)
        eval(f)
        var v = f.asArray(Float.self)
        let scale = Float(1 / Double(headDim).squareRoot())
        let half = headDim / 2
        switch b.op {
        case "q":
            for i in v.indices { v[i] *= scale }
        case "tile", "tileP":
            var tiled = [Float](); tiled.reserveCapacity(v.count * (b.heads ?? 1))
            for _ in 0 ..< (b.heads ?? 1) { tiled.append(contentsOf: v) }
            v = b.scaled == true ? tiled.map { $0 * scale } : tiled
        default: break
        }
        if b.op == "tileP" {
            // The halves of each head swapped, no sign: the sign is in the rotated projection.
            var o = v
            for head in 0 ..< v.count / headDim {
                let a = head * headDim
                for j in 0 ..< half { o[a + j] = v[a + half + j]; o[a + half + j] = v[a + j] }
            }
            v = o
        }
        return Half.bits(v)
    }

    /// Where each of the program's weights lies in its compiled weight file ("tensor|op" -> byte
    /// offset and size), read from the file's own records in the recipe's order: the long-sequence
    /// layers (ANETextBackend) take their weights from there. nil when the file and the recipe
    /// disagree anywhere.
    static func blobOffsets(_ name: String, weightFile: URL) -> [String: (offset: Int, bytes: Int)]? {
        guard let recipeURL = Bundle.module.url(forResource: name + ".recipe", withExtension: "json", subdirectory: "ANE"),
              let r = try? JSONDecoder().decode(Recipe.self, from: Data(contentsOf: recipeURL)),
              let h = try? FileHandle(forReadingFrom: weightFile) else { return nil }
        defer { try? h.close() }
        var out: [String: (offset: Int, bytes: Int)] = [:]
        var at = 64
        for b in r.blobs {
            guard (try? h.seek(toOffset: UInt64(at))) != nil, let rec = try? h.read(upToCount: 64), rec.count == 64 else { return nil }
            let magic = rec.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
            let size = Int(rec.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) })
            let offset = Int(rec.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt64.self) })
            guard magic == 0xDEADBEEF, size == b.bytes, offset == at + 64 else { return nil }
            if !b.tensor.isEmpty { out["\(b.tensor)|\(b.op)"] = (offset, size) }
            at = offset + size + (64 - (offset + size) % 64) % 64
        }
        return at == r.weightBytes ? out : nil
    }

    /// Constants of the graph, not of the weights: head indicators, broadcast epsilons, and
    /// rotate-half per head as a grouped conv's weight ((heads x d) x d: row j of a head -1 at
    /// column j + d/2, row j + d/2 +1 at column j).
    static func constant(_ b: Recipe.Blob, headDim: Int) -> [UInt16]? {
        switch b.op {
        case "indicator":
            guard let heads = b.heads else { return nil }
            let c = heads * headDim, x = Half.bits(Float(b.value ?? 1))
            var m = [UInt16](repeating: 0, count: heads * c)
            for h in 0 ..< heads {
                for j in h * headDim ..< (h + 1) * headDim {
                    if b.transpose == true { m[j * heads + h] = x } else { m[h * c + j] = x }
                }
            }
            return m
        case "fill":
            return [UInt16](repeating: Half.bits(Float(b.value ?? 0)), count: b.count ?? 0)
        case "rotate":
            guard let heads = b.heads else { return nil }
            let half = headDim / 2, one = Half.bits(Float(1)), minus = Half.bits(Float(-1))
            var m = [UInt16](repeating: 0, count: heads * headDim * headDim)
            for h in 0 ..< heads {
                for j in 0 ..< half {
                    m[(h * headDim + j) * headDim + half + j] = minus
                    m[(h * headDim + half + j) * headDim + j] = one
                }
            }
            return m
        default:
            return nil
        }
    }

    private static func awaitCompile(_ pkg: URL) throws -> URL {
        final class Box: @unchecked Sendable { var result: Result<URL, Error>? }
        let box = Box(), done = DispatchSemaphore(value: 0)
        Task.detached { do { box.result = .success(try await MLModel.compileModel(at: pkg)) } catch { box.result = .failure(error) }; done.signal() }
        done.wait()
        return try box.result!.get()
    }

    private static let manifest = """
        {
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": {
                "385EE874-E712-41A6-8B9E-4A07D0DEBFE8": {
                    "author": "com.apple.CoreML", "description": "CoreML Model Specification",
                    "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel"
                },
                "86FA77E9-12C5-47E8-81CA-AAF769287DA3": {
                    "author": "com.apple.CoreML", "description": "CoreML Model Weights",
                    "name": "weights", "path": "com.apple.CoreML/weights"
                }
            },
            "rootModelIdentifier": "385EE874-E712-41A6-8B9E-4A07D0DEBFE8"
        }
        """
}
