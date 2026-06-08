import XCTest
@testable import OmniKit

/// Correctness test for the folder-embedding projection core. Builds synthetic Gaussian blobs at
/// the real embedding dim and asserts the 2D layout is (a) all-finite and (b) separates the blobs
/// (mean intra-blob 2D distance < mean inter-blob 2D distance). Runs through the ungated reference
/// path `ProjectionEngine.layout`, so no model engine is required.
final class ProjectionEngineTests: XCTestCase {

    /// Deterministic LCG so the test is reproducible without importing a random module.
    private struct LCG {
        var state: UInt64
        mutating func nextUnit() -> Float {            // uniform [0,1)
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        mutating func gaussian() -> Float {            // Box-Muller
            let u1 = max(nextUnit(), 1e-7), u2 = nextUnit()
            return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    }

    /// `blobs` clusters of `perBlob` L2-normalized vectors of dimension `dim`, each blob centered on
    /// a distinct random unit-ish direction with small per-point jitter.
    private func makeBlobs(blobs: Int, perBlob: Int, dim: Int, seed: UInt64) -> (FolderVectors, [Int]) {
        var rng = LCG(state: seed)
        var centers = [[Float]]()
        for _ in 0 ..< blobs {
            var c = (0 ..< dim).map { _ in rng.gaussian() }
            var nrm: Float = 0; for v in c { nrm += v * v }
            let inv = 1 / nrm.squareRoot(); for i in 0 ..< dim { c[i] *= inv }
            centers.append(c)
        }
        var vectors = [Float](); vectors.reserveCapacity(blobs * perBlob * dim)
        var paths = [String](); var kinds = [String](); var label = [Int]()
        let kindList = FileKind.allCases.map { $0.rawValue }
        for b in 0 ..< blobs {
            for j in 0 ..< perBlob {
                var v = [Float](repeating: 0, count: dim)
                var nrm: Float = 0
                for i in 0 ..< dim { v[i] = centers[b][i] + 0.05 * rng.gaussian(); nrm += v[i] * v[i] }
                let inv = nrm > 0 ? 1 / nrm.squareRoot() : 0
                for i in 0 ..< dim { v[i] *= inv }
                vectors.append(contentsOf: v)
                paths.append("/blob\(b)/file\(j).txt")
                kinds.append(kindList[b % kindList.count])
                label.append(b)
            }
        }
        return (FolderVectors(paths: paths, kinds: kinds, vectors: vectors, dim: dim), label)
    }

    private func meanIntraInter(_ pts: [ProjectionPoint], _ label: [Int]) -> (intra: Double, inter: Double) {
        var intraSum = 0.0, intraN = 0, interSum = 0.0, interN = 0
        for i in 0 ..< pts.count {
            for j in (i + 1) ..< pts.count {
                let dx = Double(pts[i].position.x - pts[j].position.x)
                let dy = Double(pts[i].position.y - pts[j].position.y)
                let dist = (dx * dx + dy * dy).squareRoot()
                if label[i] == label[j] { intraSum += dist; intraN += 1 }
                else { interSum += dist; interN += 1 }
            }
        }
        return (intraSum / Double(max(1, intraN)), interSum / Double(max(1, interN)))
    }

    /// Force-layout path (N well above the smallN threshold): finite + blobs separate.
    func testForceLayoutSeparatesBlobs() {
        let dim = 1024            // real embedding dim
        let (data, label) = makeBlobs(blobs: 6, perBlob: 50, dim: dim, seed: 12345)
        let pts = ProjectionEngine.layout(data, epochs: 200, seed: 7)
        XCTAssertEqual(pts.count, data.count)
        XCTAssertTrue(pts.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite },
                      "force layout produced non-finite coordinates")
        let (intra, inter) = meanIntraInter(pts, label)
        XCTAssertGreaterThan(inter, intra,
                             "blobs not separated: mean inter \(inter) <= mean intra \(intra)")
    }

    /// PCA-only path (N <= smallN threshold): still finite and still separates.
    func testPCAFallbackPathSeparatesBlobs() {
        let dim = 1024
        let (data, label) = makeBlobs(blobs: 4, perBlob: 8, dim: dim, seed: 999)  // 32 files <= 50 -> PCA only
        let pts = ProjectionEngine.layout(data)
        XCTAssertEqual(pts.count, data.count)
        XCTAssertTrue(pts.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite },
                      "PCA-2D produced non-finite coordinates")
        let (intra, inter) = meanIntraInter(pts, label)
        XCTAssertGreaterThan(inter, intra,
                             "PCA blobs not separated: mean inter \(inter) <= mean intra \(intra)")
    }

    /// Degenerate inputs must not crash or emit NaN.
    func testEmptyAndTinyInputs() {
        XCTAssertTrue(ProjectionEngine.layout(FolderVectors(paths: [], kinds: [], vectors: [], dim: 1024)).isEmpty)
        let one = FolderVectors(paths: ["/a.txt"], kinds: ["text"], vectors: [Float](repeating: 0.1, count: 8), dim: 8)
        let pts = ProjectionEngine.layout(one)
        XCTAssertEqual(pts.count, 1)
        XCTAssertTrue(pts[0].position.x.isFinite && pts[0].position.y.isFinite)
    }
}
