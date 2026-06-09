import Foundation
import MLX
import MLXLinalg

/// One file's 2D position in the folder embedding map.
public struct ProjectionPoint: Sendable {
    public let position: SIMD2<Float>
    public let path: String
    public let kind: String     // FileKind rawValue

    public init(position: SIMD2<Float>, path: String, kind: String) {
        self.position = position
        self.path = path
        self.kind = kind
    }
}

/// A finished projection: the 2D points plus the embedding-space kNN graph used to lay them out.
/// `knn` is row-major `[count * k]` Int32 (the `k` nearest OTHER files per file, nearest first); the
/// UI reuses it to highlight a clicked file's nearest neighbors without recomputing anything. Empty
/// when there are too few files (<= k) to have a graph.
public struct ProjectionResult: Sendable {
    public let points: [ProjectionPoint]
    public let knn: [Int32]
    public let k: Int
    public init(points: [ProjectionPoint], knn: [Int32], k: Int) {
        self.points = points; self.knn = knn; self.k = k
    }
}

/// Projects per-file embedding vectors to 2D for the folder visualization.
///
/// Algorithm: umap-with-pca-fallback. (1) PCA-via-SVD 2D (`pca2D`, SVD on the .cpu stream) is the
/// eigh-free init AND the fallback layout. (2) A minimal UMAP-ish force layout (`forceLayout`,
/// scatter-add SGD, k=15/epochs=300/negRate=5) refines it, initialized from the PCA-2D layout
/// scaled to ~5.0 std (replacing the spike's random `MLX.normal*5.0` init). (3) If the force result
/// contains any non-finite value or count<=2, the PCA-2D layout is returned instead. Below ~50 files
/// force is skipped (meaningless for tiny N) and the PCA-2D layout is returned directly.
///
/// `project()` runs the fit in ~10-epoch batches behind the engine's low-priority GPU gate so an
/// interactive search preempts within one batch, and returns only the settled layout (no snapshots).
/// `layout()` is the equivalent synchronous, ungated reference path (used by tests).
public final class ProjectionEngine: @unchecked Sendable {
    private let engine: OmniEngine
    public init(engine: OmniEngine) { self.engine = engine }

    // Tunables shared by the gated and sync paths.
    private static let smallN = 50          // below this, skip force (PCA-2D only)
    private static let batchEpochs = 10     // epochs per gated batch / eval barrier
    private static let initStd: Float = 5.0 // target std of the PCA-2D force init

    // MARK: - Public entry (gated, async)

    /// Runs OFF the main actor (caller uses Task.detached) and returns ONLY the settled layout - no
    /// intermediate snapshots. The fit still runs in ~batchEpochs slices behind the low-priority GPU
    /// gate (so a search preempts within a batch) and checks cancellation between slices, but the
    /// host readback happens once at the end. The UI shows the final result, not a bloom.
    public func project(_ data: FolderVectors,
                        k: Int = 15, epochs: Int = 300, negRate: Int = 5, seed: UInt64 = 42,
                        refine: Bool = true) async -> ProjectionResult {
        let n = data.count
        guard n > 0, data.dim > 0, data.vectors.count == n * data.dim else { return ProjectionResult(points: [], knn: [], k: k) }
        let d = data.dim

        // Step 1 (gated): build X on the GPU thread, compute the PCA-2D layout, read it back.
        var Y0host = [Float]()
        let (X, Y0) = engine.runLowPriorityGPU { () -> (MLXArray, MLXArray) in
            let X = MLXArray(data.vectors, [n, d]).asType(.float32)
            let Y0 = Self.pca2D(X)
            eval(Y0)
            Y0host = Y0.asArray(Float.self)
            return (X, Y0)
        }
        let pcaPoints = Self.makePoints(Y0host, data)

        // PCA-only (default, light): stop here. The kNN step materializes hundreds of MB of GPU
        // distance tiles for a large folder, and the 300-epoch force layout adds more - enough to
        // exhaust unified memory and freeze a low-RAM Mac. UMAP refinement (better cluster separation
        // + the neighbor graph for click-to-spotlight) is opt-in via Settings. PCA is N-light + instant.
        if !refine { return ProjectionResult(points: pcaPoints, knn: [], k: 0) }

        // kNN graph (embedding space), computed ONCE: it both seeds the force layout and is returned
        // for the click-to-highlight-neighbors UI. Skipped only when there are too few files (<= k).
        var knnHost: [Int32] = []
        var knnArr = MLXArray()
        let haveKNN = n > k
        if haveKNN {
            _ = engine.runLowPriorityGPU { () -> Int in
                knnArr = Self.knn(X, k: k); eval(knnArr); knnHost = knnArr.asArray(Int32.self); return 0
            }
        }

        // Tiny N: force layout is meaningless; PCA-2D is the answer (with the graph if we have one).
        // PCA-only when there's no usable kNN graph: too few files (smallN), or n <= k so each file
        // can't have k distinct neighbors (the force layout would reshape an empty graph against n*k edges).
        if n <= Self.smallN || n <= 2 || !haveKNN { return ProjectionResult(points: pcaPoints, knn: knnHost, k: k) }
        if Task.isCancelled { return ProjectionResult(points: [], knn: [], k: k) }
        await Task.yield()

        // Step 2 (gated): edge/negative-sampling buffers from the kNN graph + scaled PCA-2D force init.
        var Y = Y0
        var edgeFrom = MLXArray(); var edgeTo = MLXArray(); var negHeads = MLXArray()
        _ = engine.runLowPriorityGPU { () -> Int in
            edgeFrom = MLXArray((0 ..< n).flatMap { Array(repeating: Int32($0), count: k) })  // [n*k]
            edgeTo   = knnArr.reshaped([-1]).asType(.int32)                                    // [n*k]
            negHeads = MLX.concatenated(Array(repeating: edgeFrom, count: negRate), axis: 0)   // [n*k*negRate]
            // Scale the PCA-2D init to ~initStd std so the force dynamics match the spike's tuned
            // learning rate/clip (which assumed a `normal*5.0` init).
            let std = Self.std2D(Y0host)
            let s = std > 0 ? Self.initStd / std : 1.0
            Y = Y0 * s
            MLX.seed(seed)
            eval(Y, edgeFrom, edgeTo, negHeads)
            return 0
        }
        if Task.isCancelled { return ProjectionResult(points: [], knn: [], k: k) }
        await Task.yield()

        // Step 3 (gated, batched): force layout in ~batchEpochs slices, one eval barrier per batch so
        // the gate is released often enough that a high-priority query preempts within a batch. No
        // per-batch readback - only the final layout is copied to the host.
        var epoch = 0
        while epoch < epochs {
            let end = min(epoch + Self.batchEpochs, epochs)
            _ = engine.runLowPriorityGPU { () -> Int in
                Y = Self.forceEpochs(Y, edgeFrom: edgeFrom, edgeTo: edgeTo, negHeads: negHeads,
                                     n: n, negRate: negRate, epochStart: epoch, epochEnd: end, totalEpochs: epochs)
                eval(Y)
                return 0
            }
            epoch = end
            if Task.isCancelled { return ProjectionResult(points: [], knn: [], k: k) }
            await Task.yield()
        }

        let host = engine.runLowPriorityGPU { () -> [Float] in eval(Y); return Y.asArray(Float.self) }
        // umap-with-pca-fallback: any non-finite element -> return the PCA-2D layout instead.
        if !host.allSatisfy({ $0.isFinite }) { return ProjectionResult(points: pcaPoints, knn: knnHost, k: k) }
        return ProjectionResult(points: Self.makePoints(host, data), knn: knnHost, k: k)
    }

    // MARK: - Synchronous reference path (ungated, no streaming)

    /// Full pca + force + fallback pipeline run synchronously without the GPU gate or snapshots.
    /// This is the reference implementation the gated `project()` mirrors; tests exercise it
    /// directly (no live model engine required).
    public static func layout(_ data: FolderVectors,
                              k: Int = 15, epochs: Int = 300, negRate: Int = 5, seed: UInt64 = 42) -> [ProjectionPoint] {
        let n = data.count
        guard n > 0, data.dim > 0, data.vectors.count == n * data.dim else { return [] }
        let d = data.dim
        let X = MLXArray(data.vectors, [n, d]).asType(.float32)
        let Y0 = pca2D(X)
        eval(Y0)
        let Y0host = Y0.asArray(Float.self)
        let pcaPoints = makePoints(Y0host, data)
        if n <= smallN || n <= 2 || n <= k { return pcaPoints }   // n <= k: can't build a k-NN graph

        let knnIdx = knn(X, k: k)
        let edgeFrom = MLXArray((0 ..< n).flatMap { Array(repeating: Int32($0), count: k) })
        let edgeTo   = knnIdx.reshaped([-1]).asType(.int32)
        let negHeads = MLX.concatenated(Array(repeating: edgeFrom, count: negRate), axis: 0)
        eval(edgeFrom, edgeTo, negHeads)

        MLX.seed(seed)
        let std = std2D(Y0host)
        let s = std > 0 ? initStd / std : 1.0
        var Y = Y0 * s
        eval(Y)
        Y = forceEpochs(Y, edgeFrom: edgeFrom, edgeTo: edgeTo, negHeads: negHeads,
                        n: n, negRate: negRate, epochStart: 0, epochEnd: epochs, totalEpochs: epochs)
        eval(Y)
        let host = Y.asArray(Float.self)
        if !host.allSatisfy({ $0.isFinite }) { return pcaPoints }
        return makePoints(host, data)
    }

    // MARK: - Core MLX kernels (ported verbatim from the spike)

    /// PCA via SVD: eigh is missing in MLXLinalg, so the top-2 components come from the SVD of the
    /// d x d covariance. SVD only runs on the CPU stream (no GPU SVD). N-independent ~105ms cost.
    static func pca2D(_ X: MLXArray) -> MLXArray {           // X: [n, d]
        let n = X.dim(0)
        let mean = MLX.mean(X, axis: 0)                       // [d]
        let Xc = X - mean                                     // [n, d]
        let cov = Xc.transposed().matmul(Xc) / Float(max(1, n - 1))  // [d, d]
        eval(cov)
        let (_, _, Vt) = MLXLinalg.svd(cov, stream: .cpu)    // SVD ONLY on .cpu; Vt: [d, d]
        let comps = Vt[0 ..< 2]                               // top-2 rows [2, d]
        let Y = Xc.matmul(comps.transposed())                // [n, 2]
        eval(Y)
        return Y
    }

    /// Chunked brute-force kNN that never materializes the full N x N distance matrix.
    static func knn(_ X: MLXArray, k: Int) -> MLXArray {     // -> [n, k] int32
        let n = X.dim(0)
        let sqNorms = MLX.sum(X * X, axis: 1)                 // [n]
        eval(sqNorms)
        // Cap the distance-tile VRAM at ~200MB. No lower floor on the chunk: a floor (e.g. 1000) would
        // override the byte cap for large n - at n=200k a [1000,n] fp32 tile is ~800MB and the argSort
        // index array doubles it, enough to OOM an 8GB Mac in UMAP mode. Total FLOPs are chunk-invariant,
        // so smaller tiles just mean more kernel launches (negligible on any GPU).
        let chunk = min(n, max(1, 200_000_000 / max(1, n * 4)))
        var idxChunks = [MLXArray]()
        var start = 0
        let xT = X.transposed()
        while start < n {
            let end = min(start + chunk, n)
            let xChunk = X[start ..< end]                     // [c, d]
            var D = sqNorms[start ..< end, .newAxis] + sqNorms[.newAxis, 0...]
                  - 2.0 * xChunk.matmul(xT)                   // [c, n] squared dists
            D = MLX.maximum(D, MLXArray(Float(0)))
            let rowsIdx = MLX.arange(start, end)[0..., .newAxis]   // [c,1] int32
            let colsIdx = MLX.arange(0, n)[.newAxis, 0...]         // [1,n] int32
            D = D + (rowsIdx .== colsIdx).asType(.float32) * 1e30  // mask self
            let idx = MLX.argSort(D, axis: 1)[0..., 0 ..< k]  // [c, k] nearest
            eval(idx)
            idxChunks.append(idx)
            start = end
        }
        return MLX.concatenated(idxChunks, axis: 0)           // [n, k]
    }

    /// Runs epochs [epochStart, epochEnd) of the UMAP-ish scatter-add SGD on `Y`, returning the
    /// updated (unevaluated) layout. Graph eval is throttled internally; the caller adds the final
    /// eval barrier for the batch. `negHeads` must equal edgeFrom tiled `negRate` times.
    static func forceEpochs(_ Y0: MLXArray, edgeFrom: MLXArray, edgeTo: MLXArray, negHeads: MLXArray,
                            n: Int, negRate: Int, epochStart: Int, epochEnd: Int, totalEpochs: Int) -> MLXArray {
        var Y = Y0
        let a: Float = 1.0, b: Float = 1.0, lr: Float = 1.0
        let nEdges = edgeFrom.dim(0)
        let nNeg = nEdges * negRate
        for epoch in epochStart ..< epochEnd {
            let alpha = lr * (1.0 - Float(epoch) / Float(totalEpochs))
            // attractive (neighbors)
            let diff = Y[edgeFrom] - Y[edgeTo]
            let d2 = MLX.maximum(MLX.sum(diff * diff, axis: 1, keepDims: true), MLXArray(Float(1e-6)))
            let posCoeff = (-2.0 * a * b) * MLX.pow(d2, b - 1.0) / (1.0 + a * MLX.pow(d2, b))
            let posGrad = MLX.clip(posCoeff * diff, min: Float(-4), max: Float(4)) * alpha
            Y = Y.at[edgeFrom].add(posGrad)
            Y = Y.at[edgeTo].subtract(posGrad)
            // repulsive (negative sampling)
            let negTo = MLX.randInt(0 ..< n, [nNeg]).asType(.int32)
            let nd = Y[negHeads] - Y[negTo]
            let nd2 = MLX.maximum(MLX.sum(nd * nd, axis: 1, keepDims: true), MLXArray(Float(1e-6)))
            let negCoeff = (2.0 * b) / ((0.001 + nd2) * (1.0 + a * MLX.pow(nd2, b)))
            let negGrad = MLX.clip(negCoeff * nd, min: Float(-4), max: Float(4)) * alpha
            Y = Y.at[negHeads].add(negGrad)
            if (epoch + 1) % 16 == 0 { eval(Y) }              // throttle graph eval to bound memory
        }
        return Y
    }

    // MARK: - Host helpers

    /// Std of a mean-centered [n,2] host layout (PCA output is mean-centered, so sqrt(mean(y^2))).
    private static func std2D(_ h: [Float]) -> Float {
        guard !h.isEmpty else { return 0 }
        var s: Float = 0
        for v in h where v.isFinite { s += v * v }
        return (s / Float(h.count)).squareRoot()
    }

    /// Zip a row-major [n,2] host layout with the file paths/kinds into ProjectionPoints.
    static func makePoints(_ h: [Float], _ data: FolderVectors) -> [ProjectionPoint] {
        let n = data.count
        guard h.count >= n * 2 else { return [] }
        var pts = [ProjectionPoint](); pts.reserveCapacity(n)
        for i in 0 ..< n {
            pts.append(ProjectionPoint(position: SIMD2(h[2 * i], h[2 * i + 1]),
                                       path: data.paths[i], kind: data.kinds[i]))
        }
        return pts
    }
}
