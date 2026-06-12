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
        // Landmarks: the rows the quadratic work runs on. L == n (no split) reproduces the
        // pre-landmark behavior exactly; with a split, the remaining rows are PLACED relative to the
        // landmark layout so every file gets a dot at near-sample cost.
        let L = min(max(data.landmarkCount, 1), n)
        let cancelled = ProjectionResult(points: [], knn: [], k: k)

        // Step 1 (gated): landmark matrix + PCA basis + landmark PCA-2D layout.
        var Y0host = [Float]()
        var XL = MLXArray(); var Y0 = MLXArray()
        var pcaMean = MLXArray(); var pcaComps = MLXArray()
        _ = engine.runLowPriorityGPU { () -> Int in
            XL = L == n ? MLXArray(data.vectors, [n, d]).asType(.float32) : Self.hostTile(data, 0, L)
            let basis = Self.pca2DBasis(XL)
            Y0 = basis.Y; pcaMean = basis.mean; pcaComps = basis.comps
            Y0host = Y0.asArray(Float.self)
            return 0
        }
        if Task.isCancelled { return cancelled }
        await Task.yield()

        // All-points PCA positions: landmarks from Y0, the rest projected EXACTLY through the same
        // basis, in memory-bounded tiles. This is both the PCA-mode result and the fallback if the
        // force layout goes non-finite.
        var pcaAll = Y0host
        if L < n {
            pcaAll.reserveCapacity(n * 2)
            let tileRows = Self.placementTileRows(d)
            var start = L
            while start < n {
                let end = min(start + tileRows, n)
                let part = engine.runLowPriorityGPU { () -> [Float] in
                    Self.pcaProjectTile(tile: Self.hostTile(data, start, end), mean: pcaMean, comps: pcaComps)
                }
                pcaAll.append(contentsOf: part)
                if Task.isCancelled { return cancelled }
                await Task.yield()
                start = end
            }
        }
        let pcaPoints = Self.makePoints(pcaAll, data)

        // PCA-only (default, light): stop here. UMAP refinement (better cluster separation + the
        // neighbor graph for click-to-spotlight) is opt-in via Settings.
        if !refine { return ProjectionResult(points: pcaPoints, knn: [], k: 0) }

        // kNN graph over the LANDMARKS, computed ONCE: it both seeds the force layout and is
        // returned for the click-to-highlight-neighbors UI. Skipped only when too few rows (<= k).
        var knnHost: [Int32] = []
        var knnArr = MLXArray()
        let haveKNN = L > k
        if haveKNN {
            _ = engine.runLowPriorityGPU { () -> Int in
                knnArr = Self.knn(XL, k: k); eval(knnArr); knnHost = knnArr.asArray(Int32.self); return 0
            }
        }

        // Tiny landmark sets: force layout is meaningless; the PCA layout is the answer. A split
        // never lands here (the landmark budget floor is far above smallN), so pcaPoints == all rows.
        if L <= Self.smallN || L <= 2 || !haveKNN { return ProjectionResult(points: pcaPoints, knn: knnHost, k: k) }
        if Task.isCancelled { return cancelled }
        await Task.yield()

        // Step 2 (gated): edge/negative-sampling buffers from the kNN graph + scaled PCA-2D force init.
        var Y = Y0
        var edgeFrom = MLXArray(); var edgeTo = MLXArray(); var negHeads = MLXArray()
        _ = engine.runLowPriorityGPU { () -> Int in
            edgeFrom = MLXArray((0 ..< L).flatMap { Array(repeating: Int32($0), count: k) })  // [L*k]
            edgeTo   = knnArr.reshaped([-1]).asType(.int32)                                    // [L*k]
            negHeads = MLX.concatenated(Array(repeating: edgeFrom, count: negRate), axis: 0)   // [L*k*negRate]
            // Scale the PCA-2D init to ~initStd std so the force dynamics match the spike's tuned
            // learning rate/clip (which assumed a `normal*5.0` init).
            let std = Self.std2D(Y0host)
            let s = std > 0 ? Self.initStd / std : 1.0
            Y = Y0 * s
            MLX.seed(seed)
            eval(Y, edgeFrom, edgeTo, negHeads)
            return 0
        }
        if Task.isCancelled { return cancelled }
        await Task.yield()

        // Step 3 (gated, batched): force layout in ~batchEpochs slices, one eval barrier per batch so
        // the gate is released often enough that a high-priority query preempts within a batch. No
        // per-batch readback - only the final layout is copied to the host.
        var epoch = 0
        while epoch < epochs {
            let end = min(epoch + Self.batchEpochs, epochs)
            _ = engine.runLowPriorityGPU { () -> Int in
                Y = Self.forceEpochs(Y, edgeFrom: edgeFrom, edgeTo: edgeTo, negHeads: negHeads,
                                     n: L, negRate: negRate, epochStart: epoch, epochEnd: end, totalEpochs: epochs)
                eval(Y)
                return 0
            }
            epoch = end
            if Task.isCancelled { return cancelled }
            await Task.yield()
        }

        var host = engine.runLowPriorityGPU { () -> [Float] in eval(Y); return Y.asArray(Float.self) }
        let finite = host.allSatisfy { $0.isFinite }

        // Place the non-landmark rows: IDW over each row's nearest landmarks (one [tile, L] GEMM per
        // tile). The nearest-landmark indices also become these rows' spotlight kNN entries - valid
        // global point indices because landmarks are the first rows. Index computation does not
        // depend on the layout, so the kNN rows stay correct even on the PCA fallback.
        if L < n {
            let tileRows = Self.placementTileRows(L)
            var XLt = MLXArray(); var YLmlx = MLXArray()
            _ = engine.runLowPriorityGPU { () -> Int in
                XLt = XL.transposed()
                YLmlx = MLXArray(host, [L, 2])
                eval(XLt, YLmlx)
                return 0
            }
            var start = L
            while start < n {
                let end = min(start + tileRows, n)
                let (pos, idx) = engine.runLowPriorityGPU { () -> ([Float], [Int32]) in
                    Self.placeTileIDW(XLt: XLt, YL: YLmlx, tile: Self.hostTile(data, start, end), k: k)
                }
                host.append(contentsOf: pos)
                knnHost.append(contentsOf: idx)
                if Task.isCancelled { return cancelled }
                await Task.yield()
                start = end
            }
        }

        // umap-with-pca-fallback: any non-finite landmark layout -> the exact PCA positions instead
        // (the spotlight graph is layout-independent and kept).
        if !finite { return ProjectionResult(points: pcaPoints, knn: knnHost, k: k) }
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
        let L = min(max(data.landmarkCount, 1), n)
        let XL = L == n ? MLXArray(data.vectors, [n, d]).asType(.float32) : hostTile(data, 0, L)
        let (Y0, pcaMean, pcaComps) = pca2DBasis(XL)
        let Y0host = Y0.asArray(Float.self)

        // All-points PCA positions (exact projection of the rest through the landmark basis).
        var pcaAll = Y0host
        if L < n {
            let tileRows = placementTileRows(d)
            var start = L
            while start < n {
                let end = min(start + tileRows, n)
                pcaAll.append(contentsOf: pcaProjectTile(tile: hostTile(data, start, end), mean: pcaMean, comps: pcaComps))
                start = end
            }
        }
        let pcaPoints = makePoints(pcaAll, data)
        if L <= smallN || L <= 2 || L <= k { return pcaPoints }   // L <= k: can't build a k-NN graph

        let knnIdx = knn(XL, k: k)
        let edgeFrom = MLXArray((0 ..< L).flatMap { Array(repeating: Int32($0), count: k) })
        let edgeTo   = knnIdx.reshaped([-1]).asType(.int32)
        let negHeads = MLX.concatenated(Array(repeating: edgeFrom, count: negRate), axis: 0)
        eval(edgeFrom, edgeTo, negHeads)

        MLX.seed(seed)
        let std = std2D(Y0host)
        let s = std > 0 ? initStd / std : 1.0
        var Y = Y0 * s
        eval(Y)
        Y = forceEpochs(Y, edgeFrom: edgeFrom, edgeTo: edgeTo, negHeads: negHeads,
                        n: L, negRate: negRate, epochStart: 0, epochEnd: epochs, totalEpochs: epochs)
        eval(Y)
        var host = Y.asArray(Float.self)
        if !host.allSatisfy({ $0.isFinite }) { return pcaPoints }

        // Place the non-landmark rows by IDW over their nearest landmarks (see project()).
        if L < n {
            let XLt = XL.transposed()
            let YLmlx = MLXArray(host, [L, 2])
            let tileRows = placementTileRows(L)
            var start = L
            while start < n {
                let end = min(start + tileRows, n)
                host.append(contentsOf: placeTileIDW(XLt: XLt, YL: YLmlx, tile: hostTile(data, start, end), k: k).pos)
                start = end
            }
        }
        return makePoints(host, data)
    }

    // MARK: - Core MLX kernels (ported verbatim from the spike)

    /// PCA via SVD: eigh is missing in MLXLinalg, so the top-2 components come from the SVD of the
    /// d x d covariance. SVD only runs on the CPU stream (no GPU SVD). N-independent ~105ms cost.
    static func pca2D(_ X: MLXArray) -> MLXArray { pca2DBasis(X).Y }

    /// pca2D plus the fitted basis (mean + top-2 components), so non-landmark rows can be projected
    /// EXACTLY through the same components later (the landmark placement path).
    ///
    /// Top-2 components via GPU BLOCK subspace iteration (block 8) on the d x d covariance plus a
    /// host-side 8x8 Rayleigh-Ritz step - a few [d,d]x[d,8] matmuls instead of the CPU-only SVD,
    /// which was a fixed ~45ms (d=768) on every first map click. A wide block converges at the
    /// (lambda_9/lambda_2)^k rate, so near-degenerate lambda_2/lambda_3 spectra (where rank-2
    /// power iteration stalls) still converge in a handful of iterations. Deterministic: fixed
    /// init, fixed iteration count; the top-2 Ritz residual is checked and any miss falls back
    /// to the exact CPU SVD. Map orientation was never a contract (SVD signs are themselves
    /// platform-dependent); the captured-variance parity vs SVD is gated in projbench.
    public static func pca2DBasis(_ X: MLXArray) -> (Y: MLXArray, mean: MLXArray, comps: MLXArray) {
        let n = X.dim(0)
        let d = X.dim(1)
        let mean = MLX.mean(X, axis: 0)                       // [d]
        let Xc = X - mean                                     // [n, d]
        let cov = Xc.transposed().matmul(Xc) / Float(max(1, n - 1))  // [d, d]
        let q = 8                                             // block width
        // Deterministic full-rank init: fixed sinusoid columns (seed-free, never degenerate
        // against a real covariance).
        var initB = [Float](repeating: 0, count: d * q)
        for i in 0 ..< d {
            for j in 0 ..< q { initB[i * q + j] = cosf(Float(i) * (0.37 + 0.29 * Float(j)) + 0.11 * Float(j * j + 1)) }
        }
        var B = MLXArray(initB, [d, q])
        // Modified Gram-Schmidt over q columns (lazy graph; no per-column syncs).
        func orthonormalize(_ M: MLXArray) -> MLXArray {
            var cols: [MLXArray] = (0 ..< q).map { M[0..., $0 ..< ($0 + 1)] }
            for j in 0 ..< q {
                for i in 0 ..< j { cols[j] = cols[j] - cols[i] * MLX.sum(cols[i] * cols[j]) }
                cols[j] = cols[j] / MLX.sqrt(MLX.maximum(MLX.sum(cols[j] * cols[j]), MLXArray(Float(1e-30))))
            }
            return MLX.concatenated(cols, axis: 1)
        }
        for _ in 0 ..< 8 { B = orthonormalize(cov.matmul(B)) }
        // Rayleigh-Ritz: T = B' C B (q x q, symmetric) eigendecomposed on the host (Jacobi),
        // top-2 Ritz vectors become the components. One readback of q*q + 2 scalars total.
        let CB = cov.matmul(B)
        let T = B.transposed().matmul(CB)                     // [q, q]
        eval(T)
        var a = T.asArray(Float.self)                         // row-major q x q
        // Symmetrize fp noise, then cyclic Jacobi (handful of sweeps; q=8 is tiny).
        for i in 0 ..< q { for j in 0 ..< i { let m = 0.5 * (a[i*q+j] + a[j*q+i]); a[i*q+j] = m; a[j*q+i] = m } }
        var V = [Float](repeating: 0, count: q * q)
        for i in 0 ..< q { V[i*q+i] = 1 }
        for _ in 0 ..< 12 {
            var off: Float = 0
            for p1 in 0 ..< q { for q1 in (p1 + 1) ..< q { off += a[p1*q+q1] * a[p1*q+q1] } }
            if off < 1e-18 { break }
            for p1 in 0 ..< q {
                for q1 in (p1 + 1) ..< q {
                    let apq = a[p1*q+q1]
                    if abs(apq) < 1e-20 { continue }
                    let app = a[p1*q+p1], aqq = a[q1*q+q1]
                    let theta = 0.5 * atan2f(2 * apq, app - aqq)
                    let c = cosf(theta), s2 = sinf(theta)
                    for k in 0 ..< q {
                        let akp = a[k*q+p1], akq = a[k*q+q1]
                        a[k*q+p1] = c * akp + s2 * akq
                        a[k*q+q1] = -s2 * akp + c * akq
                    }
                    for k in 0 ..< q {
                        let apk = a[p1*q+k], aqk = a[q1*q+k]
                        a[p1*q+k] = c * apk + s2 * aqk
                        a[q1*q+k] = -s2 * apk + c * aqk
                    }
                    for k in 0 ..< q {
                        let vkp = V[k*q+p1], vkq = V[k*q+q1]
                        V[k*q+p1] = c * vkp + s2 * vkq
                        V[k*q+q1] = -s2 * vkp + c * vkq
                    }
                }
            }
        }
        // Top-2 eigenpairs by eigenvalue (diagonal of the rotated T).
        let order = (0 ..< q).sorted { a[$0*q+$0] > a[$1*q+$1] }
        let (i0, i1) = (order[0], order[1])
        var v2 = [Float](repeating: 0, count: q * 2)
        for k in 0 ..< q { v2[k*2] = V[k*q+i0]; v2[k*2+1] = V[k*q+i1] }
        let Ritz = B.matmul(MLXArray(v2, [q, 2]))             // [d, 2] candidate components
        // Convergence check on the SUBSPACE (rotation-insensitive): ||C S - S (S'CS)|| relative.
        // Per-vector eigen-residuals never settle when lambda_2 ~ lambda_3 (the vectors are
        // rotationally degenerate), but any basis of the converged subspace captures the same
        // variance - which is what a 2D map needs. The captured-variance parity vs exact SVD is
        // gated empirically in projbench.
        let CS = cov.matmul(Ritz)
        let M2 = Ritz.transposed().matmul(CS)                 // [2, 2]
        let resid = CS - Ritz.matmul(M2)
        let relErr = MLX.sum(resid * resid) / MLX.maximum(MLX.sum(CS * CS), MLXArray(Float(1e-30)))
        eval(relErr)
        if ProcessInfo.processInfo.environment["OMNI_PCA_DEBUG"] == "1" {
            FileHandle.standardError.write(Data("PCA relErr=\(relErr.item(Float.self))\n".utf8))
        }
        var comps: MLXArray
        // 2e-4 maps to >= 99.9% captured-variance parity with the exact SVD (measured); spectra
        // that converge slower (near-isotropic synthetic data) take the exact CPU-SVD fallback.
        if relErr.item(Float.self) < 2e-4 {
            comps = Ritz.transposed()                         // [2, d]
        } else {
            let (_, _, Vt) = MLXLinalg.svd(cov, stream: .cpu)   // exact fallback
            comps = Vt[0 ..< 2]
        }
        let Y = Xc.matmul(comps.transposed())                // [n, 2]
        eval(Y)
        return (Y, mean, comps)
    }

    // MARK: - Landmark placement (all-points maps)
    //
    // The quadratic layout work (kNN + force, or the SVD fit) runs on the LANDMARK rows only
    // (data's first landmarkCount rows, an even-stride sample). The remaining rows are placed
    // relative to that layout, so every file gets a dot at near-sample cost:
    //   - PCA mode: exact - project each row through the landmark-fitted (mean, comps).
    //   - UMAP mode: inverse-distance weighting over the row's k nearest landmarks (cosine via one
    //     [tile, L] GEMM; vectors are L2-normalized). w_i = 1 / (d2_i + eps) with d2 = 2 - 2*sim,
    //     so a row that coincides with a landmark lands on it and everything else interpolates.
    // Tiles bound the GEMM memory; total FLOPs are rest x L x d (linear in rest, not quadratic).

    /// Rows [start, end) of data.vectors as a [t, d] MLXArray.
    static func hostTile(_ data: FolderVectors, _ start: Int, _ end: Int) -> MLXArray {
        let d = data.dim
        return data.vectors.withUnsafeBufferPointer { buf in
            MLXArray(Array(UnsafeBufferPointer(rebasing: buf[(start * d) ..< (end * d)])), [end - start, d])
        }
    }

    /// Tile rows for the placement GEMM: bound the [t, L] similarity matrix to ~200MB.
    static func placementTileRows(_ landmarks: Int) -> Int {
        max(1, 200_000_000 / max(1, landmarks * 4))
    }

    /// Place one tile of non-landmark rows: returns row-major [t*2] positions and the [t*k] nearest
    /// landmark indices (nearest first - they double as the spotlight kNN rows for these points,
    /// valid globally because landmarks are the first rows of the result).
    static func placeTileIDW(XLt: MLXArray, YL: MLXArray, tile: MLXArray, k: Int) -> (pos: [Float], knn: [Int32]) {
        let sims = tile.matmul(XLt)                                       // [t, L] cosine
        let part = MLX.argPartition(MLX.negative(sims), kth: k, axis: 1)[0..., 0 ..< k]   // top-k, unordered
        let simK = MLX.takeAlong(sims, part, axis: 1)                     // [t, k]
        let ord = MLX.argSort(MLX.negative(simK), axis: 1)                // nearest first
        let idx = MLX.takeAlong(part, ord, axis: 1)                       // [t, k]
        let s = MLX.takeAlong(simK, ord, axis: 1)
        let d2 = MLX.maximum(2.0 - 2.0 * s, MLXArray(Float(0)))           // squared L2 on the unit sphere
        var w = 1.0 / (d2 + 1e-4)                                         // IDW: coincident row -> its landmark
        w = w / MLX.sum(w, axis: 1, keepDims: true)                       // [t, k]
        let nbr = YL[idx.reshaped([-1])].reshaped([idx.dim(0), k, 2])     // [t, k, 2]
        let pos = MLX.sum(nbr * w.expandedDimensions(axis: 2), axis: 1)   // [t, 2]
        eval(pos, idx)
        return (pos.asArray(Float.self), idx.asType(.int32).asArray(Int32.self))
    }

    /// Exact PCA projection of one tile through the landmark-fitted basis.
    static func pcaProjectTile(tile: MLXArray, mean: MLXArray, comps: MLXArray) -> [Float] {
        let Y = (tile - mean).matmul(comps.transposed())
        eval(Y)
        return Y.asArray(Float.self)
    }

    /// Chunked brute-force kNN that never materializes the full N x N distance matrix.
    public static func knn(_ X: MLXArray, k: Int) -> MLXArray {     // -> [n, k] int32
        let n = X.dim(0)
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
            // Inputs are L2-normalized (per-file embeddings), so squared L2 = 2 - 2*cos: the
            // negated similarity ranks identically and skips the sqNorms broadcast assembly.
            // The dropped per-column ||b||^2-1 term is fp32-rounding-sized (~1e-7): it can only
            // reorder exact near-ties, the latitude the selection already has.
            var D = -xChunk.matmul(xT)                        // [c, n] -cosine (rank == squared L2)
            let rowsIdx = MLX.arange(start, end)[0..., .newAxis]   // [c,1] int32
            let colsIdx = MLX.arange(0, n)[.newAxis, 0...]         // [1,n] int32
            D = D + (rowsIdx .== colsIdx).asType(.float32) * 1e30  // mask self
            // Select the k nearest via argPartition (O(n) per row) + a small [c,k] sort to restore
            // the nearest-first order the UI relies on, instead of fully sorting every [c,n] row
            // (O(n log n) per row - the dominant kNN cost at n in the tens of thousands). Identical
            // output for distinct distances; exact-tie sets at the k-th boundary are pool-equivalent.
            let part = MLX.argPartition(D, kth: k, axis: 1)[0..., 0 ..< k]   // [c,k] smallest, unordered
            let dk = MLX.takeAlong(D, part, axis: 1)                          // [c,k] their distances
            let idx = MLX.takeAlong(part, MLX.argSort(dk, axis: 1), axis: 1) // [c,k] nearest first
            eval(idx)
            idxChunks.append(idx)
            start = end
        }
        return MLX.concatenated(idxChunks, axis: 0)           // [n, k]
    }

    /// Runs epochs [epochStart, epochEnd) of the UMAP-ish scatter-add SGD on `Y`, returning the
    /// updated (unevaluated) layout. Graph eval is throttled internally; the caller adds the final
    /// eval barrier for the batch. `negHeads` must equal edgeFrom tiled `negRate` times.
    /// One force epoch as a COMPILED graph: (Y, edgeFrom, edgeTo, negHeads, negTo, alpha) -> Y'.
    /// Shaped compile (not shapeless): the shapes are fixed for a whole projection, so MLX caches
    /// one specialization per (L, k, negRate) and fuses the elementwise chains between the
    /// gather/scatter ops - the loop was ~16 tiny kernel dispatches per epoch, pure launch
    /// overhead at [L, 2] sizes. Identical math, same op set, no reassociation: positions are
    /// bit-identical to the uncompiled loop (gated by projbench's layout digest).
    private static let forceEpochCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) { xs in
        let (Y0, edgeFrom, edgeTo, negHeads, negTo, alpha) = (xs[0], xs[1], xs[2], xs[3], xs[4], xs[5])
        var Y = Y0
        // attractive (neighbors)
        let diff = Y[edgeFrom] - Y[edgeTo]
        let d2 = MLX.maximum(MLX.sum(diff * diff, axis: 1, keepDims: true), MLXArray(Float(1e-6)))
        let posCoeff = -2.0 / (1.0 + d2)
        let posGrad = MLX.clip(posCoeff * diff, min: Float(-4), max: Float(4)) * alpha
        Y = Y.at[edgeFrom].add(posGrad)
        Y = Y.at[edgeTo].subtract(posGrad)
        // repulsive (negative sampling; negTo supplied by the caller - random state stays outside
        // the compiled graph so the sampling sequence matches the uncompiled loop exactly)
        let nd = Y[negHeads] - Y[negTo]
        let nd2 = MLX.maximum(MLX.sum(nd * nd, axis: 1, keepDims: true), MLXArray(Float(1e-6)))
        let negCoeff = 2.0 / ((0.001 + nd2) * (1.0 + nd2))
        let negGrad = MLX.clip(negCoeff * nd, min: Float(-4), max: Float(4)) * alpha
        Y = Y.at[negHeads].add(negGrad)
        return [Y]
    }

    public static func forceEpochs(_ Y0: MLXArray, edgeFrom: MLXArray, edgeTo: MLXArray, negHeads: MLXArray,
                            n: Int, negRate: Int, epochStart: Int, epochEnd: Int, totalEpochs: Int) -> MLXArray {
        var Y = Y0
        // UMAP a=1, b=1 (the spike's tuned constants) folded in algebraically: pow(d2, b-1) is
        // pow(x, 0) == 1 and pow(d2, b) is the identity, so the three pow kernels per epoch were
        // pure dispatch waste. The folded forms are bit-identical for finite d2 >= 1e-6
        // (IEEE pow(x, 1) == x, pow(x, 0) == 1, and -2.0 * 1 * 1 == -2.0 exactly).
        let lr: Float = 1.0
        let nEdges = edgeFrom.dim(0)
        let nNeg = nEdges * negRate
        for epoch in epochStart ..< epochEnd {
            let alpha = lr * (1.0 - Float(epoch) / Float(totalEpochs))
            let negTo = MLX.randInt(0 ..< n, [nNeg]).asType(.int32)
            Y = Self.forceEpochCompiled([Y, edgeFrom, edgeTo, negHeads, negTo, MLXArray(alpha)])[0]
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
