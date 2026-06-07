import Foundation
import MLX

/// Measured results of one profiling indexing pass. Keys match the upload contract exactly.
public struct ProfilingMetrics: Sendable, Codable {
    public let files: Int            // files that produced an embedding
    public let scanned: Int
    public let failed: Int
    public let seconds: Double
    public let filesPerSec: Double
    public let tokens: Int
    public let tokensPerSec: Double
    public let errorRate: Double
    public let peakVramDeltaBytes: Int   // peak GPU memory above the loaded-model baseline
}

/// The full report uploaded to /omni/profiling. Hardware + timing only; no PII.
public struct ProfilingReport: Sendable, Codable {
    public let runId: String
    public let appVersion: String
    public let datasetVersion: String
    public let hardware: HardwareProfile
    public let metrics: ProfilingMetrics

    public init(runId: String, appVersion: String, datasetVersion: String,
                hardware: HardwareProfile, metrics: ProfilingMetrics) {
        self.runId = runId
        self.appVersion = appVersion
        self.datasetVersion = datasetVersion
        self.hardware = hardware
        self.metrics = metrics
    }
}

private final class ProgressBox: @unchecked Sendable { var p = IndexProgress() }

/// Index `targetURL` with a FRESH temporary vector store (the user's real index is never touched),
/// timing the pass and measuring throughput and peak GPU-memory delta. Reuses the app's loaded
/// `engine` so the measurement reflects the real embedding path. `onProgress` fires on a background
/// thread; marshal to the main actor for UI. Returns the measured metrics.
public func runProfilingPass(engine: OmniEngine, targetURL: URL, settings: IndexSettings,
                             onProgress: @escaping @Sendable (IndexProgress) -> Void) async throws -> ProfilingMetrics {
    try await Task.detached(priority: .userInitiated) { () throws -> ProfilingMetrics in
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-profiling-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Scope the store so it is closed (deinit checkpoints + sqlite3_close) before cleanup.
        let metrics: ProfilingMetrics = try {
            let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
            let indexer = Indexer(store: store, embedder: engine)

            // VRAM baseline: drop recyclable buffers and reset the high-water mark to the current
            // (loaded-model) residency, so peakMemory afterwards reflects only what THIS pass added.
            MLX.GPU.clearCache()
            MLX.GPU.resetPeakMemory()
            let baseActive = MLX.GPU.activeMemory
            let tok0 = engine.tokensProcessed
            let t0 = Date()

            let box = ProgressBox()
            // index() is synchronous: it returns when the pass completes (force: true so every file
            // is embedded, giving a full-cost measurement rather than an incremental top-up).
            indexer.index(roots: [targetURL], settings: settings, force: true) { p in
                box.p = p
                onProgress(p)
            }

            let seconds = max(0.0001, Date().timeIntervalSince(t0))
            let tokens = max(0, engine.tokensProcessed - tok0)
            let peakDelta = max(0, MLX.GPU.peakMemory - baseActive)
            let p = box.p
            return ProfilingMetrics(
                files: p.embedded,
                scanned: p.scanned,
                failed: p.failed,
                seconds: seconds,
                filesPerSec: Double(p.embedded) / seconds,
                tokens: tokens,
                tokensPerSec: Double(tokens) / seconds,
                errorRate: p.scanned > 0 ? Double(p.failed) / Double(p.scanned) : 0,
                peakVramDeltaBytes: peakDelta
            )
        }()
        return metrics
    }.value
}
