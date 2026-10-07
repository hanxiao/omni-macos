import Foundation
import CoreGraphics
import PDFKit
import MLX

/// App facade: Google EmbeddingGemma 2 by default; Jina remains an explicit comparison option.
/// The Google runtime is a persistent local MLX Python process, not an HTTP/cloud service.
public final class OmniEngine: Embedder, @unchecked Sendable {
    private let jina: JinaEngine?
    private let gemma: EmbeddingGemmaWorker?
    public let modelDir: URL
    public let dim: Int
    private let stateLock = NSLock()
    private var vision: Bool
    private var audio: Bool
    private var queryAt = Date.distantPast
    public var isEmbeddingGemma2: Bool { gemma != nil }
    public var lastError: String? { gemma?.lastError }
    public var helperMemory: (footprint: Int, active: Int, cache: Int) { gemma?.memory ?? (0, 0, 0) }
    public func setHelperMemoryLimit(_ bytes: Int) { gemma?.setMemoryLimit(bytes) }
    public var usesRawAudio: Bool { gemma != nil }
    public var supportsPatchTags: Bool { jina != nil }
    public var supportsImages: Bool { jina?.supportsImages ?? stateLock.withLock { vision } }
    public var supportsVideo: Bool { supportsImages }
    public var supportsAudio: Bool { jina?.supportsAudio ?? stateLock.withLock { audio } }
    public var tokensProcessed: Int { jina?.tokensProcessed ?? gemma?.tokensProcessed ?? 0 }
    public var gpuBusySeconds: TimeInterval { jina?.gpuBusySeconds ?? gemma?.gpuBusySeconds ?? 0 }
    public var interactiveQueryActive: Bool { jina?.interactiveQueryActive ?? stateLock.withLock { -queryAt.timeIntervalSinceNow < 2 } }
    public var tagger: OmniTagger? {
        get { jina?.tagger }
        set { jina?.tagger = newValue }
    }
    public static var indexGateWindow: Int {
        get { JinaEngine.indexGateWindow }
        set { JinaEngine.indexGateWindow = newValue }
    }
    static var adaptiveBatch: Bool {
        get { JinaEngine.adaptiveBatch }
        set { JinaEngine.adaptiveBatch = newValue }
    }
    public static let tokOverlapEnabled = JinaEngine.tokOverlapEnabled
    static func defaultGateWindow(gpuCores: Int?) -> Int { JinaEngine.defaultGateWindow(gpuCores: gpuCores) }

    public static func variant(at directory: URL) -> ModelVariant {
        let config = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
        let json = config.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        if json?["model_type"] as? String == "embedding_gemma2" { return .embeddingGemma2 }
        return directory.path.contains("nano") ? .nano : .small
    }

    public init(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true, keepAudio: Bool = true) async throws {
        self.modelDir = modelDir; vision = keepVision; audio = keepAudio
        if Self.variant(at: modelDir) == .embeddingGemma2 {
            jina = nil
            gemma = try EmbeddingGemmaWorker(modelDir: modelDir, vision: keepVision, audio: keepAudio)
            dim = 768
        } else {
            gemma = nil
            let engine = try await JinaEngine.loadValidated(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes,
                                                            keepVision: keepVision, keepAudio: keepAudio)
            jina = engine; dim = engine.dim
        }
    }
    public static func loadValidated(modelDir: URL, gpuCacheBytes: Int = 0, keepVision: Bool = true,
                                     keepAudio: Bool = true, maxAttempts: Int = 4) async throws -> OmniEngine {
        try await OmniEngine(modelDir: modelDir, gpuCacheBytes: gpuCacheBytes, keepVision: keepVision, keepAudio: keepAudio)
    }
    public static func load() async throws -> OmniEngine {
        guard let dir = ModelLocator.resolve() else { throw OmniError.model("Download EmbeddingGemma 2 in Settings") }
        return try await loadValidated(modelDir: dir)
    }
    public func noteInteractive() { jina?.noteInteractive(); stateLock.withLock { queryAt = Date() } }
    public func queryVectorGraph(_ text: String) -> MLXArray? { jina?.queryVectorGraph(text) }
    public func embedQuery(_ text: String) -> [Float] { noteInteractive(); return embedText(text, as: .query) }
    public func embedText(_ text: String, as type: OmniInputType) -> [Float] {
        jina?.embedText(text, as: type) ?? gemma!.text([text], as: type)[0]
    }
    public func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        jina?.embedTextBatch(texts, as: type) ?? gemma!.text(texts, as: type)
    }
    public func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        if let jina { return jina.embedTextBatches(batches, as: type) }
        return batches.map { gemma!.text($0, as: type) }
    }
    public func warmText() { if let jina { jina.warmText() } else { _ = gemma!.text(["Bonjour"], as: .query) } }
    public func tokenizeOnlyForBenchmark(_ batches: [[String]], as type: OmniInputType) -> Int {
        if let jina { return jina.tokenizeOnlyForBenchmark(batches, as: type) }
        return (try? gemma!.request(["op": "tokenize", "texts": batches.flatMap { $0 }, "query": type == .query])["tokens"] as? Int) ?? 0
    }
    public func prepareImage(_ image: CGImage) -> OmniVisionPreprocess.RawPatches {
        if jina != nil { return OmniVisionPreprocess.preprocessRaw(image) }
        return OmniVisionPreprocess.RawPatches(pixels: [], gridTHW: [], encodedImage: EmbeddingGemmaWorker.png(image))
    }
    public func embedImage(_ image: CGImage) -> [Float]? {
        if let jina { return jina.embedImage(image) }
        guard supportsImages, let png = EmbeddingGemmaWorker.png(image) else { return nil }
        return gemma!.images([png])?.first
    }
    public func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
        if let jina { return jina.embedImages(raws) }
        guard supportsImages else { return nil }
        let images = raws.compactMap(\.encodedImage)
        guard images.count == raws.count else { return nil }
        return gemma!.images(images)
    }
    public func embedImagesTagged(_ raws: [OmniVisionPreprocess.RawPatches]) -> (vecs: [[Float]], tags: [[String]])? {
        if let jina { return jina.embedImagesTagged(raws) }
        return embedImages(raws).map { ($0, Array(repeating: [], count: $0.count)) }
    }
    public func embedImagesTaggedHQ(_ raws: [OmniVisionPreprocess.RawPatches], crops: [[OmniVisionPreprocess.RawPatches]]) -> (vecs: [[Float]], tags: [[String]])? {
        if let jina { return jina.embedImagesTaggedHQ(raws, crops: crops) }
        return embedImagesTagged(raws)
    }
    public func embedImagesTagScores(_ raws: [OmniVisionPreprocess.RawPatches], tagger: OmniTagger) -> [[Float]]? {
        jina?.embedImagesTagScores(raws, tagger: tagger)
    }
    public func seedTaggerPrior(_ tagger: OmniTagger) { jina?.seedTaggerPrior(tagger) }
    public func embedVideoFrames(_ frames: [CGImage]) -> [Float]? {
        if let jina { return jina.embedVideoFrames(frames) }
        guard supportsVideo else { return nil }
        let images = frames.compactMap { EmbeddingGemmaWorker.png($0) }
        guard images.count == frames.count else { return nil }
        return gemma!.images(images, video: true)?.first
    }
    public func embedVideoFramesTagged(_ frames: [CGImage]) -> (vec: [Float], tags: [String])? {
        if let jina { return jina.embedVideoFramesTagged(frames) }
        return embedVideoFrames(frames).map { ($0, []) }
    }
    public func embedAudio(_ url: URL) -> [Float]? {
        if let jina { return jina.embedAudio(url) }
        guard supportsAudio, let reader = OmniAudioPreprocess.AudioSegmentReader(url: url),
              let samples = reader.nextSegment() else { return nil }
        return gemma!.pcm(samples)
    }
    /// Google receives PCM, prepared by AudioSegmentReader(rawPCM: true), never Jina mels.
    public func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? {
        if let jina { return jina.embedAudioMel(mel, frames: frames) }
        return supportsAudio ? gemma!.pcm(mel) : nil
    }
    public func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? {
        if let jina { return jina.embedAudioMelBatch(mels, frames: frames) }
        guard supportsAudio, mels.count == frames.count else { return nil }
        var vectors: [[Float]] = []
        for samples in mels { guard let v = gemma!.pcm(samples) else { return nil }; vectors.append(v) }
        return vectors
    }
    public func embedImageQuery(_ image: CGImage, asDocument: Bool = false) -> [Float]? {
        if let jina { return jina.embedImageQuery(image, asDocument: asDocument) }
        noteInteractive()
        guard supportsImages, let png = EmbeddingGemmaWorker.png(image) else { return nil }
        return gemma!.images([png], indexing: false)?.first
    }
    public func embedVideoQuery(_ frames: [CGImage], asDocument: Bool = false) -> [Float]? {
        if let jina { return jina.embedVideoQuery(frames, asDocument: asDocument) }
        noteInteractive()
        guard supportsVideo else { return nil }
        let images = frames.compactMap { EmbeddingGemmaWorker.png($0) }
        guard images.count == frames.count else { return nil }
        return gemma!.images(images, video: true, indexing: false)?.first
    }
    public func embedAudioQuery(_ url: URL, asDocument: Bool = false) -> [Float]? {
        if let jina { return jina.embedAudioQuery(url, asDocument: asDocument) }
        noteInteractive()
        guard supportsAudio, let reader = OmniAudioPreprocess.AudioSegmentReader(url: url),
              let samples = reader.nextSegment() else { return nil }
        return gemma!.pcm(samples, indexing: false)
    }
    public func embedFileQuery(_ url: URL, asDocument: Bool = false, maxImageDimension: Int = 1568, maxVideoFrames: Int = 6) -> [Float]? {
        if let jina { return jina.embedFileQuery(url, asDocument: asDocument, maxImageDimension: maxImageDimension, maxVideoFrames: maxVideoFrames) }
        switch FileExtractor.kind(for: url) {
        case .image:
            return FileExtractor.loadImage(url, maxDimension: maxImageDimension).flatMap { embedImageQuery($0, asDocument: asDocument) }
        case .video:
            return embedVideoQuery(FileExtractor.videoFrames(url, maxFrames: maxVideoFrames, maxDimension: maxImageDimension), asDocument: asDocument)
        case .audio: return embedAudioQuery(url, asDocument: asDocument)
        case .text, .scan:
            switch (try? FileExtractor.extract(url, maxImageDimension: maxImageDimension, maxVideoFrames: maxVideoFrames)) ?? .empty {
            case .text(let text), .pagedText(let text, _): return embedText(text, as: asDocument ? .passage : .query)
            case .images(let pages): return pooledPages(pages)
            case .scannedPDF(let count):
                guard let doc = PDFDocument(url: url) else { return nil }
                let pages = (0..<min(count, 8)).compactMap { FileExtractor.renderPDFPage(doc, index: $0, maxDimension: maxImageDimension) }
                return pooledPages(pages)
            case .empty: return nil
            }
        case .none: return nil
        }
    }
    private func pooledPages(_ pages: [CGImage]) -> [Float]? {
        noteInteractive()
        guard supportsImages, !pages.isEmpty else { return nil }
        let images = pages.compactMap { EmbeddingGemmaWorker.png($0) }
        guard images.count == pages.count, let vectors = gemma!.images(images, indexing: false) else { return nil }
        var mean = [Float](repeating: 0, count: dim)
        for vector in vectors { for i in mean.indices { mean[i] += vector[i] } }
        let norm = sqrt(mean.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else { return nil }
        return mean.map { $0 / norm }
    }
    public func setTowers(keepVision: Bool, keepAudio: Bool) {
        if let jina { jina.setTowers(keepVision: keepVision, keepAudio: keepAudio); return }
        if gemma!.setTowers(vision: keepVision, audio: keepAudio) {
            stateLock.withLock { vision = keepVision; audio = keepAudio }
        }
    }
    public func indexingIdle() { if let jina { jina.indexingIdle() } else { gemma!.idle() } }
    public func recoverMediaPath() -> Bool { jina?.recoverMediaPath() ?? false }
    public func rebuildEncodersForDiagnosis() { jina?.rebuildEncodersForDiagnosis() }
    public func weightDigests() -> [(name: String, digest: UInt64, nonFinite: Int, gpuSum: Double)] { jina?.weightDigests() ?? [] }
    func runLowPriorityGPU<T>(_ work: () -> T) -> T { if let jina { return jina.runLowPriorityGPU(work) }; return work() }
    public func imageEncoderForTesting() -> OmniImageEncoder? { jina?.imageEncoderForTesting() }
    public var docPrefixForTesting: [Int] { jina?.docPrefixForTesting ?? [] }
}
