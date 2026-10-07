import Foundation
import CryptoKit

/// Downloads an omni model variant into a local directory, reporting progress.
///
/// The weights come from this repository's GitHub release: the Hugging Face checkpoint with its
/// retrieval adapter already merged and the backbone in bf16 (omni-verify exportmerged), which is
/// exactly what WeightStore builds at load, so the app loads them as stored. GitHub caps a release
/// asset at 2 GiB, so model.safetensors ships as byte parts named by a small manifest, joined here
/// and checked against its SHA-256. The Hugging Face files are the fallback when the release cannot
/// be reached at all.
public final class ModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    public struct Progress: Sendable {
        public let file: String
        public let fileIndex: Int
        public let fileCount: Int
        public let received: Int64
        public let total: Int64          // -1 if unknown
    }

    /// The Hugging Face fallback's files (model.safetensors is by far the largest).
    public static let files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "adapters/retrieval/adapter_config.json",
        "adapters/retrieval/adapter_model.safetensors",
        "model.safetensors",
    ]

    // Pin official weights and processors: a changing upstream main must not silently change the vector space.
    public static let gemmaRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    public static let gemmaFiles = ["config.json", "tokenizer.json", "tokenizer_config.json",
        "processor_config.json", "preprocessor_config.json", "chat_template.jinja",
        "config_sentence_transformers.json", "model.safetensors"]

    public static func repo(for variant: ModelVariant) -> String {
        variant == .embeddingGemma2 ? "google/embeddinggemma-2" : "jinaai/jina-embeddings-v5-omni-\(variant.rawValue)-mlx"
    }

    /// Where a downloaded variant is installed.
    public static func installDir(for variant: ModelVariant) -> URL? {
        let fm = FileManager.default
        guard let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return nil }
        return appSup.appendingPathComponent("\(variant == .embeddingGemma2 ? "OmniEmbeddingGemma2" : "Omni")/\(variant.rawValue)")
    }

    private var session: URLSession!
    // `perFile` and `continuation` are written from the async download flow but read/cleared on
    // URLSession's (separate) delegate queue, so every access goes through `lock`. This is what
    // makes the @unchecked Sendable sound, and taking the continuation under the lock guarantees
    // it is resumed at most once even if didFinish and didComplete both fire.
    private let lock = NSLock()
    private var perFile: (@Sendable (Int64, Int64) -> Void)?
    private var continuation: CheckedContinuation<URL, Error>?
    private var currentTask: URLSessionDownloadTask?
    private var isCancelled = false

    private func setProgressHandler(_ handler: (@Sendable (Int64, Int64) -> Void)?) {
        lock.withLock { perFile = handler }
    }
    private func reportProgress(_ written: Int64, _ total: Int64) {
        let handler = lock.withLock { perFile }
        handler?(written, total)
    }
    private func setContinuation(_ cont: CheckedContinuation<URL, Error>) {
        lock.withLock { continuation = cont }
    }
    private func takeContinuation() -> CheckedContinuation<URL, Error>? {
        lock.withLock { let c = continuation; continuation = nil; return c }
    }

    public override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    public static let releaseTag = "embed-weights-v1"
    public static let repository = "hanxiao/omni-macos"
    static let releaseFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    /// Release assets are a flat namespace, so the variant is folded into the asset name.
    static func releaseURL(_ variant: ModelVariant, _ name: String) -> URL? {
        URL(string: "https://github.com/\(repository)/releases/download/\(releaseTag)/\(variant.rawValue)-\(name)")
    }

    /// `<variant>-omni-model.json`: how model.safetensors was split, and what it must hash to.
    struct Manifest: Decodable {
        let parts: Int
        let partBytes: Int64
        let bytes: Int64
        let sha256: String
    }

    /// Download `variant` into `dest`: the release when it can be reached, Hugging Face otherwise.
    public func download(variant: ModelVariant, to dest: URL, onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        if variant == .embeddingGemma2 {
            try await downloadFromHub(variant: variant, to: dest, onProgress: onProgress)
            return
        }
        let manifest: Manifest
        do {
            guard let url = Self.releaseURL(variant, "omni-model.json") else { throw OmniError.model("bad release URL") }
            let (data, response) = try await session.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw OmniError.model("no release manifest") }
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            if lock.withLock({ isCancelled }) { throw URLError(.cancelled) }
            try await downloadFromHub(variant: variant, to: dest, onProgress: onProgress)
            return
        }
        try await downloadRelease(variant: variant, manifest: manifest, to: dest, onProgress: onProgress)
    }

    private func downloadRelease(variant: ModelVariant, manifest: Manifest, to dest: URL,
                                 onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)
        let count = Self.releaseFiles.count + 1
        // An adapter beside merged weights would be merged a second time; WeightStore also refuses,
        // but a half-finished Hugging Face download should not leave one behind.
        try? fm.removeItem(at: dest.appendingPathComponent("adapters"))
        for (idx, rel) in Self.releaseFiles.enumerated() {
            if lock.withLock({ isCancelled }) { throw URLError(.cancelled) }
            let fileURL = dest.appendingPathComponent(rel)
            if let size = try? fm.attributesOfItem(atPath: fileURL.path)[.size] as? Int64, size > 0 {
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: count, received: size, total: size)); continue
            }
            guard let url = Self.releaseURL(variant, rel) else { throw OmniError.model("bad URL for \(rel)") }
            setProgressHandler { received, total in
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: count, received: received, total: total))
            }
            let tmp = try await downloadOne(url)
            try? fm.removeItem(at: fileURL)
            try fm.moveItem(at: tmp, to: fileURL)
        }

        let model = dest.appendingPathComponent("model.safetensors")
        let idx = Self.releaseFiles.count
        if let size = try? fm.attributesOfItem(atPath: model.path)[.size] as? Int64, size == manifest.bytes {
            onProgress(Progress(file: "model.safetensors", fileIndex: idx, fileCount: count, received: size, total: size))
            return
        }
        try? fm.removeItem(at: model)                     // an older layout's file, or a stray
        // Parts append to a .partial file, so an interrupted download resumes at part granularity.
        let partial = dest.appendingPathComponent("model.safetensors.partial")
        if !fm.fileExists(atPath: partial.path) { fm.createFile(atPath: partial.path, contents: nil) }
        let have = (try? fm.attributesOfItem(atPath: partial.path)[.size] as? Int64) ?? 0
        let done = min(Int(have / manifest.partBytes), manifest.parts)
        let out = try FileHandle(forWritingTo: partial)
        defer { try? out.close() }
        try out.truncate(atOffset: UInt64(Int64(done) * manifest.partBytes))
        try out.seekToEnd()
        for k in stride(from: done + 1, through: manifest.parts, by: 1) {
            if lock.withLock({ isCancelled }) { throw URLError(.cancelled) }
            guard let url = Self.releaseURL(variant, "model.safetensors.part\(k)") else { throw OmniError.model("bad part URL") }
            let base = Int64(k - 1) * manifest.partBytes
            setProgressHandler { received, _ in
                onProgress(Progress(file: "model.safetensors", fileIndex: idx, fileCount: count,
                                    received: base + received, total: manifest.bytes))
            }
            let tmp = try await downloadOne(url)
            defer { try? fm.removeItem(at: tmp) }
            let input = try FileHandle(forReadingFrom: tmp)
            defer { try? input.close() }
            while let chunk = try input.read(upToCount: 16 << 20), !chunk.isEmpty { try out.write(contentsOf: chunk) }
        }
        try out.synchronize()
        guard try Self.sha256(of: partial) == manifest.sha256 else {
            try? fm.removeItem(at: partial)
            throw OmniError.model("model.safetensors failed its checksum; download it again")
        }
        try fm.moveItem(at: partial, to: model)
    }

    static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try h.read(upToCount: 16 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The original checkpoint and its adapter from the Hugging Face Hub; WeightStore merges them at load.
    private func downloadFromHub(variant: ModelVariant, to dest: URL, onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        let repo = Self.repo(for: variant)
        let fm = FileManager.default
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)

        let files = variant == .embeddingGemma2 ? Self.gemmaFiles : Self.files
        let revision = variant == .embeddingGemma2 ? Self.gemmaRevision : "main"
        for (idx, rel) in files.enumerated() {
            // A cancel that landed between two files (no live task to kill) must still stop the
            // loop, or the next file would start downloading as if nothing happened.
            if lock.withLock({ isCancelled }) { throw URLError(.cancelled) }
            let fileURL = dest.appendingPathComponent(rel)
            try fm.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if let size = try? fm.attributesOfItem(atPath: fileURL.path)[.size] as? Int64, size > 0 {
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: files.count, received: size, total: size))
                continue
            }
            guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(rel)") else {
                throw OmniError.model("bad URL for \(rel)")
            }
            setProgressHandler { received, total in
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: files.count, received: received, total: total))
            }
            let tmp = try await downloadOne(url)
            try? fm.removeItem(at: fileURL)
            try fm.moveItem(at: tmp, to: fileURL)
        }
    }

    private func downloadOne(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            self.setContinuation(cont)
            let task = session.downloadTask(with: url)
            lock.withLock { currentTask = task }
            task.resume()
        }
    }

    /// Cancel the download. The in-flight file's task errors with URLError.cancelled, which
    /// surfaces from download(variant:to:). Files that already completed are kept and skipped
    /// by the next attempt; the interrupted file restarts from scratch.
    public func cancel() {
        let task = lock.withLock { isCancelled = true; return currentTask }
        task?.cancel()
    }

    // MARK: URLSessionDownloadDelegate

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        reportProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // `location` is deleted when this returns; move it to a stable temp we own.
        let staged = FileManager.default.temporaryDirectory.appendingPathComponent("omni-dl-\(UUID().uuidString)")
        let cont = takeContinuation()
        do {
            try FileManager.default.moveItem(at: location, to: staged)
            // Reject HTML error pages (HF returns 200 + html for some failures).
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw OmniError.model("HTTP \(http.statusCode)")
            }
            cont?.resume(returning: staged)
        } catch {
            cont?.resume(throwing: error)
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { takeContinuation()?.resume(throwing: error) }
    }
}
