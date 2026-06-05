import Foundation

/// Downloads an omni model variant from the HuggingFace Hub into a local directory,
/// reporting progress. Only the files the Swift runtime needs are fetched (no Python).
public final class ModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    public struct Progress: Sendable {
        public let file: String
        public let fileIndex: Int
        public let fileCount: Int
        public let received: Int64
        public let total: Int64          // -1 if unknown
    }

    /// Runtime-required files (model.safetensors is by far the largest).
    public static let files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "adapters/retrieval/adapter_config.json",
        "adapters/retrieval/adapter_model.safetensors",
        "model.safetensors",
    ]

    public static func repo(for variant: ModelVariant) -> String {
        "jinaai/jina-embeddings-v5-omni-\(variant.rawValue)-mlx"
    }

    /// Where a downloaded variant is installed.
    public static func installDir(for variant: ModelVariant) -> URL? {
        let fm = FileManager.default
        guard let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return nil }
        return appSup.appendingPathComponent("Omni/\(variant.rawValue)")
    }

    private var session: URLSession!
    // `perFile` and `continuation` are written from the async download flow but read/cleared on
    // URLSession's (separate) delegate queue, so every access goes through `lock`. This is what
    // makes the @unchecked Sendable sound, and taking the continuation under the lock guarantees
    // it is resumed at most once even if didFinish and didComplete both fire.
    private let lock = NSLock()
    private var perFile: (@Sendable (Int64, Int64) -> Void)?
    private var continuation: CheckedContinuation<URL, Error>?

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

    /// Download `variant` into `dest`. Existing complete files are skipped (resume-ish).
    public func download(variant: ModelVariant, to dest: URL, onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        let repo = Self.repo(for: variant)
        let fm = FileManager.default
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)

        for (idx, rel) in Self.files.enumerated() {
            let fileURL = dest.appendingPathComponent(rel)
            try fm.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if let size = try? fm.attributesOfItem(atPath: fileURL.path)[.size] as? Int64, size > 0 {
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: Self.files.count, received: size, total: size))
                continue
            }
            guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(rel)") else {
                throw OmniError.model("bad URL for \(rel)")
            }
            setProgressHandler { received, total in
                onProgress(Progress(file: rel, fileIndex: idx, fileCount: Self.files.count, received: received, total: total))
            }
            let tmp = try await downloadOne(url)
            try? fm.removeItem(at: fileURL)
            try fm.moveItem(at: tmp, to: fileURL)
        }
    }

    private func downloadOne(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            self.setContinuation(cont)
            session.downloadTask(with: url).resume()
        }
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
