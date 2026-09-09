import Foundation

/// Fetches an OCR weight variant from its GitHub release into Application Support.
///
/// Same shape as `ModelDownloader` (URLSession download tasks, lock-guarded continuation) with
/// one difference that matters: the file LIST is not known up front. `omni-ocr.json` is fetched
/// first and names the shards, so a build that is re-sharded later does not need an app update.
///
/// Files that already exist with a non-zero size are skipped, so an interrupted 4 GB transfer
/// resumes at shard granularity rather than starting over.
public final class OCRModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    public struct Progress: Sendable {
        public let file: String
        public let fileIndex: Int
        public let fileCount: Int
        public let received: Int64
        public let total: Int64          // -1 when the server does not say
        /// The WHOLE download, not the file in flight. Reporting per-file progress meant a bar
        /// that emptied and refilled once per shard, which only made sense next to a "part k of n"
        /// counter - a number about how the weights happen to be packaged, which is not the
        /// reader's business.
        public let documentReceived: Int64
        public let documentTotal: Int64  // 0 until the manifest lands
    }

    private var session: URLSession!
    private let lock = NSLock()
    private var perFile: (@Sendable (Int64, Int64) -> Void)?
    /// Bytes of files already on disk, and what the manifest says the whole download weighs.
    private var completedBytes: Int64 = 0
    private var documentTotal: Int64 = 0
    private var continuation: CheckedContinuation<URL, Error>?
    private var currentTask: URLSessionDownloadTask?
    private var isCancelled = false

    public override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    public func download(variant: OCRModelCatalog.Variant, to destination: URL,
                         onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        lock.withLock { completedBytes = 0; documentTotal = 0 }

        // The manifest first - it is what names everything else, and what says how big the whole
        // download is.
        try await fetch(variant: variant, file: OCRModelCatalog.manifestFile, into: destination,
                        index: 0, count: 1, onProgress: onProgress)
        let manifest = try OCRModelCatalog.readManifest(at: destination)
        lock.withLock { completedBytes = 0; documentTotal = manifest.bytes }

        for (index, file) in manifest.files.enumerated() where file != OCRModelCatalog.manifestFile {
            if lock.withLock({ isCancelled }) { throw URLError(.cancelled) }
            try await fetch(variant: variant, file: file, into: destination,
                            index: index, count: manifest.files.count, onProgress: onProgress)
        }
    }

    private func fetch(variant: OCRModelCatalog.Variant, file: String, into destination: URL,
                       index: Int, count: Int,
                       onProgress: @escaping @Sendable (Progress) -> Void) async throws {
        let target = destination.appendingPathComponent(file)
        let fm = FileManager.default
        if let size = try? fm.attributesOfItem(atPath: target.path)[.size] as? Int64, size > 0 {
            let done = lock.withLock { completedBytes += size; return completedBytes }
            let whole = lock.withLock { documentTotal }
            onProgress(Progress(file: file, fileIndex: index, fileCount: count,
                                received: size, total: size,
                                documentReceived: done, documentTotal: whole))
            return
        }
        guard let url = OCRModelCatalog.assetURL(variant: variant, file: file) else {
            throw OmniError.model("bad asset URL for \(file)")
        }
        let base = lock.withLock { completedBytes }
        let whole = lock.withLock { documentTotal }
        lock.withLock {
            perFile = { received, total in
                onProgress(Progress(file: file, fileIndex: index, fileCount: count,
                                    received: received, total: total,
                                    documentReceived: base + received, documentTotal: whole))
            }
        }
        let staged = try await downloadOne(url)
        try? fm.removeItem(at: target)
        try fm.moveItem(at: staged, to: target)
        if let size = try? fm.attributesOfItem(atPath: target.path)[.size] as? Int64 {
            lock.withLock { completedBytes += size }
        }
    }

    private func downloadOne(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock { self.continuation = continuation }
            let task = session.downloadTask(with: url)
            lock.withLock { currentTask = task }
            task.resume()
        }
    }

    public func cancel() {
        let task = lock.withLock { isCancelled = true; return currentTask }
        task?.cancel()
    }

    /// Remove an installed variant. Deleting 4 GB is the user's call, so this is only ever
    /// reached from an explicit action.
    public static func remove(_ variant: OCRModelCatalog.Variant) throws {
        guard let dir = OCRModelCatalog.installDir(for: variant) else { return }
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    private func takeContinuation() -> CheckedContinuation<URL, Error>? {
        lock.withLock { let c = continuation; continuation = nil; return c }
    }

    // MARK: URLSessionDownloadDelegate

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                           totalBytesExpectedToWrite: Int64) {
        let handler = lock.withLock { perFile }
        handler?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didFinishDownloadingTo location: URL) {
        let staged = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-ocr-dl-\(UUID().uuidString)")
        let continuation = takeContinuation()
        do {
            try FileManager.default.moveItem(at: location, to: staged)
            // A release asset that 404s comes back as an HTML page with a 200 on some paths;
            // treat anything but a 200 as a failure rather than writing the page to disk as if
            // it were a shard.
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw OmniError.model("HTTP \(http.statusCode)")
            }
            continuation?.resume(returning: staged)
        } catch {
            continuation?.resume(throwing: error)
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { takeContinuation()?.resume(throwing: error) }
    }
}
