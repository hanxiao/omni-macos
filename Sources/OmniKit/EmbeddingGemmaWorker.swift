import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Darwin

/// Persistent local MLX process. No HTTP endpoint and no inference-time networking.
final class EmbeddingGemmaWorker: @unchecked Sendable {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private let lock = NSLock()
    private var buffered = Data()
    private var terminalError: String?
    private let log: FileHandle
    private let metricsLock = NSLock()
    private var tokenCount = 0
    private var busySeconds: TimeInterval = 0
    private var failure: String?
    private var activeBytes = 0
    private var cacheBytes = 0
    private var idlePending = false
    var tokensProcessed: Int { metricsLock.withLock { tokenCount } }
    var gpuBusySeconds: TimeInterval { metricsLock.withLock { busySeconds } }
    var lastError: String? { metricsLock.withLock { failure } }
    var memory: (footprint: Int, active: Int, cache: Int) {
        var info = rusage_info_v2()
        let rc = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(process.processIdentifier, RUSAGE_INFO_V2, $0)
            }
        }
        return metricsLock.withLock { (rc == 0 ? Int(info.ri_phys_footprint) : 0, activeBytes, cacheBytes) }
    }

    static var runtimeDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/OmniEmbeddingGemma2/runtime")
    }

    init(modelDir: URL, vision: Bool, audio: Bool) throws {
        let env = ProcessInfo.processInfo.environment
        let python = env["OMNI_GEMMA_PYTHON"].map { URL(fileURLWithPath: $0) }
            ?? Self.runtimeDir.appendingPathComponent("venv/bin/python")
        let worker = env["OMNI_GEMMA_WORKER"].map { URL(fileURLWithPath: $0) }
            ?? Bundle.main.url(forResource: "worker", withExtension: "py", subdirectory: "EmbeddingGemma2")
            ?? Bundle.main.url(forResource: "worker", withExtension: "py")
            ?? Self.runtimeDir.appendingPathComponent("worker.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: worker.path) else {
            throw OmniError.model("EmbeddingGemma 2 runtime missing. Run Scripts/setup-embeddinggemma2.sh in the fork, then relaunch.")
        }
        let logDir = Self.runtimeDir.appendingPathComponent("logs")
        try FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        let logURL = logDir.appendingPathComponent("worker-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
        self.log = try FileHandle(forWritingTo: logURL)
        process.executableURL = python
        process.arguments = ["-u", worker.path, modelDir.path, vision ? "1" : "0", audio ? "1" : "0"]
        process.standardInput = input; process.standardOutput = output; process.standardError = log
        process.environment = env.merging(["HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                          "TOKENIZERS_PARALLELISM": "false",
                                          "OMNI_GEMMA_MEMORY_LIMIT": String(OmniMemoryBudget.capBytes)]) { _, new in new }
        // A dead child must produce a write error, not SIGPIPE terminating the app.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        try process.run()
        // Close parent copies of the child's ends, so an exit produces EOF rather than a hang.
        try input.fileHandleForReading.close()
        try output.fileHandleForWriting.close()
        do {
            let hello = try readResponse(timeout: 180)
            guard hello["ready"] as? Bool == true, hello["dimension"] as? Int == 768 else {
                throw OmniError.model(hello["error"] as? String ?? "EmbeddingGemma 2 failed to initialize")
            }
        } catch {
            if process.isRunning { process.terminate() }
            throw error
        }
    }

    deinit {
        try? input.fileHandleForWriting.close()
        if process.isRunning { process.terminate() }
        try? output.fileHandleForReading.close()
        try? log.close()
    }

    private func readResponse(timeout: TimeInterval) throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            if let newline = buffered.firstIndex(of: 10) {
                let line = buffered[..<newline]
                buffered.removeSubrange(...newline)
                guard let value = try JSONSerialization.jsonObject(with: Data(line)) as? [String: Any] else {
                    throw OmniError.model("Invalid response from EmbeddingGemma 2")
                }
                return value
            }
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw OmniError.model("EmbeddingGemma 2 timed out") }
            var fd = pollfd(fd: output.fileHandleForReading.fileDescriptor, events: Int16(POLLIN), revents: 0)
            let ready = poll(&fd, 1, Int32(min(remaining * 1000, 1000)))
            if ready < 0, errno == EINTR { continue }
            guard ready >= 0 else { throw OmniError.model("EmbeddingGemma 2 pipe failed") }
            if ready == 0 { continue }
            var bytes = [UInt8](repeating: 0, count: 65536)
            let n = bytes.withUnsafeMutableBytes { Darwin.read(fd.fd, $0.baseAddress, $0.count) }
            guard n > 0 else { throw OmniError.model("EmbeddingGemma 2 worker exited; see runtime/logs") }
            buffered.append(contentsOf: bytes.prefix(n))
            guard buffered.count < 16 * 1024 * 1024 else { throw OmniError.model("EmbeddingGemma 2 response too large") }
        }
    }

    func request(_ payload: [String: Any], indexing: Bool = false) throws -> [String: Any] {
        try lock.withLock {
            if let terminalError { throw OmniError.model(terminalError) }
            guard process.isRunning else { throw OmniError.model("EmbeddingGemma 2 worker is no longer running") }
            let start = Date()
            defer { metricsLock.withLock { busySeconds += -start.timeIntervalSinceNow } }
            var receivedReply = false
            do {
                var data = try JSONSerialization.data(withJSONObject: payload)
                data.append(10)
                try input.fileHandleForWriting.write(contentsOf: data)
                let reply = try readResponse(timeout: 180)
                receivedReply = true
                if let error = reply["error"] as? String { throw OmniError.model(error) }
                metricsLock.withLock {
                    failure = nil
                    activeBytes = reply["active_bytes"] as? Int ?? activeBytes
                    cacheBytes = reply["cache_bytes"] as? Int ?? cacheBytes
                    if indexing { tokenCount += reply["tokens"] as? Int ?? 0 }
                }
                return reply
            } catch {
                metricsLock.withLock { failure = String(describing: error) }
                // Failed transport cannot be reused; complete model-error replies can be retried.
                if !receivedReply || !process.isRunning {
                    terminalError = String(describing: error)
                    if process.isRunning { process.terminate() }
                }
                throw error
            }
        }
    }

    func vectors(_ payload: [String: Any], count: Int, indexing: Bool = false) -> [[Float]]? {
        do {
            let reply = try request(payload, indexing: indexing)
            guard let rows = reply["vectors"] as? [[NSNumber]], rows.count == count else {
                throw OmniError.model("EmbeddingGemma 2 returned the wrong number of vectors")
            }
            let values = rows.map { $0.map(\.floatValue) }
            guard values.allSatisfy({ row in
                row.count == 768 && row.allSatisfy(\.isFinite)
                    && abs(sqrt(row.reduce(Float(0)) { $0 + $1 * $1 }) - 1) < 0.01
            }) else { throw OmniError.model("EmbeddingGemma 2 returned invalid vectors") }
            return values
        } catch {
            FileHandle.standardError.write(Data("EmbeddingGemma 2: \(error)\n".utf8))
            return nil
        }
    }

    static func png(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }

    func text(_ texts: [String], as type: OmniInputType) -> [[Float]] {
        if texts.isEmpty { return [] }
        return vectors(["op": "text", "texts": texts, "query": type == .query], count: texts.count, indexing: type != .query)
            ?? Array(repeating: Array(repeating: Float.nan, count: 768), count: texts.count)
    }
    func images(_ images: [Data], video: Bool = false, indexing: Bool = true) -> [[Float]]? {
        guard !images.isEmpty else { return nil }
        return vectors(["op": video ? "video" : "images", "images": images.map { $0.base64EncodedString() }],
                       count: video ? 1 : images.count, indexing: indexing)
    }
    func pcm(_ samples: [Float], indexing: Bool = true) -> [Float]? {
        guard !samples.isEmpty else { return nil }
        let data = samples.withUnsafeBytes { Data($0) }
        return vectors(["op": "audio", "pcm": data.base64EncodedString()], count: 1, indexing: indexing)?.first
    }
    func setTowers(vision: Bool, audio: Bool) -> Bool {
        do { _ = try request(["op": "towers", "vision": vision, "audio": audio]); return true }
        catch { return false }
    }
    func setMemoryLimit(_ bytes: Int) { _ = try? request(["op": "memory", "bytes": bytes]) }
    func idle() {
        let schedule = metricsLock.withLock {
            if idlePending { return false }
            idlePending = true
            return true
        }
        guard schedule else { return }
        // Called from the UI after search: never block its actor behind an indexing request.
        DispatchQueue.global(qos: .utility).async { [self] in
            _ = try? request(["op": "idle"])
            metricsLock.withLock { idlePending = false }
        }
    }
}
