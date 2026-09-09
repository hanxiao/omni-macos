import Foundation

/// A pool of worker PROCESSES for transcribing a long document.
///
/// ## Why processes and not threads
///
/// Measured on this stack, transcribing the same 20-page scan:
///
/// | lanes | in-process (MLX streams) | separate processes |
/// |---|---|---|
/// | 1 | 1.00x | 1.00x |
/// | 2 | 1.05x | **1.47x** |
/// | 3 | 1.05x | **1.66x** |
///
/// Threads inside one process saturate almost immediately because MLX submits work through a
/// single command queue, so several streams still serialise at submission. Separate processes
/// each get their own queue and the GPU interleaves them. The predecessor project found the same
/// (1.8x at six worker processes) and this port reproduces it.
///
/// ## The cost, and why the pool is sized rather than fixed
///
/// Each worker holds its own copy of the weights - 4.5 GB for the balanced build. That is free on
/// a 512 GB machine and impossible on a 16 GB one, so the worker count is derived from Metal's
/// reported working set rather than picked. A machine that can hold one copy runs one worker and
/// loses nothing but the parallelism it never had.
public struct OCRWorkerPool: Sendable {

    /// How many workers this machine can actually hold.
    ///
    /// Each worker needs the weights plus room for its KV cache and the vision tower's transients;
    /// 2 GB of headroom per worker covers both at the full context window. The cap of 4 is not
    /// arbitrary either - throughput gains flatten well before it (1.47x, 1.66x at 2 and 3), and
    /// every extra worker is another full copy of the weights.
    /// - Parameter availableBytes: the machine's usable ceiling. Defaults to what Metal reports
    ///   it can hold; injectable so the sizing rule is testable without a second machine.
    public static func recommendedWorkers(modelBytes: Int, reserveBytes: Int = 3_000_000_000,
                                          availableBytes: Int? = nil) -> Int {
        let ceiling = availableBytes
            ?? min(omniMetalWorkingSetBytes() ?? Int(ProcessInfo.processInfo.physicalMemory),
                   Int(ProcessInfo.processInfo.physicalMemory))
        let perWorker = modelBytes + 2_000_000_000
        let affordable = (ceiling - reserveBytes) / max(perWorker, 1)
        return max(1, min(affordable, 4))
    }

    /// One page's transcription, as it crosses the process boundary.
    public struct Reply: Codable, Sendable {
        public let page: Int
        public let text: String
        public let tokens: Int
        public let seconds: Double
        public let error: String?
    }

    let executable: URL
    let modelDir: URL
    let tokenizerDir: URL?
    let pdf: URL
    let dpi: Int
    let draftLength: Int

    public init(executable: URL, modelDir: URL, tokenizerDir: URL?, pdf: URL,
                dpi: Int = 200, draftLength: Int = 3) {
        self.executable = executable
        self.modelDir = modelDir
        self.tokenizerDir = tokenizerDir
        self.pdf = pdf
        self.dpi = dpi
        self.draftLength = draftLength
    }

    /// Run `pages` across `workers` processes and return the replies in page order.
    ///
    /// Pages are pulled from a shared queue, NOT dealt out round-robin at spawn time. Static
    /// assignment looks fine on paper - pages take 1-8 s and it should average out - and it is
    /// measurably wrong: on a document whose page types repeat with period 3, three lanes each
    /// received every page of one kind, so the lane holding the 8-second ledger pages ran alone
    /// while the others idled. That configuration scored 1.14x where two lanes scored 1.40x.
    /// Real documents have periodic structure; a queue does not care.
    public func run(pages: [Int], workers: Int,
                    onPage: (@Sendable (Reply) -> Void)? = nil) throws -> [Reply] {
        guard !pages.isEmpty else { return [] }
        let lanes = max(1, min(workers, pages.count))
        let collector = ReplyCollector()
        let group = DispatchGroup()
        let failures = FailureLog()

        let queue = PageQueue(pages)
        for lane in 0 ..< lanes {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                do {
                    try self.runLane(queue: queue) { reply in
                        collector.add(reply)
                        onPage?(reply)
                    }
                } catch {
                    failures.add("worker \(lane): \(error)")
                }
            }
        }
        group.wait()
        if let first = failures.first, collector.isEmpty {
            throw OmniError.model("every OCR worker failed - \(first)")
        }
        // A partially failed run keeps what it got and says what it lost, rather than throwing
        // away completed pages the way an uncaught worker death would.
        if failures.count > 0 {
            OmniLog.warn("ocr: \(failures.count) worker(s) failed; "
                         + "\(collector.count) of \(pages.count) pages returned")
        }
        return collector.sorted()
    }

    /// One lane: spawn a worker, then feed it pages one at a time, taking the next only when the
    /// previous reply lands. That is what makes the queue dynamic - a slow page holds up its own
    /// lane and nothing else.
    private func runLane(queue: PageQueue, deliver: (Reply) -> Void) throws {
        guard queue.peek() != nil else { return }
        let process = Process()
        process.executableURL = executable
        var arguments = ["--ocr-worker", modelDir.path, "--pdf", pdf.path,
                         "--dpi", String(dpi), "--draft", String(draftLength)]
        if let tokenizerDir { arguments += ["--tokenizer", tokenizerDir.path] }
        process.arguments = arguments

        let stdin = Pipe(), stdout = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        try process.run()
        defer {
            try? stdin.fileHandleForWriting.close()
            process.waitUntilExit()
        }

        let reader = LineReader(stdout.fileHandleForReading)
        let decoder = JSONDecoder()
        while let page = queue.next() {
            stdin.fileHandleForWriting.write(Data("\(page)\n".utf8))
            guard let line = reader.next() else {
                // The worker died mid-document. Put the page back so another lane can take it.
                queue.giveBack(page)
                throw OmniError.model("OCR worker exited early on page \(page + 1)")
            }
            if let reply = try? decoder.decode(Reply.self, from: line) { deliver(reply) }
        }
    }
}

/// The other half of the pool: the body a host executable runs when it sees `--ocr-worker`.
///
/// Living in OmniKit rather than in one CLI means the app can re-exec ITSELF as a worker - the
/// weights are the only large thing a worker needs, and the app bundle already contains
/// everything else.
public enum OCRWorker {
    /// Returns true when `arguments` asked for worker mode, in which case this call does the
    /// whole job and the caller should exit.
    public static func runIfRequested(_ arguments: [String]) async -> Bool {
        guard let index = arguments.firstIndex(of: "--ocr-worker"),
              index + 1 < arguments.count else { return false }
        func option(_ name: String) -> String? {
            arguments.firstIndex(of: name).flatMap {
                $0 + 1 < arguments.count ? arguments[$0 + 1] : nil
            }
        }
        let modelDir = URL(fileURLWithPath: arguments[index + 1])
        guard let pdfPath = option("--pdf") else { return true }
        let dpi = option("--dpi").flatMap(Int.init) ?? 200
        let draft = option("--draft").flatMap(Int.init) ?? 3
        let tokenizer = option("--tokenizer").map { URL(fileURLWithPath: $0) }

        do {
            let model = try await OCRModel(modelDir: modelDir, tokenizerDir: tokenizer)
            let encoder = JSONEncoder()
            while let line = readLine(strippingNewline: true), let page = Int(line) {
                let start = Date()
                let reply: OCRWorkerPool.Reply
                do {
                    let out = try model.transcribe(pdfAt: URL(fileURLWithPath: pdfPath),
                                                   pageRange: page ..< (page + 1), dpi: dpi,
                                                   draftLength: draft)
                    let text = out.pages.first?.text ?? ""
                    reply = .init(page: page, text: text,
                                  tokens: out.pages.first?.tokenCount ?? 0,
                                  seconds: Date().timeIntervalSince(start), error: nil)
                } catch {
                    reply = .init(page: page, text: "", tokens: 0,
                                  seconds: Date().timeIntervalSince(start),
                                  error: String(describing: error))
                }
                if let data = try? encoder.encode(reply) {
                    FileHandle.standardOutput.write(data)
                    FileHandle.standardOutput.write(Data("\n".utf8))
                }
            }
        } catch {
            // A worker that cannot load the model exits quietly; the pool reports the shortfall.
        }
        return true
    }
}

/// Pages waiting to be transcribed, shared by every lane.
private final class PageQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [Int]
    init(_ pages: [Int]) { self.pending = pages.reversed() }
    func next() -> Int? { lock.withLock { pending.popLast() } }
    func peek() -> Int? { lock.withLock { pending.last } }
    /// Return a page a dead worker never finished, so a surviving lane can pick it up.
    func giveBack(_ page: Int) { lock.withLock { pending.append(page) } }
}

/// Newline-delimited reads from a pipe. `readDataToEndOfFile` cannot be used here: the lane has
/// to see each reply as it arrives in order to hand out the next page.
private final class LineReader {
    private let handle: FileHandle
    private var buffer = Data()
    init(_ handle: FileHandle) { self.handle = handle }

    func next() -> Data? {
        while true {
            if let index = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = buffer[buffer.startIndex ..< index]
                buffer.removeSubrange(buffer.startIndex ... index)
                if !line.isEmpty { return Data(line) }
                continue
            }
            let chunk = handle.availableData
            if chunk.isEmpty { return nil }
            buffer.append(chunk)
        }
    }
}

private final class FailureLog: @unchecked Sendable {
    private let lock = NSLock()
    private var messages: [String] = []
    func add(_ message: String) { lock.withLock { messages.append(message) } }
    var first: String? { lock.withLock { messages.first } }
    var count: Int { lock.withLock { messages.count } }
}

private final class ReplyCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var replies: [OCRWorkerPool.Reply] = []
    func add(_ reply: OCRWorkerPool.Reply) { lock.withLock { replies.append(reply) } }
    func sorted() -> [OCRWorkerPool.Reply] { lock.withLock { replies.sorted { $0.page < $1.page } } }
    var isEmpty: Bool { lock.withLock { replies.isEmpty } }
    var count: Int { lock.withLock { replies.count } }
}
