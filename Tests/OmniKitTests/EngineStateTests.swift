import XCTest
import MLX
@testable import OmniKit

/// The engine's and the downloaders' implicit state machines (docs/review-2026-11-06): the tower
/// release against a switch back to the GPU, a Cancel with nothing to cancel, the OCR memory hold.
final class EngineStateTests: XCTestCase {
    // MARK: - 12. Switching back to the GPU during the tower release

    /// The release waits for the gate. A switch back to the GPU that lands meanwhile read the flag as
    /// down and let the programs go, then the release let the GPU towers go: no towers at all.
    func testASwitchToTheGPUDuringTheTowerReleaseLeavesTowers() async throws {
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async { Self.raceTheRelease(engine); done.resume() }
        }
    }

    /// Hold the gate with an embed, the release queued behind it, the switch landing in between. Not
    /// async: these wait, which a test's async context may not.
    private static func raceTheRelease(_ e: OmniEngine) {
        XCTAssertNotNil(e.imageEncoderForTesting())
        e.setComputeUnitForTest(.neuralEngine)
        let holding = DispatchSemaphore(value: 0), letGo = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { e.holdGateForTest(holding, until: letGo) }
        holding.wait()
        let released = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { e.releaseGPUTowers(vision: true, audio: true); released.signal() }
        Thread.sleep(forTimeInterval: 0.2)   // it is waiting for the gate
        XCTAssertFalse(e.setComputeUnit(.gpu, cache: FileManager.default.temporaryDirectory))
        letGo.signal()
        released.wait()
        XCTAssertNotNil(e.imageEncoderForTesting(), "no towers left, and no reload coming")
        XCTAssertFalse(e.gpuTowersReleased)
        XCTAssertNotNil(e.embedImage(Self.image()), "images refused")

        // Still on the Neural Engine, it goes as it should.
        e.setComputeUnitForTest(.neuralEngine)
        e.releaseGPUTowers(vision: true, audio: true)
        XCTAssertNil(e.imageEncoderForTesting())
        XCTAssertTrue(e.gpuTowersReleased)
    }

    private static func image() -> CGImage {
        let ctx = CGContext(data: nil, width: 224, height: 224, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.2, green: 0.6, blue: 0.4, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 224, height: 224))
        return ctx.makeImage()!
    }

    // MARK: - 19. The batch budget during an OCR hold

    func testAnOCRHoldLeavesIndexingsBudgetAlone() {
        let was = OmniMemoryBudget.capBytes
        let wasLimit = MLX.Memory.memoryLimit, wasCache = MLX.Memory.cacheLimit
        defer { OmniMemoryBudget.capBytes = was; MLX.Memory.memoryLimit = wasLimit; MLX.Memory.cacheLimit = wasCache }
        omniSetMemoryHeadroom(1_500_000_000, residentBytes: 2_000_000_000)
        let budget = OmniMemoryBudget.capBytes
        XCTAssertLessThan(budget, Int(ProcessInfo.processInfo.physicalMemory))
        omniSetOCRMemory()
        XCTAssertEqual(OmniMemoryBudget.capBytes, budget, "indexing's budget taken from physical memory")
        XCTAssertEqual(MLX.Memory.memoryLimit, Int(ProcessInfo.processInfo.physicalMemory), "OCR's own limit is still lifted")
    }

    // MARK: - 19. What a killed build leaves

    @available(macOS 15.0, *)
    func testLeftoverBuildFoldersAreSwept() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("ane-sweep-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: cache) }
        for name in ["nano_text.building-ABC", ".nano_vision-ABC.mlmodelc"] {
            try FileManager.default.createDirectory(at: cache.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        for name in ["nano_text", "nano_vision.mlmodelc"] {
            try FileManager.default.createDirectory(at: cache.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        try Data().write(to: cache.appendingPathComponent("nano.prepared"))
        ANEBlocks.sweepLeftovers(cache: cache)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cache.path).sorted(),
                       ["nano.prepared", "nano_text", "nano_vision.mlmodelc"])
    }

    // MARK: - 19. The tagger's prior

    func testATaggerPriorSurvivesALaunchBeforeItFreezes() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("prior-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        // Where `init(labels:rows:)` writes its cache, and the prior beside it.
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("omni-tagger-mem-\(ProcessInfo.processInfo.processIdentifier)-3.cache")
        try? FileManager.default.removeItem(at: cache.deletingPathExtension().appendingPathExtension("prior"))
        addTeardownBlock { try? FileManager.default.removeItem(at: cache.deletingPathExtension().appendingPathExtension("prior")) }
        let first = try XCTUnwrap(OmniTagger(labels: ["harbor", "engine", "anchor"], rows: [[Float]](repeating: [Float](repeating: 0.5, count: 8), count: 3)))
        XCTAssertTrue(first.needsSeed)
        let scores = [Float](repeating: 0.25, count: 6)
        for _ in 0 ..< 6 { _ = first.finalize(scores) }   // what seeding does: six neutral images
        let next = try XCTUnwrap(OmniTagger(cacheURL: cache, dim: 8))   // the next launch
        XCTAssertFalse(next.needsSeed, "seeded again at every launch until the library has 64 images")
    }

    // MARK: - 13, 17. A Cancel with no task to cancel

    /// Never answers: what `waitsForConnectivity` does with no network.
    final class Silent: URLProtocol {
        nonisolated(unsafe) static var started = 0
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() { Self.started += 1 }
        override func stopLoading() {}
    }

    private func stubbed() -> URLSessionConfiguration {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [Silent.self]
        return c
    }

    private func finishes<T: Sendable>(within seconds: TimeInterval, _ work: @escaping @Sendable () async throws -> T) async -> Bool {
        await withTaskGroup(of: Bool.self) { g in
            g.addTask { _ = try? await work(); return true }
            g.addTask { try? await Task.sleep(for: .seconds(seconds)); return false }
            let first = await g.next() ?? false
            g.cancelAll()
            return first
        }
    }

    /// Offline, first launch: click Download, then Cancel. The screen stayed on "Preparing...".
    func testCancelReachesTheManifestRequest() async throws {
        let d = ModelDownloader(configuration: stubbed())
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dest) }
        let finished = Task { () -> Error? in
            do { try await d.download(variant: .nano, to: dest) { _ in }; return nil } catch { return error }
        }
        try await Task.sleep(for: .seconds(0.5))
        d.cancel()
        let ended = await finishes(within: 5) { await finished.value != nil }
        XCTAssertTrue(ended, "Cancel did nothing while the manifest request waited for a network")
        if ended {
            let error = await finished.value
            XCTAssertEqual((error as? URLError)?.code, .cancelled, "\(String(describing: error))")
        } else { finished.cancel() }
    }

    /// A cancel that lands after the loop's check, before the next part's task exists.
    func testACancelBetweenTwoDownloadsIsNotLost() async throws {
        let d = ModelDownloader(configuration: stubbed())
        d.cancel()
        let url = URL(string: "https://example.invalid/part2")!
        let ended = await finishes(within: 5) { _ = try await d.downloadOne(url) }
        XCTAssertTrue(ended, "the next part started as if nothing had happened")

        let o = OCRModelDownloader(configuration: stubbed())
        o.cancel()
        let endedOCR = await finishes(within: 5) { _ = try await o.downloadOne(url) }
        XCTAssertTrue(endedOCR, "the next shard started as if nothing had happened")
    }

    // MARK: - 16, 19. Neural Engine stamps and switches (compiles on first use: run one at a time)

    /// The switch back to the GPU landing right after the text phase took over: nothing may be left
    /// scanning on the CPU. Between the lock and the write was microseconds; the hook holds it open.
    func testASwitchRightAfterTheTextPhaseDoesNotStrandTheStoreOnTheCPU() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async { Self.switchAtTheTextPhase(engine); done.resume() }
        }
        OmniEngine.testAfterTextPhase = nil
        OmniCompute.offGPU = false
    }

    private static func switchAtTheTextPhase(_ engine: OmniEngine) {
        let cache = ANEModeTests.cache
        OmniEngine.testAfterTextPhase = { _ = engine.setComputeUnit(.gpu, cache: cache) }
        _ = engine.setComputeUnit(.neuralEngine, cache: cache)
        XCTAssertFalse(OmniCompute.offGPU, "unit on the GPU, the store scanning on the CPU")
    }

    /// A rebuild that does not finish takes the stamp with it: its canary passes the next launch's
    /// check beside the old stamp, and the rest of the functions compile inside a pass.
    func testARebuildThatDoesNotFinishLeavesNoStamp() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir)
        let cache = ANEModeTests.cache
        let stamp = cache.appendingPathComponent("nano.prepared")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let was = try? String(contentsOf: stamp, encoding: .utf8)
        try "another build".write(to: stamp, atomically: true, encoding: .utf8)
        addTeardownBlock {
            if let was { try? was.write(to: stamp, atomically: true, encoding: .utf8) } else { try? FileManager.default.removeItem(at: stamp) }
        }
        let cancelled = engine.setComputeUnit(.neuralEngine, cache: cache, isCancelled: { true })
        XCTAssertFalse(cancelled)
        XCTAssertFalse(FileManager.default.fileExists(atPath: stamp.path), "the old stamp stands over a build that did not finish")
        _ = engine.setComputeUnit(.gpu, cache: cache)
        OmniCompute.offGPU = false
    }

    // MARK: - N14. An indexing batch beside queries: it waits, but not forever and not past a stop

    func testABatchWaitsOnQueriesButNotForeverNorPastAStop() throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        typealias B = ANETextBackend
        XCTAssertTrue(B.batchMayGo(waiting: 0, waited: 0, lastCallTook: 0.1, stopped: false), "nobody waiting")
        XCTAssertFalse(B.batchMayGo(waiting: 2, waited: 1, lastCallTook: 0.1, stopped: false), "queries waiting")
        XCTAssertFalse(B.batchMayGo(waiting: 2, waited: 0.099 * B.queryShare, lastCallTook: 0.1, stopped: false))
        XCTAssertTrue(B.batchMayGo(waiting: 2, waited: 0.1 * B.queryShare, lastCallTook: 0.1, stopped: false), "waited long enough")
        XCTAssertFalse(B.batchMayGo(waiting: 2, waited: 59, lastCallTook: 0, stopped: false), "no call timed yet: the floor")
        XCTAssertTrue(B.batchMayGo(waiting: 2, waited: 60, lastCallTook: 0, stopped: false))
        XCTAssertFalse(B.batchMayGo(waiting: 2, waited: 0, lastCallTook: 0.1, stopped: true), "stopping: queries still go first")
        XCTAssertTrue(B.batchMayGo(waiting: 2, waited: 0.1, lastCallTook: 0.1, stopped: true), "stopping: one call at a time")
    }

    /// A Pause reaches a batch already in flight: Indexer.cancel tells the embedder.
    func testCancellingAPassTellsTheEmbedder() throws {
        final class Recording: Embedder, @unchecked Sendable {
            var stops = 0
            var dim: Int { 4 }
            func embedText(_ text: String, as type: OmniInputType) -> [Float] { [1, 0, 0, 0] }
            func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in [1, 0, 0, 0] } }
            func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] { batches.map { embedTextBatch($0, as: type) } }
            func embedImage(_ image: CGImage) -> [Float]? { nil }
            func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
            func embedAudio(_ url: URL) -> [Float]? { nil }
            func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
            func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
            func indexingStopping() { stops += 1 }
        }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("stop-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try VectorStore(dbURL: dir.appendingPathComponent("test.sqlite"))
        defer { store.close() }
        let e = Recording()
        let ix = Indexer(store: store, embedder: e)
        ix.cancel(.pause)
        XCTAssertEqual(e.stops, 1)
    }

    // MARK: - N13. Neural Engine text batches against a stream of queries

    /// Not a check: `OMNI_ANE_PROBE=1` prints how far an indexing batch gets beside `k` threads of
    /// queries, each a new text so the query cache cannot answer it.
    func testProbeBatchesBesideQueries() async throws {
        guard ProcessInfo.processInfo.environment["OMNI_ANE_PROBE"] == "1" else { throw XCTSkip("a measurement") }
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let engine = try await OmniEngine.loadValidated(modelDir: modelDir)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async { Self.probeBatches(engine); done.resume() }
        }
        OmniCompute.offGPU = false
    }

    @available(macOS 15.0, *)
    private static func probeBatches(_ engine: OmniEngine) {
        guard engine.setComputeUnit(.neuralEngine, cache: ANEModeTests.cache) else { return print("no Neural Engine") }
        let words = ["harbor", "engine", "anchor", "compass", "lantern", "rigging", "ballast", "keel", "mast", "tide"]
        func text(_ i: Int, _ n: Int) -> String { (0 ..< n).map { words[($0 * 7 + i) % words.count] + "\(i % 97)" }.joined(separator: " ") }
        var salt = 0
        final class Box: @unchecked Sendable {
            let lock = NSLock(); var stop = false; var times: [Double] = []; var n = 0
        }
        // `burst`: the queries stop after that many seconds; `stopAfter`: indexing is told to stop then.
        func one(_ k: Int, burst: Double?, stopAfter: Double? = nil, label: String) {
            salt += 1
            let passages = (0 ..< 64).map { text($0 + salt * 1000, 160) }   // new texts: no vector is kept
            let box = Box()
            let group = DispatchGroup()
            let tStart = Date()
            for t in 0 ..< k {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    var i = t * 1_000_000 + salt * 10_000_000
                    while !box.lock.withLock({ box.stop }), burst.map({ -tStart.timeIntervalSinceNow < $0 }) ?? true {
                        i += 1
                        let t0 = Date()
                        _ = engine.embedText(text(i, 6) + " q\(i)", as: .query)
                        let dt = -t0.timeIntervalSinceNow
                        box.lock.withLock { box.times.append(dt); box.n += 1 }
                    }
                    group.leave()
                }
            }
            let batchDone = DispatchSemaphore(value: 0)
            let t0 = Date()
            nonisolated(unsafe) var seconds = -1.0
            DispatchQueue.global(qos: .utility).async {
                _ = engine.embedTextBatch(passages, as: .passage)
                seconds = -t0.timeIntervalSinceNow
                batchDone.signal()
            }
            var stoppedAt = -1.0
            if let stopAfter {
                Thread.sleep(forTimeInterval: stopAfter)
                engine.indexingStopping(); stoppedAt = -t0.timeIntervalSinceNow
            }
            let finished = batchDone.wait(timeout: .now() + 60) == .success
            box.lock.withLock { box.stop = true }
            group.wait()
            if !finished { batchDone.wait() }
            let sorted = box.times.sorted()
            let p50 = sorted.isEmpty ? 0 : sorted[sorted.count / 2] * 1000, p99 = sorted.isEmpty ? 0 : sorted[sorted.count * 99 / 100] * 1000
            let stopNote = stoppedAt >= 0 ? String(format: " finished %.2f s after the stop", seconds - stoppedAt) : ""
            print(String(format: "PROBE %@ k=%d batch of 64: %.2f s%@%@  queries=%d p50=%.1f ms p99=%.1f ms",
                         label, k, seconds, finished ? "" : " (over 60 s)", stopNote, box.n, p50, p99))
        }
        let ks = (ProcessInfo.processInfo.environment["OMNI_ANE_KS"] ?? "0,1,2,4,16").split(separator: ",").compactMap { Int($0) }
        for k in ks { one(k, burst: nil, label: "sustained") }
        for k in ks where k > 0 { one(k, burst: 10, label: "burst10s") }
        one(4, burst: nil, stopAfter: 2, label: "stop@2s")
    }
}
