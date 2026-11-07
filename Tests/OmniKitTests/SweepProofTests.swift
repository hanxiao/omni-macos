import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// THE SWEEP DELETES WHAT THE WALK DID NOT MEET, so each of these is a way of not meeting a file that
/// is still on disk: a walk that was stopped, a folder that could not be read, a read that stalled,
/// an embed that failed. The rows stay in all of them; a file really gone still goes.
final class SweepProofTests: XCTestCase {
    /// Text vectors always; images until `imagesFail` is set.
    final class Switchable: Embedder, @unchecked Sendable {
        let dim = 8
        private let lock = NSLock()
        private var failing = false, poison: String?
        var imagesFail: Bool { get { lock.withLock { failing } } set { lock.withLock { failing = newValue } } }
        /// Text holding this comes back non-finite, which fails the file's turn.
        var poisonText: String? { get { lock.withLock { poison } } set { lock.withLock { poison = newValue } } }
        private func v(_ s: String) -> [Float] {
            if let p = poisonText, s.contains(p) { return [Float](repeating: .nan, count: 8) }
            var out = [Float](repeating: 0, count: 8)
            for (i, b) in s.utf8.enumerated() { out[i % 8] += Float(b % 13) + 1 }
            let n = out.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return out.map { $0 / n }
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { v(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map(v) }
        func embedImage(_ image: CGImage) -> [Float]? { imagesFail ? nil : [0, 1, 0, 0, 0, 0, 0, 0] }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
            imagesFail ? nil : raws.map { _ in [0, 1, 0, 0, 0, 0, 0, 0] }
        }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var root: URL!, db: URL!
    private var store: VectorStore!
    private let embedder = Switchable()
    private var indexer: Indexer!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("sweep-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("sweep-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        indexer = Indexer(store: store, embedder: embedder)
    }

    override func tearDown() {
        Indexer.testReadDelay = nil
        store.close()
        if let items = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let u as URL in items { chmod(u.path, 0o755) }
        }
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    @discardableResult
    private func write(_ rel: String, _ text: String = "a note about the harbor budget") throws -> URL {
        let url = root.appendingPathComponent(rel)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func png(_ name: String, shade: CGFloat = 0.2) throws -> URL {
        let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: shade, green: 0.4, blue: 0.8, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let url = root.appendingPathComponent(name)
        let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
        return url
    }

    @discardableResult
    private func pass(settings: IndexSettings = IndexSettings(enabledKinds: [.text]),
                      onProgress: ((IndexProgress) -> Void)? = nil) -> IndexProgress {
        let done = expectation(description: "pass")
        nonisolated(unsafe) var last = IndexProgress()
        indexer.index(roots: [root], settings: settings) { p in
            onProgress?(p)
            if p.done { last = p; done.fulfill() }
        }
        wait(for: [done], timeout: 120)
        return last
    }

    private func indexed() -> Set<String> { Set(store.indexedFiles().keys) }

    // MARK: - The walker's report

    func testTheWalkReportsAFolderItCouldNotOpen() throws {
        try write("readable/a.txt")
        let locked = try write("locked/deep/b.txt").deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertEqual(chmod(locked.path, 0), 0)
        let outcome = FileCrawler(roots: [root]).walk { _ in }
        XCTAssertFalse(outcome.stopped)
        XCTAssertEqual(outcome.unreadableDirs, [locked.path])
        XCTAssertTrue(outcome.isUnread(locked.path + "/deep/b.txt"), "anything under the folder is unknown")
        XCTAssertFalse(outcome.isUnread(root.path + "/readable/a.txt"))
        XCTAssertFalse(outcome.isUnread(locked.path), "the folder's own row is its parent's to answer for")
    }

    func testAStoppedWalkSaysSo() throws {
        for i in 0 ..< 8 { try write("d\(i)/f.txt") }
        XCTAssertTrue(FileCrawler(roots: [root]).walk(shouldContinue: { false }) { _ in }.stopped)
        XCTAssertFalse(FileCrawler(roots: [root]).walk { _ in }.stopped)
    }

    // MARK: - What the sweep keeps

    func testAnUnreadableFolderKeepsItsRowsAndAGoneFileStillGoes() throws {
        let kept = try write("readable/kept.txt")
        let gone = try write("readable/gone.txt", "another note about the garden")
        let locked = try write("locked/inside.txt", "a note in a folder about to be locked")
        pass()
        XCTAssertEqual(indexed(), [kept.path, gone.path, locked.path])

        // The root still yields a file, so the blind-root guard does not apply.
        try FileManager.default.removeItem(at: gone)
        XCTAssertEqual(chmod(locked.deletingLastPathComponent().path, 0), 0)
        pass()
        XCTAssertTrue(indexed().contains(locked.path), "a folder that could not be read lost its rows")
        XCTAssertFalse(indexed().contains(gone.path), "control: a file really gone must still be swept")

        // Readable again and the file gone: this pass could look, and it goes.
        XCTAssertEqual(chmod(locked.deletingLastPathComponent().path, 0o755), 0)
        try FileManager.default.removeItem(at: locked)
        pass()
        XCTAssertEqual(indexed(), [kept.path])
    }

    /// The race itself: the pass's walkers stop on a cancel, the cancel is cleared while the pass is
    /// still winding down, and the consumer reads an empty queue as a finished walk.
    func testACancelClearedWhileAPassWindsDownDoesNotSweep() throws {
        var files: [String] = []
        for i in 0 ..< 200 { files.append(try write("d\(i)/f\(i).txt", "note \(i) on the harbor").path) }
        pass()
        XCTAssertEqual(indexed(), Set(files))

        // A small queue parks the walk part-way, so it is still running when the cancel lands.
        setenv("OMNI_CRAWL_QUEUE_CAP", "5", 1)
        defer { unsetenv("OMNI_CRAWL_QUEUE_CAP") }
        nonisolated(unsafe) var fired = false
        let last = pass { [indexer] p in
            // From the first wave on: the walk is parked behind it, and has met a handful of files.
            guard !fired, !p.perRoot.isEmpty, p.perRoot.values.contains(where: { $0.total > 0 }) else { return }
            fired = true
            indexer!.cancel(.pause)
            // The walk sees the cancel and stops. Then what drainPendingFSChanges did to a pass
            // winding down.
            Thread.sleep(forTimeInterval: 0.5)
            indexer!.resetCancelled()
        }
        XCTAssertTrue(fired, "the probe: the cancel landed")
        XCTAssertEqual(indexed(), Set(files), "a stopped walk swept the files it never reached")
        XCTAssertTrue(last.cancelled, "a walk that stopped did not finish, whatever the flag says now")
    }

    /// A file met by the walk is met, whatever its read then did.
    func testAStalledReadOfAChangedFileKeepsItsRows() throws {
        let stuck = try write("stuck.txt", "first version")
        try write("other.txt", "another file")
        pass()
        XCTAssertTrue(indexed().contains(stuck.path))

        try "second version, longer than the first".write(to: stuck, atomically: true, encoding: .utf8)
        indexer.stallLimit = 1
        Indexer.testReadDelay = { $0.hasSuffix("/stuck.txt") ? 4 : nil }
        pass()
        XCTAssertTrue(indexed().contains(stuck.path), "a read that stalled deleted the file's rows")
        XCTAssertEqual(indexer.quarantine.count, 1, "the probe: the read did stall")
    }

    // MARK: - An embed that failed is not an empty file

    func testAFailedImageBatchOnTheWatcherPathKeepsEveryRow() throws {
        let settings = IndexSettings(enabledKinds: [.image])
        let a = try png("a.png", shade: 0.1), b = try png("b.png", shade: 0.2)
        pass(settings: settings)
        XCTAssertEqual(indexed(), [a.path, b.path])

        // Both changed in their bytes (a touch alone is served from the store), and the tower faults
        // on their batch.
        _ = try png("a.png", shade: 0.7)
        _ = try png("b.png", shade: 0.8)
        embedder.imagesFail = true
        indexer.update(paths: [a.path, b.path], settings: settings, roots: [root.path])
        XCTAssertEqual(indexed(), [a.path, b.path], "one failed batch deleted every file in it")

        // The tower back: the same event embeds them, under their new dates.
        embedder.imagesFail = false
        indexer.update(paths: [a.path, b.path], settings: settings, roots: [root.path])
        let after = store.indexedFiles()
        let modified = try XCTUnwrap(try FileManager.default.attributesOfItem(atPath: a.path)[.modificationDate] as? Date)
        XCTAssertEqual(try XCTUnwrap(after[a.path]).modified, modified.timeIntervalSince1970, accuracy: 0.01)
    }

    /// The other meaning of empty still deletes: a file with nothing left in it has no rows to keep.
    func testAFileEmptiedOnDiskStillLosesItsRows() throws {
        let note = try write("note.txt", "a note about the harbor budget review")
        pass()
        XCTAssertTrue(indexed().contains(note.path))
        try "".write(to: note, atomically: true, encoding: .utf8)
        indexer.update(paths: [note.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertFalse(indexed().contains(note.path), "control: an emptied file must go")
    }

    // MARK: - Long files

    /// A later turn that fails keeps its place. Without the cursor the partial rows read as a finished
    /// file, under the file's real date, and it stayed cut off until it next changed.
    func testALaterTurnThatFailsIsResumedNotLeftTruncated() throws {
        var lines = (0 ..< 90_000).map { "line \($0): a note about harbor invoice \($0 * 7919 % 100003) and the garden" }
        lines[80_000] = "the unembeddable line"
        let big = root.appendingPathComponent("big.txt")
        try lines.joined(separator: "\n").write(to: big, atomically: true, encoding: .utf8)

        embedder.poisonText = "the unembeddable line"
        let first = pass()
        XCTAssertGreaterThan(first.failed, 0, "the probe: a later turn did fail")
        XCTAssertTrue(indexer.streams.inProgress(big.path), "the failed turn dropped the file's place")
        let partial = store.chunkVectors(path: big.path, dim: 8, cap: 1_000_000).count
        XCTAssertGreaterThan(partial, 0)

        embedder.poisonText = nil
        pass()
        XCTAssertFalse(indexer.streams.inProgress(big.path))
        let whole = store.chunkVectors(path: big.path, dim: 8, cap: 1_000_000).count
        XCTAssertGreaterThan(whole, partial, "the next pass read the partial rows as a finished file")
    }

    // MARK: - The prelude

    /// A caller that cleared the cancel when it committed the pass keeps any cancel set since.
    func testACancelBeforeACommittedPassStandsAndADirectPassClearsIt() throws {
        let note = try write("note.txt")
        indexer.resetCancelled()
        indexer.cancel(.pause)
        let done = expectation(description: "pass")
        nonisolated(unsafe) var cancelled = false
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.text]), cancelClearedByCaller: true) { p in
            if p.done { cancelled = p.cancelled; done.fulfill() }
        }
        wait(for: [done], timeout: 60)
        XCTAssertTrue(cancelled, "the pass erased the cancel that landed in its prelude and ran on")
        XCTAssertFalse(indexed().contains(note.path))

        // The direct form still starts clean, which every tool and test relies on.
        indexer.cancel(.pause)
        XCTAssertFalse(pass().cancelled)
        XCTAssertTrue(indexed().contains(note.path))
    }

    // MARK: - Neural Engine mode

    /// A build that cannot start leaves the Mac on the GPU, and the GPU defers nothing.
    func testAFailedNeuralEngineBuildDefersNothing() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("OMNI_NANO_MODEL_DIR not set")
        }
        let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: dir))
        // Nowhere the program can be written.
        let ok = engine.setComputeUnit(.neuralEngine, cache: URL(fileURLWithPath: "/dev/null/no-such-cache"))
        XCTAssertFalse(ok)
        XCTAssertTrue(engine.deferredKinds.isEmpty, "the app says unavailable, and these kinds were held back for good")
    }
}
