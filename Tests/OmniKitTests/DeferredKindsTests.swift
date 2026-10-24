import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// READY PER KIND: while Neural Engine mode compiles a kind's functions, a pass leaves that kind's
/// files for later (Indexer.deferredKinds) - not embedded, and not swept away if already indexed -
/// and the pass after its phase takes them.
final class DeferredKindsTests: XCTestCase {
    final class Kinds: @unchecked Sendable {
        private let lock = NSLock(); private var k: Set<FileKind> = []
        var value: Set<FileKind> { get { lock.withLock { k } } set { lock.withLock { k = newValue } } }
    }

    private var root: URL!, db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("deferred-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("deferred-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    private func png(_ name: String, shade: CGFloat) throws -> URL {
        let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: shade, green: 0.4, blue: 1 - shade, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let url = root.appendingPathComponent(name)
        let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
        return url
    }

    private func pass(_ indexer: Indexer) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 120)
    }

    func testADeferredKindWaitsAndIsNotSweptAway() throws {
        let note = root.appendingPathComponent("note.txt")
        try "The harbor budget review is on Friday.".write(to: note, atomically: true, encoding: .utf8)
        let first = try png("first.png", shade: 0.2)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: EnrichmentTests.Fake())
        let kinds = Kinds()
        indexer.deferredKinds = { kinds.value }

        // Images not ready: the text is indexed, the image is not.
        kinds.value = [.image, .audio, .video]
        pass(indexer)
        XCTAssertNotNil(store.indexedFiles()[note.path])
        XCTAssertNil(store.indexedFiles()[first.path], "a deferred kind was embedded")

        // Its phase in: the next pass takes it.
        kinds.value = [.video]
        pass(indexer)
        XCTAssertNotNil(store.indexedFiles()[first.path], "the pass after the phase did not take it")

        // Deferred again (another build), with a new image beside it: the indexed one stays, the new
        // one waits, the watcher's path leaves both alone too.
        let second = try png("second.png", shade: 0.8)
        kinds.value = [.image, .audio, .video]
        pass(indexer)
        XCTAssertNotNil(store.indexedFiles()[first.path], "a deferred kind's indexed file was swept away")
        XCTAssertNil(store.indexedFiles()[second.path])
        _ = indexer.update(paths: [second.path], settings: IndexSettings(), roots: [root.path])
        XCTAssertNil(store.indexedFiles()[second.path], "the watcher's path embedded a deferred kind")
        kinds.value = []
        pass(indexer)
        XCTAssertNotNil(store.indexedFiles()[second.path])
    }
}
