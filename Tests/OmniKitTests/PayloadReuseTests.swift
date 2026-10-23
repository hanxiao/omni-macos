import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import OmniKit

/// A file whose bytes change but whose pixels or samples do not - a photo's metadata edited, a
/// recording's tags rewritten - keeps its vectors: the rows are keyed by what the model sees
/// (Indexer.imagePayloadKey, audioPayloadKey), so the changed file is matched to its own stored
/// rows and nothing runs through the model. A real pixel edit embeds again (the negative control).
final class PayloadReuseTests: XCTestCase {
    private var root: URL!
    private var db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("payload-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("payload-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    final class Counting: Embedder, @unchecked Sendable {
        let inner = EnrichmentTests.Fake()
        var dim: Int { inner.dim }
        var audio = 0, pictures = 0, texts = 0
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { texts += 1; return inner.embedText(text, as: type) }
        func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { texts += t.count; return inner.embedTextBatch(t, as: type) }
        func embedImage(_ image: CGImage) -> [Float]? { pictures += 1; return inner.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { pictures += raws.count; return inner.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { pictures += 1; return inner.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { audio += 1; return inner.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { audio += 1; return inner.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { audio += mels.count; return inner.embedAudioMelBatch(mels, frames: frames) }
    }

    private func pass(_ indexer: Indexer) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    /// A PNG of a gradient with a square at `mark`, and a description in its metadata.
    private func png(_ url: URL, mark: Int, description: String) {
        let ctx = CGContext(data: nil, width: 640, height: 480, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for x in stride(from: 0, to: 640, by: 16) {
            ctx.setFillColor(CGColor(red: CGFloat(x) / 640, green: 0.4, blue: 0.7, alpha: 1))
            ctx.fill(CGRect(x: x, y: 0, width: 16, height: 480))
        }
        ctx.setFillColor(.white); ctx.fill(CGRect(x: mark, y: 200, width: 60, height: 60))
        let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        let props = [kCGImagePropertyPNGDictionary: [kCGImagePropertyPNGDescription: description]] as CFDictionary
        CGImageDestinationAddImage(dst, ctx.makeImage()!, props)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
    }

    /// 16 kHz mono 16-bit PCM of a tone, `seconds` long, with a LIST/INFO tag chunk after the data.
    private func wav(_ url: URL, seconds: Int, title: String) throws {
        let n = 16000 * seconds
        var pcm = Data(capacity: n * 2)
        for i in 0 ..< n {
            var s = Int16(sin(Double(i) * 2 * .pi * 330 / 16000) * 9000).littleEndian
            withUnsafeBytes(of: &s) { pcm.append(contentsOf: $0) }
        }
        func u32(_ v: Int) -> Data { var x = UInt32(v).littleEndian; return Data(bytes: &x, count: 4) }
        func u16(_ v: Int) -> Data { var x = UInt16(v).littleEndian; return Data(bytes: &x, count: 2) }
        let fmt = Data("fmt ".utf8) + u32(16) + u16(1) + u16(1) + u32(16000) + u32(32000) + u16(2) + u16(16)
        var name = Data(title.utf8); name.append(0); if name.count % 2 == 1 { name.append(0) }
        let info = Data("INFO".utf8) + Data("INAM".utf8) + u32(name.count) + name
        let list = Data("LIST".utf8) + u32(info.count) + info
        let body = Data("WAVE".utf8) + fmt + Data("data".utf8) + u32(pcm.count) + pcm + list
        try (Data("RIFF".utf8) + u32(body.count) + body).write(to: url)
    }

    private func bump(_ url: URL, by seconds: Double) throws {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let m = (attrs[.modificationDate] as? Date) ?? Date()
        try FileManager.default.setAttributes([.modificationDate: m.addingTimeInterval(seconds)], ofItemAtPath: url.path)
    }

    func testAPhotosMetadataEditKeepsItsVectorWithoutAForward() throws {
        let file = root.appendingPathComponent("IMG_0001.png")
        png(file, mark: 100, description: "before")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        let before = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(embedder.pictures, 1)
        XCTAssertFalse(before[0].chunkKey.isEmpty)

        let bytes = try Data(contentsOf: file)
        png(file, mark: 100, description: "after an edit of the metadata")
        try bump(file, by: 5)
        XCTAssertNotEqual(try Data(contentsOf: file), bytes, "the bytes did not change; the test proves nothing")
        pass(indexer)
        XCTAssertEqual(embedder.pictures, 1, "a metadata edit ran the picture through the model again")
        let after = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(after.map(\.embedding), before.map(\.embedding))
        XCTAssertEqual(after.map(\.snippet), before.map(\.snippet))

        png(file, mark: 300, description: "after an edit of the metadata")   // negative control: the pixels change
        try bump(file, by: 10)
        pass(indexer)
        XCTAssertEqual(embedder.pictures, 2, "a pixel edit was not embedded again")
    }

    func testARecordingsTagEditKeepsItsVectorsWithoutAForward() throws {
        let file = root.appendingPathComponent("memo.wav")
        try wav(file, seconds: 130, title: "Before")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        let before = try XCTUnwrap(store.storedChunks(path: file.path))
        let forwards = embedder.audio
        XCTAssertGreaterThan(forwards, 0, "nothing embedded; the test proves nothing")

        try wav(file, seconds: 130, title: "After a long rename of the title")
        try bump(file, by: 5)
        pass(indexer)
        XCTAssertEqual(embedder.audio, forwards, "a tag edit ran the recording through the model again")
        let after = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertEqual(after.map(\.embedding), before.map(\.embedding))
        XCTAssertEqual(after.map(\.locator), before.map(\.locator))
    }

    /// CONTENT THAT LEAVES AND COMES BACK (a Trash restore, a branch switched back) keeps its
    /// vectors while the store still holds what it released (VectorStore.released): a text file,
    /// a photo and a recording, each removed by a pass and then restored, run nothing through the
    /// model again. After a wipe the store keeps nothing (the negative control).
    func testContentThatComesBackIsNotEmbeddedAgain() throws {
        let words = ["harbor", "engine", "fog", "sail", "anchor", "tide", "compass", "deck", "rope", "storm"]
        let notes = root.appendingPathComponent("notes.txt")
        try (0 ..< 900).map { words[($0 * 7) % words.count] + "\($0 % 37)" }.joined(separator: " ").write(to: notes, atomically: true, encoding: .utf8)
        let photo = root.appendingPathComponent("IMG_0002.png")
        png(photo, mark: 120, description: "kept")
        let memo = root.appendingPathComponent("memo.wav")
        try wav(memo, seconds: 70, title: "Kept")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        let files = [notes, photo, memo]
        let before = try files.map { try XCTUnwrap(store.storedChunks(path: $0.path)) }
        XCTAssertGreaterThan(embedder.texts, 1, "the text file was not chunked; the test proves nothing")
        let counts = (embedder.texts, embedder.pictures, embedder.audio)

        let away = root.deletingLastPathComponent().appendingPathComponent("away-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: away, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: away) }
        // As the watcher reports them: gone, then back.
        func reconcile() { indexer.resetCancelled(); indexer.update(paths: files.map(\.path), settings: IndexSettings(), roots: [root.path]) }
        for f in files { try FileManager.default.moveItem(at: f, to: away.appendingPathComponent(f.lastPathComponent)) }
        reconcile()
        XCTAssertTrue(files.allSatisfy { store.storedChunks(path: $0.path) == nil }, "the files are still indexed")
        for f in files { try FileManager.default.moveItem(at: away.appendingPathComponent(f.lastPathComponent), to: f) }
        reconcile()
        XCTAssertEqual(embedder.texts, counts.0, "the returning text was embedded again")
        XCTAssertEqual(embedder.pictures, counts.1, "the returning photo was embedded again")
        XCTAssertEqual(embedder.audio, counts.2, "the returning recording was embedded again")
        let after = try files.map { try XCTUnwrap(store.storedChunks(path: $0.path)) }
        for (a, b) in zip(after, before) {
            XCTAssertEqual(a.map(\.embedding), b.map(\.embedding))
            XCTAssertEqual(a.map(\.snippet), b.map(\.snippet))
        }

        // Negative control: a wipe forgets what was released, so the same round trip embeds again.
        store.wipeChunks()
        for f in files { try FileManager.default.moveItem(at: f, to: away.appendingPathComponent(f.lastPathComponent)) }
        reconcile()
        for f in files { try FileManager.default.moveItem(at: away.appendingPathComponent(f.lastPathComponent), to: f) }
        reconcile()
        XCTAssertGreaterThan(embedder.pictures, counts.1, "after a wipe the photo still came back without a forward")
    }
}
