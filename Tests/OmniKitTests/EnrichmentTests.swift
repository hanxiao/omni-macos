import XCTest
import CoreGraphics
import CoreText
import AppKit
import AVFoundation
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// Screenshot text, soundtracks, one-minute audio windows - and the enrichment that brings an
/// index built without them up to date, once, safely redone.
final class EnrichmentTests: XCTestCase {
    /// Text by a deterministic hash, images and audio a fixed direction: what matters here is
    /// which chunks exist and which vectors are kept, not what they mean.
    final class Fake: Embedder, @unchecked Sendable {
        let dim = 8
        private func v(_ s: String) -> [Float] {
            var out = [Float](repeating: 0, count: 8)
            for (i, b) in s.utf8.enumerated() { out[i % 8] += Float(b % 13) + 1 }
            let n = out.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return out.map { $0 / n }
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { v(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map(v) }
        func embedImage(_ image: CGImage) -> [Float]? { [0, 1, 0, 0, 0, 0, 0, 0] }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { raws.map { _ in [0, 1, 0, 0, 0, 0, 0, 0] } }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { [0, 0, 1, 0, 0, 0, 0, 0] }
        func embedAudio(_ url: URL) -> [Float]? { [0, 0, 0, 1, 0, 0, 0, 0] }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { [0, 0, 0, 1, 0, 0, 0, 0] }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { mels.map { _ in [0, 0, 0, 1, 0, 0, 0, 0] } }
    }

    private var root: URL!
    private var db: URL!

    override func setUpWithError() throws {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("enrich-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        root = r
        db = r.deletingLastPathComponent().appendingPathComponent("enrich-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: db)
    }

    /// A 1280x800 window with lines of text, marked as a screen capture.
    private func screenshot(_ lines: [String], name: String) throws -> URL {
        let ctx = CGContext(data: nil, width: 1280, height: 800, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(.white); ctx.fill(CGRect(x: 0, y: 0, width: 1280, height: 800))
        let attr = NSAttributedString(string: lines.joined(separator: "\n"), attributes: [.font: NSFont.systemFont(ofSize: 28)])
        let frame = CTFramesetterCreateFrame(CTFramesetterCreateWithAttributedString(attr), CFRange(location: 0, length: 0),
                                             CGPath(rect: CGRect(x: 80, y: 80, width: 1100, height: 600), transform: nil), nil)
        CTFrameDraw(frame, ctx)
        let url = root.appendingPathComponent(name)
        let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
        let flag = try PropertyListSerialization.data(fromPropertyList: true, format: .binary, options: 0)
        _ = flag.withUnsafeBytes { setxattr(url.path, "com.apple.metadata:kMDItemIsScreenCapture", $0.baseAddress, flag.count, 0, 0) }
        return url
    }

    /// `seconds` of one still frame a second with a tone soundtrack (16 kHz AAC).
    private func video(seconds: Int, name: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let vIn = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: vIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64])
        let aIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 16000,
                                                                          AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 32000])
        vIn.expectsMediaDataInRealTime = false; aIn.expectsMediaDataInRealTime = false
        writer.add(vIn); writer.add(aIn)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        var asbd = AudioStreamBasicDescription(mSampleRate: 16000, mFormatID: kAudioFormatLinearPCM,
                                               mFormatFlags: kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
                                               mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 1,
                                               mBitsPerChannel: 32, mReserved: 0)
        var fmt: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &fmt)
        let per = 16000
        // A second of each in turn: the writer interleaves its inputs, and an input written far
        // ahead of the other stops taking data until the other catches up.
        for t in 0 ..< seconds {
            while !vIn.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
            adaptor.append(pb!, withPresentationTime: CMTime(value: CMTimeValue(t), timescale: 1))
            let samples = (0 ..< per).map { Float(sin(Double(t * per + $0) * 2 * .pi * 440 / 16000)) * 0.3 }
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: per * 4, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: per * 4, flags: 0, blockBufferOut: &block)
            samples.withUnsafeBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block!, offsetIntoDestination: 0, dataLength: per * 4) }
            var sb: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: block!, formatDescription: fmt!,
                                                                 sampleCount: per, presentationTimeStamp: CMTime(value: CMTimeValue(t * per), timescale: 16000),
                                                                 packetDescriptions: nil, sampleBufferOut: &sb)
            while !aIn.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            aIn.append(sb!)
        }
        vIn.markAsFinished(); aIn.markAsFinished()
        let done = expectation(description: "video")
        writer.finishWriting { done.fulfill() }
        wait(for: [done], timeout: 60)
        XCTAssertEqual(writer.status, .completed, "\(String(describing: writer.error))")
        return url
    }

    private func pass(_ indexer: Indexer, _ settings: IndexSettings) {
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    func testScreenshotTextAndSoundtrackAreIndexedWithTheFile() throws {
        let shot = try screenshot(["Booking confirmed", "Confirmation code: K7Q2LM", "Seat 32A"], name: "Screenshot 2026-09-20 at 21.15.40.png")
        let film = try video(seconds: 130, name: "talk.mp4")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Fake()
        pass(Indexer(store: store, embedder: embedder), IndexSettings())

        let s = try XCTUnwrap(store.storedChunks(path: shot.path))
        XCTAssertEqual(s.first?.kind, "image")
        let text = s.filter { $0.kind == "text" }
        XCTAssertFalse(text.isEmpty, "no text chunks for the screenshot")
        XCTAssertTrue(text.allSatisfy { $0.locator == "Text" })
        XCTAssertTrue(text.map(\.snippet).joined(separator: " ").contains("K7Q2LM"), text.map(\.snippet).description)
        let found = store.search(embedder.embedText("K7Q2LM", as: .query), topK: 10, textQuery: "K7Q2LM")
        XCTAssertEqual(found.first?.path, shot.path, "the code on the screenshot was not found")
        XCTAssertEqual(found.first?.fileKind, "image", "shown as what the file is")
        XCTAssertEqual(found.first?.locator, "Text", "opened at the picture, not at the text that holds the code")

        let v = try XCTUnwrap(store.storedChunks(path: film.path))
        XCTAssertEqual(v.first?.kind, "video")
        XCTAssertEqual(v.filter { $0.kind == "audio" }.map(\.locator), ["0:00", "1:00", "2:00"], "one soundtrack chunk a minute")
    }

    /// RE-TAGGING A FILE (the backfill and Generate Tags: forceFreshEmbed + hqMediaTags) re-runs the
    /// picture's forward for its tags and nothing else: a screenshot keeps its screen text and a video
    /// its soundtrack windows, neither embedded again, when the file is unchanged since indexing.
    func testRetagKeepsScreenTextAndSoundtrackWithoutEmbeddingThem() throws {
        let shot = try screenshot(["Booking confirmed", "Confirmation code: K7Q2LM", "Seat 32A"], name: "Screenshot 2026-09-20 at 21.15.40.png")
        let film = try video(seconds: 130, name: "talk.mp4")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        final class Counting: Embedder, @unchecked Sendable {
            let inner = Fake()
            var dim: Int { inner.dim }
            var audio = 0, texts = 0, pictures = 0
            func embedText(_ text: String, as type: OmniInputType) -> [Float] { texts += 1; return inner.embedText(text, as: type) }
            func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { texts += t.count; return inner.embedTextBatch(t, as: type) }
            func embedImage(_ image: CGImage) -> [Float]? { pictures += 1; return inner.embedImage(image) }
            func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { pictures += raws.count; return inner.embedImages(raws) }
            func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { pictures += 1; return inner.embedVideoFrames(frames) }
            func embedAudio(_ url: URL) -> [Float]? { audio += 1; return inner.embedAudio(url) }
            func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { audio += 1; return inner.embedAudioMel(mel, frames: frames) }
            func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { audio += mels.count; return inner.embedAudioMelBatch(mels, frames: frames) }
        }
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, IndexSettings())
        let shotBefore = try XCTUnwrap(store.storedChunks(path: shot.path))
        let filmBefore = try XCTUnwrap(store.storedChunks(path: film.path))
        XCTAssertFalse(shotBefore.filter { $0.kind == "text" }.isEmpty, "no screen text indexed; the test proves nothing")
        XCTAssertEqual(filmBefore.filter { $0.kind == "audio" }.count, 3, "no soundtrack indexed; the test proves nothing")
        XCTAssertGreaterThanOrEqual(embedder.audio, 3)

        var retag = IndexSettings()
        retag.forceFreshEmbed = true
        retag.hqMediaTags = true
        let (audio0, texts0, pictures0) = (embedder.audio, embedder.texts, embedder.pictures)
        indexer.resetCancelled()
        indexer.update(paths: [shot.path, film.path], settings: retag, force: true, roots: [root.path])
        XCTAssertGreaterThan(embedder.pictures, pictures0, "the retag ran no picture forward")

        let shotAfter = try XCTUnwrap(store.storedChunks(path: shot.path))
        let filmAfter = try XCTUnwrap(store.storedChunks(path: film.path))
        XCTAssertEqual(shotAfter.filter { $0.kind == "text" }.map(\.snippet), shotBefore.filter { $0.kind == "text" }.map(\.snippet),
                       "the retag lost or changed the screenshot's text")
        XCTAssertEqual(filmAfter.filter { $0.kind == "audio" }.map(\.embedding), filmBefore.filter { $0.kind == "audio" }.map(\.embedding),
                       "the retag lost or changed the soundtrack")
        XCTAssertEqual(embedder.audio, audio0, "the retag embedded the soundtrack again")
        XCTAssertEqual(embedder.texts, texts0, "the retag embedded the screen text again")
    }

    /// An index built with neither, then enriched: pictures and frames kept as they were, the
    /// rest added, the ledger done - and a second enrichment of the same files changes nothing.
    func testEnrichmentAddsWhatIsMissingOnceAndKeepsWhatWasThere() throws {
        let shot = try screenshot(["Order #114-3328-9912 has shipped"], name: "Screenshot 2026-09-29 at 19.48.06.png")
        // A byte-identical copy: its own rows, the same content digest.
        let twin = root.appendingPathComponent("Screenshot 2026-09-29 at 19.48.06 copy.png")
        try FileManager.default.copyItem(at: shot, to: twin)
        let film = try video(seconds: 70, name: "clip.mp4")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Fake()
        var old = IndexSettings()
        old.screenText = false; old.soundtrack = false
        pass(Indexer(store: store, embedder: embedder), old)
        let before = try XCTUnwrap(store.storedChunks(path: shot.path))
        XCTAssertEqual(before.count, 1, "the old layout had text (control)")
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: film.path)).filter { $0.kind == "audio" }.count, 0)
        XCTAssertTrue(store.enrichLedger.backfilled, "done, for the settings it ran with")
        store.enrichLedger.prepare(IndexSettings())
        XCTAssertFalse(store.enrichLedger.backfilled, "a ledger kept for other settings would never add the text")

        pass(Indexer(store: store, embedder: embedder), IndexSettings())
        let after = try XCTUnwrap(store.storedChunks(path: shot.path))
        XCTAssertEqual(after.first?.embedding, before.first?.embedding, "the picture was embedded again")
        XCTAssertGreaterThan(after.count, 1)
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: film.path)).filter { $0.kind == "audio" }.map(\.locator), ["0:00", "1:00"])
        XCTAssertTrue(store.enrichLedger.backfilled)
        XCTAssertGreaterThan(try XCTUnwrap(store.storedChunks(path: twin.path)).count, 1, "the copy was taken for done with the original")
        let byCode = store.search(embedder.embedText("114-3328-9912", as: .query), topK: 5, textQuery: "114-3328-9912")
        XCTAssertEqual(Set(byCode.prefix(2).map(\.path)), [shot.path, twin.path], "both copies hold the code")

        // Redone (the ledger lost, as after a kill between a write and its record): same chunks.
        store.enrichLedger.backfilled = false
        let shotChunks = after.map { "\($0.kind)|\($0.locator)|\($0.snippet)" }
        let filmChunks = try XCTUnwrap(store.storedChunks(path: film.path)).map { "\($0.kind)|\($0.locator)" }
        let indexer = Indexer(store: store, embedder: embedder)
        for path in [shot.path, film.path] {
            let f = try XCTUnwrap(store.storedFiles(paths: [path])[path])
            let d = try XCTUnwrap(store.contentDigests(paths: [path])[path])
            XCTAssertTrue(indexer.enrichOne((path, f.kind, f.modified, f.size, 70), digest: d, settings: IndexSettings()))
        }
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: shot.path)).map { "\($0.kind)|\($0.locator)|\($0.snippet)" }, shotChunks)
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: film.path)).map { "\($0.kind)|\($0.locator)" }, filmChunks)
    }

    // MARK: - A screenshot read whole before (0.20.0), read again in pieces

    /// An index as 0.20.0 left it - each screenshot read whole, under that reading's keys, done in
    /// the ledger - becomes, in one enrichment, exactly what indexing the same files today makes:
    /// the picture not embedded again, its text read in pieces, the keys today's. A copy takes the
    /// rows of the one done first; a picture that is not a screenshot is not touched; a second
    /// pass does nothing.
    func testAScreenshotReadWholeBeforeBecomesWhatIndexingItTodayMakes() throws {
        final class Counting: Embedder, @unchecked Sendable {
            let inner = Fake()
            var dim: Int { inner.dim }
            var pictures = 0, texts = 0
            func embedText(_ text: String, as type: OmniInputType) -> [Float] { texts += 1; return inner.embedText(text, as: type) }
            func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { texts += t.count; return inner.embedTextBatch(t, as: type) }
            func embedImage(_ image: CGImage) -> [Float]? { pictures += 1; return inner.embedImage(image) }
            func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { pictures += raws.count; return inner.embedImages(raws) }
            func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { inner.embedVideoFrames(frames) }
            func embedAudio(_ url: URL) -> [Float]? { inner.embedAudio(url) }
            func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { inner.embedAudioMel(mel, frames: frames) }
            func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { inner.embedAudioMelBatch(mels, frames: frames) }
        }
        let shot = try screenshot(["Booking confirmed", "Confirmation code: K7Q2LM", "Seat 32A"], name: "Screenshot 2026-11-01 at 10.00.00.png")
        let twin = root.appendingPathComponent("Screenshot 2026-11-01 at 10.00.00 copy.png")
        try FileManager.default.copyItem(at: shot, to: twin)
        let other = try screenshot(["Order #114-3328-9912 has shipped"], name: "Screenshot 2026-11-02 at 09.30.00.png")
        // Not a screenshot: the same kind of picture, without the mark.
        let photo = try screenshot(["A sign by the road"], name: "sign.png")
        XCTAssertEqual(removexattr(photo.path, "com.apple.metadata:kMDItemIsScreenCapture", 0), 0)
        let paths = [shot.path, twin.path, other.path, photo.path]

        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        // NEGATIVE CONTROL: the reading 0.20.0 had, to build the index an upgrade finds. Restored at once.
        ScreenText.inPieces = false
        defer { ScreenText.inPieces = true }
        pass(Indexer(store: store, embedder: embedder), IndexSettings())
        ScreenText.inPieces = true
        let before = Dictionary(uniqueKeysWithValues: try paths.map { ($0, try XCTUnwrap(store.storedChunks(path: $0))) })
        let digestsBefore = store.contentDigests(paths: paths)
        for path in [shot.path, twin.path, other.path] {
            XCTAssertTrue(before[path]?.contains { $0.kind == "text" } == true, "the old index has its text (control)")
            let d = try XCTUnwrap(digestsBefore[path])
            XCTAssertEqual(store.enrichLedger.done([EnrichLedger.key(digest: d, path: path)]).count, 1, "done, as 0.20.0 recorded it (control)")
            XCTAssertTrue(store.enrichLedger.done([EnrichLedger.key(digest: d, path: path, screenshot: true)]).isEmpty)
        }
        XCTAssertTrue(store.enrichLedger.backfilled)

        // The same files indexed today, in an index of their own.
        let freshDir = db.appendingPathComponent("fresh")
        try FileManager.default.createDirectory(at: freshDir, withIntermediateDirectories: true)
        let fresh = try VectorStore(dbURL: freshDir.appendingPathComponent("index.sqlite"))
        defer { fresh.close() }
        pass(Indexer(store: fresh, embedder: Fake()), IndexSettings())
        let digestsToday = fresh.contentDigests(paths: paths)

        // The upgrade: the new scope clears "backfilled" at the next launch (EnrichLedger.prepare).
        store.enrichLedger.backfilled = false
        let pictures = embedder.pictures
        pass(Indexer(store: store, embedder: embedder), IndexSettings())
        XCTAssertEqual(embedder.pictures, pictures, "a picture was embedded again")
        let digestsAfter = store.contentDigests(paths: paths)
        func shape(_ rows: [IndexedChunk]?) -> [String] {
            (rows ?? []).sorted { $0.chunkIndex < $1.chunkIndex }.map { "\($0.kind)|\($0.chunkIndex)|\($0.locator)|\($0.snippet)|\($0.chunkKey)|\($0.embedding)" }
        }
        for path in [shot.path, twin.path, other.path] {
            XCTAssertNotEqual(digestsAfter[path], digestsBefore[path], "still under the old reading's key")
            XCTAssertEqual(digestsAfter[path], digestsToday[path])
            XCTAssertEqual(shape(store.storedChunks(path: path)), shape(fresh.storedChunks(path: path)), "not what indexing it today makes")
            XCTAssertNotEqual(store.storedChunks(path: path)?.first { $0.kind == "image" }?.chunkKey, before[path]?.first { $0.kind == "image" }?.chunkKey)
            let d = try XCTUnwrap(digestsAfter[path])
            XCTAssertEqual(store.enrichLedger.done([EnrichLedger.key(digest: d, path: path, screenshot: true)]).count, 1)
        }
        XCTAssertEqual(digestsAfter[shot.path], digestsAfter[twin.path], "copies share a content")
        XCTAssertEqual(digestsAfter[photo.path], digestsBefore[photo.path], "a picture that is not a screenshot was touched")
        XCTAssertEqual(shape(store.storedChunks(path: photo.path)), shape(before[photo.path]))
        XCTAssertTrue(store.enrichLedger.backfilled)
        let found = store.search(embedder.embedText("K7Q2LM", as: .query), topK: 10, textQuery: "K7Q2LM")
        XCTAssertEqual(Set(found.prefix(2).map(\.path)), [shot.path, twin.path], "the code is found under the new key")

        // Done: asked again, nothing is read or embedded.
        store.enrichLedger.backfilled = false
        let texts = embedder.texts
        let indexer = Indexer(store: store, embedder: embedder)
        var asked: [String] = []
        XCTAssertEqual(indexer.enrichMedia(settings: IndexSettings(), onFile: { asked.append($0) }), 0)
        XCTAssertEqual(asked, [])
        XCTAssertEqual(embedder.texts, texts)
        XCTAssertEqual(embedder.pictures, pictures)
    }

    /// Text the reader no longer finds is taken out with the old reading, not left beside the picture.
    func testAScreenshotWithNoTextNowLosesTheTextItHad() throws {
        let shot = try screenshot(["Booking confirmed", "Confirmation code: K7Q2LM"], name: "Screenshot 2026-11-03 at 08.00.00.png")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Fake())
        pass(indexer, IndexSettings())
        let had = try XCTUnwrap(store.storedChunks(path: shot.path))
        XCTAssertGreaterThan(had.count, 1, "no text indexed; the test proves nothing")
        let f = try XCTUnwrap(store.storedFiles(paths: [shot.path])[shot.path])
        let d = try XCTUnwrap(store.contentDigests(paths: [shot.path])[shot.path])
        // A key of the stored form, other than the one it has.
        let old = try XCTUnwrap(had.first { $0.kind == "image" }?.chunkKey)
        let today = String(old.dropLast()) + (old.hasSuffix("0") ? "1" : "0")
        indexer.resetCancelled()
        XCTAssertTrue(indexer.addScreenText(CrawledFile(path: shot.path, modified: f.modified, size: f.size), digest: d, pieces: [],
                                            settings: IndexSettings(), pictureKey: today))
        let has = try XCTUnwrap(store.storedChunks(path: shot.path))
        XCTAssertEqual(has.map(\.kind), ["image"])
        XCTAssertEqual(has.first?.chunkKey, today)
        XCTAssertEqual(has.first?.embedding, had.first { $0.kind == "image" }?.embedding)
    }

    // MARK: - Failures are retried

    /// `Fake` whose audio can be refused.
    final class AudioRefused: Embedder, @unchecked Sendable {
        let inner = Fake()
        var refuse = true
        var dim: Int { inner.dim }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { inner.embedText(text, as: type) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { inner.embedTextBatch(texts, as: type) }
        func embedImage(_ image: CGImage) -> [Float]? { inner.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { inner.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { inner.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { refuse ? nil : inner.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { refuse ? nil : inner.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { refuse ? nil : inner.embedAudioMelBatch(mels, frames: frames) }
    }

    private func runPass(_ store: VectorStore, _ embedder: any Embedder, _ settings: IndexSettings) {
        let done = expectation(description: "pass")
        Indexer(store: store, embedder: embedder).index(roots: [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 120)
    }

    /// The enrichment that failed on a file finished anyway, and the file went without for good.
    func testAFileTheEnrichmentFailedOnIsTriedAgain() throws {
        let clip = try video(seconds: 90, name: "clip.mp4")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        var without = IndexSettings(enabledKinds: [.video])
        without.soundtrack = false
        let embedder = AudioRefused()
        runPass(store, embedder, without)
        XCTAssertFalse(try XCTUnwrap(store.storedChunks(path: clip.path)).contains { $0.kind == "audio" })
        // The soundtrack comes on: the phase runs, and the tower refuses it.
        let with = IndexSettings(enabledKinds: [.video])
        runPass(store, embedder, with)
        XCTAssertFalse(store.enrichLedger.backfilled, "a file not done was called done")
        embedder.refuse = false
        runPass(store, embedder, with)
        XCTAssertTrue(try XCTUnwrap(store.storedChunks(path: clip.path)).contains { $0.kind == "audio" }, "it was never tried again")
        XCTAssertTrue(store.enrichLedger.backfilled)
    }

    /// One that always fails stops being tried, or the phase runs at the end of every pass for good.
    func testAFileThatAlwaysFailsStopsBeingTried() throws {
        _ = try video(seconds: 90, name: "clip.mp4")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        var without = IndexSettings(enabledKinds: [.video])
        without.soundtrack = false
        let embedder = AudioRefused()
        runPass(store, embedder, without)
        let with = IndexSettings(enabledKinds: [.video])
        for expected in [false, false, true] {
            runPass(store, embedder, with)
            XCTAssertEqual(store.enrichLedger.backfilled, expected)
        }
    }

    /// A soundtrack that failed at index time: "the enrichment pass retries it" only if one runs.
    func testASoundtrackThatFailedAtIndexTimeIsTriedAgain() throws {
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        try "a note".write(to: root.appendingPathComponent("note.txt"), atomically: true, encoding: .utf8)
        let settings = IndexSettings(enabledKinds: [.text, .video])
        let embedder = AudioRefused()
        runPass(store, embedder, settings)
        XCTAssertTrue(store.enrichLedger.backfilled, "the index was not settled before the new file")
        let clip = try video(seconds: 90, name: "clip.mp4")
        runPass(store, embedder, settings)
        XCTAssertFalse(try XCTUnwrap(store.storedChunks(path: clip.path)).isEmpty)
        embedder.refuse = false
        runPass(store, embedder, settings)
        XCTAssertTrue(try XCTUnwrap(store.storedChunks(path: clip.path)).contains { $0.kind == "audio" }, "the soundtrack was never tried again")
    }
}
