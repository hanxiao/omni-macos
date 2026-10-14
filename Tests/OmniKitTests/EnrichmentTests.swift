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
}
