import XCTest
import AVFoundation
@testable import OmniKit

/// Long recordings, videos and scans take turns like long text: a turn's worth now, the rest at
/// the end of the queue, resumed where it stopped.
final class MediaTurnTests: XCTestCase {
    final class Counting: Embedder, @unchecked Sendable {
        let dim = 8
        private let lock = NSLock()
        private var audio = 0
        var audioCalls: Int { lock.withLock { audio } }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { [1, 0, 0, 0, 0, 0, 0, 0] }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in [1, 0, 0, 0, 0, 0, 0, 0] } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { frames.isEmpty ? nil : [0, 0, 1, 0, 0, 0, 0, 0] }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { lock.withLock { audio += 1 }; return [0, 0, 0, 1, 0, 0, 0, 0] }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? {
            lock.withLock { audio += mels.count }; return mels.map { _ in [0, 0, 0, 1, 0, 0, 0, 0] }
        }
    }

    private func folder(_ name: String) throws -> (URL, URL) {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let db = r.deletingLastPathComponent().appendingPathComponent("\(name)-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        let root = r
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        return (root, db)
    }

    /// `minutes` of a tone, 16 kHz mono AAC.
    private func recording(minutes: Int, at url: URL) throws {
        let fmt = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let file = try AVAudioFile(forWriting: url, settings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 16000,
                                                              AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 24000])
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 16000)!
        for s in 0 ..< minutes * 60 {
            buf.frameLength = 16000
            let d = buf.floatChannelData![0]
            for k in 0 ..< 16000 { d[k] = Float(sin(Double(s * 16000 + k) * 2 * .pi * 330 / 16000)) * 0.2 }
            try file.write(from: buf)
        }
    }

    /// `minutes` of black frames, one a second, with a tone soundtrack.
    private func video(minutes: Int, at url: URL) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let vIn = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: vIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64])
        let aIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 16000,
                                                                          AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 24000])
        writer.add(vIn); writer.add(aIn)
        XCTAssertTrue(writer.startWriting()); writer.startSession(atSourceTime: .zero)
        var asbd = AudioStreamBasicDescription(mSampleRate: 16000, mFormatID: kAudioFormatLinearPCM,
                                               mFormatFlags: kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
                                               mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        var fmt: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &fmt)
        for t in 0 ..< minutes * 60 {
            while !vIn.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
            adaptor.append(pb!, withPresentationTime: CMTime(value: CMTimeValue(t), timescale: 1))
            let samples = (0 ..< 16000).map { Float(sin(Double(t * 16000 + $0) * 2 * .pi * 330 / 16000)) * 0.2 }
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: 64000, blockAllocator: nil, customBlockSource: nil, offsetToData: 0, dataLength: 64000, flags: 0, blockBufferOut: &block)
            samples.withUnsafeBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block!, offsetIntoDestination: 0, dataLength: 64000) }
            var sb: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: block!, formatDescription: fmt!, sampleCount: 16000,
                                                                 presentationTimeStamp: CMTime(value: CMTimeValue(t * 16000), timescale: 16000), packetDescriptions: nil, sampleBufferOut: &sb)
            while !aIn.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
            aIn.append(sb!)
        }
        vIn.markAsFinished(); aIn.markAsFinished()
        let done = expectation(description: "video")
        writer.finishWriting { done.fulfill() }
        wait(for: [done], timeout: 120)
        XCTAssertEqual(writer.status, .completed)
    }

    /// 44 minutes: 11 segments, more than a turn. Frames in two turns, the soundtrack with the last.
    func testALongVideoTakesTurnsAndKeepsItsSoundtrack() throws {
        let (root, db) = try folder("vturn")
        let film = root.appendingPathComponent("talk.mp4")
        try video(minutes: 44, at: film)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Counting())
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.video])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 600)
        let chunks = try XCTUnwrap(store.storedChunks(path: film.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(chunks.filter { $0.kind == "video" }.map(\.locator), (0 ..< 11).map { "\($0 * 4):00" })
        XCTAssertEqual(chunks.filter { $0.kind == "audio" }.count, 44, "one soundtrack chunk a minute")
        XCTAssertEqual(indexer.streams.count, 0)
    }

    func testALongRecordingIsIndexedWholeOverTurns() throws {
        let (root, db) = try folder("mturn")
        let long = root.appendingPathComponent("lecture.m4a")
        try recording(minutes: 25, at: long)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Counting())
        let done = expectation(description: "pass")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.audio])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
        let chunks = try XCTUnwrap(store.storedChunks(path: long.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(chunks.count, 25, "\(chunks.map(\.locator))")
        XCTAssertEqual(chunks.first?.locator, "0:00"); XCTAssertEqual(chunks.last?.locator, "24:00")
        XCTAssertEqual(indexer.streams.count, 0, "a cursor was left behind")
        XCTAssertNotNil(store.contentKeys(paths: [long.path])[long.path], "the finished recording's content key was not recorded")
    }

    func testAStoppedRecordingResumesAtItsTurn() throws {
        let (root, db) = try folder("mresume")
        let long = root.appendingPathComponent("podcast.m4a")
        try recording(minutes: 25, at: long)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = Counting()
        var indexer = Indexer(store: store, embedder: embedder)
        let first = expectation(description: "pass 1")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.audio])) { p in
            if p.embedded >= 1 { indexer.cancel(.pause) }
            if p.done { first.fulfill() }
        }
        wait(for: [first], timeout: 300)
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: long.path)).count, Indexer.audioWindowsPerTurn, "the first turn was not stored")
        XCTAssertEqual(indexer.streams.count, 1, "no cursor to resume from")
        XCTAssertNil(store.contentKeys(paths: [long.path])[long.path], "a part-read recording's key was recorded")
        let afterFirst = embedder.audioCalls
        indexer = Indexer(store: store, embedder: embedder)   // a new launch
        let second = expectation(description: "pass 2")
        indexer.index(roots: [root], settings: IndexSettings(enabledKinds: [.audio])) { p in if p.done { second.fulfill() } }
        wait(for: [second], timeout: 300)
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: long.path)).count, 25)
        XCTAssertEqual(embedder.audioCalls - afterFirst, 5, "the resumed pass embedded the first turn again")
        XCTAssertEqual(indexer.streams.count, 0)
    }
}
