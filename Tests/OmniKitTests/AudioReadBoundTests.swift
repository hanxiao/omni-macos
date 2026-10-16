import XCTest
import AVFAudio
@testable import OmniKit

/// The segment reader reads at most `readFrames` per AVAudioFile call. One large request sent
/// AudioToolbox's AAC decoder into a read that never returned while its realtime-safe allocator
/// grew to 31 GB (a silent iPhone screen recording's soundtrack, Omni 0.17.0); the bound must not
/// change a single sample of what the reader returns.
final class AudioReadBoundTests: XCTestCase {
    /// 70 s of stereo 44.1 kHz, a different tone per channel: several reads per segment, more than
    /// one segment, a short tail, a downmix and a resample.
    private func writeTone(_ url: URL) throws {
        let fmt = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
        let file = try AVAudioFile(forWriting: url, settings: fmt.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let chunk: AVAudioFrameCount = 44_100
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: chunk)!
        buf.frameLength = chunk
        for second in 0 ..< 70 {
            for i in 0 ..< Int(chunk) {
                let t = Double(second * Int(chunk) + i) / 44_100
                buf.floatChannelData![0][i] = Float(0.4 * sin(2 * .pi * 440 * t))
                buf.floatChannelData![1][i] = Float(0.3 * sin(2 * .pi * 660 * t + 0.5))
            }
            try file.write(from: buf)
        }
    }

    func testTheReadBoundChangesNoSample() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tone-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTone(url)
        XCTAssertLessThanOrEqual(OmniAudioPreprocess.AudioSegmentReader.readFrames, 65_536)
        // 0.17.0 read 60 s per call; the reader now reads `readFrames`.
        let wide = try XCTUnwrap(OmniAudioPreprocess.AudioSegmentReader(url: url, readFrames: 2_646_000))
        let bounded = try XCTUnwrap(OmniAudioPreprocess.AudioSegmentReader(url: url))
        XCTAssertEqual(wide.segmentCount, bounded.segmentCount)
        var segments = 0
        while let a = wide.nextSegment() {
            let b = try XCTUnwrap(bounded.nextSegment(), "segment \(segments)")
            XCTAssertEqual(a, b, "segment \(segments) differs")
            segments += 1
        }
        XCTAssertNil(bounded.nextSegment())
        XCTAssertEqual(segments, 2)   // 70 s: one full minute and a 10 s tail
    }

    /// A parked reader holds nothing open and reads nothing, but still answers its plan.
    func testACloseReaderKeepsItsPlan() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tone-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTone(url)
        let r = try XCTUnwrap(OmniAudioPreprocess.AudioSegmentReader(url: url))
        r.close()
        XCTAssertEqual(r.segmentCount, 2)
        XCTAssertNil(r.nextSegment())
        r.seek(toSegment: 1)
        XCTAssertNil(r.nextMelSegment())
    }
}
