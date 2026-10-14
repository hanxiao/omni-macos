import Foundation
import AudioToolbox

/// A media file's length from its header, for the time-left estimate: what indexing audio and video
/// costs follows their length, not their size (a minute of WAV is ten times a minute of AAC). Read
/// before the file is decoded, so it has to be cheap: 0.12 ms a file cold on an external disk,
/// 0.02 warm, measured over 96 real files; the MP4 reading agreed with AVFoundation's to the
/// millisecond. nil when the header does not say - the estimate then counts the file as one
/// segment of video or as no audio, rather than guess.
public enum MediaDuration {
    public static func seconds(_ path: String, ext: String) -> Double? {
        if FileExtractor.audioExtensions.contains(ext) { return audio(path) }
        if ["mp4", "mov", "m4v", "3gp"].contains(ext) { return isoMovie(path) }
        return nil
    }

    /// Core Audio's own estimate (it reads the header and, for some formats, the packet table).
    static func audio(_ path: String) -> Double? {
        var f: AudioFileID?
        guard AudioFileOpenURL(URL(fileURLWithPath: path) as CFURL, .readPermission, 0, &f) == noErr, let f else { return nil }
        defer { AudioFileClose(f) }
        var d = 0.0
        var size = UInt32(MemoryLayout<Double>.size)
        guard AudioFileGetProperty(f, kAudioFilePropertyEstimatedDuration, &size, &d) == noErr, d.isFinite, d >= 0 else { return nil }
        return d
    }

    /// The movie header (moov/mvhd) of an MP4 or QuickTime file: walk the top-level boxes to
    /// `moov`, wherever it is (a file not written for streaming keeps it at the end), then its
    /// children to `mvhd`, whose timescale and duration give the length. A handful of small reads.
    static func isoMovie(_ path: String) -> Double? {
        guard let h = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? h.close() }
        guard let end = try? h.seekToEnd() else { return nil }
        func read(_ at: UInt64, _ n: Int) -> Data? {
            guard (try? h.seek(toOffset: at)) != nil, let d = try? h.read(upToCount: n) else { return nil }
            return d
        }
        func be(_ d: Data, _ o: Int, _ n: Int) -> UInt64 {
            var v: UInt64 = 0
            for i in 0 ..< n { v = v << 8 | UInt64(d[d.startIndex + o + i]) }
            return v
        }
        func tag(_ d: Data, _ o: Int) -> UInt32 { UInt32(be(d, o, 4)) }
        let moov: UInt32 = 0x6D6F_6F76, mvhd: UInt32 = 0x6D76_6864
        /// (size, header length) of the box at `at`, or nil.
        func box(_ at: UInt64, within limit: UInt64) -> (size: UInt64, header: UInt64, type: UInt32)? {
            guard at + 8 <= limit, let d = read(at, 16), d.count >= 8 else { return nil }
            var size = be(d, 0, 4), header: UInt64 = 8
            if size == 1 { guard d.count >= 16 else { return nil }; size = be(d, 8, 8); header = 16 }
            else if size == 0 { size = limit - at }
            guard size >= header, at + size <= limit else { return nil }
            return (size, header, tag(d, 4))
        }
        var at: UInt64 = 0
        var steps = 0
        while let b = box(at, within: end), steps < 64 {
            steps += 1
            if b.type == moov {
                var c = at + b.header
                let stop = at + b.size
                while let child = box(c, within: stop), steps < 128 {
                    steps += 1
                    if child.type == mvhd {
                        guard let m = read(c + child.header, 32), m.count >= 20 else { return nil }
                        let version = m[m.startIndex]
                        let (scale, length): (UInt64, UInt64) = version == 1
                            ? (m.count >= 32 ? be(m, 20, 4) : 0, m.count >= 32 ? be(m, 24, 8) : 0)
                            : (be(m, 12, 4), be(m, 16, 4))
                        return scale > 0 ? Double(length) / Double(scale) : nil
                    }
                    c += child.size
                }
                return nil
            }
            at += b.size
        }
        return nil
    }
}
