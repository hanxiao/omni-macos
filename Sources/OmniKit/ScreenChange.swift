import Foundation
import CoreVideo

/// What a frame looks like, kept small enough to compare with every later one: its green channel
/// sampled on a grid about 960 wide, read as tiles of 16 x 16 samples. Two captures of a screen that
/// did not change are the same bytes, so a tile counts as changed on a handful of moved samples, and
/// the share of changed tiles is how much of the screen is new.
public struct ScreenSignature: Sendable {
    public static let tile = 16
    public let width: Int      // samples
    public let height: Int
    let samples: [UInt8]
    /// Per tile: it shows something. A tile of one flat colour (an empty margin, a blank page, a
    /// black screen) shows nothing, whatever it showed before.
    let detail: [Bool]
    /// The share of tiles that show something, 0...1.
    public let detailShare: Double

    public init?(_ pixels: CVPixelBuffer) {
        guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixels) else { return nil }
        let w = CVPixelBufferGetWidth(pixels), h = CVPixelBufferGetHeight(pixels), stride = CVPixelBufferGetBytesPerRow(pixels)
        let step = max(1, w / 960)
        let sw = w / step, sh = h / step
        guard sw >= Self.tile, sh >= Self.tile else { return nil }
        let source = base.assumingMemoryBound(to: UInt8.self)
        var out = [UInt8](repeating: 0, count: sw * sh)
        out.withUnsafeMutableBufferPointer { o in
            for y in 0 ..< sh {
                let row = source + y * step * stride + 1      // B G R A: green
                let dst = o.baseAddress! + y * sw
                for x in 0 ..< sw { dst[x] = row[x * step * 4] }
            }
        }
        let t = Self.tile, columns = sw / t, rows = sh / t
        var shows = [Bool](repeating: false, count: columns * rows)
        var showing = 0
        out.withUnsafeBufferPointer { a in
            for ty in 0 ..< rows {
                for tx in 0 ..< columns {
                    var lo = UInt8.max, hi = UInt8.min
                    for y in ty * t ..< ty * t + t {
                        let o = y * sw + tx * t
                        for x in 0 ..< t { lo = min(lo, a[o + x]); hi = max(hi, a[o + x]) }
                    }
                    if hi - lo > Self.step { shows[ty * columns + tx] = true; showing += 1 }
                }
            }
        }
        width = sw; height = sh; samples = out; detail = shows
        detailShare = Double(showing) / Double(columns * rows)
    }

    /// Two samples further apart than this differ; nearer is the same colour drawn twice.
    static let step: UInt8 = 16

    /// The share of tiles that differ from `other`'s, 0...1. 1 when the two are not the same size.
    public func changedShare(from other: ScreenSignature) -> Double { share(from: other, showingOnly: false) }

    /// The share of tiles that differ from `other`'s AND show something: what this frame has that
    /// `other` does not. A window closed over an empty desktop changed the screen and added nothing.
    public func newShare(from other: ScreenSignature) -> Double { share(from: other, showingOnly: true) }

    private func share(from other: ScreenSignature, showingOnly: Bool) -> Double {
        guard other.width == width, other.height == height else { return showingOnly ? detailShare : 1 }
        let t = Self.tile, columns = width / t, rows = height / t
        var changed = 0
        samples.withUnsafeBufferPointer { a in
            other.samples.withUnsafeBufferPointer { b in
                for ty in 0 ..< rows {
                    for tx in 0 ..< columns where !showingOnly || detail[ty * columns + tx] {
                        var moved = 0
                        for y in ty * t ..< ty * t + t {
                            let o = y * width + tx * t
                            for x in 0 ..< t where abs(Int(a[o + x]) - Int(b[o + x])) > Int(Self.step) { moved += 1 }
                        }
                        if moved >= 2 { changed += 1 }
                    }
                }
            }
        }
        return Double(changed) / Double(columns * rows)
    }
}

/// WHICH SHOTS GO INTO THE INDEX. Every shot goes into the screen history; indexing one costs a
/// picture embedding and a reading of its text, and a shot that shows what an indexed one already
/// shows adds a second answer to the same question. A shot is indexed when enough of it is new
/// against EVERY recently indexed shot (so going back to a window already indexed is not new), or,
/// when the user stops, when anything at all is new: the state a screen was left in is the one
/// that holds everything typed into it. New means changed and showing something, so a screen that
/// went blank, or never showed anything, is not indexed at all.
public struct ScreenGate: Sendable {
    /// The share of the screen that has to be new while the user is working, and when they stop.
    /// Measured on generated sessions (omni-verify shotbench; docs/screenshots.md).
    public static let working = 0.10
    public static let settled = 0.005
    static let memory = 8

    private var indexed: [ScreenSignature] = []
    public init() {}

    /// How much of `shot` none of the recently indexed shots show.
    public func novelty(_ shot: ScreenSignature) -> Double {
        indexed.reduce(shot.detailShare) { min($0, shot.newShare(from: $1)) }
    }

    /// True when the shot is to be indexed; it is then remembered as one.
    public mutating func admit(_ shot: ScreenSignature, settled: Bool, working: Double = ScreenGate.working) -> Bool {
        guard novelty(shot) >= (settled ? ScreenGate.settled : working) else { return false }
        remember(shot)
        return true
    }

    /// A shot the index already holds: what a recorder that starts again reads back.
    public mutating func remember(_ shot: ScreenSignature) {
        indexed.append(shot)
        if indexed.count > Self.memory { indexed.removeFirst() }
    }
}
