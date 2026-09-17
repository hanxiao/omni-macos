import Foundation

/// CONTENT-DEFINED CHUNKING, the cutter for chunk generation 2.
///
/// The fixed grid cuts at `i * step`, so inserting one line near the top of a file shifts every
/// boundary below it and changes every chunk key below it. Measured on 121 real files of 60 KB or
/// more, one line inserted at 10% of the way through re-embeds 101.7 of 120.9 chunks. The same edit
/// under this cutter re-embeds 1.2 of 91.6. That is the whole reason this file exists: chunk
/// identity has to survive an insertion, or content addressing buys nothing for a file anyone edits.
///
/// This is the boundary-shift problem, and the fix is the one FastCDC describes (Xia et al., USENIX
/// ATC 2016): let the CONTENT pick the boundary. A rolling hash runs over the bytes and a cut is
/// declared wherever the hash has a run of low bits, so a boundary depends on the few dozen bytes
/// around it and on nothing else - not on the offset, not on what came before.
///
/// Three FastCDC techniques are used, and one is deliberately not:
///   - Gear hashing: one table lookup, one shift, one add per byte.
///   - Cut-point skipping: no hashing at all below `minBytes`, which is most of the work skipped.
///   - Normalized chunking at level 2: a strict mask before the target size and a lax one after,
///     which pulls the size distribution in towards the target. The paper measures level 2 as the
///     sweet spot, and notes that pushing normalization to its limit degenerates to fixed-size
///     chunking, i.e. straight back to the problem this solves.
///   - NOT the rolling-hash-free variants (AE, RAM, MII). They trade deduplication for throughput,
///     and throughput is not the constraint here: chunking is nanoseconds against milliseconds of
///     GPU per chunk.
///
/// SIZES ARE IN UTF-8 BYTES, not Characters. The hash has to see bytes, and a Character is one to
/// four of them. Every emitted boundary is still guaranteed to sit on a scalar boundary, so no
/// chunk can split a multi-byte character and no chunk is ever invalid UTF-8.
public enum ContentChunker {

    // MARK: - Parameters

    /// No cut is looked for below this. Also the floor on chunk size, except for a file's last chunk.
    public static let minBytes = 900
    /// Where the mask relaxes. The average chunk lands near here.
    public static let targetBytes = 1800
    /// A cut is forced here whether or not the hash agrees, so one chunk cannot swallow a file.
    public static let maxBytes = 4000
    /// Normalized chunking level. The paper's measured sweet spot.
    public static let normalization = 2
    /// How far past a content cut to look for a line break. See `snapToLine`.
    public static let lineSnapWindow = 300

    /// Names the cutter and its parameters inside the chunk key, so generation 1 and generation 2
    /// chunks occupy disjoint key spaces and can coexist in one index during migration without ever
    /// colliding. Changing any number above changes this string, which is what forces a re-cut
    /// rather than silently serving chunks cut under different rules.
    public static var fingerprint: String {
        "cdc\(minBytes)-\(targetBytes)-\(maxBytes)-n\(normalization)-l\(lineSnapWindow)"
    }

    // MARK: - Gear table

    /// 256 fixed 64-bit values, one per byte. DERIVED, NOT RANDOM: chunk boundaries decide chunk
    /// keys, and chunk keys have to be identical on every machine and every launch or nothing
    /// deduplicates against anything. A `random` table would re-cut the whole corpus on each run.
    /// SplitMix64 from a fixed seed gives a good scramble reproducibly, in eight lines, with no
    /// 256-entry literal to mistype.
    static let gear: [UInt64] = {
        var state: UInt64 = 0          // canonical SplitMix64 seeding: advance, then mix
        var table = [UInt64](repeating: 0, count: 256)
        for i in 0 ..< 256 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            table[i] = z ^ (z >> 31)
        }
        return table
    }()

    /// Bit count for the strict (pre-target) and lax (post-target) masks. `targetBytes` sets the
    /// base: a mask of b bits fires about once every 2^b bytes.
    static let maskBits = 64 - UInt64(targetBytes).leadingZeroBitCount - 1   // floor(log2(target))
    static let maskStrict: UInt64 = (1 << UInt64(maskBits + normalization)) - 1
    static let maskLax: UInt64 = (1 << UInt64(maskBits - normalization)) - 1

    // MARK: - Cutting

    /// Byte offsets where `bytes` should be cut, excluding 0 and including `bytes.count`.
    /// Every returned offset sits on a UTF-8 scalar boundary.
    static func cutPoints(_ bytes: [UInt8]) -> [Int] {
        var cuts: [Int] = []
        var start = 0
        let n = bytes.count
        while start < n {
            // A tail that cannot reach the minimum is not worth cutting: it would leave a runt
            // whose only content is the end of the file.
            if n - start <= minBytes { cuts.append(n); break }
            let hardEnd = Swift.min(start + maxBytes, n)
            let relax = Swift.min(start + targetBytes, hardEnd)
            var hash: UInt64 = 0
            var i = start + minBytes          // cut-point skipping: nothing below the floor
            var cut = 0
            while i < relax {
                hash = (hash << 1) &+ gear[Int(bytes[i])]
                if hash & maskStrict == 0 { cut = i; break }
                i += 1
            }
            if cut == 0 {
                while i < hardEnd {
                    hash = (hash << 1) &+ gear[Int(bytes[i])]
                    if hash & maskLax == 0 { cut = i; break }
                    i += 1
                }
            }
            if cut == 0 { cut = hardEnd }
            cut = snapToLine(bytes, from: cut, limit: hardEnd)
            cut = alignToScalar(bytes, cut)
            // alignToScalar can only move a cut backwards, and snapToLine forwards; neither may
            // leave the cut at or before where this chunk started, or the loop cannot advance.
            if cut <= start { cut = Swift.min(start + minBytes, n) ; cut = alignForward(bytes, cut) }
            cuts.append(cut)
            start = cut
        }
        if cuts.isEmpty { cuts.append(n) }
        return cuts
    }

    /// Move a content cut forward to just after the next newline, if one is close.
    ///
    /// The hash puts boundaries mid-sentence, which costs nothing for backup deduplication and a
    /// real amount for retrieval: a chunk starting halfway through a word reads badly as a snippet
    /// and embeds worse. Snapping keeps the boundary CONTENT-DEFINED - the search starts from the
    /// content cut, so an insertion elsewhere does not move it - while landing it where the text
    /// already breaks. Measured cost: insertion re-embeds 1.5 chunks instead of 1.2, against 101.7
    /// for the grid, which is a rounding error on the win.
    private static func snapToLine(_ bytes: [UInt8], from cut: Int, limit: Int) -> Int {
        let stop = Swift.min(cut + lineSnapWindow, limit)
        var i = cut
        while i < stop {
            if bytes[i] == 0x0A { return i + 1 }
            i += 1
        }
        return cut
    }

    /// UTF-8 continuation bytes are 10xxxxxx. Walk BACK to the start of the character the cut
    /// landed inside, so no chunk ever ends mid-character.
    private static func alignToScalar(_ bytes: [UInt8], _ at: Int) -> Int {
        var i = Swift.min(at, bytes.count)
        while i > 0, i < bytes.count, bytes[i] & 0xC0 == 0x80 { i -= 1 }
        return i
    }

    /// The same, forward. Used only on the degenerate path where aligning back would stall the loop.
    private static func alignForward(_ bytes: [UInt8], _ at: Int) -> Int {
        var i = Swift.min(at, bytes.count)
        while i < bytes.count, bytes[i] & 0xC0 == 0x80 { i += 1 }
        return i
    }

    // MARK: - API

    /// One piece of a cut-up text: its content and the UTF-8 offset it started at, which is what a
    /// caller turns into a locator ("Line 240", "Page 7").
    public struct Piece: Sendable, Equatable {
        public let text: String
        public let byteOffset: Int
        public init(text: String, byteOffset: Int) { self.text = text; self.byteOffset = byteOffset }
    }

    /// Cut `text` at content-defined boundaries. A text at or under `minBytes` is one piece.
    public static func cut(_ text: String) -> [Piece] {
        let bytes = Array(text.utf8)
        guard bytes.count > minBytes else {
            return text.isEmpty ? [] : [Piece(text: text, byteOffset: 0)]
        }
        var out: [Piece] = []
        var start = 0
        for cut in cutPoints(bytes) {
            guard cut > start else { continue }
            out.append(Piece(text: String(decoding: bytes[start ..< cut], as: UTF8.self),
                             byteOffset: start))
            start = cut
        }
        return out
    }
}
