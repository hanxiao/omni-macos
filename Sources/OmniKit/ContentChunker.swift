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
/// THE HASH SEES BYTES; THE SIZE GATES COUNT CHARACTERS. Both halves matter and they are not the
/// same decision - see `Params`. Every emitted boundary is guaranteed to sit on a scalar boundary,
/// so no chunk can split a multi-byte character and no chunk is ever invalid UTF-8.
public enum ContentChunker {

    // MARK: - Parameters

    /// The sizes a cut is measured against, and the string that names them inside a chunk key.
    ///
    /// SIZES ARE IN CHARACTERS, NOT BYTES, and that is a correction rather than a preference. The
    /// hash must see bytes - it is a byte-wise rolling hash - but the SIZE GATES are what decide
    /// how much text a chunk holds, and the grid this replaces has always counted characters. Left
    /// in bytes, a Chinese document (3 bytes a character) cut to a 1800-BYTE target holds 600
    /// characters where the same setting gives an English one 1800: a third of the context per
    /// chunk, three times the chunks, three times the vectors, on exactly the corpora least able to
    /// spare the precision. Counting scalars costs one comparison per byte - a scalar starts at
    /// every byte that is not a 10xxxxxx continuation - and makes the cutter script-neutral.
    ///
    /// `lineSnapWindow` stays in BYTES: it is a search distance for the next newline, not a size.
    public struct Params: Sendable, Equatable {
        public let minChars: Int
        public let targetChars: Int
        public let maxChars: Int
        public let normalization: Int
        public let lineSnapWindow: Int

        public init(minChars: Int, targetChars: Int, maxChars: Int,
                    normalization: Int, lineSnapWindow: Int) {
            self.minChars = minChars; self.targetChars = targetChars; self.maxChars = maxChars
            self.normalization = normalization; self.lineSnapWindow = lineSnapWindow
        }

        /// Names the cutter and every number that moves a boundary, so chunks cut under different
        /// settings occupy disjoint key spaces and can coexist in one index. Changing a size
        /// re-cuts rather than silently mixing chunks cut under different rules into one key
        /// space - which is what the grid's key already does with its `c<maxChars>` field.
        public var fingerprint: String {
            "cdc\(minChars)-\(targetChars)-\(maxChars)-n\(normalization)-l\(lineSnapWindow)"
        }

        /// DERIVED FROM THE USER'S SETTING, because that setting has to keep working.
        /// "Characters per chunk" is in Settings > Performance with four values, and a cutter that
        /// ignored it would make the control silently do nothing. The target IS the setting - which
        /// is why the label lost the word "Max" when this shipped; the
        /// floor is half of it, and the ceiling twice it plus the line-snap slack - which at the
        /// default 1800 reproduces the 900 / 1800 / 4000 the parameter study measured.
        public static func forMaxChars(_ n: Int) -> Params {
            let t = Swift.max(200, n)
            return Params(minChars: t / 2, targetChars: t, maxChars: t * 2 + 400,
                          normalization: 2, lineSnapWindow: 300)
        }

        public static let `default` = Params.forMaxChars(1800)

        /// Bit count for the strict (pre-target) and lax (post-target) masks. The target sets the
        /// base: a mask of b bits fires about once every 2^b characters.
        var maskBits: UInt64 { UInt64(64 - UInt64(targetChars).leadingZeroBitCount - 1) }
        var maskStrict: UInt64 { (1 << (maskBits + UInt64(normalization))) - 1 }
        var maskLax: UInt64 { (1 << (maskBits - UInt64(normalization))) - 1 }
    }

    /// The shipped defaults, kept as top-level names because the tests and the notes above quote
    /// them. Every one of them is `Params.default`'s.
    public static var minBytes: Int { Params.default.minChars }
    public static var targetBytes: Int { Params.default.targetChars }
    public static var maxBytes: Int { Params.default.maxChars }
    public static var normalization: Int { Params.default.normalization }
    public static var lineSnapWindow: Int { Params.default.lineSnapWindow }
    public static var fingerprint: String { Params.default.fingerprint }

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

    // MARK: - Cutting

    /// Byte offsets where `bytes` should be cut, excluding 0 and including `bytes.count`.
    /// Every returned offset sits on a UTF-8 scalar boundary.
    ///
    /// The walk carries TWO cursors over the same bytes: `i` for the hash, which must see every
    /// byte, and `chars`, the number of scalars since the chunk started, which is what the min,
    /// target and max gates are compared against. A scalar begins at every byte that is not a
    /// 10xxxxxx continuation, so the count is one mask and one compare per byte.
    static func cutPoints(_ bytes: [UInt8], _ p: Params = .default) -> [Int] {
        var cuts: [Int] = []
        var start = 0
        let n = bytes.count
        let maskStrict = p.maskStrict, maskLax = p.maskLax
        @inline(__always) func isScalarStart(_ b: UInt8) -> Bool { b & 0xC0 != 0x80 }
        while start < n {
            // A tail that cannot reach the minimum is not worth cutting: it would leave a runt
            // whose only content is the end of the file. Measured in characters like every other
            // gate, so the test is how much TEXT is left, not how many bytes encode it.
            var tailChars = 0
            var t = start
            while t < n, tailChars <= p.minChars {
                if isScalarStart(bytes[t]) { tailChars += 1 }
                t += 1
            }
            if tailChars <= p.minChars { cuts.append(n); break }

            // Cut-point skipping: no hashing at all below the floor. Walk to it counting scalars.
            var i = start
            var chars = 0
            while i < n, chars < p.minChars {
                if isScalarStart(bytes[i]) { chars += 1 }
                i += 1
            }
            var hash: UInt64 = 0
            var cut = 0
            var hardEnd = n
            while i < n {
                if isScalarStart(bytes[i]) {
                    chars += 1
                    if chars > p.maxChars { hardEnd = i; break }
                }
                hash = (hash << 1) &+ gear[Int(bytes[i])]
                let mask = chars <= p.targetChars ? maskStrict : maskLax
                if hash & mask == 0, isScalarStart(bytes[i]) { cut = i; break }
                i += 1
            }
            if cut == 0 { cut = hardEnd }
            cut = snapToLine(bytes, from: cut, limit: Swift.min(hardEnd + p.lineSnapWindow, n), p)
            cut = alignToScalar(bytes, cut)
            // alignToScalar can only move a cut backwards, and snapToLine forwards; neither may
            // leave the cut at or before where this chunk started, or the loop cannot advance.
            if cut <= start {
                cut = alignForward(bytes, Swift.min(start + Swift.max(1, p.minChars), n))
            }
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
    private static func snapToLine(_ bytes: [UInt8], from cut: Int, limit: Int, _ p: Params) -> Int {
        let stop = Swift.min(cut + p.lineSnapWindow, limit)
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

    /// Cut `text` at content-defined boundaries. A text at or under the floor is one piece.
    public static func cut(_ text: String, _ p: Params = .default) -> [Piece] {
        let bytes = Array(text.utf8)
        guard !text.isEmpty else { return [] }
        guard bytes.count > p.minChars else { return [Piece(text: text, byteOffset: 0)] }
        var out: [Piece] = []
        var start = 0
        for cut in cutPoints(bytes, p) {
            guard cut > start else { continue }
            out.append(Piece(text: String(decoding: bytes[start ..< cut], as: UTF8.self),
                             byteOffset: start))
            start = cut
        }
        return out
    }
}
