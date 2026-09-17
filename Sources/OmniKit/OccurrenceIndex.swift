import Foundation

/// THE POINTERS, RESIDENT.
///
/// v4's in-memory model is one row per chunk with the owning path on it, so a score and a file are
/// the same thing. Under v5 a slot is a CONTENT and may belong to many files, so the two have to be
/// separated: the scan produces a score per slot, and this structure turns slots into files.
///
/// Two parallel Int32 arrays rather than an array of structs, and rather than a dictionary. On the
/// measured index this is 9,725,096 occurrences, so a struct with a String would be gigabytes and a
/// dictionary would be a cache miss per lookup. Two Int32 columns are 78 MB and both hot loops walk
/// them linearly.
///
/// THE FILTER PROBLEM THIS EXISTS TO SOLVE. Every filter the app offers - folder scope, kind,
/// extension, date - is a property of a FILE, while the scan now runs over CONTENTS. A content can
/// sit in one file inside the scope and another outside it. Filtering after the scan is wrong:
/// truncating at top-k silently drops a chunk whose only in-scope occurrence ranked below the cut.
/// So the file-level decision is propagated DOWN to a per-slot mask before the scan, which is the
/// same pre-filtering choice Qdrant and Weaviate make, and the same shape `pathAllowGPULocked`
/// already has for files.
public struct OccurrenceIndex: Sendable {

    /// Owning file id per occurrence.
    public private(set) var file: [Int32] = []
    /// Slot (chunk id) per occurrence, lockstep with `file`.
    public private(set) var slot: [Int32] = []
    /// One past the highest slot, i.e. the length of the score vector the scan produces.
    public private(set) var slotCount: Int = 0

    public init() {}

    public init(file: [Int32], slot: [Int32], slotCount: Int) {
        precondition(file.count == slot.count, "occurrence columns must be lockstep")
        self.file = file; self.slot = slot; self.slotCount = slotCount
    }

    public var count: Int { file.count }

    public mutating func append(file f: Int32, slot s: Int32) {
        file.append(f); slot.append(s)
        if Int(s) >= slotCount { slotCount = Int(s) + 1 }
    }

    // MARK: - Filtering

    /// Which slots any allowed file reaches. A slot is scannable when at least ONE file that
    /// contains it passes the filter; the scan cannot be finer than that, because one vector is
    /// shared by every occurrence.
    ///
    /// This is deliberately permissive, and `expand` is what makes it correct: a slot let through
    /// by an out-of-scope sibling still only ever reports the files that actually qualify.
    public func slotMask(fileAllowed: (Int32) -> Bool) -> [Bool] {
        var allowedFile: [Int32: Bool] = [:]
        var mask = [Bool](repeating: false, count: slotCount)
        for i in 0 ..< file.count {
            let f = file[i]
            let ok: Bool
            if let cached = allowedFile[f] { ok = cached }
            else { ok = fileAllowed(f); allowedFile[f] = ok }
            if ok { mask[Int(slot[i])] = true }
        }
        return mask
    }

    /// The same, when the caller already holds a dense per-file decision. One linear pass, no
    /// hashing: this runs on a filtered query over millions of occurrences.
    public func slotMask(fileAllowedDense: [Bool]) -> [Bool] {
        var mask = [Bool](repeating: false, count: slotCount)
        for i in 0 ..< file.count {
            let f = Int(file[i])
            if f < fileAllowedDense.count, fileAllowedDense[f] { mask[Int(slot[i])] = true }
        }
        return mask
    }

    // MARK: - Expansion

    /// One file's best-scoring slot, with that score.
    public struct FileHit: Equatable, Sendable {
        public let file: Int32
        public let slot: Int32
        public let score: Float
    }

    /// Turn a score per slot into a best score per FILE, honouring the filter a second time.
    ///
    /// The second check is not redundant. `slotMask` lets a slot through when ANY file containing
    /// it qualifies, so without re-checking here a search scoped to one folder would report the
    /// same content under a file in a folder the user excluded. That is the failure this design
    /// has to avoid, and it costs one predicate per occurrence.
    public func expand(scores: [Float], fileAllowed: (Int32) -> Bool,
                       slotAllowed: ((Int32) -> Bool)? = nil) -> [FileHit] {
        var best: [Int32: (slot: Int32, score: Float)] = [:]
        var allowedFile: [Int32: Bool] = [:]
        for i in 0 ..< file.count {
            let s = slot[i]
            guard Int(s) < scores.count else { continue }
            if let slotAllowed, !slotAllowed(s) { continue }
            let f = file[i]
            let ok: Bool
            if let cached = allowedFile[f] { ok = cached }
            else { ok = fileAllowed(f); allowedFile[f] = ok }
            guard ok else { continue }
            let sc = scores[Int(s)]
            if let cur = best[f], cur.score >= sc { continue }
            best[f] = (s, sc)
        }
        return best.map { FileHit(file: $0.key, slot: $0.value.slot, score: $0.value.score) }
            .sorted { $0.score == $1.score ? $0.file < $1.file : $0.score > $1.score }
    }

    /// Every file holding this content, for showing what a hit stands for.
    public func files(ofSlot s: Int32) -> [Int32] {
        var out: [Int32] = []
        for i in 0 ..< slot.count where slot[i] == s { out.append(file[i]) }
        return out
    }

    /// Reference count per slot, derived. `chunk.refs` is a cache of exactly this, and the point of
    /// being able to recompute it is that an undercount frees a vector another file still points at
    /// and nothing says so.
    public func referenceCounts() -> [Int32: Int] {
        var out: [Int32: Int] = [:]
        for s in slot { out[s, default: 0] += 1 }
        return out
    }
}
