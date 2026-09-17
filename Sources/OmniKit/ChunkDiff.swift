import Foundation

/// WHAT CHANGES WHEN A FILE CHANGES.
///
/// Under v4 a re-index of a file rewrote all of its chunks. Under v5 a file is a LIST OF POINTERS
/// into a content-addressed store, so re-indexing it is a set difference: work out which contents
/// are new to the store, which pointers move, and which contents nobody references any more. A file
/// that was moved or copied re-points at chunks that already exist and costs no GPU at all.
///
/// MULTIPLICITY IS THE TRAP, and it is not hypothetical on this corpus. The same content occurs
/// many times in one file - a repeated config block in an agent log occurs 8,145 times across the
/// index - so reference counting has to be by COUNT, not by membership. A plan that treated
/// "present in old" and "present in new" as booleans would decrement a chunk to zero and free a
/// slot that the same file still points at, twice. Nothing would crash; a later search would score
/// a different chunk's vector and report it under this file.
///
/// The plan is deliberately pure: it takes the two key lists and one question about the store, and
/// returns what to do. The store applies it, because only the store knows the GLOBAL reference
/// count and therefore which contents actually reach zero.
public struct ChunkDiff: Equatable, Sendable {

    /// Distinct contents that nothing in the store has yet, in first-seen order. These are the only
    /// ones that cost a forward pass.
    public let embed: [String]
    /// Net change to each key's reference count. Applied by the store on top of the global count;
    /// a key reaching zero there releases its slot.
    public let refDelta: [String: Int]
    /// Pointers this file already had that it still has, counted with multiplicity. Reporting only.
    public let reused: Int

    /// - Parameters:
    ///   - old: the file's current pointers, ordinal -> key. Empty when the file is new.
    ///   - new: the pointers it should have. Empty when the file is being removed.
    ///   - isStored: whether the store already holds a vector for this content, from any file.
    public static func plan(old: [String], new: [String],
                            isStored: (String) -> Bool) -> ChunkDiff {
        var delta: [String: Int] = [:]
        var oldCount: [String: Int] = [:]
        for k in old { oldCount[k, default: 0] += 1; delta[k, default: 0] -= 1 }
        for k in new { delta[k, default: 0] += 1 }
        delta = delta.filter { $0.value != 0 }

        // Embed a content only when the store has none. Held as a set as well as an array so a key
        // repeated inside this file is embedded once, which is the multiplicity trap in the other
        // direction: the same new content at three ordinals is one forward pass, not three.
        var embed: [String] = []
        var scheduled = Set<String>()
        for k in new where !scheduled.contains(k) {
            scheduled.insert(k)
            // Already in the store, or about to be by virtue of this file keeping it.
            if isStored(k) { continue }
            embed.append(k)
        }

        var reused = 0
        var remaining = oldCount
        for k in new {
            if let c = remaining[k], c > 0 { remaining[k] = c - 1; reused += 1 }
        }
        return ChunkDiff(embed: embed, refDelta: delta, reused: reused)
    }

    /// Nothing to do. A file whose content did not change produces this, which is what makes a
    /// re-crawl of an unchanged tree free.
    public var isEmpty: Bool { embed.isEmpty && refDelta.isEmpty }
}
