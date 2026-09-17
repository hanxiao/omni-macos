import XCTest
@testable import OmniKit

/// The set difference that decides what gets embedded. Every case here is a real shape from the
/// index: a file re-saved unchanged, a log appended to, a document edited in the middle, a file
/// copied, a file deleted. A wrong reference count here frees a slot another file still points at,
/// which is silent.
final class ChunkDiffTests: XCTestCase {

    /// The store holds nothing.
    private let empty: (String) -> Bool = { _ in false }
    /// The store holds everything, which is what a moved or copied file meets.
    private let full: (String) -> Bool = { _ in true }

    // MARK: - The cheap cases

    func testAnUnchangedFileCostsNothing() {
        let keys = ["a", "b", "c"]
        let d = ChunkDiff.plan(old: keys, new: keys, isStored: full)
        XCTAssertTrue(d.isEmpty)
        XCTAssertEqual(d.embed, [])
        XCTAssertEqual(d.refDelta, [:])
        XCTAssertEqual(d.reused, 3)
    }

    func testACopiedFileEmbedsNothing() {
        // The file is new - no old pointers - but every content already exists. This is the case
        // that used to cost a full re-embed because reuse was scoped to the file's own path.
        let d = ChunkDiff.plan(old: [], new: ["a", "b", "c"], isStored: full)
        XCTAssertEqual(d.embed, [], "a copy of existing content was scheduled for embedding")
        XCTAssertEqual(d.refDelta, ["a": 1, "b": 1, "c": 1])
    }

    func testABrandNewFileEmbedsEverything() {
        let d = ChunkDiff.plan(old: [], new: ["a", "b"], isStored: empty)
        XCTAssertEqual(d.embed, ["a", "b"])
        XCTAssertEqual(d.refDelta, ["a": 1, "b": 1])
    }

    // MARK: - Edits

    func testAnAppendEmbedsOnlyTheTail() {
        let d = ChunkDiff.plan(old: ["a", "b"], new: ["a", "b", "c"]) { $0 != "c" }
        XCTAssertEqual(d.embed, ["c"])
        XCTAssertEqual(d.refDelta, ["c": 1])
        XCTAssertEqual(d.reused, 2)
    }

    func testAnEditInTheMiddleTouchesOnlyThatChunk() {
        // What content-defined boundaries buy: the chunks either side are untouched.
        let d = ChunkDiff.plan(old: ["a", "b", "c"], new: ["a", "B2", "c"]) { $0 != "B2" }
        XCTAssertEqual(d.embed, ["B2"])
        XCTAssertEqual(d.refDelta, ["b": -1, "B2": 1])
        XCTAssertEqual(d.reused, 2)
    }

    func testReorderingCostsNothing() {
        // Only the pointers move. The contents are the same, so no vector changes.
        let d = ChunkDiff.plan(old: ["a", "b", "c"], new: ["c", "a", "b"], isStored: full)
        XCTAssertEqual(d.embed, [])
        XCTAssertEqual(d.refDelta, [:], "a reorder changed a reference count")
    }

    func testADeletedFileReleasesEveryPointer() {
        let d = ChunkDiff.plan(old: ["a", "b"], new: [], isStored: full)
        XCTAssertEqual(d.embed, [])
        XCTAssertEqual(d.refDelta, ["a": -1, "b": -1])
        XCTAssertEqual(d.reused, 0)
    }

    // MARK: - Multiplicity, which is where this goes wrong silently

    func testRepeatedContentInOneFileIsCountedNotMembered() {
        // The config block that occurs 8,145 times across this index occurs many times per FILE
        // too. Treating "present" as a boolean decrements to zero while the file still points at it.
        let d = ChunkDiff.plan(old: ["a", "a", "a"], new: ["a", "a"], isStored: full)
        XCTAssertEqual(d.refDelta, ["a": -1], "multiplicity was collapsed to membership")
    }

    func testRepeatedNewContentIsEmbeddedOnce() {
        // The same new content at three ordinals is ONE forward pass and three pointers.
        let d = ChunkDiff.plan(old: [], new: ["x", "x", "x"], isStored: empty)
        XCTAssertEqual(d.embed, ["x"], "a repeated new content was embedded more than once")
        XCTAssertEqual(d.refDelta, ["x": 3])
    }

    func testGrowingTheCountOfARepeatedContent() {
        let d = ChunkDiff.plan(old: ["a"], new: ["a", "a", "a"], isStored: full)
        XCTAssertEqual(d.embed, [])
        XCTAssertEqual(d.refDelta, ["a": 2])
    }

    func testAContentThatMovesWithinTheFileIsNotReEmbedded() {
        let d = ChunkDiff.plan(old: ["a", "b", "b"], new: ["b", "b", "a"], isStored: full)
        XCTAssertTrue(d.isEmpty, "a move produced work: \(d)")
    }

    // MARK: - Mixed

    func testAPartlyRewrittenFile() {
        let old = ["h1", "p1", "p2", "p3", "f1"]
        let new = ["h1", "p1", "pX", "p3", "f1", "f2"]
        let stored = Set(old)
        let d = ChunkDiff.plan(old: old, new: new) { stored.contains($0) }
        XCTAssertEqual(d.embed, ["pX", "f2"])
        XCTAssertEqual(d.refDelta, ["p2": -1, "pX": 1, "f2": 1])
        XCTAssertEqual(d.reused, 4)
    }

    func testAContentDroppedHereButHeldElsewhereStillDecrements() {
        // The plan only reports the DELTA. Whether the content survives is the store's question,
        // because only the store knows the global count - which is the whole reason this is split.
        let d = ChunkDiff.plan(old: ["shared"], new: [], isStored: full)
        XCTAssertEqual(d.refDelta, ["shared": -1])
    }

    func testEmbedOrderIsStable() {
        // Embedding runs in batches and the batch is length-sorted downstream; a nondeterministic
        // order here would make an indexing pass non-reproducible for no reason.
        for _ in 0 ..< 20 {
            let d = ChunkDiff.plan(old: [], new: ["m", "a", "z", "a", "q"], isStored: empty)
            XCTAssertEqual(d.embed, ["m", "a", "z", "q"])
        }
    }

    func testNothingAtAll() {
        XCTAssertTrue(ChunkDiff.plan(old: [], new: [], isStored: empty).isEmpty)
    }
}
