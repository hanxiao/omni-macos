import XCTest
import CryptoKit
@testable import OmniKit

final class ChunkKeyTests: XCTestCase {

    /// The v4 formula, written out a SECOND time and on purpose. `ChunkKey.grid` must keep matching
    /// this, so changing the stored key format takes two deliberate edits rather than one careless
    /// one. Everything about the migration depends on it: every existing index's 9.13M keys were
    /// computed this way, and the migration reuses those vectors by looking the key up. A byte of
    /// drift here turns a no-GPU migration into a full re-embed of the corpus, silently.
    private func v4Formula(_ text: String, maxChars: Int, overlap: Int, dim: Int) -> String {
        var h = SHA256()
        h.update(data: Data("1|c\(maxChars)|o\(overlap)|m\(dim)|".utf8))
        h.update(data: Data(text.utf8))
        return h.finalize().prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    func testGeneration1KeysAreBitIdenticalToV4() {
        for text in ["", "a", "the quarterly revenue report is attached",
                     String(repeating: "x", count: 5000), "中文文档检索", "emoji 👨‍👩‍👧‍👦 tail"] {
            XCTAssertEqual(ChunkKey.grid(text, maxChars: 1800, overlap: 200, dim: 768),
                           v4Formula(text, maxChars: 1800, overlap: 200, dim: 768))
        }
    }

    func testGeneration1KeysTrackTheirSettings() {
        let t = "same text, different settings"
        let a = ChunkKey.grid(t, maxChars: 1800, overlap: 200, dim: 768)
        XCTAssertNotEqual(a, ChunkKey.grid(t, maxChars: 2000, overlap: 200, dim: 768))
        XCTAssertNotEqual(a, ChunkKey.grid(t, maxChars: 1800, overlap: 100, dim: 768))
        XCTAssertNotEqual(a, ChunkKey.grid(t, maxChars: 1800, overlap: 200, dim: 512))
    }

    func testTheTwoGenerationsNeverCollide() {
        // One index holds both while it migrates. The same bytes under two cutters are different
        // chunks with different neighbours, and must never be served for each other.
        let t = "a passage that exists under both cutters"
        XCTAssertNotEqual(ChunkKey.grid(t, maxChars: 1800, overlap: 200, dim: 768),
                          ChunkKey.text(t, cutter: ContentChunker.fingerprint, dim: 768))
    }

    func testGeneration2KeysTrackTheCutter() {
        // Not a tautology: it fails if the fingerprint is left out of the prefix, which is the
        // mistake that would let chunks cut at 1800 be served for chunks cut at 3600.
        let a = ContentChunker.Params.forMaxChars(1800).fingerprint
        let b = ContentChunker.Params.forMaxChars(3600).fingerprint
        XCTAssertEqual(ChunkKey.text("x", cutter: a, dim: 768).count, 32)
        XCTAssertNotEqual(ChunkKey.text("x", cutter: a, dim: 768),
                          ChunkKey.text("x", cutter: b, dim: 768))
        XCTAssertNotEqual(ChunkKey.text("x", cutter: a, dim: 768),
                          ChunkKey.text("x", cutter: a, dim: 512))
    }

    func testKeysAreStableAcrossCalls() {
        let t = String(repeating: "stability ", count: 400)
        XCTAssertEqual(ChunkKey.text(t, cutter: ContentChunker.fingerprint, dim: 768),
                       ChunkKey.text(t, cutter: ContentChunker.fingerprint, dim: 768))
        XCTAssertEqual(ChunkKey.media(kind: .image, payload: Data([1, 2, 3]), preprocess: "p", dim: 768),
                       ChunkKey.media(kind: .image, payload: Data([1, 2, 3]), preprocess: "p", dim: 768))
    }

    func testMediaKeysSeparateKindAndPreprocess() {
        let d = Data([9, 9, 9])
        let base = ChunkKey.media(kind: .image, payload: d, preprocess: "d1568", dim: 768)
        XCTAssertNotEqual(base, ChunkKey.media(kind: .video, payload: d, preprocess: "d1568", dim: 768))
        XCTAssertNotEqual(base, ChunkKey.media(kind: .image, payload: d, preprocess: "d1024", dim: 768))
    }

    func testSlicedMediaHashingMatchesTheWholePayload() {
        // A 240 s mel arrives in pieces; hashing it piecewise must equal hashing it whole, or the
        // same segment keys differently depending on how it was read.
        let whole = Data((0 ..< 3000).map { UInt8($0 % 251) })
        let slices = [whole[0 ..< 1000], whole[1000 ..< 1731], whole[1731 ..< 3000]].map { Data($0) }
        XCTAssertEqual(ChunkKey.media(kind: .audio, slices: slices, preprocess: "s240", dim: 768),
                       ChunkKey.media(kind: .audio, payload: whole, preprocess: "s240", dim: 768))
    }

    func testAKeyIs128Bits() {
        XCTAssertEqual(ChunkKey.text("anything", cutter: ContentChunker.fingerprint,
                                     dim: 768).count, 32)   // 16 bytes as hex
        XCTAssertEqual(StoreSchema.hexToBytes(
            ChunkKey.text("anything", cutter: ContentChunker.fingerprint, dim: 768)).count, 16)
    }
}

final class SlotAllocatorTests: XCTestCase {

    func testAFreshAllocatorExtendsTheFile() {
        var a = SlotAllocator()
        XCTAssertEqual([a.allocate(), a.allocate(), a.allocate()], [0, 1, 2])
        XCTAssertEqual(a.highWater, 3)
    }

    func testAReleasedSlotIsReusedRatherThanGrowingTheFile() {
        // The whole reason a free list exists: v4 accumulated 96,256 holes that nothing took back.
        var a = SlotAllocator()
        for _ in 0 ..< 5 { _ = a.allocate() }
        a.release(1); a.release(3)
        a.commit()
        XCTAssertEqual(a.allocate(), 1)
        XCTAssertEqual(a.allocate(), 3)
        XCTAssertEqual(a.highWater, 5, "the file grew while holes were available")
    }

    func testAReleasedSlotIsNotReusedBeforeCommit() {
        // THE QUARANTINE. A search in flight may still hold this id in a candidate list; handing it
        // out now means a different chunk's vector is written underneath that reader, which scores
        // the wrong content and reports it under the wrong file. Nothing crashes.
        var a = SlotAllocator()
        for _ in 0 ..< 3 { _ = a.allocate() }
        a.release(0)
        XCTAssertEqual(a.allocate(), 3, "a quarantined slot was handed out in the same transaction")
        a.commit()
        XCTAssertEqual(a.allocate(), 0)
    }

    func testARollbackKeepsTheSlotsOwned() {
        var a = SlotAllocator()
        for _ in 0 ..< 3 { _ = a.allocate() }
        a.release(1)
        a.rollback()
        a.commit()
        XCTAssertEqual(a.allocate(), 3, "a rolled-back release still freed the slot")
    }

    func testTheSameSlotIsNeverHandedOutTwice() {
        var a = SlotAllocator()
        var seen = Set<Int>()
        for _ in 0 ..< 200 { XCTAssertTrue(seen.insert(a.allocate()).inserted) }
        for id in [3, 17, 99, 100] { a.release(id) }
        a.commit()
        var again = Set<Int>()
        for _ in 0 ..< 4 { XCTAssertTrue(again.insert(a.allocate()).inserted) }
        XCTAssertEqual(again, [3, 17, 99, 100])
    }

    func testReleasingSomethingOutOfRangeIsIgnored() {
        var a = SlotAllocator()
        _ = a.allocate()
        a.release(-1); a.release(50)
        a.commit()
        XCTAssertEqual(a.allocate(), 1, "an out-of-range release corrupted the free list")
    }

    func testCoverageCatchesALeak() {
        var a = SlotAllocator()
        for _ in 0 ..< 5 { _ = a.allocate() }
        XCTAssertTrue(a.covers(liveIDs: [0, 1, 2, 3, 4]))
        // Slot 2's chunk is gone but nobody released it: invisible, the file just never shrinks.
        XCTAssertFalse(a.covers(liveIDs: [0, 1, 3, 4]))
        XCTAssertEqual(a.leaked(liveIDs: [0, 1, 3, 4]), [2])
    }

    func testCoverageCatchesADoubleOwnedSlot() {
        var a = SlotAllocator()
        for _ in 0 ..< 3 { _ = a.allocate() }
        a.release(1); a.commit()
        XCTAssertFalse(a.covers(liveIDs: [0, 1, 2]), "a free slot that a chunk still owns passed")
    }

    func testReconcileRebuildsFromTheChunkTableAlone() {
        // The free list is a cache of a fact SQLite already holds, so it never has to survive a
        // crash - it only has to be rebuildable.
        let a = SlotAllocator.reconciled(liveIDs: [0, 2, 5], highWater: 6)
        XCTAssertEqual(a.available.sorted(), [1, 3, 4])
        XCTAssertTrue(a.covers(liveIDs: [0, 2, 5]))
    }

    func testReconcileOfAFullFileLeavesNothingFree() {
        let a = SlotAllocator.reconciled(liveIDs: Set(0 ..< 4), highWater: 4)
        XCTAssertTrue(a.available.isEmpty)
        var b = a
        XCTAssertEqual(b.allocate(), 4)
    }
}
