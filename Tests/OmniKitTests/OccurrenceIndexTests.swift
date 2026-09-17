import XCTest
@testable import OmniKit

/// Turning scores over CONTENTS into results about FILES. The leak this guards against is specific
/// and silent: a chunk shared between an in-scope file and an out-of-scope one must never report
/// the out-of-scope file, and must never be dropped just because a sibling was excluded.
final class OccurrenceIndexTests: XCTestCase {

    /// file 10 -> slots 0,1   file 20 -> slots 1,2   (slot 1 is shared)
    private func shared() -> OccurrenceIndex {
        var ix = OccurrenceIndex()
        ix.append(file: 10, slot: 0)
        ix.append(file: 10, slot: 1)
        ix.append(file: 20, slot: 1)
        ix.append(file: 20, slot: 2)
        return ix
    }

    func testSlotCountTracksTheHighestSlot() {
        XCTAssertEqual(shared().slotCount, 3)
        XCTAssertEqual(shared().count, 4)
    }

    // MARK: - Masking

    func testAScopedSearchStillScansASharedChunk() {
        // Scoped to file 10. Slot 1 is shared with file 20 and must still be scanned, or the
        // content is invisible inside the folder the user actually asked about.
        let mask = shared().slotMask { $0 == 10 }
        XCTAssertEqual(mask, [true, true, false])
    }

    func testAnExcludedFileContributesNoSlots() {
        XCTAssertEqual(shared().slotMask { _ in false }, [false, false, false])
    }

    func testTheDenseMaskAgreesWithThePredicate() {
        var dense = [Bool](repeating: false, count: 21)
        dense[10] = true
        XCTAssertEqual(shared().slotMask(fileAllowedDense: dense), shared().slotMask { $0 == 10 })
    }

    func testADenseMaskShorterThanTheFileIdsDoesNotCrash() {
        XCTAssertEqual(shared().slotMask(fileAllowedDense: [true, true]), [false, false, false])
    }

    // MARK: - Expansion, and the leak it prevents

    func testASharedChunkNeverReportsTheExcludedFile() {
        // THE FAILURE THIS DESIGN HAS TO AVOID. Slot 1 is scanned because file 10 qualifies. If
        // expansion did not re-check the filter, file 20 - which the user excluded - would appear
        // in the results carrying slot 1's score.
        let hits = shared().expand(scores: [0.1, 0.9, 0.8]) { $0 == 10 }
        XCTAssertEqual(hits.map(\.file), [10])
        XCTAssertEqual(hits.first?.slot, 1)
        XCTAssertEqual(hits.first?.score ?? 0, 0.9, accuracy: 1e-6)
    }

    func testEachFileKeepsItsOwnBestSlot() {
        let hits = shared().expand(scores: [0.95, 0.10, 0.80]) { _ in true }
        XCTAssertEqual(hits.map(\.file), [10, 20])
        XCTAssertEqual(hits.first(where: { $0.file == 10 })?.slot, 0)
        XCTAssertEqual(hits.first(where: { $0.file == 20 })?.slot, 2)
    }

    func testResultsAreRankedByScore() {
        let hits = shared().expand(scores: [0.2, 0.3, 0.9]) { _ in true }
        XCTAssertEqual(hits.map(\.file), [20, 10])
    }

    func testTiesBreakDeterministically() {
        // Two files scoring identically must not reorder between runs; a flapping result list is
        // indistinguishable from a ranking bug when someone reports it.
        for _ in 0 ..< 20 {
            XCTAssertEqual(shared().expand(scores: [0.5, 0.5, 0.5]) { _ in true }.map(\.file), [10, 20])
        }
    }

    func testASlotFilterAppliesOnTopOfTheFileFilter() {
        // Kind is a property of the CONTENT, so it filters slots, not files.
        let hits = shared().expand(scores: [0.9, 0.8, 0.7], fileAllowed: { _ in true },
                                   slotAllowed: { $0 != 0 })
        XCTAssertEqual(hits.first(where: { $0.file == 10 })?.slot, 1, "a disallowed slot was used")
    }

    func testAShortScoreVectorIsIgnoredNotCrashed() {
        XCTAssertEqual(shared().expand(scores: [0.5]) { _ in true }.map(\.file), [10])
    }

    func testNoFilesQualifyGivesNoHits() {
        XCTAssertTrue(shared().expand(scores: [1, 1, 1]) { _ in false }.isEmpty)
    }

    // MARK: - Derived facts

    func testFilesOfASharedSlot() {
        XCTAssertEqual(shared().files(ofSlot: 1).sorted(), [10, 20])
        XCTAssertEqual(shared().files(ofSlot: 0), [10])
    }

    func testReferenceCountsAreDerivable() {
        // chunk.refs is a cache of this. An undercount frees a vector another file still points at.
        XCTAssertEqual(shared().referenceCounts(), [0: 1, 1: 2, 2: 1])
    }

    func testAnEmptyIndexIsWellBehaved() {
        let ix = OccurrenceIndex()
        XCTAssertEqual(ix.slotCount, 0)
        XCTAssertTrue(ix.slotMask { _ in true }.isEmpty)
        XCTAssertTrue(ix.expand(scores: []) { _ in true }.isEmpty)
        XCTAssertTrue(ix.referenceCounts().isEmpty)
    }

    // MARK: - Scale sanity

    func testAOneToOneIndexBehavesLikeV4() {
        // A v4 index loaded into this model has exactly one occurrence per slot, and must behave
        // identically to the row-per-chunk model it replaces. This is what lets the read path be
        // rewritten before any data is migrated.
        var ix = OccurrenceIndex()
        for i in 0 ..< 1000 { ix.append(file: Int32(i / 4), slot: Int32(i)) }
        let scores = (0 ..< 1000).map { Float($0) / 1000 }
        let hits = ix.expand(scores: scores) { _ in true }
        XCTAssertEqual(hits.count, 250, "one file per four chunks")
        XCTAssertEqual(hits.first?.slot, 999)
        XCTAssertEqual(ix.referenceCounts().values.allSatisfy { $0 == 1 }, true)
    }
}
