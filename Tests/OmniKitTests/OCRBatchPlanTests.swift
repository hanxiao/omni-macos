import XCTest
@testable import OmniKit

/// The batch width is a memory decision and a document decision, and getting either wrong makes
/// transcription SLOWER - a narrow batch gives up speculation and gets almost nothing back.
final class OCRBatchPlanTests: XCTestCase {

    private let weights = 4_530_000_000

    func testShortDocumentsDoNotBatch() {
        // Below the width that pays, batching costs speculation and returns nothing: measured
        // 158 at B=2 and 185 at B=4 against 194 for the single speculative path.
        for pages in [1, 2, 4, 7] {
            XCTAssertEqual(OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: pages,
                                                         availableBytes: 512_000_000_000), 1,
                           "a \(pages)-page document must not batch")
        }
    }

    func testSixteenGigMachineStillBatches() {
        // The whole point: a machine that cannot hold a SECOND copy of the weights can still hold
        // a dozen KV slots. Metal's working set on a 16 GB Mac is well under the installed memory.
        let width = OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 40,
                                                  availableBytes: 11_000_000_000)
        XCTAssertGreaterThanOrEqual(width, OCRBatchPlan.worthwhile)
        XCTAssertEqual(OCRWorkerPool.recommendedWorkers(modelBytes: weights,
                                                        availableBytes: 11_000_000_000), 1,
                       "the same machine must not afford a second worker process")
    }

    func testTinyMachineDoesNotBatchAtAll() {
        XCTAssertEqual(OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 40,
                                                     availableBytes: 6_000_000_000), 1)
    }

    func testWidthNeverExceedsThePageCount() {
        let width = OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 9,
                                                  availableBytes: 512_000_000_000)
        XCTAssertLessThanOrEqual(width, 9)
        XCTAssertGreaterThanOrEqual(width, OCRBatchPlan.worthwhile)
    }

    /// The planner must subtract what ANOTHER model is holding. Metal reports a device
    /// capability, not free memory, so a resident embedding model is invisible to it.
    func testCoresidentModelNarrowsTheBatch() {
        let sixteenGig = 11_000_000_000          // a 16 GB laptop's working set
        OCRBatchPlan.coresidentBytes = 0
        let alone = OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 100,
                                                  availableBytes: sixteenGig)
        OCRBatchPlan.coresidentBytes = 3_600_000_000   // embedding weights, upcast
        let shared = OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 100,
                                                   availableBytes: sixteenGig)
        OCRBatchPlan.coresidentBytes = 0
        XCTAssertGreaterThan(alone, shared,
                             "a resident embedding model must narrow the batch, not be ignored")
        XCTAssertGreaterThanOrEqual(alone, OCRBatchPlan.worthwhile)
    }

    /// A big machine is unaffected: it caps at 32 either way.
    func testCoresidentModelDoesNotNarrowABigMachine() {
        OCRBatchPlan.coresidentBytes = 3_600_000_000
        let width = OCRBatchPlan.recommendedWidth(modelBytes: weights, pageCount: 100,
                                                  availableBytes: 512_000_000_000)
        OCRBatchPlan.coresidentBytes = 0
        XCTAssertEqual(width, 32)
    }
}
