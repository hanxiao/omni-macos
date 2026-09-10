import MLX
import XCTest
@testable import OmniKit

/// The shared-cursor KV cache, and specifically the case that crashed the app: a group whose
/// pages do NOT all have the same prompt length.
///
/// A page's tile grid comes from its aspect ratio, so dropping several documents at once puts
/// prompts of different lengths in one group - 1007 tokens for A4 portrait against 1197 for a
/// squarer page. An earlier version fixed each row's output start at seed time, which a later
/// longer prompt then invalidated, and asserted the lengths were equal.
final class OCRBatchCacheTests: XCTestCase {

    private let heads = 10
    private let dim = 128

    private func block(_ tokens: Int) -> MLXArray {
        MLX.zeros([heads, tokens, dim], dtype: .float32) + 0.25
    }

    private func step(_ cache: OCRBatchKVCache, rows: Int) {
        let k = MLX.zeros([rows, heads, 1, dim], dtype: .float32) + 0.5
        cache.appendStep(k, k)
    }

    func testRaggedPromptsSeedWithoutTrapping() {
        // The exact shape from the crash report: cursor 1007, then a 1197-token prompt.
        let cache = OCRBatchKVCache(batch: 2)
        cache.seed(slot: 0, keys: block(1007), values: block(1007))
        cache.seed(slot: 1, keys: block(1197), values: block(1197))
        XCTAssertEqual(cache.cursor, 1197, "the cursor follows the longest prompt")
        XCTAssertEqual(cache.lengths, [1007, 1197], "each row keeps its own prompt length")
    }

    func testShorterRowMasksItsGapOnceDecodingStarts() {
        let cache = OCRBatchKVCache(batch: 2)
        cache.seed(slot: 0, keys: block(100), values: block(100))
        cache.seed(slot: 1, keys: block(160), values: block(160))

        // Before any token is generated neither row has a gap: row 0 is valid over [0, 100)
        // and the cursor is where its output will begin.
        XCTAssertEqual(cache.lengths, [100, 160])

        step(cache, rows: 2)
        XCTAssertEqual(cache.cursor, 161)
        XCTAssertEqual(cache.lengths, [101, 161], "both rows gained exactly one token")

        // Row 0 must not be able to attend to [100, 160): those columns hold nothing it wrote.
        guard let mask = cache.mask() else {
            return XCTFail("ragged rows need a mask")
        }
        let values = mask.asArray(Float.self)
        let t = cache.cursor
        for j in 100 ..< 160 {
            XCTAssertEqual(values[j], -Float.infinity, "row 0 column \(j) must be masked")
        }
        XCTAssertEqual(values[99], 0, "row 0's own prompt stays visible")
        XCTAssertEqual(values[160], 0, "row 0's own first token stays visible")
        for j in 0 ..< t {
            XCTAssertEqual(values[t + j], 0, "row 1 is level and needs no mask")
        }
    }

    func testLevelPromptsNeedNoMask() {
        // The common case must keep the fused attention kernel's fastest path.
        let cache = OCRBatchKVCache(batch: 3)
        for slot in 0 ..< 3 { cache.seed(slot: slot, keys: block(64), values: block(64)) }
        XCTAssertNil(cache.mask())
        step(cache, rows: 3)
        XCTAssertNil(cache.mask(), "rows that started together stay level")
    }

    func testCanAdmitRefusesAPromptPastTheCursor() {
        let cache = OCRBatchKVCache(batch: 1)
        cache.seed(slot: 0, keys: block(120), values: block(120))
        step(cache, rows: 1)
        XCTAssertTrue(cache.canAdmit(promptTokens: 120))
        XCTAssertTrue(cache.canAdmit(promptTokens: 60))
        XCTAssertFalse(cache.canAdmit(promptTokens: 500),
                       "a prompt past the cursor would overwrite itself on the next step")
    }
}
