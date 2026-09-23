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

    // MARK: - compaction

    /// Each row's step token carries its own value, so what a row can see identifies itself.
    private func stamped(_ cache: OCRBatchKVCache, rows: Int, _ n: inout Int) {
        n += 1
        let k = MLXArray((0 ..< rows).map { Float(10_000 * ($0 + 1) + n) }, [rows, 1, 1, 1])
            * MLX.ones([rows, heads, 1, dim], dtype: .float32)
        cache.appendStep(k, k)
    }

    /// The keys a row attends, in buffer order: its columns below the cursor that the mask
    /// leaves open.
    private func visible(_ cache: OCRBatchKVCache, row: Int) -> [Float] {
        guard let view = cache.view else { return [] }
        let t = cache.cursor
        let keys = view.keys[row, 0, 0 ..< t, 0].asArray(Float.self)
        let mask = cache.mask()?.asArray(Float.self)
        return (0 ..< t).compactMap { j in
            mask.map { $0[row * t + j] == 0 } ?? true ? keys[j] : nil
        }
    }

    /// Under continuous batching the shared cursor counts the steps of the whole RUN, so without
    /// compaction the buffer follows the document: a 200-page scan at width 32 reached thousands
    /// of positions, each row able to see at most one page's worth of them. Compaction must hand
    /// every row exactly the history it had.
    func testCompactionKeepsEveryRowsHistory() {
        let cache = OCRBatchKVCache(batch: 2)
        var n = 0
        cache.seed(slot: 0, keys: block(10), values: block(10))
        cache.seed(slot: 1, keys: block(10), values: block(10))
        for _ in 0 ..< 150 { stamped(cache, rows: 2, &n) }
        // Both rows are recycled: new, shorter pages, admitted at different points.
        cache.seed(slot: 1, keys: block(5) + 1, values: block(5) + 1)
        for _ in 0 ..< 40 { stamped(cache, rows: 2, &n) }
        cache.seed(slot: 0, keys: block(8) + 2, values: block(8) + 2)
        while cache.cursor < (cache.keys?.dim(2) ?? 0) { stamped(cache, rows: 2, &n) }

        let length = cache.keys!.dim(2)
        let before = [visible(cache, row: 0), visible(cache, row: 1)]
        let lengths = cache.lengths
        let cursor = cache.cursor
        stamped(cache, rows: 2, &n)                      // the write that has to make room

        XCTAssertEqual(cache.keys!.dim(2), length, "room came from packing, not from growing")
        XCTAssertLessThan(cache.cursor, cursor, "the cursor came down to the longest live row")
        XCTAssertEqual(cache.lengths, lengths.map { $0 + 1 })
        for row in 0 ..< 2 {
            XCTAssertEqual(visible(cache, row: row),
                           before[row] + [Float(10_000 * (row + 1) + n)],
                           "row \(row) sees what it saw before, plus its new token")
        }
    }

    func testRowsLongerThanTheRoomAreGrownNotPacked() {
        let cache = OCRBatchKVCache(batch: 1)
        var n = 0
        cache.seed(slot: 0, keys: block(10), values: block(10))
        while cache.cursor < (cache.keys?.dim(2) ?? 0) { stamped(cache, rows: 1, &n) }
        let length = cache.keys!.dim(2)
        let before = visible(cache, row: 0)
        stamped(cache, rows: 1, &n)
        XCTAssertGreaterThan(cache.keys!.dim(2), length, "a row that fills the buffer needs it grown")
        XCTAssertEqual(visible(cache, row: 0), before + [Float(10_000 + n)])
    }
}
