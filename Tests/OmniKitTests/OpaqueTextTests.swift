import XCTest
@testable import OmniKit

/// The guard that keeps base64 payload out of the index. Each case here is a real string shape
/// from the 2.6M-file index that prompted it, so a future threshold change has to answer for
/// what it breaks.
final class OpaqueTextTests: XCTestCase {

    // MARK: - What must be dropped

    func testABase64urlBlobIsPayload() {
        let blob = "KBktK7MXhJ-aEguUZIdL_wj9BZ3g7dg1JCNBDqMEsNa9JhGFpBAgkmW8M_CZRddrW0Db1z69f5R-H58jzgr76cfZSphj1BZOQ"
        XCTAssertTrue(OpaqueText.isPayload(blob))
    }

    func testAThinkingSignatureLineIsPayload() {
        // The observed shape: a short key, then the blob that dwarfs it.
        let line = "thinkingSignature\t" + String(repeating: "fPTsDCRnRTStlfS5BM0CMT4tzZote5XImRL4zC34", count: 4)
        XCTAssertTrue(OpaqueText.isPayload(line))
    }

    func testPemBodyIsPayloadDespiteLineBreaks() {
        // 64-character lines still clear the run floor, so wrapping does not rescue a key blob.
        let body = (0 ..< 8).map { _ in String(repeating: "MIIEowIBAAKCAQEAx7Vn9Q2pLmKfR4sTgWbYcHdJeNoZuPvXkAqBnCsDtEuFgHiJ", count: 1) }
            .joined(separator: "\n")
        XCTAssertTrue(OpaqueText.isPayload(body))
    }

    // MARK: - What must survive

    func testProseIsNotPayload() {
        let prose = "The invoice covers February 2026 and is due on the last day of the month."
        XCTAssertFalse(OpaqueText.isPayload(prose))
    }

    func testALongPathIsNotPayload() {
        // '/' is not a run character precisely so that this stays indexable: counted as one, the
        // whole path would be a single 68-character run.
        let path = "/Volumes/han2tb/jina-dataroom-harness/runs/c559b0ff288b/events.jsonl"
        XCTAssertFalse(OpaqueText.isPayload(path))
        XCTAssertEqual(OpaqueText.payloadFraction(path), 0, accuracy: 0.001)
    }

    func testUUIDsAreNotPayload() {
        let line = "session 85430390-b975-4eae-b082-34e65749c949 opened at 12:04"
        XCTAssertFalse(OpaqueText.isPayload(line))
    }

    func testSourceCodeIsNotPayload() {
        let code = "let hits = backend.search(query, topK: min(topK * 3, 150), filter: filter)"
        XCTAssertFalse(OpaqueText.isPayload(code))
    }

    func testCJKProseIsNotPayload() {
        // Multi-byte scalars must break runs, not extend them.
        XCTAssertFalse(OpaqueText.isPayload("博通第三季度营收创下新高，数据中心业务增长显著。"))
    }

    // MARK: - Threshold behaviour

    func testARunJustUnderTheFloorCountsForNothing() {
        let justUnder = String(repeating: "a", count: OpaqueText.minRun - 1)
        XCTAssertEqual(OpaqueText.payloadFraction(justUnder), 0, accuracy: 0.001)
    }

    func testARunAtTheFloorCounts() {
        let atFloor = String(repeating: "a", count: OpaqueText.minRun)
        XCTAssertEqual(OpaqueText.payloadFraction(atFloor), 1, accuracy: 0.001)
    }

    func testProseCarryingOneBlobSurvivesWhenProseDominates() {
        let blob = String(repeating: "Z", count: 60)
        let prose = String(repeating: "the quarterly revenue report is attached. ", count: 6)
        XCTAssertFalse(OpaqueText.isPayload(prose + blob))
    }

    // MARK: - filter

    func testFilterDropsPayloadChunksAndKeepsProse() {
        let pieces = ["a real passage about invoices and payment terms",
                      String(repeating: "Q", count: 80),
                      "another real passage about deployment"]
        let kept = OpaqueText.filter(pieces) { $0 }
        XCTAssertEqual(kept.count, 2)
        XCTAssertFalse(kept.contains { $0.hasPrefix("QQQ") })
    }

    func testAnAllPayloadFileKeepsOneChunk() {
        // The file still needs a row for the filename channel to reach it.
        let pieces = [String(repeating: "Q", count: 80), String(repeating: "R", count: 80)]
        XCTAssertEqual(OpaqueText.filter(pieces) { $0 }.count, 1)
    }

    func testASingleChunkFileIsNeverEmptied() {
        let pieces = [String(repeating: "Q", count: 80)]
        XCTAssertEqual(OpaqueText.filter(pieces) { $0 }.count, 1)
    }
}
