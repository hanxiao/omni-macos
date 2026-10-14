import XCTest
@testable import OmniKit

final class WeakFoldTests: XCTestCase {
    private func hit(_ path: String, _ score: Float, _ kind: String) -> SearchHit {
        SearchHit(path: path, score: score, snippet: "", kind: kind, chunkIndex: 0, modified: 0)
    }

    func testWeakIsUnderSixTenthsOfTheBestOnItsKindsScale() {
        let hits = [hit("a", 0.70, "text"), hit("b", 0.45, "text"), hit("c", 0.40, "text"), hit("d", 0.30, "audio"), hit("e", 0.20, "image")]
        // 0.6 of 0.70 is 0.42: b stays, c folds. Audio 0.30 is 0.488 on the text scale: stays.
        // Image 0.20 is 0.325: folds.
        XCTAssertEqual(VectorStore.weakFold(hits), [false, false, true, false, true])
    }

    func testTheFirstHitIsNeverWeak() {
        // A photo that stands out on its own scale is the reference, but the top hit still shows.
        let hits = [hit("doc", 0.34, "text"), hit("photo", 0.30, "image")]
        XCTAssertEqual(VectorStore.weakFold(hits), [false, false])
        let lower = [hit("doc", 0.25, "text"), hit("photo", 0.30, "image")]
        XCTAssertEqual(VectorStore.weakFold(lower), [false, false], "rank 1 folded under a higher-scaled photo")
    }

    func testNothingToFold() {
        XCTAssertEqual(VectorStore.weakFold([]), [])
        XCTAssertEqual(VectorStore.weakFold([hit("a", 0.1, "text")]), [false])
    }
}
