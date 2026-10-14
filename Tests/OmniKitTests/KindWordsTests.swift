import XCTest
@testable import OmniKit

final class KindWordsTests: XCTestCase {
    func testTheKindAQueryNames() {
        func kind(_ q: String) -> String? { KindWords.requested(in: q).map { $0.screenshot ? "screenshot" : ($0.ext ?? $0.kind) } }
        XCTAssertEqual(kind("photo of a penguin"), "image")
        XCTAssertEqual(kind("Video of a zebra"), "video")
        XCTAssertEqual(kind("voice memo about groceries"), "audio")
        XCTAssertEqual(kind("the pdf about climbing"), "pdf")
        XCTAssertEqual(kind("screenshot of the payment error"), "screenshot", "a screenshot is asked for, not any image")
        XCTAssertEqual(kind("企鹅的照片"), "image")
        XCTAssertEqual(kind("ペンギンの動画"), "video")
        XCTAssertEqual(kind("Foto vom Strand"), "image")
        // Words that only look like one.
        XCTAssertNil(kind("photography tips"))
        XCTAssertNil(kind("clipboard history"))
        XCTAssertNil(kind("a penguin"))
    }
}
