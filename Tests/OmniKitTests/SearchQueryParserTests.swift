import XCTest
@testable import OmniKit

/// Correctness gate for the search query-language parser. The colon-adjacency + whitelist rules are
/// the make-or-break behavior (a parser that turns "12:30" or "type: theory" into a filter is worse
/// than no parser), so they are tested explicitly.
final class SearchQueryParserTests: XCTestCase {

    private func q(_ s: String) -> ParsedQuery { SearchQueryParser.parse(s) }

    func testPlainTextHasNoQualifiers() {
        let p = q("quarterly revenue report")
        XCTAssertEqual(p.semanticText, "quarterly revenue report")
        XCTAssertTrue(p.qualifiers.isEmpty)
    }

    func testBasicQualifierAndSemanticRemainder() {
        let p = q("sunset photos type:image after:30d")
        XCTAssertEqual(p.semanticText, "sunset photos")
        XCTAssertEqual(p.qualifiers, [
            .init(key: "type", value: "image", negated: false),
            .init(key: "after", value: "30d", negated: false),
        ])
    }

    func testColonWithSpaceStaysProse() {
        // The single most important rule: a space after the colon means it is NOT a qualifier.
        let p = q("notes about type: theory")
        XCTAssertEqual(p.semanticText, "notes about type: theory")
        XCTAssertTrue(p.qualifiers.isEmpty)
    }

    func testNonKeyColonsStayProse() {
        let p = q("meeting at 12:30 ratio 3:1 see http://example.com")
        XCTAssertTrue(p.qualifiers.isEmpty)
        XCTAssertEqual(p.semanticText, "meeting at 12:30 ratio 3:1 see http://example.com")
    }

    func testUnknownKeyIsSemantic() {
        let p = q("color:red running shoes")
        XCTAssertTrue(p.qualifiers.isEmpty)
        XCTAssertEqual(p.semanticText, "color:red running shoes")
    }

    func testQuotedValueWithSpaces() {
        let p = q("invoice in:\"~/Documents/Project X\" total")
        XCTAssertEqual(p.semanticText, "invoice total")
        XCTAssertEqual(p.qualifiers, [.init(key: "in", value: "~/Documents/Project X", negated: false)])
    }

    func testTypeMultiValue() {
        let p = q("type:image,video beach")
        XCTAssertEqual(p.semanticText, "beach")
        XCTAssertEqual(p.qualifiers, [.init(key: "type", value: "image,video", negated: false)])
    }

    func testTypeNegation() {
        let p = q("meeting notes -type:audio")
        XCTAssertEqual(p.semanticText, "meeting notes")
        XCTAssertEqual(p.qualifiers, [.init(key: "type", value: "audio", negated: true)])
    }

    func testNegatedBareWordIsSemantic() {
        // Omni has no full-text exclusion, so a leading '-' on a non-qualifier stays literal text.
        let p = q("-france report")
        XCTAssertTrue(p.qualifiers.isEmpty)
        XCTAssertEqual(p.semanticText, "-france report")
    }

    func testAliasesCanonicalize() {
        let p = q("kind:image folder:/tmp min:50% sort:date")
        XCTAssertEqual(p.qualifiers, [
            .init(key: "type", value: "image", negated: false),
            .init(key: "in", value: "/tmp", negated: false),
            .init(key: "score", value: "50%", negated: false),
            .init(key: "sort", value: "date", negated: false),
        ])
        XCTAssertEqual(p.semanticText, "")
    }

    func testTrailingIncompleteKeyIsSemantic() {
        // "type:" with no value yet (mid-typing) is not a qualifier; it stays in the text.
        let p = q("budget type:")
        XCTAssertTrue(p.qualifiers.isEmpty)
        XCTAssertEqual(p.semanticText, "budget type:")
    }

    func testKeyCaseInsensitiveValuePreserved() {
        let p = q("TYPE:Image")
        XCTAssertEqual(p.qualifiers, [.init(key: "type", value: "Image", negated: false)])
    }

    func testEmptyAndWhitespace() {
        XCTAssertEqual(q("").semanticText, "")
        XCTAssertTrue(q("").qualifiers.isEmpty)
        XCTAssertEqual(q("    ").semanticText, "")
    }

    func testQualifierBetweenWords() {
        let p = q("red type:image car")
        XCTAssertEqual(p.semanticText, "red car")
        XCTAssertEqual(p.qualifiers, [.init(key: "type", value: "image", negated: false)])
    }
}
