import XCTest
@testable import OmniKit

/// `score:` in the query box. Three spellings of one number, and they have to agree.
final class ScoreQualifierTests: XCTestCase {
    /// XCTAssertEqual has no overload taking an Optional AND an accuracy, and comparing parsed
    /// floats exactly is the kind of test that passes until a division is reassociated.
    private func expect(_ raw: String, _ want: Double, file: StaticString = #filePath, line: UInt = #line) {
        guard let got = ScoreQualifier.parse(raw) else {
            return XCTFail("score:\(raw) did not parse", file: file, line: line)
        }
        XCTAssertEqual(got, want, accuracy: 1e-9, "score:\(raw)", file: file, line: line)
    }

    /// THE ALIASES. All three mean a 70% floor.
    func testEverySpellingOfSeventyPercentAgrees() {
        expect("70%", 0.7)
        expect("0.7", 0.7)
        expect("70", 0.7)
    }

    /// THE BUG. A bare integer clamped to 1.0, so `score:70` asked for a 100% floor and returned
    /// nothing - the opposite of what was typed, with no error. Anything above 1 can only have
    /// been meant as a percentage; 1.0 is the most a cosine reaches.
    func testABareIntegerIsAPercentageNotAClampedFraction() {
        expect("50", 0.5)
        expect("100", 1.0)
        expect("1", 1.0)     // 1 is already a fraction, and the maximum either way
    }

    func testFractionsAndSpacingSurvive() {
        expect("0.55", 0.55)
        expect(" 55 % ", 0.55)
        expect("0", 0)
    }

    /// Out of range is clamped rather than refused - a floor above 1 is just "everything".
    func testOutOfRangeClamps() {
        expect("250", 1.0)
        expect("-5", 0)
    }

    /// Not a number is not a filter: the qualifier is ignored rather than silently meaning zero.
    func testNonNumbersAreRejected() {
        XCTAssertNil(ScoreQualifier.parse("high"))
        XCTAssertNil(ScoreQualifier.parse("%"))
        XCTAssertNil(ScoreQualifier.parse(""))
        XCTAssertNil(ScoreQualifier.parse("nan"))
    }
}
