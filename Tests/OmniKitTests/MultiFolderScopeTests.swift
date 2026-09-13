import XCTest
@testable import OmniKit

/// Issue #18, "Parent consumes child": with one indexed root you could scope a search to that root
/// or to a single folder under it, never to TWO. Adding both as roots does not help - the parent
/// subsumes the children - so two sibling project folders under one root were not expressible.
///
/// These are about `SearchFilter`, which is where the scope is actually applied: the store asks
/// `acceptsPath` once per file, and the GPU mask is keyed on the same fields.
final class MultiFolderScopeTests: XCTestCase {

    private func filter(_ folders: [String]) -> SearchFilter {
        var f = SearchFilter()
        f.folderPrefixes = folders
        return f
    }

    func testTwoSiblingsBothMatch() {
        let f = filter(["/Users/x/Docs/Alpha", "/Users/x/Docs/Beta"])
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Alpha/notes.txt"))
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Beta/notes.txt"))
        XCTAssertFalse(f.acceptsPath("/Users/x/Docs/Gamma/notes.txt"),
                       "a folder that was not scoped was accepted")
        XCTAssertFalse(f.acceptsPath("/Users/x/Elsewhere/notes.txt"))
    }

    /// The folder itself, not only what is under it - same as the single-folder behaviour.
    func testTheFolderItselfMatches() {
        let f = filter(["/Users/x/Docs/Alpha", "/Users/x/Docs/Beta"])
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Alpha"))
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Beta"))
    }

    /// A SIBLING whose name merely starts with the same characters must not match. This is why the
    /// boundary form ("<folder>/") exists rather than a bare `hasPrefix`.
    func testANamePrefixIsNotAFolderPrefix() {
        let f = filter(["/Users/x/Docs"])
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/a.txt"))
        XCTAssertFalse(f.acceptsPath("/Users/x/Docs2/a.txt"),
                       "Docs2 was accepted under a Docs scope")
        XCTAssertFalse(f.acceptsPath("/Users/x/DocsOther"))
    }

    func testEmptyMeansUnscoped() {
        let f = filter([])
        XCTAssertTrue(f.acceptsPath("/anywhere/at/all.txt"))
        XCTAssertTrue(f.isEmpty, "an unscoped filter must still take the fast path")
    }

    /// One folder has to keep behaving exactly as it did, including through the old spelling.
    func testTheSingleFolderSpellingStillWorks() {
        var f = SearchFilter()
        f.folderPrefix = "/Users/x/Docs/Alpha"
        XCTAssertEqual(f.folderPrefixes, ["/Users/x/Docs/Alpha"])
        XCTAssertEqual(f.folderPrefix, "/Users/x/Docs/Alpha")
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Alpha/deep/in/here.txt"))
        XCTAssertFalse(f.acceptsPath("/Users/x/Docs/Beta/here.txt"))
        f.folderPrefix = nil
        XCTAssertTrue(f.folderPrefixes.isEmpty)
        XCTAssertTrue(f.acceptsPath("/anything.txt"))
    }

    /// A parent AND its own child, which is the shape the issue is named for. The parent already
    /// covers the child, so the answer must simply be the parent's - not a double-count or a miss.
    func testAParentAndItsChildTogether() {
        let f = filter(["/Users/x/Docs", "/Users/x/Docs/Alpha"])
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Alpha/a.txt"))
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/Beta/b.txt"))
        XCTAssertFalse(f.acceptsPath("/Users/x/Other/c.txt"))
    }

    /// The QUERY layer has to carry both folders through as well. The parser always accumulated
    /// every qualifier - it was the consumer that took the last `in:` and dropped the rest, which
    /// is where issue #18 actually lived.
    func testRepeatedInQualifiersBothSurviveTheParser() {
        let parsed = SearchQueryParser.parse("in:/Users/x/Alpha in:/Users/x/Beta porsche")
        let folders = parsed.qualifiers.filter { $0.key == "in" }.map(\.value)
        XCTAssertEqual(folders, ["/Users/x/Alpha", "/Users/x/Beta"],
                       "a repeated in: qualifier was collapsed at the parser")
        XCTAssertEqual(parsed.semanticText, "porsche",
                       "the folders leaked into the text that gets embedded")
    }

    /// Quoted paths with spaces, since that is what the box writes back for a real folder.
    func testQuotedFoldersWithSpaces() {
        let parsed = SearchQueryParser.parse("in:\"/Users/x/My Docs\" in:\"/Users/x/Other Docs\" report")
        XCTAssertEqual(parsed.qualifiers.filter { $0.key == "in" }.map(\.value),
                       ["/Users/x/My Docs", "/Users/x/Other Docs"])
        XCTAssertEqual(parsed.semanticText, "report")
    }

    /// macOS stores filenames NFD, so a combining mark right after the separator is ordinary. The
    /// byte-wise boundary exists because `hasPrefix` clusters that mark onto the "/" and answers
    /// NOT-under, while SQLite's byte range says it is.
    func testACombiningMarkAfterTheSeparator() {
        let f = filter(["/Users/x/Docs"])
        XCTAssertTrue(f.acceptsPath("/Users/x/Docs/\u{0301}odd.txt"),
                      "a path whose first character after the separator is a combining mark was "
                      + "rejected - the grapheme-cluster trap the byte-wise compare exists to avoid")
    }
}
