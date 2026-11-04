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

    /// When the box turns a qualifier into a chip: once the edit has left it, wherever it is, and
    /// never while it is the word being typed.
    func testFinishedQualifier() {
        let f = { SearchQueryParser.hasFinishedQualifier($0) }
        XCTAssertTrue(f("type:image "))
        XCTAssertTrue(f("type:image photos"))          // pasted, or typed faster than each keystroke reports
        XCTAssertTrue(f("photos type:image "))
        XCTAssertTrue(f("in:\"my folder\" notes"))
        XCTAssertFalse(f("type:image"))                 // still being typed
        XCTAssertFalse(f("type:i"))
        XCTAssertFalse(f("notes type:i"))
        XCTAssertFalse(f("in:\"my "))                  // a quoted value still open
        XCTAssertFalse(f("notes about type: theory"))  // prose, finished or not
        XCTAssertFalse(f("meeting at 12:30 "))
        XCTAssertFalse(f(""))
    }

    /// Typed inside the box, between words: the word at the caret is still being typed, even with
    /// a space after it, and is finished once the caret leaves it.
    func testFinishedQualifierInsideTheBox() {
        let f = SearchQueryParser.hasFinishedQualifier
        XCTAssertFalse(f("beach type:i photos", "beach type: photos"))
        XCTAssertFalse(f("beach type:image photos", "beach type:imag photos"))
        XCTAssertFalse(f("beach type:imge photos", "beach type:imgge photos"))   // a deletion inside it
        XCTAssertTrue(f("beach type:image  photos", "beach type:image photos"))  // its space typed
        XCTAssertTrue(f("type:image photos", "type:image photo"))                // typing the next word
        XCTAssertFalse(f("in:\"my folder\"", "in:\"my folder"))                // the quote just closed
        XCTAssertTrue(f("type:image photos", "abc"))                              // pasted over the box
        XCTAssertFalse(f("beach photos", "beach photo"))
    }

    /// `in:` by the name the sidebar shows: a root, a path inside it, a localized name; a path or a
    /// name no root has resolves to nothing here (the caller reads it as a path).
    func testFolderByName() {
        let roots: [(path: String, name: String)] = [
            ("/Users/u/Documents", "Dokumente"),
            ("/Volumes/x/corpus2", "corpus2"),
            ("/Users/u/Library/Mobile Documents/com~apple~CloudDocs", "iCloud Drive"),
        ]
        let f = { SearchQueryParser.folderPaths(named: $0, roots: roots) }
        XCTAssertEqual(f("corpus2"), ["/Volumes/x/corpus2"])
        XCTAssertEqual(f("CORPUS2/"), ["/Volumes/x/corpus2"])
        XCTAssertEqual(f("corpus2/Taxes/2024"), ["/Volumes/x/corpus2/Taxes/2024"])
        XCTAssertEqual(f("Documents/Taxes"), ["/Users/u/Documents/Taxes"])   // the folder's own name
        XCTAssertEqual(f("Dokumente"), ["/Users/u/Documents"])               // the name shown
        XCTAssertEqual(f("iCloud Drive"), ["/Users/u/Library/Mobile Documents/com~apple~CloudDocs"])
        XCTAssertEqual(f("Taxes"), [])
        XCTAssertEqual(f("/Volumes/x/corpus2"), [])
        XCTAssertEqual(f("~/Documents"), [])
        XCTAssertEqual(f(""), [])
        // Two roots with one name: both, as `in:A in:B` would be.
        let twins: [(path: String, name: String)] = [("/a/Notes", "Notes"), ("/b/Notes", "Notes")]
        XCTAssertEqual(SearchQueryParser.folderPaths(named: "notes", roots: twins), ["/a/Notes", "/b/Notes"])
    }

    /// Keystroke by keystroke, as the field reports them: a filter typed between two words is not
    /// finished until its own space, and a word typed after it is.
    func testFinishedQualifierKeystrokes() {
        func typeInto(_ box: String, at caret: Int, _ keys: String) -> [Bool] {
            var text = Array(box), at = caret, out: [Bool] = []
            for k in keys {
                let before = String(text)
                text.insert(k, at: at); at += 1
                out.append(SearchQueryParser.hasFinishedQualifier(String(text), previous: before))
                // The field reports some keystrokes twice: the second is no edit.
                XCTAssertFalse(SearchQueryParser.hasFinishedQualifier(String(text), previous: String(text)))
            }
            return out
        }
        XCTAssertEqual(typeInto("beach photos", at: 5, " type:image"), Array(repeating: false, count: 11))
        XCTAssertEqual(typeInto("beach photos", at: 5, " type:image ").last, true)
        XCTAssertEqual(typeInto("", at: 0, "type:image photos"),
                       Array(repeating: false, count: 10) + Array(repeating: true, count: 7))
    }
}
