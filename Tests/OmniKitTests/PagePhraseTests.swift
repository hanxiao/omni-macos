import CoreText
import XCTest
@testable import OmniKit

/// The words a passage's double-click sends to Preview (#32): Preview highlights them, so they must
/// be the passage's own words, on its page, once, and on no page before it.
final class PagePhraseTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("pagephrase-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    /// A PDF with these pages, each drawn line by line as given.
    private func pdf(_ pages: [[String]]) -> PageTexts {
        let url = dir.appendingPathComponent("\(UUID().uuidString).pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 9, nil)
        for lines in pages {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 760
            for line in lines {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: y); CTLineDraw(l, ctx); y -= 14
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
        return PageTexts(url)
    }

    private let filler = [
        "Adulterants in unregulated herbal and traditional medicaments pose",
        "a threat of toxic interstitial nephritis as exemplified by aristolochic",
        "acid contamination of herbal slimming preparations and food sources",
    ]

    /// Every word of `phrase` is one of the passage's, in order.
    private func assertFrom(_ phrase: String?, _ snippet: String, file: StaticString = #filePath, line: UInt = #line) {
        guard let phrase else { return XCTFail("no phrase", file: file, line: line) }
        let words = PagePhrase.lead(snippet, count: .max).joined(separator: " ")
        XCTAssertTrue(words.contains(phrase), "\"\(phrase)\" is not the passage's", file: file, line: line)
    }

    /// The reporter's first row: a heading on its own line over its paragraph, the passage starting at
    /// the heading. Four words across two lines were never a run on one line, and the phrase came
    /// from the top of the page.
    func testHeadingOverItsParagraph() {
        let texts = pdf([["Nothing about kidneys on the first page of this book at all"],
                         filler + ["THROMBOTIC MICROANGIOPATHY",
                                   "Thrombotic microangiopathy (TMA) is a pathologic lesion character-",
                                   "ized by endothelial cell injury in the terminal arterioles"]])
        let snippet = "THROMBOTIC MICROANGIOPATHY Thrombotic microangiopathy (TMA) is a pathologic lesion character- ized by endothelial cell injury in the terminal"
        let phrase = PagePhrase.unique(texts, page: 1, near: snippet, budget: 5)
        assertFrom(phrase, snippet)
        XCTAssertNotEqual(phrase, "adulterants in unregulated herbal and")
    }

    /// The reporter's second row: a comma among the first words broke the run they were looked for in.
    func testCommaInTheFirstWords() {
        let texts = pdf([["exchange is used concurrently with rituximab, these treatments",
                          "must be separated by at least 48 h to avoid removal of rituximab",
                          "the sinonasal tissues, it should never be used alone to treat active",
                          "granulomatosis with polyangiitis involving other organs"]])
        let snippet = "the sinonasal tissues, it should never be used alone to treat active granulomatosis with polyangiitis"
        let phrase = PagePhrase.unique(texts, page: 0, near: snippet, budget: 5)
        assertFrom(phrase, snippet)
        XCTAssertEqual(phrase, "it should never be used")
    }

    /// A window of a long chunk can start inside a word.
    func testPassageStartingInsideAWord() {
        let texts = pdf([filler + ["Knockout mutations in mice give rise to new disease models for",
                                   "investigating the pathogenetic roles of individual molecules"]])
        let snippet = "tions in mice give rise to new disease models for investigating the pathogenetic roles"
        let phrase = PagePhrase.unique(texts, page: 0, near: snippet, budget: 5)
        XCTAssertEqual(phrase, "mutations in mice give rise")
    }

    /// A passage that is a window of a chunk running on to the next page is found there.
    func testPassageOnTheNextPage() {
        let texts = pdf([filler,
                         ["Despite differences in their underlying pathogenic mechanisms the two",
                          "cerebral small-vessel diseases produce a similar range of lesions"]])
        let snippet = "cerebral small-vessel diseases produce a similar range of lesions"
        let phrase = PagePhrase.unique(texts, page: 0, near: snippet, budget: 5)
        XCTAssertEqual(phrase, "diseases produce a similar range")
    }

    /// Words that also occur earlier on the page, or on a page before it, would land the search
    /// there: the next words of the passage are used.
    func testOnceOnThePageAndNowhereBefore() {
        let texts = pdf([["the drug should never be used alone in children"],
                         ["it should never be used in pregnancy at any dose",
                          "the sinonasal tissues, it should never be used alone to treat active"]])
        let snippet = "the sinonasal tissues, it should never be used alone to treat active"
        let phrase = PagePhrase.unique(texts, page: 1, near: snippet, budget: 5)
        XCTAssertEqual(phrase, "it should never be used alone to")
    }

    /// A window running over a page break: its first words end one page and the rest open the next,
    /// so its opening is on neither. It is found by its later words, on the next page.
    func testWindowOverAPageBreak() {
        let texts = pdf([filler + ["the two models leave a substantial gap between"],
                         ["24",
                          "Figure 20: Performance comparison of Qwen2 in the basic scenario across",
                          "different agent types and three frameworks"]])
        let snippet = "substantial gap between 24 Figure 20: Performance comparison of Qwen2 in the basic scenario across"
        let phrase = PagePhrase.unique(texts, page: 0, near: snippet, budget: 5)
        XCTAssertEqual(phrase, "performance comparison of qwen2 in")
    }

    /// No passage words that qualify: the page from the top, as before.
    func testFallsBackToThePage() {
        let texts = pdf([filler])
        XCTAssertEqual(PagePhrase.unique(texts, page: 0, near: "words that are on no page", budget: 5),
                       "adulterants in unregulated herbal and")
        XCTAssertNil(PagePhrase.unique(pdf([[]]), page: 0, near: "a scan", budget: 5))
    }
}
