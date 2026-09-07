import XCTest
@testable import OmniKit

/// CJK behaviour of the filename channel.
///
/// Han, kana and hangul are written without spaces, which breaks two ASCII-shaped assumptions:
///   1. The fusion gate accepts a single token at 4+ CHARACTERS. "理财" (wealth management),
///      "税务局" (tax bureau) and "肖涵" (a name) are complete, specific words that the raw count
///      rejected, while "budget" passed at six letters.
///   2. FTS5's unicode61 tokenizer keeps a whole run as ONE token and the channel queries by
///      prefix, so "记录" - a real word in the middle of "会议记录" - could never match.
///
/// Measured on the live 2,666,141-file index (1,501 CJK basenames): inner-word recall@10 went from
/// 0.0% to 67.0%, whole-basename recall from 98.5% to 98.7%, and long prose queries stayed at 0
/// lexical hits, which is what the run-length cap protects.
final class LexicalCJKTests: XCTestCase {

    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("lexcjk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    private func makeIndex() -> LexicalIndex {
        LexicalIndex(indexURL: dir.appendingPathComponent("index.sqlite"))
    }

    // MARK: The gate

    func testShortCJKWordsAreTreatedAsNameShaped() {
        for q in ["理财", "税务局", "肖涵", "报表", "记录", "会議", "무역"] {
            XCTAssertTrue(LexicalIndexProbe.shouldFuse(q), "\(q) is a word and should reach the channel")
        }
    }

    func testASCIIGateIsUnchanged() {
        // The threshold and every prose rule must behave exactly as before.
        XCTAssertTrue(LexicalIndexProbe.shouldFuse("budget"))
        XCTAssertTrue(LexicalIndexProbe.shouldFuse("notes from the meeting.md"))
        XCTAssertFalse(LexicalIndexProbe.shouldFuse("cat"))          // 3 letters, below the bar
        XCTAssertFalse(LexicalIndexProbe.shouldFuse("ab"))
        XCTAssertFalse(LexicalIndexProbe.shouldFuse("what is the budget for this quarter"))
        XCTAssertFalse(LexicalIndexProbe.shouldFuse("photos of my dog"))
    }

    func testSingleCJKCharacterIsNotEnough() {
        // One character is the CJK equivalent of a two-letter token: too ambiguous to lead.
        XCTAssertFalse(LexicalIndexProbe.shouldFuse("表"))
    }

    // MARK: Bigram expansion

    func testBigramsOnlyApplyToShortCJKRuns() {
        XCTAssertEqual(LexicalIndexProbe.cjkBigrams("会议记录"), ["会议", "议记", "记录"])
        XCTAssertEqual(LexicalIndexProbe.cjkBigrams("报表"), ["报表"])
        // ASCII never expands, so no ASCII name gains a term and no ASCII ranking moves.
        XCTAssertEqual(LexicalIndexProbe.cjkBigrams("budget"), [])
        XCTAssertEqual(LexicalIndexProbe.cjkBigrams("OmniEngine"), [])
        // A long run is a sentence, not a name: expanding it would let prose match filenames.
        XCTAssertEqual(LexicalIndexProbe.cjkBigrams("上周的会议记录在哪里"), [])
    }

    func testInteriorCJKWordFindsTheFile() throws {
        let lex = makeIndex()
        let paths = ["/Users/me/Documents/会议记录.md",
                     "/Users/me/Documents/理财报表.pdf",
                     "/Users/me/Documents/budget-report.pdf"]
        lex.rebuildIfStale(paths: paths, stamp: 1)

        // The failure this fixes: a word in the middle of the name.
        XCTAssertEqual(lex.match("记录", limit: 10), [paths[0]])
        XCTAssertEqual(lex.match("报表", limit: 10), [paths[1]])
        // What already worked must still work.
        XCTAssertEqual(lex.match("会议记录", limit: 10), [paths[0]])
        XCTAssertEqual(lex.match("budget", limit: 10), [paths[2]])
        XCTAssertEqual(lex.match("report", limit: 10), [paths[2]])
    }

    func testProseDoesNotReachFilenamesThroughBigrams() throws {
        let lex = makeIndex()
        let paths = ["/Users/me/Documents/会议记录.md", "/Users/me/Documents/理财报表.pdf"]
        lex.rebuildIfStale(paths: paths, stamp: 1)
        // A whole sentence is one long token: no bigrams, so it matches by prefix or not at all.
        XCTAssertEqual(lex.match("上周的会议记录在哪里", limit: 10), [])
        XCTAssertEqual(lex.match("帮我找一下去年的报税材料", limit: 10), [])
    }

    // MARK: Fusion strength

    func testTermMatchCreditsCJKContainmentAndNothingElse() {
        // The containment case, which is what stops a retrieved CJK hit from scoring zero.
        XCTAssertTrue(LexicalIndexProbe.termMatches(query: "记录", basename: "会议记录"))
        XCTAssertTrue(LexicalIndexProbe.termMatches(query: "会议", basename: "会议记录"))
        XCTAssertTrue(LexicalIndexProbe.termMatches(query: "报表", basename: "报表"))
        // Not the other way round: a longer query is not accounted for by a shorter name.
        XCTAssertFalse(LexicalIndexProbe.termMatches(query: "会议记录", basename: "记录"))
        // ASCII keeps strict equality, so no existing score can shift.
        XCTAssertFalse(LexicalIndexProbe.termMatches(query: "budget", basename: "budgeting"))
        XCTAssertFalse(LexicalIndexProbe.termMatches(query: "port", basename: "report"))
        XCTAssertTrue(LexicalIndexProbe.termMatches(query: "report", basename: "report"))
        // A single character is below the bar for containment, or "表" would match everything.
        XCTAssertFalse(LexicalIndexProbe.termMatches(query: "表", basename: "理财报表"))
    }
}
