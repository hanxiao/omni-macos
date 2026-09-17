import XCTest
@testable import OmniKit

/// The cutter decides chunk identity, so these are correctness tests, not quality tests. A cut that
/// loses a byte loses it from the index silently, and a cut that is not reproducible re-embeds the
/// whole corpus on the next launch.
final class ContentChunkerTests: XCTestCase {

    private func prose(_ n: Int, seed: UInt64 = 7) -> String {
        var s = seed
        let words = ["index", "chunk", "vector", "search", "embedding", "folder", "pointer",
                     "locator", "occurrence", "content", "boundary", "document", "passage"]
        var out = ""
        while out.utf8.count < n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            out += words[Int((s >> 33) % UInt64(words.count))] + " "
            if (s >> 60) % 8 == 0 { out += "\n" }
        }
        return out
    }


    /// Text with FEW newlines, the shape of a .jsonl log: one long record per line. This is the
    /// corpus that separates a content-defined cut from a grid, because `snapToLine` finds nothing
    /// to snap to and the rolling hash has to carry the boundary on its own.
    private func longLines(_ n: Int, seed: UInt64 = 7) -> String {
        var s = seed
        var out = ""
        while out.utf8.count < n {
            var line = ""
            while line.utf8.count < 6_000 {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                line += "\"field\":\"" + String(s >> 20, radix: 36) + "\","
            }
            out += line + "\n"
        }
        return out
    }

    // MARK: - Correctness

    func testCuttingLosesNothing() {
        // The invariant everything else depends on: the pieces reassemble into the original, byte
        // for byte. A cutter that drops a byte drops it from the index and nothing ever says so.
        for n in [500, 2_000, 10_000, 60_000] {
            let text = prose(n)
            let joined = ContentChunker.cut(text).map(\.text).joined()
            XCTAssertEqual(joined, text, "round trip failed at \(n) bytes")
        }
    }

    func testByteOffsetsAreTrue() {
        let text = prose(40_000)
        let bytes = Array(text.utf8)
        for piece in ContentChunker.cut(text) {
            let end = piece.byteOffset + piece.text.utf8.count
            XCTAssertLessThanOrEqual(end, bytes.count)
            XCTAssertEqual(Array(bytes[piece.byteOffset ..< end]), Array(piece.text.utf8),
                           "piece at \(piece.byteOffset) does not match the source at that offset")
        }
    }

    func testTheCutIsDeterministic() {
        // Not a tautology: the gear table is DERIVED. If it were seeded randomly this fails, and
        // every launch would re-cut and re-embed the whole corpus.
        let text = prose(50_000)
        XCTAssertEqual(ContentChunker.cut(text).map(\.byteOffset),
                       ContentChunker.cut(text).map(\.byteOffset))
    }

    func testTheGearTableIsFixed() {
        // Pin two entries. A change here silently invalidates every chunk key in every index.
        XCTAssertEqual(ContentChunker.gear.count, 256)
        XCTAssertEqual(ContentChunker.gear[0], 0xE220A8397B1DCDAF)
        XCTAssertEqual(ContentChunker.gear[255], 0x5A5832BB47BCF19E)
        XCTAssertEqual(Set(ContentChunker.gear).count, 256, "gear values must be distinct")
    }

    func testNoEmptyPieces() {
        for n in [0, 1, 899, 901, 5_000, 50_000] {
            for piece in ContentChunker.cut(prose(n)) {
                XCTAssertFalse(piece.text.isEmpty)
            }
        }
    }

    func testEmptyTextGivesNoPieces() {
        XCTAssertTrue(ContentChunker.cut("").isEmpty)
    }

    func testShortTextIsOnePiece() {
        let short = "a short note about invoices"
        XCTAssertEqual(ContentChunker.cut(short).map(\.text), [short])
    }

    // MARK: - UTF-8 safety

    func testCJKIsNeverSplitMidCharacter() {
        // Every Chinese character is 3 bytes. A cut landing inside one produces invalid UTF-8,
        // which String(decoding:) silently turns into replacement characters - so the test is
        // that no replacement character appears anywhere.
        let text = String(repeating: "中文文档检索系统测试内容。", count: 4000)
        let pieces = ContentChunker.cut(text)
        XCTAssertGreaterThan(pieces.count, 1)
        XCTAssertEqual(pieces.map(\.text).joined(), text)
        for piece in pieces {
            XCTAssertFalse(piece.text.contains("\u{FFFD}"), "a cut landed inside a character")
        }
    }

    func testEmojiAndCombiningMarksSurvive() {
        let text = String(repeating: "family 👨‍👩‍👧‍👦 and cafe\u{301} and 🇯🇵 flag. ", count: 1200)
        let pieces = ContentChunker.cut(text)
        XCTAssertEqual(pieces.map(\.text).joined(), text)
        for piece in pieces { XCTAssertFalse(piece.text.contains("\u{FFFD}")) }
    }

    // MARK: - Size bounds

    func testEveryChunkButTheLastRespectsTheBounds() {
        let text = prose(200_000)
        let pieces = ContentChunker.cut(text)
        XCTAssertGreaterThan(pieces.count, 20)
        for piece in pieces.dropLast() {
            let n = piece.text.utf8.count
            XCTAssertGreaterThanOrEqual(n, ContentChunker.minBytes, "chunk below the floor")
            // maxBytes plus the line-snap window is the true ceiling: snapping may carry a cut
            // past the hard stop to reach a line break.
            XCTAssertLessThanOrEqual(n, ContentChunker.maxBytes + ContentChunker.lineSnapWindow)
        }
    }

    func testAverageSizeIsNearTheTarget() {
        let text = prose(400_000)
        let pieces = ContentChunker.cut(text)
        let mean = Double(text.utf8.count) / Double(pieces.count)
        XCTAssertGreaterThan(mean, 1_000, "mean \(mean) far under target")
        XCTAssertLessThan(mean, 3_200, "mean \(mean) far over target")
    }

    func testAFileWithNoNewlinesStillCuts() {
        // snapToLine finds nothing; the content cut has to stand on its own.
        let text = String(repeating: "abcdefghij", count: 6_000)
        let pieces = ContentChunker.cut(text)
        XCTAssertGreaterThan(pieces.count, 1)
        XCTAssertEqual(pieces.map(\.text).joined(), text)
    }

    // MARK: - The property this exists for

    func testAnInsertionNearTheTopMovesAlmostNothing() {
        // ON LONG-LINE TEXT, which is the case that separates this cutter from a grid. On
        // newline-dense prose a grid plus a line snap is already fairly stable, so a test written
        // on prose alone passes whether or not the content hash works - it did, and it was
        // measuring nothing. .jsonl logs are 40% of the real index and look like this.
        for text in [longLines(200_000), prose(120_000)] {
            let before = Set(ContentChunker.cut(text).map(\.text))
            let at = text.utf8.count / 10
            let idx = text.utf8.index(text.utf8.startIndex, offsetBy: at)
            let edited = String(decoding: Array(text.utf8[..<idx]) + Array("INSERTED,".utf8)
                                         + Array(text.utf8[idx...]), as: UTF8.self)
            let after = ContentChunker.cut(edited)
            let fresh = after.filter { !before.contains($0.text) }
            XCTAssertLessThanOrEqual(fresh.count, 4,
                "\(fresh.count) of \(after.count) chunks changed; boundaries are not content-defined")
        }
    }

    func testAnAppendMovesOnlyTheTail() {
        let text = prose(120_000)
        let before = Set(ContentChunker.cut(text).map(\.text))
        let after = ContentChunker.cut(text + "one appended line\n")
        XCTAssertLessThanOrEqual(after.filter { !before.contains($0.text) }.count, 2)
    }

    func testAPrefixIsSharedWithItsOwnExtension() {
        // The reason cross-file dedup works at all: a file that starts the same way as another
        // produces the same leading chunks.
        let a = prose(80_000)
        let b = a + prose(20_000, seed: 99)
        let ka = ContentChunker.cut(a).map(\.text)
        let kb = ContentChunker.cut(b).map(\.text)
        let shared = zip(ka, kb).prefix { $0 == $1 }.count
        XCTAssertGreaterThan(shared, ka.count - 3, "only \(shared) of \(ka.count) leading chunks shared")
    }

    // MARK: - Sizes are in CHARACTERS

    func testCJKChunksHoldTheSameNUMBEROFCHARACTERSAsEnglishOnes() {
        // THE CORRECTION, and it fails outright if the size gates count bytes. Chinese is 3 bytes
        // a character, so a 1800-BYTE target holds 600 characters - a third of the context per
        // chunk and three times the vectors, on exactly the corpora least able to spare either.
        // The hash still sees bytes; only the gates count scalars.
        // VARIED text, not `String(repeating:)`. A periodic string never satisfies the hash mask,
        // so every chunk runs to the hard ceiling and the test would only ever be comparing two
        // ceilings - which passes for the wrong reason.
        func varied(_ words: [String], _ chars: Int, seed: UInt64) -> String {
            var s = seed
            var out = ""
            var n = 0                  // tracked, not re-counted: `out.count` in the condition is
            while n < chars {          // O(length) per turn and made this test take 33 seconds
                s = s &* 6364136223846793005 &+ 1442695040888963407
                let w = words[Int((s >> 33) % UInt64(words.count))]
                out += w; n += w.count
                if (s >> 60) % 9 == 0 { out += "\n"; n += 1 }
            }
            return out
        }
        let cjk = varied(["中文", "文档", "检索", "系统", "测试", "内容", "向量", "索引", "。", "，"],
                         120_000, seed: 3)
        let english = varied(["index ", "chunk ", "vector ", "search ", "embedding ", "folder ",
                              "content ", "document ", "passage ", ". "], 120_000, seed: 3)
        func meanChars(_ text: String) -> Double {
            let pieces = ContentChunker.cut(text)
            XCTAssertGreaterThan(pieces.count, 10, "too few pieces to average")
            return Double(text.count) / Double(pieces.count)
        }
        let c = meanChars(cjk), e = meanChars(english)
        XCTAssertGreaterThan(c, 1_000, "CJK chunks hold \(c) characters; the gates are counting bytes")
        XCTAssertLessThan(abs(c - e) / e, 0.5,
                          "CJK mean \(c) against English \(e): the cutter is not script-neutral")
    }

    func testAScalarIsNeverCountedTwice() {
        // The scalar counter drives every gate, so an off-by-one in it silently resizes every
        // chunk. Counted a second way - Swift's own scalar view - over a mixed-script text.
        let text = String(repeating: "mixed 中文 and emoji 🇯🇵 and cafe\u{301} text here. ", count: 900)
        let pieces = ContentChunker.cut(text)
        XCTAssertEqual(pieces.map(\.text).joined(), text)
        XCTAssertEqual(pieces.reduce(0) { $0 + $1.text.unicodeScalars.count }, text.unicodeScalars.count)
    }

    // MARK: - The user's setting

    func testTheSizeSettingIsHonoured() {
        // "Max characters per chunk" is a four-value picker in Settings, so a cutter that ignored
        // it would make the control silently do nothing - which is what a fixed-parameter cutter
        // would have done.
        let text = prose(400_000)
        var means: [Double] = []
        for setting in [1200, 1800, 3600] {
            let p = ContentChunker.Params.forMaxChars(setting)
            let pieces = ContentChunker.cut(text, p)
            means.append(Double(text.count) / Double(pieces.count))
        }
        XCTAssertLessThan(means[0], means[1], "1200 did not give smaller chunks than 1800")
        XCTAssertLessThan(means[1], means[2], "1800 did not give smaller chunks than 3600")
        XCTAssertGreaterThan(means[2] / means[0], 1.8, "tripling the setting barely moved the size")
    }

    func testTheDefaultParamsAreTheMeasuredOnes() {
        // 900 / 1800 / 4000 is what the parameter study in docs/schema-v5.md measured; deriving
        // them from the default setting must not quietly change them.
        let d = ContentChunker.Params.default
        XCTAssertEqual([d.minChars, d.targetChars, d.maxChars], [900, 1800, 4000])
    }

    // MARK: - Fingerprint

    func testTheFingerprintNamesEveryParameter() {
        let fp = ContentChunker.fingerprint
        for n in [ContentChunker.minBytes, ContentChunker.targetBytes, ContentChunker.maxBytes] {
            XCTAssertTrue(fp.contains(String(n)), "\(fp) does not name \(n)")
        }
        XCTAssertTrue(fp.hasPrefix("cdc"))
    }

    func testADifferentSettingIsADifferentKeySpace() {
        // Chunks cut to different sizes are different chunks, so they must not be able to collide
        // on a key. The grid's key already carries its `c<maxChars>`; this is the same promise.
        let a = ContentChunker.Params.forMaxChars(1800).fingerprint
        let b = ContentChunker.Params.forMaxChars(3600).fingerprint
        XCTAssertNotEqual(a, b)
    }

}
