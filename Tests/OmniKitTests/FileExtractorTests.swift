import XCTest
@testable import OmniKit

final class FileExtractorTests: XCTestCase {
    private func tempTxt(_ content: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("omni-fe-\(UUID().uuidString).txt")
        try content.data(using: .utf8)!.write(to: url)
        return url
    }

    /// A UTF-8 file LARGER than maxTextBytes whose cap cut splits a multi-byte codepoint must still
    /// decode as UTF-8, not fall through to whole-file Latin-1 mojibake. "中" is 3 bytes (E4 B8 AD)
    /// and maxTextBytes (2_000_000) is not a multiple of 3, so the cut lands mid-codepoint. Before the
    /// fix, String(data:.utf8) returned nil at the boundary and the whole file decoded as Latin-1
    /// (s.first would be "ä" = 0xE4), garbling every CJK/emoji char.
    func testLargeMultibyteUTF8NotMojibake() throws {
        let n = FileExtractor.maxTextBytes / 3 + 5000   // safely over the cap, cut splits a codepoint
        let url = try tempTxt(String(repeating: "中", count: n))
        defer { try? FileManager.default.removeItem(at: url) }
        guard case .text(let s) = try FileExtractor.extract(url) else { return XCTFail("expected .text") }
        XCTAssertEqual(s.first, "中", "large UTF-8 file decoded as Latin-1 mojibake (boundary cut not trimmed)")
        XCTAssertFalse(s.contains("ä"), "Latin-1 mojibake marker present")
        XCTAssertGreaterThan(s.count, 600_000)          // ~maxTextBytes/3 codepoints survive
        XCTAssertTrue(s.allSatisfy { $0 == "中" }, "non-CJK char in a pure-CJK file means mojibake")
    }

    /// Small UTF-8 (incl. accents, CJK, emoji) is unaffected by the boundary trim.
    func testSmallUTF8Intact() throws {
        let url = try tempTxt("héllo 世界 🌍")
        defer { try? FileManager.default.removeItem(at: url) }
        guard case .text(let s) = try FileExtractor.extract(url) else { return XCTFail("expected .text") }
        XCTAssertEqual(s, "héllo 世界 🌍")
    }

    /// A genuinely non-UTF-8 (Latin-1) file still decodes via the Latin-1 fallback rather than empty.
    func testLatin1FileStillDecodes() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("omni-fe-\(UUID().uuidString).txt")
        try Data([0x63, 0x61, 0x66, 0xE9]).write(to: url)   // "caf" + 0xE9 ('é' in Latin-1, invalid UTF-8)
        defer { try? FileManager.default.removeItem(at: url) }
        guard case .text(let s) = try FileExtractor.extract(url) else { return XCTFail("expected .text") }
        XCTAssertEqual(s, "café")
    }
}
