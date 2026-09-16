import XCTest
@testable import OmniKit

/// `locator` has to mean the same thing on every hit. It used to be empty whenever a file was
/// small enough to be one chunk, so a consumer could not tell "no position" from "the position is
/// the start of the file" - it just saw a missing field on short files and a present one on long
/// ones.
final class DisplayLocatorTests: XCTestCase {
    private func loc(_ stored: String, _ chunks: Int, _ kind: String, _ path: String) -> String {
        VectorStore.displayLocator(stored: stored, chunkCount: chunks, kind: kind, path: path)
    }

    /// A STORED LOCATOR IS NEVER OVERWRITTEN. It came from the chunker, which knew the real offset.
    func testStoredLocatorsWin() {
        XCTAssertEqual(loc("Line 6368", 396, "text", "/a/VectorStore.swift"), "Line 6368")
        XCTAssertEqual(loc("Page 7", 39, "text", "/a/paper.pdf"), "Page 7")
    }

    /// THE FILL. One chunk means the match starts at the beginning of the file, which is a fact,
    /// not a guess - so rows written before the chunker emitted one get it at read time rather
    /// than the whole index needing a rebuild.
    func testASingleChunkFileGetsItsPosition() {
        XCTAssertEqual(loc("", 1, "text", "/a/README.md"), "Line 1")
        XCTAssertEqual(loc("", 1, "text", "/a/one-pager.pdf"), "Page 1")
        XCTAssertEqual(loc("", 1, "scan", "/a/scanned.pdf"), "Page 1")
    }

    /// Several chunks and nothing stored is genuinely unknown - an opaque origin, where an offset
    /// maps to nothing a reader can see. Inventing "Line 1" there would be a lie.
    func testMultiChunkWithNothingStoredStaysEmpty() {
        XCTAssertEqual(loc("", 12, "text", "/a/report.docx"), "")
    }

    /// Office documents convert to text whose offsets mean nothing, single chunk or not.
    func testOpaqueFormatsStayEmpty() {
        for ext in ["docx", "rtf", "pages", "xlsx", "pptx", "key"] {
            XCTAssertEqual(loc("", 1, "text", "/a/file.\(ext)"), "", "\(ext) has no visible position")
        }
    }

    /// A photo has no line and no page.
    func testMediaHasNoPosition() {
        XCTAssertEqual(loc("", 1, "image", "/a/cat.jpg"), "")
        XCTAssertEqual(loc("", 1, "video", "/a/clip.mp4"), "")
        XCTAssertEqual(loc("", 1, "audio", "/a/note.m4a"), "")
    }
}
