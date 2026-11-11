import XCTest
import CoreGraphics
import CoreText
import AppKit
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// A screenshot's text is read at 1.5 pixels a point, in square pieces. These pin the size it is
/// read at, the pieces, and that a line crossing pieces comes back once and whole.
final class ScreenTextTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("screentext-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() { try? FileManager.default.removeItem(at: dir) }

    /// A desktop of `width` x `height` pixels with `lines` drawn from the left edge, `size` pixels tall.
    static func desktop(_ lines: [String], width: Int, height: Int, size: CGFloat, to url: URL, dpi: Double? = nil) throws {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(.white); ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let pitch = size * 3
        for (i, line) in lines.enumerated() {
            let attr = NSAttributedString(string: line, attributes: [.font: NSFont.systemFont(ofSize: size)])
            ctx.textPosition = CGPoint(x: size * 3, y: CGFloat(height) - pitch * CGFloat(i + 2))
            CTLineDraw(CTLineCreateWithAttributedString(attr), ctx)
        }
        let dst = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        let properties = dpi.map { [kCGImagePropertyDPIWidth: $0, kCGImagePropertyDPIHeight: $0] as CFDictionary }
        CGImageDestinationAddImage(dst, try XCTUnwrap(ctx.makeImage()), properties)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
    }

    func testAScreenshotIsReadAtOneAndAHalfPixelsAPoint() {
        // A pixel a point: no density stated, or 72.
        XCTAssertEqual(ScreenText.readSide(width: 1920, height: 1080, dpi: nil), 2880)
        XCTAssertEqual(ScreenText.readSide(width: 1920, height: 1080, dpi: 72), 2880)
        XCTAssertEqual(ScreenText.readSide(width: 1280, height: 800, dpi: nil), 1920)
        XCTAssertEqual(ScreenText.readSide(width: 2560, height: 1440, dpi: nil), 3840)
        XCTAssertEqual(ScreenText.readSide(width: 1440, height: 2560, dpi: 0), 3840, "a display turned on its side")
        // A Retina display's file, which has more than it is read at.
        XCTAssertEqual(ScreenText.readSide(width: 3840, height: 2160, dpi: 144), 2880)
        XCTAssertEqual(ScreenText.readSide(width: 3024, height: 1964, dpi: 144), 2268)
        XCTAssertEqual(ScreenText.readSide(width: 5120, height: 2880, dpi: 144), 3840)
        // Its density lost on the way: read as the wider desktop it then looks like, in more pieces.
        XCTAssertEqual(ScreenText.readSide(width: 3840, height: 2160, dpi: nil), 5760)
        // A capture of a whole long page: what fits the reader's memory, its shape kept.
        let side = ScreenText.readSide(width: 1920, height: 60_000, dpi: nil)
        XCTAssertLessThan(side, 60_000)
        XCTAssertLessThanOrEqual(Double(side) * Double(side) * 1920 / 60_000, Double(ScreenText.maxPixels) * 1.001)
        XCTAssertGreaterThan(Double(side) * Double(side) * 1920 / 60_000, Double(ScreenText.maxPixels) * 0.99)
    }

    func testTheFileIsReadAtThatSizeEnlargedWhenItHasFewerPixels() throws {
        let plain = dir.appendingPathComponent("plain.png"), retina = dir.appendingPathComponent("retina.png")
        try Self.desktop(["a"], width: 1280, height: 800, size: 13, to: plain)
        try Self.desktop(["a"], width: 3840, height: 2400, size: 26, to: retina, dpi: 144)
        let a = try XCTUnwrap(ScreenText.picture(plain)), b = try XCTUnwrap(ScreenText.picture(retina))
        XCTAssertEqual([a.width, a.height], [1920, 1200], "enlarged")
        XCTAssertEqual([b.width, b.height], [2880, 1800], "1920 points at 2x, reduced")
        XCTAssertNil(ScreenText.picture(dir.appendingPathComponent("none.png")))
        XCTAssertNil(ScreenText.recognize(screenshot: dir.appendingPathComponent("none.png")))
    }

    func testPiecesCoverThePictureAndOverlap() {
        XCTAssertEqual(ScreenText.tiles(width: 1024, height: 640).count, 1)
        XCTAssertEqual(ScreenText.tiles(width: 900, height: 1024).count, 1)
        for (w, h, count) in [(1920, 1200, 6), (2268, 1473, 6), (2880, 1620, 8), (3840, 2160, 15), (1620, 2880, 8), (1024, 1025, 2)] {
            let tiles = ScreenText.tiles(width: w, height: h)
            XCTAssertEqual(tiles.count, count, "\(w)x\(h)")
            XCTAssertTrue(tiles.allSatisfy { $0.width == CGFloat(min(w, 1024)) && $0.height == CGFloat(min(h, 1024)) && $0.minX >= 0 && $0.minY >= 0
                                             && $0.maxX <= CGFloat(w) && $0.maxY <= CGFloat(h) })
            XCTAssertEqual(tiles.map(\.minX).min(), 0); XCTAssertEqual(tiles.map(\.maxX).max(), CGFloat(w))
            XCTAssertEqual(tiles.map(\.minY).min(), 0); XCTAssertEqual(tiles.map(\.maxY).max(), CGFloat(h))
            // Neighbours share at least the overlap, along both axes.
            let row = tiles.filter { $0.minY == 0 }.sorted { $0.minX < $1.minX }
            for (a, b) in zip(row, row.dropFirst()) { XCTAssertGreaterThanOrEqual(a.maxX - b.minX, CGFloat(ScreenText.pieceOverlap), "\(w)x\(h)") }
            let column = tiles.filter { $0.minX == 0 }.sorted { $0.minY < $1.minY }
            for (a, b) in zip(column, column.dropFirst()) { XCTAssertGreaterThanOrEqual(a.maxY - b.minY, CGFloat(ScreenText.pieceOverlap), "\(w)x\(h)") }
        }
    }

    /// Lines that run the width of a 1920-point desktop cross the seams between its four columns of
    /// pieces, and the rows of lines cross the seam between its two rows: each line comes back
    /// once, in order, and no word is read twice. Every word is numbered, so one read twice shows.
    func testALineAcrossPiecesIsReadOnce() throws {
        let words = ["harbor", "invoice", "lantern", "meadow", "orchard", "paddle", "quarry", "ribbon", "saddle", "timber",
                     "valley", "walnut", "anchor", "basket", "candle", "dagger", "engine", "falcon", "garden", "hammer"]
        var lines: [String] = []
        for row in 0 ..< 20 {
            lines.append((0 ..< 19).map { "\(words[($0 + row * 7) % words.count])\(row * 100 + $0)" }.joined(separator: " "))
        }
        // A line with no spaces is one word to the reader, cut on a seam character by character:
        // the Thousand Character Classic, which repeats none.
        let classic = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔龙师火帝鸟官人皇始制文字乃服衣裳推位让国有虞陶唐"
        XCTAssertEqual(Set(classic).count, classic.count)
        lines.append(classic)
        let url = dir.appendingPathComponent("desktop.png")
        try Self.desktop(lines, width: 1920, height: 1080, size: 16, to: url)
        let image = try XCTUnwrap(ScreenText.picture(url))
        XCTAssertEqual(ScreenText.tiles(width: image.width, height: image.height).count, 8)
        let out = try XCTUnwrap(ScreenText.recognize(screenshot: url)).components(separatedBy: "\n")
        XCTAssertEqual(out.count, lines.count, "one line out for each drawn")
        // The reader may take a letter for another; the pieces put back together may not lose a
        // word, read one twice, or change their order.
        let drawn = lines.dropLast().flatMap { $0.split(separator: " ").map(String.init) }
        let got = out.dropLast().flatMap { $0.split(separator: " ").map(String.init) }
        XCTAssertEqual(got.count, drawn.count, "as many words as were drawn")
        XCTAssertEqual(Set(got).count, got.count, "none twice")
        let same = zip(drawn, got).filter { $0 == $1 }.count
        XCTAssertGreaterThanOrEqual(Double(same) / Double(drawn.count), 0.97, "word for word, in order")
        let last = out.last ?? ""
        XCTAssertEqual(Set(last).count, last.count, "no character twice")
        XCTAssertFalse(last.contains(" "), "joined on the seam without a gap")
        XCTAssertGreaterThanOrEqual(Double(classic.filter { last.contains($0) }.count) / Double(classic.count), 0.9)
        XCTAssertEqual(last.filter { classic.contains($0) }, classic.filter { last.contains($0) }, "in order")
    }

    /// The reader boxes a Latin word, not its letters: a word longer than what two pieces share
    /// is cut at the edge of both, and is one word again from the two readings.
    func testAWordLongerThanWhatPiecesShareIsReadOnce() throws {
        // By its two readings alone: the letter at each cut misread, the boxes a little off.
        let whole = "content.scrollContentBackground(.hidden)"
        let a = ScreenText.Word(text: "content.scrollContentBackground(.hido", minX: 590, maxX: 1022, cutRight: true)
        let b = ScreenText.Word(text: "ltent.scrollContentBackground(.hidden)", minX: 612, maxX: 1061, cutLeft: true)
        XCTAssertEqual(ScreenText.stitch(a, b), whole)
        XCTAssertEqual(ScreenText.stitch(ScreenText.Word(text: "abcdefgh", minX: 0, maxX: 80, cutRight: true),
                                         ScreenText.Word(text: "efghijkl", minX: 40, maxX: 120, cutLeft: true)), "abcdefghijkl")
        // Drawn: identifiers and a URL across the seams of a 1920x1080 desktop, short lines beside
        // them that the reader gives in parts.
        let long = ["scrollContentBackgroundHiddenFrameMaxWidthInfinityPaddingHorizontalTwelveListRowSeparatorVisible",
                    "https://example.com/reports/quarterly/2026/october/harbor-invoice-lantern-meadow-orchard-paddle-quarry",
                    "import SwiftUI", "case folder(URL)"]
        var lines: [String] = []
        for row in 0 ..< 12 { lines.append(String(repeating: " ", count: row * 3) + long[row % long.count]) }
        let url = dir.appendingPathComponent("long.png")
        try Self.desktop(lines, width: 1920, height: 1080, size: 16, to: url)
        let out = try XCTUnwrap(ScreenText.recognize(screenshot: url)).components(separatedBy: "\n")
        XCTAssertEqual(out.count, lines.count, "one line out for each drawn, in order")
        /// The characters two strings share, in order.
        func common(_ x: [Character], _ y: [Character]) -> Int {
            var row = [Int](repeating: 0, count: y.count + 1)
            for c in x { var diagonal = 0; for (j, d) in y.enumerated() { let up = row[j + 1]; row[j + 1] = c == d ? diagonal + 1 : max(up, row[j]); diagonal = up } }
            return row[y.count]
        }
        for (drawn, got) in zip(lines, out) {
            let want = Array(drawn.trimmingCharacters(in: .whitespaces)), read = Array(got)
            XCTAssertLessThanOrEqual(abs(read.count - want.count), 2, "nothing twice, nothing lost: \(got)")
            // The reader may take a letter for another (an I for an l): one, or one in twenty.
            XCTAssertLessThanOrEqual(want.count - common(want, read), max(1, want.count / 20), got)
        }
    }
}
