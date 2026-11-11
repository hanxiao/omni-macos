import Foundation
import Vision
import CoreGraphics
import ImageIO

/// The text on a screenshot, read on the device by Vision: the error code, the booking reference,
/// the chat line someone remembers - what a picture's embedding sees only as a picture of text.
/// Measured on the eval's nine screenshots (1280x800, English, Chinese, Japanese): 68-138 ms each,
/// every line read correctly but a few characters of a Swift listing. Accurate recognition, the
/// language detected per image, NO language correction: it rewrote code and codes toward words
/// ("= 0" read as "= O", "+=" as "+ ="), which is what identifier lookup needs verbatim, and
/// cost 19% (omni-verify ocrcost: 827 -> 673 ms for the nine, the other eight read the same).
/// The compute device was measured too: Neural Engine, GPU and CPU all ~120 ms an image.
public enum ScreenText {
    /// The recognized lines, top to bottom, one per line; "" when there is none.
    public static func recognize(_ image: CGImage) -> String {
        lines(image).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }

    private static func lines(_ image: CGImage) -> [VNRecognizedTextObservation] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.automaticallyDetectsLanguage = true
        do { try VNImageRequestHandler(cgImage: image).perform([request]) } catch { return [] }
        return request.results ?? []
    }

    // MARK: - A screenshot, read in pieces

    /// THE READER LOSES SMALL TEXT ON A LARGE PICTURE. Vision works at a size of its own, and what it
    /// reads depends on the text's size in pixels and on the shape of what it is given, not on how
    /// much is there. Read whole at the index's 1568, 38 real screenshots of this Mac gave 79.0% of
    /// the words two other readings agree on, and 48 drawn pages of 10-13 point text 44% of their
    /// words - none at all on 20 of them; of an editor's words 97% on a 1512-point desktop, 80% on
    /// a 2560-point one, 1% on a 3840-point one. At 1.5 pixels a point in 1024 squares: 94.9% of
    /// the real ones, 95-98% of the pages, 97-98% of the editor's on every desktop.
    /// Pieces as wide as the picture lost whole pages at every size; `minimumTextHeight`, the
    /// request's revision and its languages changed nothing (docs/screenshots.md, "The reader").

    /// The pixels a point a screenshot is read at: the file's own when it has as many, enlarged
    /// when it has fewer (a 1x display's screenshot, a recorded still).
    public static let pixelsAPoint = 1.5
    /// The piece the reader is given, on each side, and what neighbours share at least.
    public static let pieceSide = 1024
    public static let pieceOverlap = 196
    /// No picture is read larger than this many pixels (128 MB as it is held): every desktop is
    /// well under it at `pixelsAPoint` (a 5120-point one is 17 million), a capture of a whole long
    /// page is not, and is read at what fits.
    public static let maxPixels = 32_000_000

    /// NEGATIVE CONTROL, for tests: off, a screenshot is read whole at the index's picture size
    /// under the keys 0.20.0 gave it, which is how a test builds the index that an upgrade finds.
    nonisolated(unsafe) package static var inPieces = true

    /// What a screenshot's content and chunk keys carry for its text. t1: read whole at the
    /// index's picture size; t2: read in pieces.
    static var keyMark: String { inPieces ? "|t2" : "|t1" }

    /// The long side a screenshot is read at. `dpi` is what the file states: 144 on a Retina
    /// display's screenshot, none or 72 on one at a pixel a point (a recorded still is one).
    public static func readSide(width: Int, height: Int, dpi: Double?) -> Int {
        let long = Double(max(width, height)), short = Double(min(width, height))
        let read = long / max((dpi ?? 72) / 72, 1) * pixelsAPoint
        let fits = (Double(maxPixels) * long / max(short, 1)).squareRoot()
        return Int(min(read, fits).rounded())
    }

    /// The screenshot at `url` as it is read; nil when it cannot be opened.
    public static func picture(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = (p[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let h = (p[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue, w > 0, h > 0 else { return nil }
        let read = readSide(width: w, height: h, dpi: (p[kCGImagePropertyDPIWidth] as? NSNumber)?.doubleValue)
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                        kCGImageSourceThumbnailMaxPixelSize: min(max(w, h), read),
                                        kCGImageSourceCreateThumbnailWithTransform: true]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return fitted(image, long: read)
    }

    /// `image` with its long side `long`, enlarged when it is smaller.
    public static func fitted(_ image: CGImage, long: Int) -> CGImage? {
        let has = max(image.width, image.height)
        guard has != long, has > 0 else { return image }
        let k = Double(long) / Double(has)
        let w = max(1, Int((Double(image.width) * k).rounded())), h = max(1, Int((Double(image.height) * k).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return context.makeImage()
    }

    /// The text of a screenshot file; nil when it cannot be opened.
    public static func recognize(screenshot url: URL) -> String? {
        picture(url).map { recognize(pieces: $0) }
    }

    /// The text of a picture read in pieces and put back together: a line that crosses pieces is
    /// one line again, and a word both pieces read is there once.
    ///
    /// VISION BOXES A LATIN WORD, NOT ITS LETTERS (`boundingBox(for:)` of one letter is the word's),
    /// so a word is taken whole from one piece: from the one that has all of it, and when it is
    /// longer than what the pieces share and each has it cut at its edge (an identifier, a URL),
    /// from both readings laid over each other (`stitch`). Chinese and Japanese characters have
    /// boxes of their own and are split where the pieces meet. A box moves a pixel or two from one
    /// piece to the next: what is within `slack` of a seam is kept by both and matched by place.
    public static func recognize(pieces image: CGImage, side: Int = pieceSide, overlap: Int = pieceOverlap) -> String {
        let rects = tiles(width: image.width, height: image.height, side: side, overlap: overlap)
        guard rects.count > 1 else { return recognize(image) }
        let xs = Array(Set(rects.map { Int($0.minX) })).sorted(), ys = Array(Set(rects.map { Int($0.minY) })).sorted()
        let w = Double(rects[0].width), h = Double(rects[0].height)
        /// What each piece owns along an axis: up to the middle of what it shares with the next.
        func bounds(_ starts: [Int], _ length: Double) -> [Double] {
            [-Double.infinity] + (1 ..< starts.count).map { (Double(starts[$0] + starts[$0 - 1]) + length) / 2 } + [Double.infinity]
        }
        let bx = bounds(xs, w), by = bounds(ys, h)
        let slack = 4.0
        var runs: [Run] = []
        // One piece at a time: Vision reads one picture at a time whoever asks (47 stills indexed in
        // 42.0 s with 1 piece at once, 41.1 / 41.2 / 41.0 s with 2 / 4 / 8).
        for rect in rects {
            guard let column = xs.firstIndex(of: Int(rect.minX)), let row = ys.firstIndex(of: Int(rect.minY)),
                  let tile = image.cropping(to: rect) else { continue }
            let lo = bx[column], hi = bx[column + 1]
            let left = Double(rect.minX), right = Double(rect.maxX)
            /// A box of the piece (Vision's: 0...1, from the bottom left) in the picture, from the top left.
            func place(_ box: CGRect) -> (minX: Double, maxX: Double, minY: Double, maxY: Double) {
                (left + Double(box.minX) * w, left + Double(box.maxX) * w,
                 Double(rect.minY) + (1 - Double(box.maxY)) * h, Double(rect.minY) + (1 - Double(box.minY)) * h)
            }
            for line in lines(tile) {
                guard let best = line.topCandidates(1).first else { continue }
                let at = place(line.boundingBox)
                let midY = (at.minY + at.maxY) / 2
                guard midY >= by[row] - slack, midY < by[row + 1] + slack else { continue }    // the piece above or below has it
                var run = Run(words: [], minX: at.minX, maxX: at.maxX, midY: midY, height: at.maxY - at.minY, column: column, row: row)
                if at.minX >= lo + 2 * slack, at.maxX < hi - 2 * slack {
                    run.words = [Word(text: best.string, minX: at.minX, maxX: at.maxX)]; runs.append(run); continue
                }
                if at.maxX <= lo - slack || at.minX >= hi + slack { continue }
                // The line reaches a neighbour's part: word by word.
                let text = best.string
                func box(_ range: Range<String.Index>) -> (minX: Double, maxX: Double)? {
                    guard let b = try? best.boundingBox(for: range)?.boundingBox else { return nil }
                    let p = place(b)
                    return (p.minX, p.maxX)
                }
                var i = text.startIndex
                while i < text.endIndex {
                    guard !text[i].isWhitespace else { i = text.index(after: i); continue }
                    let end = text[i...].firstIndex(where: \.isWhitespace) ?? text.endIndex
                    defer { i = end }
                    guard let b = box(i ..< end), b.maxX > lo - slack, b.minX < hi + slack else { continue }
                    var word = Word(text: String(text[i ..< end]), minX: b.minX, maxX: b.maxX)
                    if b.minX < lo + slack || b.maxX >= hi - slack {
                        let second = text.index(after: i)
                        if second < end, let one = box(i ..< second), one.maxX - one.minX < 0.8 * (b.maxX - b.minX) {
                            // Characters with boxes of their own: each to the piece its middle is in.
                            var part = "", first = Double.infinity, last = -Double.infinity
                            var c = i
                            while c < end {
                                let next = text.index(after: c)
                                if let cb = box(c ..< next), (cb.minX + cb.maxX) / 2 >= lo, (cb.minX + cb.maxX) / 2 < hi {
                                    part.append(text[c]); first = min(first, cb.minX); last = max(last, cb.maxX)
                                }
                                c = next
                            }
                            guard !part.isEmpty else { continue }
                            word = Word(text: part, minX: first, maxX: last, openLeft: b.minX < lo, openRight: b.maxX >= hi)
                        } else {
                            word.cutLeft = left > 0 && b.minX <= left + slack
                            word.cutRight = right < Double(image.width) && b.maxX >= right - slack
                        }
                    }
                    run.words.append(word)
                }
                guard let first = run.words.first, let last = run.words.last else { continue }
                run.minX = first.minX; run.maxX = last.maxX
                runs.append(run)
            }
        }
        func level(_ a: Run, _ b: Run) -> Bool { abs(a.midY - b.midY) < 0.5 * min(a.height, b.height) }
        func shared(_ a: (minX: Double, maxX: Double), _ b: (minX: Double, maxX: Double)) -> Double { min(a.maxX, b.maxX) - max(a.minX, b.minX) }
        // A line on the seam between two rows of pieces is in both: the upper one's stays.
        runs = runs.filter { r in
            r.midY >= by[r.row] + 2 * slack || !runs.contains {
                $0.row + 1 == r.row && $0.column == r.column && level($0, r)
                    && shared(($0.minX, $0.maxX), (r.minX, r.maxX)) > 0.5 * min($0.maxX - $0.minX, r.maxX - r.minX)
            }
        }
        // A line's parts, left to right, are one line again where they meet.
        var joined: [Run] = []
        for run in runs.sorted(by: { $0.minX < $1.minX }) {
            /// Where in `l` the word `b` of the piece to its right is read too.
            func twin(_ b: Word, in l: Run) -> Int? {
                l.words.lastIndex { a in
                    let over = shared((a.minX, a.maxX), (b.minX, b.maxX))
                    return over > 0.5 * min(a.maxX - a.minX, b.maxX - b.minX) || (a.cutRight && b.cutLeft && over > 0)
                }
            }
            guard let head = run.words.first, let k = joined.firstIndex(where: {
                level($0, run) && run.minX - $0.maxX < 2 * run.height
                    && (run.minX - $0.maxX > -run.height || ($0.column != run.column && twin(head, in: $0) != nil))
            }) else { joined.append(run); continue }
            var rest = run.words[...]
            if joined[k].column != run.column {
                // What both pieces read is there once: from the one that has all of the word.
                while let b = rest.first, let j = twin(b, in: joined[k]) {
                    let a = joined[k].words[j]
                    if a.cutRight, b.cutLeft {
                        joined[k].words[j] = Word(text: stitch(a, b), minX: a.minX, maxX: b.maxX, cutLeft: a.cutLeft, cutRight: b.cutRight)
                    } else if a.cutRight { joined[k].words[j] = b }
                    rest = rest.dropFirst()
                }
            }
            if let a = joined[k].words.last, let b = rest.first, a.openRight, b.openLeft {
                joined[k].words[joined[k].words.count - 1] = Word(text: a.text + b.text, minX: a.minX, maxX: b.maxX, openLeft: a.openLeft, openRight: b.openRight)
                rest = rest.dropFirst()
            }
            joined[k].words += rest
            joined[k].maxX = max(joined[k].maxX, run.maxX); joined[k].column = run.column
        }
        // Top to bottom, and left to right along a row: two boxes of one row are never exactly level.
        var rows: [[Run]] = []
        for run in joined.sorted(by: { $0.midY < $1.midY }) {
            if let first = rows.last?.first, level(first, run) { rows[rows.count - 1].append(run) } else { rows.append([run]) }
        }
        return rows.flatMap { $0.sorted { $0.minX < $1.minX } }.map { $0.words.map(\.text).joined(separator: " ") }.joined(separator: "\n")
    }

    /// One word from two readings of it, `a` cut at its right end and `b` at its left: laid over
    /// each other where the boxes say they overlap, give or take three characters, at the offset
    /// where most characters agree, and each taken up to the middle of what they share (the
    /// characters at a cut are the ones misread).
    static func stitch(_ a: Word, _ b: Word) -> String {
        let x = Array(a.text), y = Array(b.text)
        guard !x.isEmpty, !y.isEmpty else { return a.text + b.text }
        let per = ((a.maxX - a.minX) / Double(x.count) + (b.maxX - b.minX) / Double(y.count)) / 2
        let most = min(x.count, y.count)
        let guess = min(most, max(1, Int(((a.maxX - b.minX) / max(per, 1)).rounded())))
        var best = (share: -1.0, k: guess)
        for k in (max(1, guess - 3) ... min(most, guess + 3)).sorted(by: { abs($0 - guess) < abs($1 - guess) }) {
            let share = Double(zip(x.suffix(k), y.prefix(k)).filter { $0 == $1 }.count) / Double(k)
            if share > best.share { best = (share, k) }
        }
        let half = best.k / 2
        return String(x.prefix(x.count - best.k + half)) + String(y.suffix(y.count - half))
    }

    /// A line of one piece, or the part of it the piece owns.
    private struct Run {
        var words: [Word]
        var minX: Double, maxX: Double, midY: Double, height: Double
        var column: Int, row: Int
    }

    /// A word of a line; a whole line that is nowhere near a seam.
    struct Word {
        var text: String
        var minX: Double, maxX: Double
        var cutLeft = false, cutRight = false      // the piece's edge runs through it
        var openLeft = false, openRight = false    // split by character, the rest in the next piece
    }

    /// The pieces a picture larger than `side` is read in: `side` on each edge that is longer, laid
    /// evenly and sharing at least `overlap`, so a word on a seam is whole in one.
    public static func tiles(width: Int, height: Int, side: Int = pieceSide, overlap: Int = pieceOverlap) -> [CGRect] {
        func starts(_ length: Int) -> [Int] {
            guard length > side else { return [0] }
            let n = Int((Double(length - overlap) / Double(side - overlap)).rounded(.up))
            return (0 ..< n).map { Int((Double($0) * Double(length - side) / Double(n - 1)).rounded()) }
        }
        let w = min(width, side), h = min(height, side)
        return starts(height).flatMap { y in starts(width).map { CGRect(x: $0, y: y, width: w, height: h) } }
    }
}
