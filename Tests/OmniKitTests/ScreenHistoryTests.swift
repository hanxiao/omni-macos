import XCTest
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// The screen history: every changed shot in a video segment, the new ones flagged for the index
/// under a path no file is behind. These pin what is stored, what is indexed and where, that a
/// history survives being reopened, cut short and pruned, and that what 0.20.2 left is taken in.
final class ScreenHistoryTests: XCTestCase {
    private var dir: URL!
    private var history: ScreenHistory!
    private let width = 640, height = 400

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("shots-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        history = open()
    }

    override func tearDownWithError() throws {
        history.stop()
        try? FileManager.default.removeItem(at: dir)
    }

    private func open(segmentFrames: Int = ScreenHistory.maxSegmentFrames, in folder: URL? = nil) -> ScreenHistory {
        let d = folder ?? dir!
        return ScreenHistory(history: d.appendingPathComponent("Screen History"), stills: d.appendingPathComponent("Screenshots"),
                             segmentFrames: segmentFrames)
    }

    /// A desktop: a page of bars laid out from `page`, and `typed` more short bars in one corner,
    /// which is what a line of typing changes.
    private func frame(page: Int, typed: Int = 0, width: Int? = nil, height: Int? = nil) -> CVPixelBuffer {
        let w = width ?? self.width, h = height ?? self.height
        let pixels = ScreenCodec.makePixelBuffer(width: w, height: h)!
        CVPixelBufferLockBaseAddress(pixels, [])
        defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
        let cg = CGContext(data: CVPixelBufferGetBaseAddress(pixels), width: w, height: h, bitsPerComponent: 8,
                           bytesPerRow: CVPixelBufferGetBytesPerRow(pixels), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        cg.setFillColor(CGColor(gray: 1, alpha: 1))
        cg.fill(CGRect(x: 0, y: 0, width: w, height: h))
        var seed = UInt64(page) &* 0x9E37_79B9_7F4A_7C15 &+ 1
        func next() -> Int { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Int(seed >> 33) }
        cg.setFillColor(CGColor(gray: 0.1, alpha: 1))
        for row in 0 ..< (h - 80) / 16 {
            var x = 16 + next() % 40
            while x < w - 60 {
                let len = 12 + next() % 70
                cg.fill(CGRect(x: x, y: 70 + row * 16, width: min(len, w - 16 - x), height: 8))
                x += len + 10
            }
        }
        for i in 0 ..< typed { cg.fill(CGRect(x: 16 + i * 14, y: 20, width: 10, height: 20)) }
        return pixels
    }

    private func meanDifference(_ a: CVPixelBuffer, _ b: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(a, .readOnly); CVPixelBufferLockBaseAddress(b, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(a, .readOnly); CVPixelBufferUnlockBaseAddress(b, .readOnly) }
        let w = CVPixelBufferGetWidth(a), h = CVPixelBufferGetHeight(a)
        let pa = CVPixelBufferGetBaseAddress(a)!.assumingMemoryBound(to: UInt8.self), sa = CVPixelBufferGetBytesPerRow(a)
        let pb = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self), sb = CVPixelBufferGetBytesPerRow(b)
        var total = 0
        for y in 0 ..< h { for x in 0 ..< w { total += abs(Int(pa[y * sa + x * 4 + 1]) - Int(pb[y * sb + x * 4 + 1])) } }
        return Double(total) / Double(w * h)
    }

    /// The paths the index holds or is to hold, as a pass is given them.
    private func indexed(_ h: ScreenHistory? = nil) -> [String] {
        ((h ?? history).indexedShots() ?? []).map(\.path)
    }

    /// The video's files: the history's folder also keeps the list of indexed shots (shots.catalog).
    private func segments(in folder: URL? = nil) -> [String] {
        let all = (try? FileManager.default.contentsOfDirectory(atPath: (folder ?? dir).appendingPathComponent("Screen History").path)) ?? []
        return all.filter { $0.hasSuffix("." + ScreenHistory.segmentExtension) }.sorted()
    }

    private func everyShot(_ h: ScreenHistory) -> [ScreenHistory.Shot] {
        h.shots(from: 0, to: t0 + 1000 * 86_400).sorted { ($0.time, $0.display) < ($1.time, $1.display) }
    }

    /// Bytes, not characters: the index compares paths as SQLite does, and Swift calls two
    /// spellings of one accent equal.
    private func bytes(_ paths: [String]) -> [[UInt8]] { paths.map { Array($0.utf8) }.sorted { $0.lexicographicallyPrecedes($1) } }

    private let t0 = 1_800_000_000.0     // a fixed afternoon; the tests never read the clock

    // MARK: The codec

    func testFramesComeBackAsTheyWentIn() throws {
        let encoder = try XCTUnwrap(ScreenFrameEncoder(width: width, height: height))
        let decoder = ScreenFrameDecoder()
        for i in 0 ..< 6 {
            let source = frame(page: 1, typed: i * 3)
            let encoded = try XCTUnwrap(encoder.encode(source))
            XCTAssertEqual(encoded.isKey, i == 0)
            XCTAssertEqual(encoded.parameterSets.isEmpty, i != 0, "a key frame carries its parameter sets")
            let back = try XCTUnwrap(decoder.decode(encoded.data, parameterSets: encoded.parameterSets))
            XCTAssertLessThan(meanDifference(source, back), 4)
        }
    }

    func testATypedLineCostsLessThanTheFrameItIsTypedOn() throws {
        let encoder = try XCTUnwrap(ScreenFrameEncoder(width: width, height: height))
        let key = try XCTUnwrap(encoder.encode(frame(page: 1)))
        let next = try XCTUnwrap(encoder.encode(frame(page: 1, typed: 4)))
        XCTAssertLessThan(next.data.count * 4, key.data.count)
    }

    // MARK: What is stored

    /// `record` answers nil for both; only one of them is something wrong.
    func testAnUnchangedScreenIsNotAFailureAndAHistoryThatCannotWriteIs() throws {
        XCTAssertNotNil(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNil(history.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNil(history.lastFailure)

        // A file where the history's folder should be: no segment can be made.
        let blocked = dir.appendingPathComponent("blocked")
        try Data("x".utf8).write(to: blocked)
        let broken = ScreenHistory(history: blocked, stills: dir.appendingPathComponent("Screenshots 2"))
        XCTAssertNil(broken.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertEqual(broken.lastFailure, "The video could not be started.")
        // And it clears with the next shot that is stored.
        try FileManager.default.removeItem(at: blocked)
        XCTAssertNotNil(broken.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNil(broken.lastFailure)
        broken.stop()
    }

    func testAScreenThatDidNotChangeIsNotStored() throws {
        XCTAssertNotNil(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNil(history.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNotNil(history.record(frame(page: 1, typed: 3), time: t0 + 10, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).map(\.time), [t0, t0 + 10])
    }

    func testTheFirstShotAndANewPageAreIndexedAndTypingIsNot() throws {
        let first = try XCTUnwrap(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(first.shot.indexed)
        XCTAssertNotNil(first.path)
        let typing = try XCTUnwrap(history.record(frame(page: 1, typed: 6), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertFalse(typing.shot.indexed)
        XCTAssertNil(typing.path)
        XCTAssertNil(history.path(of: typing.shot))
        XCTAssertLessThan(typing.novelty, ScreenGate.working)
        XCTAssertGreaterThan(typing.novelty, ScreenGate.settled)
        let page = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 10, display: 1, app: "Safari", title: "News", settled: false))
        XCTAssertTrue(page.shot.indexed)
        XCTAssertEqual(indexed(), [first.path, page.path])
        XCTAssertFalse(history.stillsPending, "no folder and no file is made for the index")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), ["Screen History"])
    }

    /// An indexed shot must not cost the video its next frame: a typed line after it is still a
    /// typed line, not the whole screen again.
    func testTheShotAfterAnIndexedOneIsStillSmall() throws {
        for (w, h) in [(640, 400), (1920, 1080)] {
            let first = try XCTUnwrap(history.record(frame(page: 1, width: w, height: h), time: t0, display: UInt32(w), app: "Xcode", title: "a.swift", settled: false))
            XCTAssertNotNil(first.path)
            let typing = try XCTUnwrap(history.record(frame(page: 1, typed: 4, width: w, height: h), time: t0 + 5, display: UInt32(w), app: "Xcode", title: "a.swift", settled: false))
            XCTAssertNil(typing.path)
            XCTAssertLessThan(typing.bytes * 4, first.bytes, "\(w)x\(h)")
        }
    }

    func testGoingBackToAnIndexedWindowIsNotNew() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        _ = history.record(frame(page: 2), time: t0 + 5, display: 1, app: "Safari", title: "News", settled: false)
        let back = try XCTUnwrap(history.record(frame(page: 1), time: t0 + 10, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertFalse(back.shot.indexed, "the shot is stored: the screen changed")
        XCTAssertEqual(indexed().count, 2)
    }

    /// Typing never crosses the working threshold a shot at a time; the state it is left in is
    /// what gets indexed, whether the last shot was taken before the user stopped or after.
    func testWhatTheUserLeftOnScreenIsIndexedWhenTheyStop() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        _ = history.record(frame(page: 1, typed: 6), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false)
        // The screen is as the last shot left it: that shot is the one indexed.
        let same = try XCTUnwrap(history.record(frame(page: 1, typed: 6), time: t0 + 10, display: 1, app: "Xcode", title: "a.swift", settled: true))
        XCTAssertEqual(same.shot.time, t0 + 5)
        XCTAssertTrue(same.shot.indexed)
        XCTAssertEqual(same.path, history.path(of: same.shot))
        XCTAssertNotNil(same.path)
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).map(\.indexed), [true, true])
        XCTAssertNil(history.record(frame(page: 1, typed: 6), time: t0 + 40, display: 1, app: "Xcode", title: "a.swift", settled: true),
                     "and only once")
        // The screen moved on since the last shot: a new shot, indexed.
        let moved = try XCTUnwrap(history.record(frame(page: 1, typed: 12), time: t0 + 60, display: 1, app: "Xcode", title: "a.swift", settled: true))
        XCTAssertEqual(moved.shot.time, t0 + 60)
        XCTAssertTrue(moved.shot.indexed)
        let live = indexed()
        XCTAssertEqual(live.count, 3)
        XCTAssertEqual(live[1], same.path)
        // The flag written into the stored record is what a new reader sees.
        history.stop()
        let reopened = open()
        XCTAssertEqual(reopened.shots(from: t0, to: t0 + 100).map(\.indexed), [true, true, true])
        XCTAssertEqual(indexed(reopened), live)
    }

    // MARK: Where the index holds a shot

    /// The path 0.20.2 gave the file of a shot, which is the path its row in the index has: the
    /// day's folder, then the second, the app and the title. No file is at it.
    func testAnIndexedShotHasThePathItsFileHadAndNoFile() throws {
        let big = frame(page: 1, width: 2560, height: 1440)
        let r = try XCTUnwrap(history.record(big, time: t0, display: 1, app: "Xcode", title: "OmniKit/Sidebar.swift: edited\n", settled: false, scale: 2))
        let path = try XCTUnwrap(r.path)
        let name = ScreenHistory0202.name(time: t0, app: "Xcode", title: "OmniKit Sidebar.swift edited")
        XCTAssertEqual(path, history.root + "/" + name.folder + "/" + name.base + ".heic")
        XCTAssertEqual(history.root, dir.path + "/Screenshots")
        XCTAssertTrue(history.isShotPath(path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: history.root))
        XCTAssertEqual(history.shot(atPath: path), r.shot)
        let entry = try XCTUnwrap(history.entry(atPath: path))
        XCTAssertEqual(entry.time, t0)
        XCTAssertGreaterThan(entry.bytes, 0)
        let size = try XCTUnwrap(history.size(of: r.shot))
        XCTAssertEqual([size.width, size.height], [1280, 720], "a pixel a point of a display at two")
        let picture = try XCTUnwrap(history.picture(of: r.shot))
        XCTAssertEqual([picture.image.width, picture.image.height, picture.width, picture.height], [1280, 720, 2560, 1440])
        // Not a shot's path: another folder, a name that is not a time, a time nothing was taken at,
        // and the right second under another title.
        for other in ["/tmp/photo.heic", history.root + "/notes.heic", history.root + "/2020-01-01/2020-01-01 10.00.00 X.heic",
                      history.root + "/" + name.folder + "/" + String(name.base.prefix(19)) + " Safari.heic"] {
            XCTAssertNil(history.shot(atPath: other), other)
            XCTAssertNil(history.entry(atPath: other), other)
        }
    }

    /// A name is spelled from the time to the millisecond, as 0.20.2's formatter spelled it: the
    /// last half millisecond of a second is named for the next one. And as the file system spells
    /// it: an accent is its own character there, which is not what the title's own bytes are.
    func testAPathIsSpelledAsTheFileItReplacesWas() throws {
        let cases: [(Double, String, String)] = [
            (t0 + 0.9996, "Xcode", "late"), (t0 + 1.9994, "Xcode", "early"), (t0 + 3, "", ""), (t0 + 4, "Safari", ""), (t0 + 5, "", "Untitled"),
            (t0 + 6, "Caf\u{E9}", "\u{D55C}\u{AE00} \u{622A}\u{56FE} na\u{EF}ve"),
        ]
        // What a folder lists for a file of each name: made as 0.20.2 made it, read with readdir.
        let scratch = dir.appendingPathComponent("names")
        let picture = try XCTUnwrap(ScreenCodec.cgImage(frame(page: 9, width: 320, height: 200)))
        for (i, c) in cases.enumerated() {
            let r = try XCTUnwrap(history.record(frame(page: i + 1), time: c.0, display: 1, app: c.1, title: c.2, settled: false))
            XCTAssertTrue(r.shot.indexed)
            try ScreenHistory0202.writeStill(picture, in: scratch, time: c.0, app: r.shot.app, title: r.shot.title)
        }
        let listed = ScreenHistory0202.listed(scratch)
        XCTAssertEqual(listed.count, cases.count)
        let expected = listed.map { history.root + "/" + $0 }
        XCTAssertEqual(bytes(indexed()), bytes(expected))
        // Controls. The half millisecond: the first is named for the second after its own, the
        // second for its own. The accents: the folder does not list the bytes the title has.
        XCTAssertEqual(listed.filter { $0.hasSuffix("Xcode - late.heic") }.map { $0.dropLast(18).suffix(2) }, ["01"])
        XCTAssertEqual(listed.filter { $0.hasSuffix("Xcode - early.heic") }.map { $0.dropLast(19).suffix(2) }, ["01"])
        let accented = ScreenHistory0202.name(time: cases[5].0, app: cases[5].1, title: cases[5].2)
        let own = accented.folder + "/" + accented.base + ".heic"
        XCTAssertTrue(listed.contains(own), "the same name to Swift")
        XCTAssertFalse(listed.contains { Array($0.utf8) == Array(own.utf8) }, "and not the same bytes")
        history.stop()
        let reopened = open()
        XCTAssertEqual(bytes(indexed(reopened)), bytes(expected))
        for path in expected { XCTAssertEqual(reopened.shot(atPath: path).flatMap { reopened.path(of: $0) }.map { Array($0.utf8) }, Array(path.utf8)) }
    }

    /// Each segment is named on the clock it was recorded on, whatever clock reads it later.
    func testAChangeOfTimeZoneStartsASegmentNamedOnItsOwnClock() throws {
        let system = NSTimeZone.default
        defer { NSTimeZone.default = system }
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Asia/Tokyo"))
        let tokyo = try XCTUnwrap(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        let tokyoName = ScreenHistory0202.name(time: t0, app: "Xcode", title: "a.swift")
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Pacific/Pago_Pago"))
        let samoa = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 5, display: 1, app: "Safari", title: "News", settled: false))
        let samoaName = ScreenHistory0202.name(time: t0 + 5, app: "Safari", title: "News")
        XCTAssertNotEqual(tokyo.shot.segment, samoa.shot.segment)
        XCTAssertNotEqual(tokyoName.folder, samoaName.folder, "control: the two clocks are a day apart at this hour")
        XCTAssertEqual(tokyo.path, history.root + "/" + tokyoName.folder + "/" + tokyoName.base + ".heic")
        XCTAssertEqual(samoa.path, history.root + "/" + samoaName.folder + "/" + samoaName.base + ".heic")
        history.stop()
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Europe/Berlin"))
        let reopened = open()
        XCTAssertEqual(indexed(reopened), [tokyo.path, samoa.path])
        XCTAssertEqual(reopened.shot(atPath: try XCTUnwrap(tokyo.path))?.time, t0)
        XCTAssertEqual(reopened.shot(atPath: try XCTUnwrap(samoa.path))?.time, t0 + 5)
    }

    /// A shot that leaves the app is the file 0.20.2 kept for every indexed shot.
    func testAShotLeavesAsAScreenCaptureFileDatedAtTheShot() throws {
        let big = frame(page: 1, width: 2560, height: 1440)
        let r = try XCTUnwrap(history.record(big, time: t0 + 0.25, display: 1, app: "Xcode", title: "a.swift", settled: false, scale: 2))
        let typing = try XCTUnwrap(history.record(frame(page: 1, typed: 4, width: 2560, height: 1440), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false, scale: 2))
        let out = dir.appendingPathComponent("out")
        let url = try XCTUnwrap(history.export(r.shot, to: out))
        XCTAssertEqual(url.lastPathComponent, (try XCTUnwrap(r.path) as NSString).lastPathComponent)
        XCTAssertNotNil(getxattr(url.path, ScreenHistory.screenCaptureAttribute, nil, 0, 0, 0) > 0 ? true : nil, "marked as a screen capture")
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual(try XCTUnwrap((attributes[.modificationDate] as? Date)?.timeIntervalSince1970), t0 + 0.25, accuracy: 0.001)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual([image.width, image.height], [1280, 720])
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: out.path).count, 1, "no temporary file is left beside it")
        // Asked again, the file made is the file given.
        let made = try XCTUnwrap(attributes[.systemFileNumber] as? Int)
        XCTAssertEqual(history.export(r.shot, to: out), url)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: url.path)[.systemFileNumber] as? Int, made)
        XCTAssertNil(history.export(typing.shot, to: out), "a shot the index does not hold has no name to leave under")
    }

    func testAStillIsAPixelAPointAndNeverMoreThanTheFrame() {
        XCTAssertEqual(ScreenHistory.stillSide(width: 1920, height: 1080, scale: 1), 1920)
        XCTAssertEqual(ScreenHistory.stillSide(width: 3840, height: 2160, scale: 2), 1920)
        XCTAssertEqual(ScreenHistory.stillSide(width: 3024, height: 1964, scale: 2), 1512)
        XCTAssertEqual(ScreenHistory.stillSide(width: 5120, height: 2880, scale: 2), 2560)
        XCTAssertEqual(ScreenHistory.stillSide(width: 3440, height: 1440, scale: 1), 3440)
        XCTAssertEqual(ScreenHistory.stillSide(width: 1440, height: 2560, scale: 1), 2560, "a display turned on its side")
        XCTAssertEqual(ScreenHistory.stillSide(width: 2560, height: 1440, scale: 0), 2560, "no scale known: the frame")
        XCTAssertEqual(ScreenHistory.stillSide(width: 2560, height: 1440, scale: 0.5), 2560)
    }

    func testNamesAreOneLineWithNoSeparators() {
        XCTAssertEqual(ScreenHistory.clean("  a/b:c\n\td  ", limit: 40), "a b c d")
        XCTAssertEqual(ScreenHistory.clean(".hidden", limit: 40), "hidden")
        XCTAssertLessThanOrEqual(ScreenHistory.clean(String(repeating: "word ", count: 40), limit: 80).count, 80)
        XCTAssertEqual(ScreenHistory.clean("截图 - 设置", limit: 40), "截图 - 设置")
    }

    // MARK: Reading back

    func testAReopenedHistoryShowsEveryShotInAnyOrder() throws {
        var sources: [CVPixelBuffer] = []
        for i in 0 ..< 40 {
            let f = frame(page: 1 + i / 10, typed: i % 10)
            sources.append(f)
            XCTAssertNotNil(history.record(f, time: t0 + Double(i) * 5, display: 1, app: "App \(i / 10)", title: "Window \(i)", settled: false))
        }
        history.stop()
        let reopened = open()
        let shots = reopened.shots(from: t0, to: t0 + 1000)
        XCTAssertEqual(shots.count, 40)
        XCTAssertEqual(shots[17].app, "App 1")
        XCTAssertEqual(shots[17].title, "Window 17")
        // Forward, backward, across a key frame, the same one twice.
        for i in [0, 1, 2, 39, 38, 31, 5, 5, 29, 30, 12] {
            let back = try XCTUnwrap(reopened.pixels(shots[i]), "shot \(i)")
            XCTAssertLessThan(meanDifference(sources[i], back), 4, "shot \(i)")
        }
        XCTAssertEqual(reopened.shots(from: t0 + 50, to: t0 + 99).map(\.index), Array(10 ... 19))
        XCTAssertEqual(reopened.days().count, 1)
    }

    func testARecordCutShortEndsTheSegmentAndLosesNothingBeforeIt() throws {
        for i in 0 ..< 5 {
            _ = history.record(frame(page: 1, typed: i), time: t0 + Double(i) * 5, display: 1, app: "Xcode", title: "a.swift", settled: false)
        }
        history.stop()
        let folder = dir.appendingPathComponent("Screen History")
        let file = folder.appendingPathComponent(try XCTUnwrap(segments().first))
        let whole = try Data(contentsOf: file)
        for cut in [1, 3, 7, 20] {
            try whole.prefix(whole.count - cut).write(to: file)
            let reopened = open()
            let shots = reopened.shots(from: t0, to: t0 + 100)
            XCTAssertEqual(shots.count, 4, "cut \(cut)")
            XCTAssertNotNil(reopened.pixels(shots[3]))
        }
        // A kill, then the next launch: a new segment beside the cut one, both read.
        let next = open()
        XCTAssertNotNil(next.record(frame(page: 1, typed: 9), time: t0 + 100, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertEqual(next.shots(from: t0, to: t0 + 200).count, 5)
        next.stop()
        try Data("not a segment".utf8).write(to: folder.appendingPathComponent("5-1.omniscreen"))
        XCTAssertEqual(open().shots(from: 0, to: t0 + 200).count, 5)
    }

    func testASegmentEndsAtItsLengthAtANewSizeAndForEachDisplay() throws {
        let short = open(segmentFrames: 3)
        for i in 0 ..< 7 {
            _ = short.record(frame(page: 1, typed: i), time: t0 + Double(i) * 5, display: 1, app: "", title: "", settled: false)
        }
        _ = short.record(frame(page: 1, width: 800, height: 500), time: t0 + 40, display: 1, app: "", title: "", settled: false)
        _ = short.record(frame(page: 3), time: t0 + 41, display: 2, app: "", title: "", settled: false)
        _ = short.record(frame(page: 1), time: t0 + 2 * 3600, display: 1, app: "", title: "", settled: false)
        let shots = short.shots(from: t0, to: t0 + 3 * 3600)
        XCTAssertEqual(shots.count, 10)
        XCTAssertEqual(Set(shots.map(\.segment)).count, 6)
        XCTAssertEqual(shots.filter { $0.display == 2 }.count, 1)
        for s in shots { XCTAssertNotNil(short.pixels(s)) }
        short.stop()
        XCTAssertEqual(open().shots(from: t0, to: t0 + 3 * 3600).map(\.id), shots.map(\.id))
    }

    // MARK: Sessions

    func testASessionIsTheShotsWithNoLongGapAndWhatWasInFront() throws {
        var t = t0
        for (i, app) in ["Xcode", "Xcode", "Safari", "Xcode"].enumerated() {
            _ = history.record(frame(page: i + 1), time: t, display: 1, app: app, title: app == "Xcode" ? "a.swift" : "Docs", settled: false)
            t += 120
        }
        _ = history.record(frame(page: 9), time: t0 + 3600, display: 1, app: "Mail", title: "Inbox", settled: false)
        let work = try XCTUnwrap(history.session(around: t0 + 130))
        XCTAssertEqual(work.start, t0)
        XCTAssertEqual(work.end, t0 + 360)
        XCTAssertEqual(work.shots, 4)
        XCTAssertEqual(work.apps.map(\.name), ["Xcode", "Safari"])
        XCTAssertEqual(work.apps[0].titles, ["a.swift"])
        XCTAssertEqual(try XCTUnwrap(history.session(around: t0 + 3600)).apps.map(\.name), ["Mail"])
        XCTAssertNil(history.session(around: t0 + 1800))
        XCTAssertEqual(work.nearest.time, t0 + 120)
    }

    /// The lookup reads two hours either side first: a stretch longer than that is still whole.
    func testALongSessionIsFollowedPastTheFirstReach() throws {
        let step = 240.0, count = 120                    // eight hours, a shot every four minutes
        for i in 0 ..< count {
            _ = history.record(frame(page: i % 7 + 1), time: t0 + Double(i) * step, display: 1, app: "Xcode", title: "a.swift", settled: false)
        }
        _ = history.record(frame(page: 9), time: t0 + Double(count) * step + 3600, display: 1, app: "Mail", title: "Inbox", settled: false)
        let work = try XCTUnwrap(history.session(around: t0 + 60 * step))
        XCTAssertEqual(work.start, t0)
        XCTAssertEqual(work.end, t0 + Double(count - 1) * step)
        XCTAssertEqual(work.shots, count)
        // Several moments at once: a stretch is walked once, and each moment keeps its own shot.
        let mail = t0 + Double(count) * step + 3600
        let many = history.sessions(around: [t0 + 10 * step, t0 + 60 * step + 5, mail, mail - 1800, t0 + 3 * step])
        XCTAssertEqual(many.map { $0?.shots }, [count, count, 1, nil, count])
        XCTAssertEqual(many.map { $0?.nearest.time }, [t0 + 10 * step, t0 + 60 * step, mail, nil, t0 + 3 * step])
        XCTAssertEqual(many[1]?.apps.map(\.name), ["Xcode"])
    }

    // MARK: Retention

    func testPruneRemovesWhatIsOlderAndKeepsTheRest() throws {
        let short = open(segmentFrames: 2)
        let day = 86_400.0
        for i in 0 ..< 3 { _ = short.record(frame(page: i + 1), time: t0 + Double(i), display: 1, app: "Old", title: "", settled: false) }
        for i in 0 ..< 3 { _ = short.record(frame(page: i + 5), time: t0 + 10 * day + Double(i), display: 1, app: "New", title: "", settled: false) }
        let all = indexed(short)
        XCTAssertEqual(all.count, 6)
        XCTAssertEqual(segments().count, 4)
        // What went is told by path, for the index to drop.
        XCTAssertEqual(short.prune(olderThan: t0 + 5 * day)?.sorted(), Array(all.prefix(3)).sorted())
        XCTAssertEqual(segments().count, 2)
        XCTAssertEqual(short.shots(from: 0, to: t0 + 20 * day).map(\.app), ["New", "New", "New"])
        XCTAssertEqual(indexed(short), Array(all.suffix(3)))
        for path in all.prefix(3) { XCTAssertNil(short.shot(atPath: path)) }
        // The segment being written is never pruned, whatever the cutoff.
        XCTAssertEqual(short.prune(olderThan: t0 + 100 * day), Array(all[3 ..< 5]))
        XCTAssertEqual(short.shots(from: 0, to: t0 + 20 * day).count, 1)
        XCTAssertNotNil(short.record(frame(page: 20), time: t0 + 10 * day + 5, display: 1, app: "New", title: "", settled: false))
        short.stop()
    }

    /// A recorder that starts again - Screenshots turned off and on, or the next launch - knows what
    /// the index already holds of the screen. With an empty gate each start indexed the screen it
    /// found one more time.
    func testARecorderThatStartsAgainDoesNotIndexWhatTheIndexHolds() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        _ = history.record(frame(page: 2), time: t0 + 5, display: 1, app: "Safari", title: "News", settled: false)
        XCTAssertEqual(indexed().count, 2)
        history.stop()
        let again = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 60, display: 1, app: "Safari", title: "News", settled: false))
        XCTAssertFalse(again.shot.indexed, "off and on: the shot is stored, in a new segment")
        history.stop()

        let launched = open()
        let first = try XCTUnwrap(launched.record(frame(page: 1), time: t0 + 120, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertFalse(first.shot.indexed, "the next launch")
        XCTAssertNil(launched.record(frame(page: 1), time: t0 + 125, display: 1, app: "Xcode", title: "a.swift", settled: true),
                     "nor when the user stops on it")
        let new = try XCTUnwrap(launched.record(frame(page: 3), time: t0 + 130, display: 1, app: "Notes", title: "", settled: false))
        XCTAssertTrue(new.shot.indexed)
        XCTAssertEqual(indexed(launched).count, 3)
        // Another display has shown none of it.
        let other = try XCTUnwrap(launched.record(frame(page: 1), time: t0 + 135, display: 2, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(other.shot.indexed)
        launched.stop()
    }

    // MARK: Deleting a span

    /// Forty shots, a key frame every thirty: a span cut from the middle of a run of changes, one
    /// that takes a key frame, one at each end, and the shots left decode to what they showed.
    func testADeletedSpanLeavesTheOtherShotsAsTheyWere() throws {
        var sources: [CVPixelBuffer] = []
        for i in 0 ..< 40 {
            let f = frame(page: 1 + i / 10, typed: i % 10)
            sources.append(f)
            XCTAssertNotNil(history.record(f, time: t0 + Double(i) * 5, display: 1, app: "App", title: "Window \(i)", settled: false))
        }
        history.stop()
        // One recording, a copy of it for each span.
        let recorded = dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + "-recorded", isDirectory: true)
        try FileManager.default.moveItem(at: dir, to: recorded)
        defer { try? FileManager.default.removeItem(at: recorded) }
        for (label, span) in [("inside a run", 12 ... 14), ("over a key frame", 28 ... 31), ("the first shots", 0 ... 2),
                              ("the last shots", 37 ... 39), ("one shot", 7 ... 7), ("a key frame alone", 30 ... 30)] {
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.copyItem(at: recorded, to: dir)
            let h = open()
            let before = indexed(h)
            let indexedInSpan = h.shots(from: t0 + Double(span.lowerBound) * 5, to: t0 + Double(span.upperBound) * 5).filter(\.indexed).count
            let deleted = h.delete(from: t0 + Double(span.lowerBound) * 5, to: t0 + Double(span.upperBound) * 5)
            XCTAssertEqual(deleted.shots, span.count, label)
            XCTAssertEqual(deleted.lost, 0, label)
            XCTAssertEqual(deleted.paths.count, indexedInSpan, label)
            XCTAssertTrue(Set(deleted.paths).isSubset(of: before), label)
            // The list a pass reads was kept beside the video (shots.catalog) and is true again.
            XCTAssertEqual(indexed(h), before.filter { !deleted.paths.contains($0) }, label)
            XCTAssertEqual(indexed(open()), indexed(h), label)
            for path in deleted.paths { XCTAssertNil(h.shot(atPath: path), label) }
            let kept = (0 ..< 40).filter { !span.contains($0) }
            // The open history stepping forward, and one that reads the file fresh stepping back.
            for (reader, backward) in [(h, false), (open(), true)] {
                let shots = reader.shots(from: t0, to: t0 + 1000)
                XCTAssertEqual(shots.map(\.title), kept.map { "Window \($0)" }, label)
                XCTAssertEqual(shots.map(\.index), Array(0 ..< kept.count), label)
                for (n, i) in backward ? Array(kept.enumerated().reversed()) : Array(kept.enumerated()) {
                    let back = try XCTUnwrap(reader.pixels(shots[n]), "\(label): shot \(i)")
                    XCTAssertLessThan(meanDifference(sources[i], back), 4, "\(label): shot \(i)")
                }
            }
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent("Screen History").path).sorted(),
                           segments() + ["shots.catalog"], "\(label): no copy is left beside the segment")
            XCTAssertEqual(segments().count, 1, label)
            // Recording goes on, in a new segment.
            XCTAssertNotNil(h.record(frame(page: 30), time: t0 + 500, display: 1, app: "App", title: "After", settled: false))
            XCTAssertEqual(h.shots(from: t0, to: t0 + 1000).count, kept.count + 1, label)
            h.stop()
        }
    }

    func testADeletedSpanTakesWholeSegmentsAndOnlyTheDisplayAsked() throws {
        let short = open(segmentFrames: 4)
        for i in 0 ..< 12 {
            _ = short.record(frame(page: i + 1), time: t0 + Double(i) * 5, display: 1, app: "One", title: "\(i)", settled: false)
            _ = short.record(frame(page: i + 50), time: t0 + Double(i) * 5 + 0.04, display: 2, app: "Two", title: "\(i)", settled: false)
        }
        XCTAssertEqual(indexed(short).count, 24)
        // One display's shot: the other display's of the same look stays, path and all.
        let one = short.delete(from: t0 + 5, to: t0 + 5, display: 1)
        XCTAssertEqual(one.shots, 1)
        XCTAssertEqual(one.paths.count, 1)
        XCTAssertEqual(short.shots(from: t0, to: t0 + 100).filter { $0.display == 2 }.count, 12)
        XCTAssertEqual(indexed(short).count, 23)
        // Every display, across three segments of each: the middle ones go as files.
        XCTAssertEqual(segments().count, 6)
        let span = short.delete(from: t0 + 10, to: t0 + 50)
        XCTAssertEqual(span.shots, 17, "the second display's shot of the last look was taken after the span ended")
        XCTAssertEqual(span.paths.count, 17)
        XCTAssertEqual(segments().count, 4)
        let left = short.shots(from: t0, to: t0 + 100)
        XCTAssertEqual(left.filter { $0.display == 1 }.map(\.title), ["0", "11"])
        XCTAssertEqual(left.filter { $0.display == 2 }.map(\.title), ["0", "1", "10", "11"])
        for s in left { XCTAssertNotNil(short.pixels(s)); XCTAssertEqual(short.path(of: s).flatMap { short.shot(atPath: $0) }, s) }
        XCTAssertEqual(indexed(short).count, 6)
        XCTAssertEqual(short.delete(from: t0 + 2000, to: t0 + 3000).shots, 0)
        // What was deleted is not remembered as indexed: the same screen is new again.
        let again = try XCTUnwrap(short.record(frame(page: 6), time: t0 + 200, display: 1, app: "One", title: "5", settled: false))
        XCTAssertTrue(again.shot.indexed)
        short.stop()
    }

    /// Two displays showing the same title in one second share a name; the later shot's is
    /// numbered, and the number is the shot's own: it does not move when the other shot goes.
    func testAShotsPathIsItsOwnWhenTwoShareAName() throws {
        let a = try XCTUnwrap(history.record(frame(page: 1), time: t0 + 0.25, display: 1, app: "Keynote", title: "Deck", settled: false))
        let b = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 0.31, display: 2, app: "Keynote", title: "Deck", settled: false))
        let pathA = try XCTUnwrap(a.path), pathB = try XCTUnwrap(b.path)
        XCTAssertEqual(pathB, String(pathA.dropLast(5)) + " 2.heic")
        XCTAssertEqual(history.shot(atPath: pathA), a.shot)
        XCTAssertEqual(history.shot(atPath: pathB), b.shot)
        let whole = try XCTUnwrap(history.picture(of: b.shot, maxDimension: 4096))
        XCTAssertEqual([whole.width, whole.height, whole.image.width], [width, height, width])
        XCTAssertEqual(history.delete(from: t0 + 0.25, to: t0 + 0.25, display: 1).paths, [pathA])
        XCTAssertNil(history.shot(atPath: pathA))
        XCTAssertEqual(history.shot(atPath: pathB)?.time, t0 + 0.31)
        XCTAssertEqual(indexed(), [pathB])
        history.stop()
        let reopened = open()
        XCTAssertEqual(indexed(reopened), [pathB])
        XCTAssertEqual(reopened.shot(atPath: pathB)?.display, 2)
        // The plain name is free again, and the next shot of that second takes it.
        let c = try XCTUnwrap(reopened.record(frame(page: 3), time: t0 + 0.6, display: 3, app: "Keynote", title: "Deck", settled: false))
        XCTAssertEqual(c.path, pathA)
        reopened.stop()
    }

    /// A name has eight numbers. A ninth shot of one second and one title is stored like any
    /// other and is not in the index: there is no path left for it.
    func testNoMoreThanEightShotsShareAName() throws {
        var paths: [String] = []
        for d in 1 ... 9 {
            let r = try XCTUnwrap(history.record(frame(page: d), time: t0 + Double(d) * 0.01, display: UInt32(d), app: "Keynote", title: "Deck", settled: false),
                                  "display \(d): \(history.lastFailure ?? "")")
            XCTAssertEqual(r.shot.indexed, d <= 8, "display \(d)")
            if let p = r.path { paths.append(p) }
        }
        XCTAssertEqual(Set(paths).count, 8)
        XCTAssertEqual(indexed().sorted(), paths.sorted())
        XCTAssertEqual(history.shots(from: t0, to: t0 + 1).count, 9)
    }

    /// The index reads a shot at a pixel a point; the shot is whole in the history.
    func testAShotsPictureAtAnySizeComesFromItsFrame() throws {
        let big = frame(page: 1, width: 2560, height: 1440)
        let r = try XCTUnwrap(history.record(big, time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false, scale: 2))
        let path = try XCTUnwrap(r.path)
        let shot = try XCTUnwrap(history.shot(atPath: path))
        let whole = try XCTUnwrap(history.picture(of: shot, maxDimension: 4096))
        XCTAssertEqual([whole.image.width, whole.image.height, whole.width, whole.height], [2560, 1440, 2560, 1440])
        let fitted = try XCTUnwrap(history.picture(of: shot, maxDimension: 2000))
        XCTAssertEqual([fitted.image.width, fitted.image.height, fitted.width], [2000, 1125, 2560])
        let read = try XCTUnwrap(history.picture(of: shot))
        XCTAssertEqual([read.image.width, read.image.height, read.width], [1280, 720, 2560])
        history.clear()
        XCTAssertNil(history.shot(atPath: path), "a cleared history has no shot at the path")
        XCTAssertNil(history.picture(of: shot))
    }

    func testClearLeavesNothingAndRecordingGoesOn() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        XCTAssertEqual(indexed().count, 1)
        XCTAssertGreaterThan(history.diskUse(), 0)
        history.clear()
        XCTAssertTrue(history.isEmpty)
        XCTAssertEqual(indexed(), [])
        XCTAssertEqual(history.diskUse(), 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent("Screen History").path), [])
        let again = try XCTUnwrap(history.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(again.shot.indexed, "nothing is remembered as indexed")
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).count, 1)
    }

    // MARK: Readers

    private final class Tally: @unchecked Sendable {
        private let lock = NSLock()
        private var wrong: [String] = [], frames = 0, nothing = 0, stopped = false
        func add(_ what: String) { lock.withLock { wrong.append(what) } }
        func count(frame: Bool) { lock.withLock { if frame { frames += 1 } else { nothing += 1 } } }
        func stop() { lock.withLock { stopped = true } }
        var isStopped: Bool { lock.withLock { stopped } }
        var read: (wrong: [String], frames: Int, nothing: Int) { lock.withLock { (wrong, frames, nothing) } }
    }

    /// Thirty looks, each a page of its own, five to a segment: the first ten on one day and the
    /// rest ten days later. Returns the frames as they went in.
    private func recordThirtyPages(_ h: ScreenHistory) -> [CVPixelBuffer] {
        var sources: [CVPixelBuffer] = []
        for i in 0 ..< 30 {
            let f = frame(page: 100 + i)
            sources.append(f)
            XCTAssertNotNil(h.record(f, time: thirtyTime(i), display: 1, app: "App", title: "Window \(i)", settled: false))
        }
        XCTAssertGreaterThan(meanDifference(sources[0], sources[1]), 12, "one page is not taken for the next")
        return sources
    }

    private func thirtyTime(_ i: Int) -> Double { t0 + (i < 10 ? 0 : 10 * 86_400) + Double(i) * 5 }

    /// A page of results that are shots is drawn several at a time, each a jump to another key
    /// frame. Every frame read beside others is the frame read alone, and when the reads are over
    /// one reader is kept, not one for each.
    func testFramesReadAtOnceAreTheFramesReadOneAtATime() throws {
        let h = open(segmentFrames: 5)
        defer { h.stop() }
        let sources = recordThirtyPages(h)
        let shots = everyShot(h)
        XCTAssertEqual(shots.count, 30)
        XCTAssertEqual(Set(shots.map(\.segment)).count, 6)
        let tally = Tally()
        let strides = [7, 11, 13, 17, 19, 23, 29, 1]       // each goes through every shot, in its own order
        for round in 0 ..< 3 {
            DispatchQueue.concurrentPerform(iterations: 8) { worker in
                for n in 0 ..< 30 {
                    let i = (n * strides[worker] + worker * 4 + round) % 30
                    guard let back = h.pixels(shots[i]) else { tally.add("round \(round) reader \(worker): no frame \(i)"); continue }
                    tally.count(frame: true)
                    if meanDifference(sources[i], back) >= 4 { tally.add("round \(round) reader \(worker): frame \(i) is another") }
                }
            }
            XCTAssertLessThanOrEqual(h.readersKept, 1, "round \(round)")
        }
        XCTAssertEqual(tally.read.wrong, [])
        XCTAssertEqual(tally.read.frames, 3 * 8 * 30)
        // And one at a time afterwards, back to front.
        for i in (0 ..< 30).reversed() { XCTAssertLessThan(meanDifference(sources[i], try XCTUnwrap(h.pixels(shots[i]))), 4, "frame \(i)") }
    }

    /// Shots deleted, pruned and cleared while they are being read: a read gives the shot's own
    /// frame or nothing, never another shot's, and the deletion is not kept waiting by the reads.
    func testReadsWhileShotsAreDeletedGiveTheShotOrNothing() throws {
        let h = open(segmentFrames: 5)
        let sources = recordThirtyPages(h)
        h.stop()
        let shots = everyShot(h)
        let tally = Tally()
        let group = DispatchGroup()
        let strides = [7, 11, 13, 17]
        for worker in 0 ..< 4 {
            DispatchQueue.global().async(group: group) {
                var n = 0
                while !tally.isStopped {
                    let i = (n * strides[worker] + worker * 4) % 30
                    n += 1
                    guard let back = h.pixels(shots[i]) else { tally.count(frame: false); continue }
                    tally.count(frame: true)
                    if self.meanDifference(sources[i], back) >= 4 { tally.add("reader \(worker): frame \(i) is another") }
                }
            }
        }
        func waitForReads(_ more: Int) {
            let from = tally.read.frames + tally.read.nothing
            let until = Date().addingTimeInterval(20)
            while tally.read.frames + tally.read.nothing < from + more, Date() < until { usleep(2000) }
        }
        waitForReads(60)
        XCTAssertGreaterThan(tally.read.frames, 0, "the readers were reading")
        XCTAssertEqual(tally.read.nothing, 0, "and every shot was there")
        // Two shots out of the middle of a segment: the three after them are encoded again.
        let deleted = h.delete(from: thirtyTime(21), to: thirtyTime(22))
        XCTAssertEqual([deleted.shots, deleted.lost], [2, 0])
        waitForReads(60)
        XCTAssertEqual(h.prune(olderThan: t0 + 5 * 86_400)?.count, 10)
        waitForReads(60)
        let left = everyShot(h)
        XCTAssertEqual(left.map(\.title), ((10 ..< 21).map { $0 } + (23 ..< 30).map { $0 }).map { "Window \($0)" })
        for (n, i) in ((10 ..< 21).map { $0 } + (23 ..< 30).map { $0 }).enumerated() {
            XCTAssertLessThan(meanDifference(sources[i], try XCTUnwrap(h.pixels(left[n]), "frame \(i)")), 4, "frame \(i)")
        }
        // A shot held from before its segment lost two: its place is another's now, or nobody's.
        XCTAssertNil(h.pixels(shots[22]))
        XCTAssertNil(h.pixels(shots[24]))
        XCTAssertNotNil(h.pixels(shots[20]))
        h.clear()
        waitForReads(60)
        tally.stop()
        XCTAssertEqual(group.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(tally.read.wrong, [])
        for s in shots + left { XCTAssertNil(h.pixels(s)) }
        XCTAssertEqual(h.readersKept, 0)
        XCTAssertEqual(segments(), [])
    }

    /// A segment rewritten behind a table read before it: the read finds the record is not where
    /// the table has it, reads the segment again, and gives the shot's frame or nothing. (The app
    /// has one history a folder; a second one here is the table that was not told.)
    func testAReadFindsItsSegmentRewrittenBehindItsTable() throws {
        var sources: [CVPixelBuffer] = []
        for i in 0 ..< 12 {
            let f = frame(page: 200 + i)
            sources.append(f)
            XCTAssertNotNil(history.record(f, time: t0 + Double(i) * 5, display: 1, app: "App", title: "Window \(i)", settled: false))
        }
        history.stop()
        let behind = open()
        let old = everyShot(behind)                 // its table, and no file open yet
        XCTAssertEqual(old.count, 12)
        let deleted = open().delete(from: t0 + 4 * 5, to: t0 + 6 * 5)
        XCTAssertEqual([deleted.shots, deleted.lost], [3, 0])
        // Before the deleted shots the records are where they were.
        for i in 0 ..< 4 { XCTAssertLessThan(meanDifference(sources[i], try XCTUnwrap(behind.pixels(old[i]))), 4, "frame \(i)") }
        XCTAssertEqual(everyShot(behind).count, 12, "the table is still the one read before the deletion")
        // After them they are not: nothing of another shot comes back, and the table is read again.
        for i in 4 ..< 12 { XCTAssertNil(behind.pixels(old[i]), "frame \(i)") }
        let now = everyShot(behind)
        XCTAssertEqual(now.map(\.title), ([0, 1, 2, 3] + Array(7 ..< 12)).map { "Window \($0)" })
        for (n, i) in ([0, 1, 2, 3] + Array(7 ..< 12)).enumerated() {
            XCTAssertLessThan(meanDifference(sources[i], try XCTUnwrap(behind.pixels(now[n]))), 4, "frame \(i)")
        }
        behind.stop()
    }

    // MARK: The list of indexed shots

    /// A recording with everything the update has to tell apart: looks typed on, two displays in
    /// one second under one title, a title with accents, a shot named for the second after its
    /// own. With three frames a segment it is three segments of the first display and one of the
    /// second, and seven of its nine shots are indexed.
    @discardableResult
    private func recordSevenIndexed(_ h: ScreenHistory) throws -> [ScreenHistory.Recorded] {
        let cafe = "Caf\u{E9}", title = "\u{D55C}\u{AE00} \u{622A}\u{56FE}"
        let looks: [(at: Double, display: UInt32, page: Int, typed: Int, app: String, title: String)] = [
            (0.25, 1, 1, 0, "Keynote", "Deck"), (0.31, 2, 6, 0, "Keynote", "Deck"),
            (5, 1, 1, 3, "Keynote", "Deck"),
            (10, 1, 2, 0, cafe, title), (10.02, 2, 7, 0, cafe, title),
            (15.9996, 1, 3, 0, "Xcode", "late"), (20, 1, 3, 2, "Xcode", "late"),
            (25, 1, 4, 0, "Safari", "News"), (30, 1, 5, 0, "Safari", "News"),
        ]
        let recorded = try looks.map {
            try XCTUnwrap(h.record(frame(page: $0.page, typed: $0.typed), time: t0 + $0.at, display: $0.display, app: $0.app, title: $0.title, settled: false))
        }
        XCTAssertEqual(recorded.map(\.shot.indexed), [true, true, false, true, true, true, false, true, true])
        return recorded
    }

    private func segmentData(in folder: URL? = nil) throws -> [String: Data] {
        var out: [String: Data] = [:]
        for name in segments(in: folder) {
            out[name] = try Data(contentsOf: (folder ?? dir).appendingPathComponent("Screen History").appendingPathComponent(name))
        }
        return out
    }

    /// Paths below the root, for histories in different folders.
    private func below(_ h: ScreenHistory) -> [[UInt8]] {
        bytes(indexed(h).map { String($0.dropFirst(h.root.count + 1)) })
    }

    /// What a pass is given is read from shots.catalog, which is only ever a copy of the video.
    func testTheListOfIndexedShotsIsTheSameFromTheVideoAndFromItsCatalog() throws {
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let live = indexed(h)                           // the segments being written: from their tables
        XCTAssertEqual(live.count, 7)
        XCTAssertEqual(Set(live.map { Array($0.utf8) }).count, 7)
        h.stop()
        let catalog = dir.appendingPathComponent("Screen History/shots.catalog")
        try? FileManager.default.removeItem(at: catalog)
        let cold = indexed(open())                      // closed segments: from the video
        XCTAssertEqual(bytes(cold), bytes(live))
        let written = try Data(contentsOf: catalog)
        XCTAssertEqual(indexed(open()), cold)           // and from the catalog
        XCTAssertEqual(indexed(h), cold)
        XCTAssertEqual(try Data(contentsOf: catalog), written)

        // POSITIVE CONTROL: the catalog is what is read. One that says something else, and is whole,
        // is believed.
        var forged = Data(written.dropLast(8))
        let name = try XCTUnwrap(forged.range(of: Data("Safari - News".utf8)))
        forged.replaceSubrange(name, with: Data("Forged - News".utf8))
        var sum: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in forged { sum = (sum ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        withUnsafeBytes(of: sum.littleEndian) { forged.append(contentsOf: $0) }
        try forged.write(to: catalog)
        XCTAssertEqual(indexed(open()).filter { $0.hasSuffix("Forged - News.heic") }.count, 1)

        // Damaged, cut short, not a catalog or gone: the list is the video's, and the catalog is
        // written again.
        for damage in ["a byte", "cut short", "not a catalog", "gone"] {
            switch damage {
            case "a byte": var d = written; d[d.count / 2] ^= 0x40; try d.write(to: catalog)
            case "cut short": try written.prefix(written.count - 9).write(to: catalog)
            case "not a catalog": try Data("hello".utf8).write(to: catalog)
            default: try FileManager.default.removeItem(at: catalog)
            }
            XCTAssertEqual(indexed(open()), cold, damage)
            XCTAssertEqual(try Data(contentsOf: catalog), written, damage)
        }

        // A segment that is gone is not listed from the catalog's memory of it.
        let first = try XCTUnwrap(segments().first)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("Screen History").appendingPathComponent(first))
        let left = indexed(open())
        XCTAssertEqual(left.count, 5)
        XCTAssertLessThan(try Data(contentsOf: catalog).count, written.count)
        XCTAssertEqual(indexed(open()), left)
    }

    // MARK: The files of 0.20.2

    /// 0.20.2 kept a picture file for every indexed shot, and the index holds each file's path.
    /// The update gives each shot that path, deletes the files, and leaves the video as this build
    /// would have recorded it.
    func testTheUpdateGivesEveryShotItsFilesPathAndDeletesTheFiles() throws {
        let fm = FileManager.default
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let paths = indexed(h)
        h.stop()
        let recorded = try segmentData()
        let files = try ScreenHistory0202.downgrade(h)
        XCTAssertEqual(files.count, 7)
        XCTAssertNotEqual(try segmentData(), recorded, "control: the segments are 0.20.2's now")
        let stills = dir.appendingPathComponent("Screenshots")
        let listed = ScreenHistory0202.listed(stills)
        XCTAssertEqual(bytes(listed.map { h.root + "/" + $0 }), bytes(paths), "this build names a shot as 0.20.2 named its file")
        XCTAssertEqual(listed.filter { $0.hasSuffix(" 2.heic") }.count, 2)

        // Before: nothing is known of what the index holds, and nothing is said.
        let old = open()
        XCTAssertTrue(old.stillsPending)
        XCTAssertNil(old.indexedShots())
        XCTAssertEqual(everyShot(old).count, 9)
        for shot in everyShot(old) { XCTAssertNil(old.path(of: shot)); XCTAssertFalse(shot.indexed, "indexed is having a path") }
        for path in paths { XCTAssertNil(old.shot(atPath: path)) }

        var steps: [String] = []
        XCTAssertEqual(old.migrateStills { steps.append($0) }, .done(kept: 7, dropped: 0))
        XCTAssertEqual(steps.filter { $0 == "header" }.count, 4)
        XCTAssertEqual(steps.filter { $0 == "shot" }.count, 7)
        XCTAssertEqual(Array(steps.suffix(2)), ["moved", "deleted"])
        XCTAssertFalse(old.stillsPending)
        XCTAssertFalse(fm.fileExists(atPath: stills.path))
        XCTAssertEqual(everyShot(old).filter(\.indexed).count, 7)
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: dir.path), ["Screen History"])
        XCTAssertEqual(try segmentData(), recorded, "the video is what this build would have recorded")
        XCTAssertEqual(bytes(indexed(old)), bytes(paths))
        for (rel, shot) in files {
            let path = old.root + "/" + rel
            let found = try XCTUnwrap(old.shot(atPath: path), rel)
            XCTAssertEqual([found.time, Double(found.display)], [shot.time, Double(shot.display)], rel)
            XCTAssertEqual(old.path(of: found).map { Array($0.utf8) }, Array(path.utf8), rel)
            XCTAssertNotNil(old.picture(of: found), rel)
        }
        XCTAssertEqual(old.migrateStills(), .none)
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: dir.path), ["Screen History"], "asked again, no folder is made")
        XCTAssertEqual(bytes(indexed(open())), bytes(paths))

        // Recording goes on beside what was taken in.
        let next = try XCTUnwrap(old.record(frame(page: 8), time: t0 + 40, display: 1, app: "Notes", title: "Next", settled: false))
        XCTAssertTrue(next.shot.indexed)
        XCTAssertEqual(indexed(old).count, 8)
        old.stop()
    }

    /// The recorder may be the first to ask: the files are taken in before a shot is recorded
    /// beside them, so that none is given a name one of them has.
    func testAShotRecordedBeforeTheUpdateRanTakesTheFilesInFirst() throws {
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let paths = indexed(h)
        try ScreenHistory0202.downgrade(h)
        let old = open()
        XCTAssertTrue(old.stillsPending)
        // The same second, app and title as the first shot of the first display.
        let twin = try XCTUnwrap(old.record(frame(page: 9), time: t0 + 0.6, display: 3, app: "Keynote", title: "Deck", settled: false))
        XCTAssertFalse(old.stillsPending)
        XCTAssertTrue(twin.shot.indexed)
        let plain = try XCTUnwrap(paths.first { old.shot(atPath: $0)?.time == t0 + 0.25 })
        XCTAssertEqual(twin.path, String(plain.dropLast(5)) + " 3.heic")
        XCTAssertEqual(bytes(indexed(old)), bytes(paths + [try XCTUnwrap(twin.path)]))
        old.stop()
    }

    /// A shot whose file is gone was taken out of the index by whoever deleted it, and stays out. A
    /// file no shot is behind (its segment was pruned) goes with the folder, and so does anything
    /// else in it.
    func testAShotWhoseFileIsGoneLeavesTheIndexAndAFileWithNoShotGoesWithTheFolder() throws {
        let fm = FileManager.default
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let paths = indexed(h)
        let lost = try XCTUnwrap(paths.first { h.shot(atPath: $0)?.time == t0 + 25 })
        try ScreenHistory0202.downgrade(h) { $0.time == self.t0 + 25 }
        let stills = dir.appendingPathComponent("Screenshots")
        let picture = try XCTUnwrap(ScreenCodec.cgImage(frame(page: 9)))
        let orphan = try ScreenHistory0202.writeStill(picture, in: stills, time: t0 + 500, app: "Orphan", title: "")
        try Data("x".utf8).write(to: orphan.deletingLastPathComponent().appendingPathComponent("notes.txt"))
        try Data("x".utf8).write(to: stills.appendingPathComponent(".DS_Store"))
        try Data("x".utf8).write(to: stills.appendingPathComponent("loose.heic"))
        let old = open()
        XCTAssertEqual(old.migrateStills(), .done(kept: 6, dropped: 1))
        XCTAssertEqual(everyShot(old).first { $0.time == t0 + 25 }?.indexed, false)
        XCTAssertEqual(everyShot(old).count, 9, "the shot is in the history as it was")
        XCTAssertEqual(bytes(indexed(old)), bytes(paths.filter { $0 != lost }))
        XCTAssertNil(old.shot(atPath: lost))
        XCTAssertNil(old.shot(atPath: old.root + "/" + orphan.deletingLastPathComponent().lastPathComponent + "/" + orphan.lastPathComponent))
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: dir.path), ["Screen History"])
        XCTAssertEqual(bytes(indexed(open())), bytes(paths.filter { $0 != lost }))
    }

    /// A restore or a copy can leave every file dated at the restore. Which file of a second was
    /// which shot's is then not known, and is not guessed: each shot takes a path no file had,
    /// which the index reads again from the frame, and the old rows go with their paths.
    func testFilesWhoseDatesWereLostAreNotGuessedAt() throws {
        let fm = FileManager.default
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let paths = indexed(h)
        try ScreenHistory0202.downgrade(h)
        let stills = dir.appendingPathComponent("Screenshots")
        for rel in ScreenHistory0202.listed(stills) {
            try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: t0 + 90_000)], ofItemAtPath: stills.appendingPathComponent(rel).path)
        }
        let old = open()
        XCTAssertEqual(old.migrateStills(), .done(kept: 7, dropped: 0))
        let now = indexed(old)
        XCTAssertEqual(now.count, 7)
        XCTAssertTrue(Set(bytes(now)).isDisjoint(with: bytes(paths)))
        for path in paths { XCTAssertNil(old.shot(atPath: path)) }
        // The next names of each second: after the one file of a shot alone, after the two of twins.
        XCTAssertEqual(now.map { String($0.dropLast(5).suffix(2)) }.sorted(), [" 2", " 2", " 2", " 3", " 3", " 4", " 4"])
        XCTAssertEqual(Set(now.compactMap { old.shot(atPath: $0)?.time }).count, 7)
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: dir.path), ["Screen History"])
        XCTAssertEqual(bytes(indexed(open())), bytes(now))
    }

    /// The files were named on the clock of the day they were written. The Mac that updates may be
    /// on another: each file's date beside its name says which clock that was.
    func testFilesNamedOnAnotherClockAreFoundByTheirDates() throws {
        let system = NSTimeZone.default
        defer { NSTimeZone.default = system }
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Asia/Tokyo"))
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        let paths = indexed(h)
        try ScreenHistory0202.downgrade(h)
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Pacific/Pago_Pago"))
        let old = open()
        XCTAssertEqual(old.migrateStills(), .done(kept: 7, dropped: 0))
        XCTAssertEqual(bytes(indexed(old)), bytes(paths))
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Europe/Berlin"))
        XCTAssertEqual(bytes(indexed(open())), bytes(paths))
    }

    /// 0.20.2 did not end a segment when the time zone changed: one segment can hold files named on
    /// two clocks, and a segment has one. The shots of the other clock take new paths, spelled on
    /// the segment's, and are read again from their frames.
    func testASegmentRecordedAcrossAChangeOfTimeZoneKeepsOneClock() throws {
        let system = NSTimeZone.default
        defer { NSTimeZone.default = system }
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Asia/Tokyo"))
        let a = try XCTUnwrap(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        let b = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 5, display: 1, app: "Safari", title: "News", settled: false))
        XCTAssertEqual(a.shot.segment, b.shot.segment)
        let pictures = try [a, b].map { try XCTUnwrap(history.picture(of: $0.shot)?.image) }
        history.stop()
        let stills = dir.appendingPathComponent("Screenshots")
        try ScreenHistory0202.writeStill(pictures[0], in: stills, time: t0, app: "Xcode", title: "a.swift")
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "Pacific/Pago_Pago"))
        let other = try ScreenHistory0202.writeStill(pictures[1], in: stills, time: t0 + 5, app: "Safari", title: "News")
        try ScreenHistory0202.strip(dir.appendingPathComponent("Screen History"))
        let old = open()
        XCTAssertEqual(old.migrateStills(), .done(kept: 2, dropped: 0))
        XCTAssertEqual(indexed(old), [try XCTUnwrap(a.path), try XCTUnwrap(b.path)], "both on Tokyo's clock, as this build records them")
        XCTAssertNil(old.shot(atPath: old.root + "/" + other.deletingLastPathComponent().lastPathComponent + "/" + other.lastPathComponent),
                     "the path the index held for the second: its row goes, and the shot is read again")
        XCTAssertEqual(old.shot(atPath: try XCTUnwrap(b.path))?.time, t0 + 5)
    }

    /// KILLED AFTER ANYTHING IT WRITES, THE UPDATE ENDS THE SAME. The disk is copied at every step
    /// as a kill there would leave it; each copy is then opened and updated as the next launch
    /// would, and must end with the same video, the same paths and no folder.
    func testTheUpdateKilledAfterAnyStepEndsTheSame() throws {
        let fm = FileManager.default
        let h = open(segmentFrames: 3)
        try recordSevenIndexed(h)
        // Every branch: a file that is gone, and one whose date was lost.
        let files = try ScreenHistory0202.downgrade(h) { $0.time == self.t0 + 25 }
        let stills = dir.appendingPathComponent("Screenshots")
        let undated = try XCTUnwrap(files.first { $0.shot.time == t0 + 10.02 }).rel
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: t0 + 90_000)], ofItemAtPath: stills.appendingPathComponent(undated).path)
        let before = try segmentData()

        let copies = dir.appendingPathComponent("killed")
        var steps: [String] = []
        let whole = open()
        let result = whole.migrateStills { step in
            let copy = copies.appendingPathComponent("\(steps.count)")
            try? fm.createDirectory(at: copy, withIntermediateDirectories: true)
            for item in ((try? fm.contentsOfDirectory(atPath: self.dir.path)) ?? []) where item != "killed" {
                try? fm.copyItem(at: self.dir.appendingPathComponent(item), to: copy.appendingPathComponent(item))
            }
            steps.append(step)
        }
        XCTAssertEqual(result, .done(kept: 6, dropped: 1))
        XCTAssertEqual(steps.count, 4 + 7 + 2)
        let want = (video: try segmentData(), paths: below(whole))
        XCTAssertEqual(want.paths.count, 6)
        XCTAssertEqual(want.paths.filter { $0.suffix(7).elementsEqual(Array(" 3.heic".utf8)) }.count, 1, "the undated twin's new path")

        // Controls: the first copy is part-way, and a copied file keeps its date (the update reads it).
        XCTAssertNotEqual(try segmentData(in: copies.appendingPathComponent("0")), want.video)
        XCTAssertNotEqual(try segmentData(in: copies.appendingPathComponent("0")), before)
        let kept = try XCTUnwrap(files.first { $0.shot.time == t0 + 0.25 })
        let date = try XCTUnwrap(fm.attributesOfItem(atPath: copies.appendingPathComponent("0/Screenshots/" + kept.rel).path)[.modificationDate] as? Date)
        XCTAssertEqual(date.timeIntervalSince1970, t0 + 0.25, accuracy: 0.001)

        for (i, step) in steps.enumerated() {
            let copy = copies.appendingPathComponent("\(i)")
            let label = "killed after step \(i) (\(step))"
            let again = open(in: copy)
            let result = again.migrateStills()
            if step == "moved" || step == "deleted" {
                XCTAssertEqual(result, .none, label)
            } else if case .done = result {} else {
                XCTFail("\(label): \(result)")
            }
            XCTAssertEqual(try fm.contentsOfDirectory(atPath: copy.path), ["Screen History"], label)
            XCTAssertEqual(Set(try fm.contentsOfDirectory(atPath: copy.appendingPathComponent("Screen History").path)).subtracting(segments(in: copy)),
                           [], label)
            XCTAssertEqual(try segmentData(in: copy), want.video, label)
            XCTAssertEqual(below(again), want.paths, label)
            XCTAssertEqual(below(open(in: copy)), want.paths, label)
        }
    }
}
