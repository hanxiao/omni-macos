import XCTest
import CoreVideo
import CoreGraphics
@testable import OmniKit

/// The screen history: every changed shot in a video segment, the new ones also as files for the
/// index. These pin what is stored, what is indexed, and that a history survives being reopened,
/// cut short and pruned.
final class ScreenHistoryTests: XCTestCase {
    private var dir: URL!
    private var history: ScreenHistory!
    private let width = 640, height = 400

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("shots-\(UUID().uuidString)", isDirectory: true)
        history = open()
    }

    override func tearDownWithError() throws {
        history.stop()
        try? FileManager.default.removeItem(at: dir)
    }

    private func open(segmentFrames: Int = ScreenHistory.maxSegmentFrames) -> ScreenHistory {
        ScreenHistory(history: dir.appendingPathComponent("Screen History"), stills: dir.appendingPathComponent("Screenshots"),
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

    private func stills() -> [URL] {
        let root = dir.appendingPathComponent("Screenshots")
        guard let walk = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return [] }
        return walk.compactMap { $0 as? URL }.filter { $0.pathExtension == "heic" }.sorted { $0.path < $1.path }
    }

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

    func testAScreenThatDidNotChangeIsNotStored() throws {
        XCTAssertNotNil(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNil(history.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertNotNil(history.record(frame(page: 1, typed: 3), time: t0 + 10, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).map(\.time), [t0, t0 + 10])
    }

    func testTheFirstShotAndANewPageAreIndexedAndTypingIsNot() throws {
        let first = try XCTUnwrap(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(first.shot.indexed)
        XCTAssertNotNil(first.still)
        let typing = try XCTUnwrap(history.record(frame(page: 1, typed: 6), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertFalse(typing.shot.indexed)
        XCTAssertNil(typing.still)
        XCTAssertLessThan(typing.novelty, ScreenGate.working)
        XCTAssertGreaterThan(typing.novelty, ScreenGate.settled)
        let page = try XCTUnwrap(history.record(frame(page: 2), time: t0 + 10, display: 1, app: "Safari", title: "News", settled: false))
        XCTAssertTrue(page.shot.indexed)
        XCTAssertEqual(stills().count, 2)
    }

    /// Writing the picture file of an indexed shot must not cost the video its next frame: a typed
    /// line after it is still a typed line, not the whole screen again.
    func testTheShotAfterAnIndexedOneIsStillSmall() throws {
        for (w, h) in [(640, 400), (1920, 1080)] {
            let first = try XCTUnwrap(history.record(frame(page: 1, width: w, height: h), time: t0, display: UInt32(w), app: "Xcode", title: "a.swift", settled: false))
            XCTAssertNotNil(first.still)
            let typing = try XCTUnwrap(history.record(frame(page: 1, typed: 4, width: w, height: h), time: t0 + 5, display: UInt32(w), app: "Xcode", title: "a.swift", settled: false))
            XCTAssertNil(typing.still)
            XCTAssertLessThan(typing.bytes * 4, first.bytes, "\(w)x\(h)")
        }
    }

    func testGoingBackToAnIndexedWindowIsNotNew() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        _ = history.record(frame(page: 2), time: t0 + 5, display: 1, app: "Safari", title: "News", settled: false)
        let back = try XCTUnwrap(history.record(frame(page: 1), time: t0 + 10, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertFalse(back.shot.indexed, "the shot is stored: the screen changed")
        XCTAssertEqual(stills().count, 2)
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
        XCTAssertNotNil(same.still)
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).map(\.indexed), [true, true])
        XCTAssertNil(history.record(frame(page: 1, typed: 6), time: t0 + 40, display: 1, app: "Xcode", title: "a.swift", settled: true),
                     "and only once")
        // The screen moved on since the last shot: a new shot, indexed.
        let moved = try XCTUnwrap(history.record(frame(page: 1, typed: 12), time: t0 + 60, display: 1, app: "Xcode", title: "a.swift", settled: true))
        XCTAssertEqual(moved.shot.time, t0 + 60)
        XCTAssertTrue(moved.shot.indexed)
        XCTAssertEqual(stills().count, 3)
        // The flag written into the stored record is what a new reader sees.
        history.stop()
        XCTAssertEqual(open().shots(from: t0, to: t0 + 100).map(\.indexed), [true, true, true])
    }

    // MARK: The file the index reads

    func testAnIndexedShotIsAScreenCaptureFileDatedAtTheShot() throws {
        let big = frame(page: 1, width: 2560, height: 1440)
        let r = try XCTUnwrap(history.record(big, time: t0, display: 1, app: "Xcode", title: "OmniKit/Sidebar.swift: edited\n", settled: false))
        let url = try XCTUnwrap(r.still)
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        let stamp = f.string(from: Date(timeIntervalSince1970: t0))
        XCTAssertEqual(url.lastPathComponent, "\(stamp) Xcode - OmniKit Sidebar.swift edited.heic")
        XCTAssertEqual(url.deletingLastPathComponent().lastPathComponent, String(stamp.prefix(10)))
        XCTAssertTrue(Indexer.isScreenshot(url.path), "the indexer reads its text as a screenshot's")
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual((attributes[.modificationDate] as? Date)?.timeIntervalSince1970, t0)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(max(image.width, image.height), ScreenHistory.stillDimension)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path).count, 1,
                       "no temporary file is left beside it")
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
        let file = folder.appendingPathComponent(try XCTUnwrap(FileManager.default.contentsOfDirectory(atPath: folder.path).first))
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
        XCTAssertEqual(stills().count, 6)
        XCTAssertEqual(short.prune(olderThan: t0 + 5 * day), 5)      // 2 segments, 3 files
        XCTAssertEqual(short.shots(from: 0, to: t0 + 20 * day).map(\.app), ["New", "New", "New"])
        XCTAssertEqual(stills().count, 3)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent("Screenshots").path).count, 1,
                       "the emptied day folder goes too")
        // The segment being written is never pruned, whatever the cutoff.
        short.prune(olderThan: t0 + 100 * day)
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
        XCTAssertEqual(stills().count, 2)
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
        XCTAssertEqual(stills().count, 3)
        // Another display has shown none of it.
        let other = try XCTUnwrap(launched.record(frame(page: 1), time: t0 + 135, display: 2, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(other.shot.indexed)
        launched.stop()
    }

    /// What the gate reads back is what the index holds: a shot whose file is gone is not in it.
    func testAScreenWhoseFileIsGoneIsIndexedAgain() throws {
        let first = try XCTUnwrap(history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false))
        history.stop()
        try FileManager.default.removeItem(at: XCTUnwrap(first.still))
        let again = try XCTUnwrap(open().record(frame(page: 1), time: t0 + 60, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(again.shot.indexed)
    }

    func testClearLeavesNothingAndRecordingGoesOn() throws {
        _ = history.record(frame(page: 1), time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false)
        XCTAssertGreaterThan(history.diskUse().history, 0)
        history.clear()
        XCTAssertTrue(history.isEmpty)
        XCTAssertEqual(stills(), [])
        XCTAssertEqual(history.diskUse().history, 0)
        let again = try XCTUnwrap(history.record(frame(page: 1), time: t0 + 5, display: 1, app: "Xcode", title: "a.swift", settled: false))
        XCTAssertTrue(again.shot.indexed, "nothing is remembered as indexed")
        XCTAssertEqual(history.shots(from: t0, to: t0 + 60).count, 1)
    }
}
