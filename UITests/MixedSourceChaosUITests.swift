import XCTest
import AppKit
import AVFoundation

/// Chaotic UI exercise over a MIXED-MODALITY corpus, driven while the index churns.
///
/// `ChaosUITests` covers a text-only corpus. This one exists for the ingestion refactor: every
/// modality now reaches the store through one `decode()` and one `ContentSource`, so the shape that
/// can break is no longer "text search misbehaves" but "one modality's rows disagree with another's
/// after the paths were merged". Text, images of several sizes, and a video are indexed together,
/// then the UI is driven across list and grid (the two views that read `hit.width/height`, which is
/// exactly the metadata the refactor made a source-owned decision), with kind filters that force
/// results from one modality at a time.
///
/// Isolation is `ChaosUITests`': launch arguments land in the NSUserDefaults ARGUMENT domain, which
/// is process-local and never written back, so a run cannot touch a real install's index or roots.
/// The Photos channel is deliberately NOT exercised - it needs TCC authorization that a test
/// machine may not have granted, and a prompt would hang the run. Its logic is covered by
/// `ContentSourceTests` and `PhotoTargetSizeTests`, which need no library.
final class MixedSourceChaosUITests: XCTestCase {

    private var root = URL(fileURLWithPath: NSTemporaryDirectory())
    private var corpus = URL(fileURLWithPath: NSTemporaryDirectory())
    private var scratchDB = URL(fileURLWithPath: NSTemporaryDirectory())
    private var churnStop = false

    override func setUpWithError() throws {
        continueAfterFailure = false
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-mixedchaos-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        scratchDB = root.appendingPathComponent("db", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratchDB, withIntermediateDirectories: true)

        for i in 0 ..< 60 {
            let body = (0 ..< 40).map { l in
                "Line \(l) of document \(i): distributed vector search over quantized replicas, "
                + "porsche sports car, quarterly revenue, memory budget, recipe with tomatoes."
            }.joined(separator: "\n")
            try body.write(to: corpus.appendingPathComponent("doc\(i).txt"), atomically: true, encoding: .utf8)
        }

        // Images ACROSS the maxImageDimension boundary (1568). The decode reduces a large one and
        // leaves a small one alone, and after the refactor both go through the same source call -
        // so both sizes have to survive indexing and render in the grid.
        for (i, side) in [320, 900, 1600, 2400].enumerated() {
            try writePNG(corpus.appendingPathComponent("pic\(i)-\(side).png"), width: side, height: side * 3 / 4)
        }
        try writeVideo(corpus.appendingPathComponent("clip.mov"))
    }

    override func tearDownWithError() throws {
        churnStop = true
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Corpus builders

    private func writePNG(_ url: URL, width: Int, height: Int) throws {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw XCTSkip("could not create a bitmap context")
        }
        // Some structure, so distinct images do not collapse into one content-dedup group.
        for b in 0 ..< 24 {
            ctx.setFillColor(red: CGFloat((b * 37) % 255) / 255.0,
                             green: CGFloat((b * 91 + width) % 255) / 255.0,
                             blue: CGFloat((b * 53 + height) % 255) / 255.0, alpha: 1)
            ctx.fill(CGRect(x: CGFloat(b) * CGFloat(width) / 24.0, y: 0,
                            width: CGFloat(width) / 24.0, height: CGFloat(height)))
        }
        guard let img = ctx.makeImage() else { throw XCTSkip("could not render a bitmap") }
        let rep = NSBitmapImageRep(cgImage: img)
        guard let data = rep.representation(using: .png, properties: [:]) else { throw XCTSkip("no PNG encoder") }
        try data.write(to: url)
    }

    /// A short H.264 clip, so the video modality is genuinely present rather than mocked.
    private func writeVideo(_ url: URL) throws {
        let w = 320, h = 240
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else {
            throw XCTSkip("no AVAssetWriter")
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        guard writer.canAdd(input) else { throw XCTSkip("writer rejected the input") }
        writer.add(input)
        guard writer.startWriting() else { throw XCTSkip("writer did not start") }
        writer.startSession(atSourceTime: .zero)

        for f in 0 ..< 24 {
            var pb: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool,
                  CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess,
                  let buf = pb else { break }
            CVPixelBufferLockBaseAddress(buf, [])
            if let base = CVPixelBufferGetBaseAddress(buf),
               let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) {
                ctx.setFillColor(red: CGFloat(f) / 24.0, green: 0.35, blue: 0.7, alpha: 1)
                ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            }
            CVPixelBufferUnlockBaseAddress(buf, [])
            while !input.isReadyForMoreMediaData { usleep(5_000) }
            adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: 12))
        }
        input.markAsFinished()
        let done = expectation(description: "video written")
        writer.finishWriting { done.fulfill() }
        wait(for: [done], timeout: 30)
    }

    // MARK: - Driving

    private func launchIsolated() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = [
            "-omni.dbDir", scratchDB.path,
            "-omni.roots", "(\"\(corpus.path)\")",
            "-omni.serving.enabled", "NO",
            "-omni.uiChaos", "YES",
        ]
        app.launch()
        return app
    }

    /// Churn across MODALITIES while the UI is driven: text edits, a new image, deletions and a
    /// rename, so reconcile is handling several kinds at once rather than only text.
    private func startChurn() {
        churnStop = false
        Thread.detachNewThread { [corpus] in
            var n = 0
            let fm = FileManager.default
            while !self.churnStop {
                switch n % 5 {
                case 0:
                    let f = corpus.appendingPathComponent("doc\(n % 60).txt")
                    if let h = try? FileHandle(forWritingTo: f) {
                        h.seekToEndOfFile(); h.write(Data("\nappended \(n)\n".utf8)); try? h.close()
                    }
                case 1:
                    try? "fresh document \(n) about metal kernels"
                        .write(to: corpus.appendingPathComponent("new\(n).txt"), atomically: true, encoding: .utf8)
                case 2:
                    try? self.writePNG(corpus.appendingPathComponent("churn\(n).png"),
                                       width: 200 + (n % 5) * 130, height: 180 + (n % 3) * 90)
                case 3:
                    try? fm.removeItem(at: corpus.appendingPathComponent("new\(n - 2).txt"))
                default:
                    try? fm.moveItem(at: corpus.appendingPathComponent("churn\(n - 2).png"),
                                     to: corpus.appendingPathComponent("moved\(n).png"))
                }
                n += 1
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
    }

    private func focusSearch(_ app: XCUIApplication) {
        let field = app.windows.firstMatch.searchFields.firstMatch
        if field.exists, field.isHittable { field.click() } else { app.typeKey("f", modifierFlags: .command) }
        usleep(150_000)
    }

    private func clearQuery(_ app: XCUIApplication) {
        focusSearch(app)
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete, modifierFlags: [])
    }

    func testMixedModalityChaosUnderIndexing() throws {
        let app = launchIsolated()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 60), "app did not come up")
        startChurn()

        // Kind filters force the UI to render rows from ONE modality at a time - the case where a
        // modality whose metadata regressed shows up as an empty or broken row rather than being
        // hidden among text hits.
        let queries = ["porsche", "type:image", "type:video", "type:text metal kernels",
                       "memory budget", "type:image colorful stripes", "recipe tomatoes"]
        let deadline = Date().addingTimeInterval(180)
        var rounds = 0

        while Date() < deadline {
            switch rounds % 7 {
            case 0:
                // Search as you type, then abandon part way.
                clearQuery(app)
                for ch in queries[rounds % queries.count] {
                    app.typeText(String(ch))
                    usleep(UInt32.random(in: 25_000 ... 90_000))
                }
                usleep(UInt32.random(in: 200_000 ... 800_000))
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 1:
                // Flip between list and grid WITH media on screen. Grid reads the stored pixel
                // size for every cell, so a modality whose meta regressed shows up here.
                clearQuery(app)
                app.typeText("type:image")
                usleep(900_000)
                for _ in 0 ..< 3 {
                    app.typeKey("1", modifierFlags: .command); usleep(350_000)
                    app.typeKey("2", modifierFlags: .command); usleep(350_000)
                }
            case 2:
                // Modality switch storm: never let one kind's results settle before asking for the
                // next. Each switch re-runs the query against a different slice of the store.
                for q in ["type:image", "type:video", "type:text", "type:image"] {
                    clearQuery(app)
                    app.typeText(q)
                    usleep(220_000)
                }
            case 3:
                // Walk results and open a preview - the path that materializes an actual item.
                clearQuery(app)
                app.typeText("type:image")
                usleep(800_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
                for _ in 0 ..< Int.random(in: 2 ... 5) {
                    app.typeKey(XCUIKeyboardKey.downArrow, modifierFlags: [])
                    usleep(90_000)
                }
                app.typeText(" ")            // Quick Look
                usleep(700_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 4:
                // Toolbar controls by position, scoped to the toolbar so the driver's own
                // "_XCUI:CloseWindow" is never clicked (it took the app down in ChaosUITests).
                let buttons = app.windows.firstMatch.toolbars.firstMatch.buttons
                let safe = (0 ..< buttons.count).map { buttons.element(boundBy: $0) }.filter {
                    guard $0.exists else { return false }
                    return !$0.identifier.hasPrefix("_XCUI")
                }
                if let b = safe.randomElement(), b.isHittable { b.click() }
                usleep(300_000)
                app.typeKey(XCUIKeyboardKey.escape, modifierFlags: [])
            case 5:
                // Escape spam then refocus: the state machine sees stop, stop, stop, start.
                for _ in 0 ..< 5 { app.typeKey(XCUIKeyboardKey.escape, modifierFlags: []); usleep(60_000) }
                clearQuery(app)
                app.typeText("m")
            default:
                let rows = app.windows.firstMatch.outlines.firstMatch.cells
                if rows.count > 0 {
                    let r = rows.element(boundBy: Int.random(in: 0 ..< min(rows.count, 4)))
                    if r.exists, r.isHittable { r.click() }
                }
                usleep(300_000)
            }

            rounds += 1
            // "The app died" and "the driver closed its window" look identical from here.
            guard app.state == .runningForeground else {
                XCTFail("app left the foreground after round \(rounds) (state \(app.state.rawValue)); "
                        + "windows=\(app.windows.count)")
                return
            }
        }

        churnStop = true
        sleep(2)

        // Still WORKING at the end, not merely still alive - and working for every modality, which
        // is the property the single decode path is responsible for.
        for q in ["type:text", "type:image", "porsche"] {
            clearQuery(app)
            app.typeText(q)
            sleep(3)
            XCTAssertEqual(app.state, .runningForeground, "app died running \(q)")
            XCTAssertTrue(app.windows.firstMatch.exists, "window gone running \(q)")
        }
        print("[mixed-chaos] completed \(rounds) interaction rounds")
    }
}
