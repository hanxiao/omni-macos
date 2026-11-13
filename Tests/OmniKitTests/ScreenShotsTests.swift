import XCTest
import AppKit
import CoreText
import CoreVideo
@testable import OmniKit

/// THE SCREEN HISTORY IN THE INDEX. A shot has a path and no file: the indexer asks the history for
/// everything it asks a file system for. These run the real passes over a real history, with a
/// counting model, and hold an index built from 0.20.2's picture files to going on as it was.
final class ScreenShotsTests: XCTestCase {
    final class Counting: Embedder, @unchecked Sendable {
        let inner = EnrichmentTests.Fake()
        var dim: Int { inner.dim }
        var texts = 0, pictures = 0
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { texts += 1; return inner.embedText(text, as: type) }
        func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { texts += t.count; return inner.embedTextBatch(t, as: type) }
        func embedImage(_ image: CGImage) -> [Float]? { pictures += 1; return inner.embedImage(image) }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { pictures += raws.count; return inner.embedImages(raws) }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { pictures += 1; return inner.embedVideoFrames(frames) }
        func embedAudio(_ url: URL) -> [Float]? { inner.embedAudio(url) }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { inner.embedAudioMel(mel, frames: frames) }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { inner.embedAudioMelBatch(mels, frames: frames) }
    }

    private var dir: URL!
    private var history: ScreenHistory!
    private var store: VectorStore!
    private var registered: [ScreenHistory] = []
    private let t0 = 1_800_000_000.0

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("shotindex-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        history = open()
        store = try VectorStore(dbURL: dir.appendingPathComponent("db/index.sqlite"))
    }

    override func tearDownWithError() throws {
        for h in registered { ScreenShots.remove(h); h.stop() }
        history.stop()
        store.close()
        try? FileManager.default.removeItem(at: dir)
    }

    private func open() -> ScreenHistory {
        ScreenHistory(history: dir.appendingPathComponent("Screen History"), stills: dir.appendingPathComponent("Screenshots"))
    }

    private func register(_ h: ScreenHistory) {
        ScreenShots.add(h)
        registered.append(h)
    }

    private var root: URL { URL(fileURLWithPath: history.root) }

    /// A 1280x800 desktop: lines of text over a page of bars laid out from `page`, so that two
    /// pages are two looks to the recorder whatever their words are.
    private func screen(_ lines: [String], page: Int, width w: Int = 1280, height h: Int = 800) -> CVPixelBuffer {
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
        for row in 0 ..< 18 {
            var x = 40 + next() % 60
            while x < w - 120 {
                let len = 30 + next() % 140
                cg.fill(CGRect(x: x, y: 40 + row * 18, width: min(len, w - 40 - x), height: 9))
                x += len + 20
            }
        }
        let attr = NSAttributedString(string: lines.joined(separator: "\n"), attributes: [.font: NSFont.systemFont(ofSize: 28)])
        let frame = CTFramesetterCreateFrame(CTFramesetterCreateWithAttributedString(attr), CFRange(location: 0, length: 0),
                                             CGPath(rect: CGRect(x: 80, y: 430, width: 1100, height: 320), transform: nil), nil)
        CTFrameDraw(frame, cg)
        return pixels
    }

    private struct Look { let at: Double; let app: String; let title: String; let lines: [String]; let code: String }
    private var looks: [Look] {
        [Look(at: 0.25, app: "Mail", title: "Booking", lines: ["Booking confirmed", "Confirmation code: K7Q2LM", "Seat 32A"], code: "K7Q2LM"),
         Look(at: 5, app: "Caf\u{E9}", title: "\u{D55C}\u{AE00} \u{622A}\u{56FE}", lines: ["Order #114-3328-9912 has shipped"], code: "114-3328-9912"),
         Look(at: 10, app: "Notes", title: "Door", lines: ["The garage door code", "is ZX81QP until Friday"], code: "ZX81QP")]
    }

    /// Records the three looks; returns each one's path.
    @discardableResult
    private func record(_ h: ScreenHistory? = nil) throws -> [String] {
        try looks.enumerated().map { i, look in
            let r = try XCTUnwrap((h ?? history).record(screen(look.lines, page: i + 1), time: t0 + look.at, display: 1,
                                                        app: look.app, title: look.title, settled: false))
            XCTAssertTrue(r.shot.indexed, "look \(i) was not taken for the index; the test proves nothing")
            return try XCTUnwrap(r.path)
        }
    }

    private func pass(_ indexer: Indexer, _ settings: IndexSettings = IndexSettings(), roots: [URL]? = nil) {
        let done = expectation(description: "pass")
        indexer.resetCancelled()
        indexer.index(roots: roots ?? [root], settings: settings) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    private struct Row: Equatable { let kind: String; let snippet: String; let locator: String; let embedding: [Float] }
    private func rows(_ path: String) -> [Row]? {
        store.storedChunks(path: path)?.map { Row(kind: $0.kind, snippet: $0.snippet, locator: $0.locator, embedding: $0.embedding) }
    }
    private func text(_ path: String) -> String {
        (rows(path) ?? []).filter { $0.kind == "text" }.map(\.snippet).joined(separator: " ")
    }
    private func files(_ paths: [String]) -> [String: [Double]] {
        store.storedFiles(paths: Set(paths)).mapValues { [$0.modified, Double($0.size)] }
    }

    // MARK: The registry and the source

    func testAPathUnderARegisteredRootIsAShotAndTheHistoryAnswersForIt() throws {
        let paths = try record()
        XCTAssertFalse(ScreenShots.isShotPath(paths[0]), "control: nothing is registered yet")
        if case .unknown = ScreenShots.presence(paths[0]) {} else { XCTFail("an unregistered path is not known to be gone") }
        register(history)
        XCTAssertTrue(ScreenShots.isShotPath(paths[0]))
        XCTAssertFalse(ScreenShots.isShotPath(history.root), "the root is not a shot")
        XCTAssertFalse(ScreenShots.isShotPath(history.root + "x/a.heic"))
        XCTAssertTrue(ScreenShots.history(holding: paths[0]) === history)
        XCTAssertTrue(ScreenShots.history(root: history.root) === history)
        XCTAssertNil(ScreenShots.history(root: dir.path))
        guard case .held(let entry) = ScreenShots.presence(paths[0]) else { return XCTFail("not held") }
        XCTAssertEqual([entry.path, "\(entry.time)"], [paths[0], "\(t0 + 0.25)"])
        XCTAssertGreaterThan(entry.bytes, 0)
        XCTAssertFalse(ScreenShots.isGone(paths[0]))
        XCTAssertTrue(ScreenShots.isGone(history.root + "/2027-01-15/2027-01-15 09.00.00 Nobody - Nothing.heic"))
        XCTAssertTrue(ScreenShots.isGone(history.root + "/not a shot's name"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: paths[0]), "no file is behind the path")
        // The folders a search was scoped to under 0.20.2: the root and a day in it, and nothing else.
        let day = (paths[0] as NSString).deletingLastPathComponent
        XCTAssertTrue(ScreenShots.isFolder(history.root))
        XCTAssertTrue(ScreenShots.isFolder(day))
        XCTAssertFalse(ScreenShots.isFolder(paths[0]), "a shot is not a folder")
        XCTAssertFalse(ScreenShots.isFolder(day + "/gone.heic"), "nor is a shot that is gone")
        XCTAssertFalse(ScreenShots.isFolder(history.root + "x"))
        XCTAssertFalse(ScreenShots.isFolder(dir.path))
        ScreenShots.remove(history)
        XCTAssertFalse(ScreenShots.isFolder(history.root), "control: only while registered")
        XCTAssertFalse(ScreenShots.isShotPath(paths[0]))
    }

    func testTheSourceGivesAShotsPictureAtAPixelAPointAndAKeyOfItsOwn() throws {
        register(history)
        // A screen with something on it: a blank one is stored and not indexed (ScreenGate).
        let big = screen(["Retina"], page: 1, width: 2560, height: 1440)
        let a = try XCTUnwrap(history.record(big, time: t0, display: 1, app: "Xcode", title: "a.swift", settled: false, scale: 2))
        let b = try XCTUnwrap(history.record(screen(["x"], page: 2), time: t0 + 5, display: 2, app: "Xcode", title: "a.swift", settled: false))
        let fa = CrawledFile(path: try XCTUnwrap(a.path), modified: t0, size: 1), fb = CrawledFile(path: try XCTUnwrap(b.path), modified: t0 + 5, size: 1)
        let settings = IndexSettings()
        let source = ContentSources.source(for: fa)
        XCTAssertTrue(source is ScreenContentSource)
        XCTAssertFalse(ContentSources.source(for: CrawledFile(path: dir.path + "/a.heic", modified: 0, size: 1)) is ScreenContentSource)
        let probe = try XCTUnwrap(source.probe(fa, settings: settings))
        XCTAssertEqual(probe.kind, .image)
        XCTAssertEqual([probe.width, probe.height], [1280, 720], "a Retina display's shot at a pixel a point")
        guard case .images(let images) = source.content(fa, kind: .image, settings: settings) else { return XCTFail("no picture") }
        XCTAssertEqual(images.map { [$0.width, $0.height] }, [[1280, 720]])
        var small = settings
        small.maxImageDimension = 640
        guard case .images(let fitted) = source.content(fa, kind: .image, settings: small) else { return XCTFail("no picture") }
        XCTAssertEqual(fitted.map { [$0.width, $0.height] }, [[640, 360]])

        let ka = try XCTUnwrap(source.contentKey(fa, kind: .image, dim: 8, chunkOverlap: 0, settings: settings))
        let kb = try XCTUnwrap(ContentSources.source(for: fb).contentKey(fb, kind: .image, dim: 8, chunkOverlap: 0, settings: settings))
        XCTAssertNotEqual(ka, kb)
        XCTAssertNotEqual(ka, source.contentKey(fa, kind: .image, dim: 8, chunkOverlap: 0, settings: small), "another size is another reading")
        var noText = settings
        noText.screenText = false
        XCTAssertNotEqual(ka, source.contentKey(fa, kind: .image, dim: 8, chunkOverlap: 0, settings: noText), "and so is one with no text")
        XCTAssertNil(source.contentKey(fa, kind: .text, dim: 8, chunkOverlap: 0, settings: settings))

        // A shot the history has let go of: not read, which is not "empty".
        let gone = CrawledFile(path: history.root + "/2027-01-15/2027-01-15 09.00.00 Nobody - Nothing.heic", modified: 0, size: 1)
        XCTAssertNil(ContentSources.source(for: gone).probe(gone, settings: settings))
        if case .unreadable = ContentSources.source(for: gone).content(gone, kind: .image, settings: settings) {} else { XCTFail("not unreadable") }

        // The key is the shot's, whoever opens the history.
        history.stop()
        ScreenShots.remove(history)
        let again = open()
        register(again)
        XCTAssertEqual(ContentSources.source(for: fa).contentKey(fa, kind: .image, dim: 8, chunkOverlap: 0, settings: settings), ka)
    }

    // MARK: A pass

    func testAPassIndexesEachShotOnceWithItsPictureAndItsWords() throws {
        register(history)
        let paths = try record()
        // A look typed on: stored, and not for the index.
        let typed = history.record(screen(looks[2].lines + ["x"], page: 3), time: t0 + 12, display: 1, app: "Notes", title: "Door", settled: false)
        XCTAssertEqual(typed?.shot.indexed, false)
        // A folder beside it, as the app's passes have.
        let folder = dir.appendingPathComponent("docs")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try "The lease runs until March.".write(to: folder.appendingPathComponent("lease.txt"), atomically: true, encoding: .utf8)

        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer, roots: [folder, root])
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3)
        XCTAssertEqual(store.fileCount(kind: "image"), 3)
        XCTAssertNotNil(store.storedChunks(path: folder.appendingPathComponent("lease.txt").path))
        for (i, path) in paths.enumerated() {
            let r = try XCTUnwrap(rows(path), "look \(i)")
            XCTAssertEqual(r.first?.kind, "image")
            XCTAssertEqual(r.filter { $0.kind == "image" }.count, 1)
            XCTAssertTrue(text(path).contains(looks[i].code), "look \(i): \(text(path))")
            XCTAssertTrue(r.filter { $0.kind == "text" }.allSatisfy { $0.locator == "Text" })
            let found = store.search(embedder.inner.embedText(looks[i].code, as: .query), topK: 10, textQuery: looks[i].code)
            XCTAssertEqual(found.first?.path.utf8.elementsEqual(path.utf8), true, "look \(i) was not found by its words")
            XCTAssertEqual(found.first?.fileKind, "image")
            guard case .held(let entry) = ScreenShots.presence(path) else { XCTFail("not held"); continue }
            XCTAssertEqual(files([path])[path], [t0 + looks[i].at, Double(entry.bytes)], "dated at the shot, sized as its frame")
        }
        XCTAssertEqual(embedder.pictures, 3)
        XCTAssertFalse(FileManager.default.fileExists(atPath: history.stills.path), "indexing made no folder")

        // Nothing is read twice: by the next pass, by the enrichment, by an update naming them.
        let (pictures, texts) = (embedder.pictures, embedder.texts)
        let before = paths.map { rows($0) }
        pass(indexer, roots: [folder, root])
        var asked: [String] = []
        XCTAssertEqual(indexer.enrichMedia(settings: IndexSettings(), onFile: { asked.append($0) }), 0)
        XCTAssertFalse(indexer.update(paths: paths, settings: IndexSettings(), roots: [folder.path, history.root]))
        XCTAssertEqual([embedder.pictures, embedder.texts], [pictures, texts])
        XCTAssertEqual(asked, [])
        XCTAssertEqual(paths.map { rows($0) }, before)

        // A reopened history, as after a restart, is the same shots.
        history.stop()
        ScreenShots.remove(history)
        let again = open()
        register(again)
        pass(Indexer(store: store, embedder: embedder), roots: [folder, root])
        XCTAssertEqual([embedder.pictures, embedder.texts], [pictures, texts])
        XCTAssertEqual(paths.map { rows($0) }, before)
    }

    func testANewShotIsIndexedByItsPathAlone() throws {
        register(history)
        let paths = try record()
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        XCTAssertTrue(indexer.update(paths: [paths[1]], settings: IndexSettings(), roots: [history.root]))
        XCTAssertEqual(store.fileCount(underFolder: history.root), 1)
        XCTAssertEqual(rows(paths[1])?.first?.kind, "image")
        XCTAssertTrue(text(paths[1]).contains(looks[1].code), text(paths[1]))
        XCTAssertEqual(embedder.pictures, 1)
        // Named again, it is what the index holds.
        XCTAssertFalse(indexer.update(paths: [paths[1]], settings: IndexSettings(), roots: [history.root]))
        XCTAssertEqual(embedder.pictures, 1)
    }

    func testPicturesSwitchedOffIndexNoShotAndAShotThatIsGoneStillLeaves() throws {
        register(history)
        let paths = try record()
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        let off = IndexSettings(enabledKinds: [.text])
        pass(indexer, off)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 0)
        XCTAssertEqual(embedder.pictures, 0)
        pass(indexer)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3, "control: with pictures on they are indexed")
        let deleted = history.delete(from: t0 + 4, to: t0 + 6)
        XCTAssertEqual(deleted.paths, [paths[1]])
        XCTAssertEqual(history.indexedShots()?.map(\.path), [paths[0], paths[2]])
        pass(indexer, off)
        XCTAssertNil(rows(paths[1]), "its rows stayed searchable")
        XCTAssertEqual(store.fileCount(underFolder: history.root), 2, "the shots still held keep their rows")
    }

    // MARK: A shot that is gone

    func testADeletedShotLeavesTheIndexByAnyOfTheThreeWays() throws {
        register(history)
        let paths = try record()
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3)

        // Held: nothing removes it.
        XCTAssertEqual(indexer.removeVanished(paths, roots: [history.root]), 0)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3)

        // 1. The paths a deletion returns, handed to an update.
        let first = history.delete(from: t0, to: t0 + 1)
        XCTAssertEqual(first.paths, [paths[0]])
        XCTAssertTrue(indexer.update(paths: first.paths, settings: IndexSettings(), roots: [history.root]))
        XCTAssertNil(rows(paths[0]))
        // 2. The same paths as a watcher's removal.
        let second = history.delete(from: t0 + 4, to: t0 + 6)
        XCTAssertEqual(second.paths, [paths[1]])
        XCTAssertEqual(indexer.removeVanished(paths, roots: [history.root]), 1, "the one that is gone, and not the one still held")
        XCTAssertNil(rows(paths[1]))
        XCTAssertNotNil(rows(paths[2]))
        // 3. Nobody said: the next pass finds it gone.
        XCTAssertEqual(history.delete(from: t0 + 9, to: t0 + 11).paths, [paths[2]])
        pass(indexer)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 0)
        XCTAssertEqual(embedder.pictures, 3, "nothing was read again on the way")
    }

    func testAClearedHistoryLeavesNoRows() throws {
        register(history)
        try record()
        let indexer = Indexer(store: store, embedder: Counting())
        pass(indexer)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3)
        history.clear()
        pass(indexer)
        XCTAssertEqual(store.fileCount(underFolder: history.root), 0)
    }

    // MARK: A forced reading

    /// RE-TAGGING (forceFreshEmbed + hqMediaTags) runs the picture's forward again and nothing else:
    /// the shot's words stay, read once.
    func testAForcedReadingRunsThePictureAgainAndKeepsTheWords() throws {
        register(history)
        let paths = try record()
        let embedder = Counting()
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        let before = paths.map { text($0) }
        XCTAssertTrue(before[0].contains(looks[0].code), "no words indexed; the test proves nothing")
        var retag = IndexSettings()
        retag.forceFreshEmbed = true
        retag.hqMediaTags = true
        let (pictures, texts) = (embedder.pictures, embedder.texts)
        indexer.resetCancelled()
        indexer.update(paths: paths, settings: retag, force: true, roots: [history.root])
        XCTAssertGreaterThan(embedder.pictures, pictures, "the forced reading ran no picture forward")
        XCTAssertEqual(paths.map { text($0) }, before, "the forced reading lost or changed the words")
        XCTAssertEqual(embedder.texts, texts, "the words were embedded again")
        XCTAssertEqual(paths.map { rows($0)?.first?.kind }, ["image", "image", "image"])
    }

    // MARK: An index built by 0.20.2

    /// An index as 0.20.2 built it: from the picture files, as files of a folder. Returns the rows
    /// it holds, by path.
    private func indexThePictureFiles(_ embedder: Counting) throws -> (paths: [String], rows: [[Row]?], files: [String: [Double]], digests: [String: String]) {
        let paths = try record()
        try ScreenHistory0202.downgrade(history)
        XCTAssertFalse(ScreenShots.isShotPath(paths[0]), "the files are indexed as files")
        pass(Indexer(store: store, embedder: embedder), roots: [history.stills])
        XCTAssertEqual(store.fileCount(underFolder: history.root), 3)
        for (i, path) in paths.enumerated() {
            XCTAssertNotNil(rows(path), "0.20.2's index holds look \(i) under the path this build spells")
            XCTAssertTrue(text(path).contains(looks[i].code), "look \(i): \(text(path))")
        }
        return (paths, paths.map { rows($0) }, files(paths), store.contentDigests(paths: paths))
    }

    /// THE UPDATE EMBEDS NOTHING AGAIN. Every row of the index is what it was, under the path it
    /// was under, and the files are gone.
    func testAnIndexBuiltFromThePictureFilesIsTheSameIndexAfterTheUpdate() throws {
        let embedder = Counting()
        let built = try indexThePictureFiles(embedder)
        let (pictures, texts) = (embedder.pictures, embedder.texts)
        XCTAssertEqual(pictures, 3)

        let updated = open()
        register(updated)
        XCTAssertEqual(updated.migrateStills(), .done(kept: 3, dropped: 0))
        XCTAssertFalse(FileManager.default.fileExists(atPath: updated.stills.path))
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        var asked: [String] = []
        XCTAssertEqual(indexer.enrichMedia(settings: IndexSettings(), onFile: { asked.append($0) }), 0)
        XCTAssertFalse(indexer.update(paths: built.paths, settings: IndexSettings(), roots: [updated.root]))
        XCTAssertEqual(indexer.removeVanished(built.paths, roots: [updated.root]), 0)
        XCTAssertEqual([embedder.pictures, embedder.texts], [pictures, texts], "the update sent a shot through the model again")
        XCTAssertEqual(asked, [])
        XCTAssertEqual(built.paths.map { rows($0) }, built.rows)
        XCTAssertEqual(files(built.paths), built.files)
        XCTAssertEqual(store.contentDigests(paths: built.paths), built.digests)
        XCTAssertEqual(store.fileCount(underFolder: updated.root), 3)

        // POSITIVE CONTROL: what kept them is that a held shot comes as the index holds it. The
        // frame's own date and size are not the file's, and a shot given them would read as changed.
        for path in built.paths {
            guard case .held(let entry) = ScreenShots.presence(path) else { XCTFail("not held"); continue }
            XCTAssertNotEqual(built.files[path]?[1], Double(entry.bytes), "control: the file's size is not the frame's")
        }
        // And a shot read from its frame says what its file said.
        XCTAssertTrue(indexer.update(paths: [built.paths[0]], settings: IndexSettings(), force: true, roots: [updated.root]))
        XCTAssertEqual(embedder.pictures, pictures + 1)
        XCTAssertTrue(text(built.paths[0]).contains(looks[0].code), text(built.paths[0]))
        XCTAssertEqual(rows(built.paths[0])?.first?.kind, "image")
    }

    /// UNTIL THE FILES ARE TAKEN IN, NOTHING IS REMOVED ON THE HISTORY'S WORD. It cannot say yet
    /// which path is which shot's: a pass, an update and a watcher's removal all leave the rows.
    func testWhileTheFilesAreNotTakenInNoRowIsRemoved() throws {
        let fm = FileManager.default
        let embedder = Counting()
        let built = try indexThePictureFiles(embedder)
        let (pictures, texts) = (embedder.pictures, embedder.texts)
        // One file deleted by hand, as a user of 0.20.2 could.
        let lost = built.paths[1]
        try fm.removeItem(atPath: lost)

        let pending = open()
        register(pending)
        XCTAssertTrue(pending.stillsPending)
        let indexer = Indexer(store: store, embedder: embedder)
        pass(indexer)
        XCTAssertFalse(indexer.update(paths: built.paths, settings: IndexSettings(), roots: [pending.root]))
        XCTAssertEqual(indexer.removeVanished(built.paths, roots: [pending.root]), 0)
        var asked: [String] = []
        XCTAssertEqual(indexer.enrichMedia(settings: IndexSettings(), onFile: { asked.append($0) }), 0)
        XCTAssertEqual(store.fileCount(underFolder: pending.root), 3)
        XCTAssertEqual(built.paths.map { rows($0) }, built.rows)
        XCTAssertEqual([embedder.pictures, embedder.texts], [pictures, texts])
        XCTAssertTrue(pending.stillsPending, "the indexer does not run the update")

        // POSITIVE CONTROL: taken in, the shot whose file was gone leaves, and only that one.
        XCTAssertEqual(pending.migrateStills(), .done(kept: 2, dropped: 1))
        pass(indexer)
        XCTAssertNil(rows(lost))
        XCTAssertEqual(store.fileCount(underFolder: pending.root), 2)
        XCTAssertEqual([built.paths[0], built.paths[2]].map { rows($0) }, [built.rows[0], built.rows[2]])
        XCTAssertEqual([embedder.pictures, embedder.texts], [pictures, texts])
    }

    /// Files whose dates were lost cannot be told apart, and each shot takes a new path: the rows
    /// under the old paths go, and the shots are read again from their frames, words and all.
    func testShotsGivenNewPathsAreReadAgainFromTheirFramesAndTheOldRowsGo() throws {
        let fm = FileManager.default
        let embedder = Counting()
        let built = try indexThePictureFiles(embedder)
        for path in built.paths {
            try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: t0 + 90_000)], ofItemAtPath: path)
        }
        let updated = open()
        register(updated)
        XCTAssertEqual(updated.migrateStills(), .done(kept: 3, dropped: 0))
        let now = (updated.indexedShots() ?? []).sorted { $0.time < $1.time }.map(\.path)
        XCTAssertEqual(now.count, 3)
        XCTAssertTrue(Set(now).isDisjoint(with: built.paths))
        pass(Indexer(store: store, embedder: embedder))
        XCTAssertEqual(store.fileCount(underFolder: updated.root), 3)
        XCTAssertEqual(embedder.pictures, 6)
        for (i, path) in now.enumerated() {
            XCTAssertNil(rows(built.paths[i]))
            XCTAssertEqual(rows(path)?.first?.kind, "image")
            XCTAssertTrue(text(path).contains(looks[i].code), "look \(i): \(text(path))")
        }
    }
}
