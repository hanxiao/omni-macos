import XCTest
import AppKit
import AVFoundation
@testable import OmniKit

/// The indexer's implicit state machines, each driven to the outcome the review traced: the watcher
/// path's long files, read errors, vanished roots, failures taken for success (docs/review-2026-11-06).
final class IndexerStateTests: XCTestCase {
    /// Text-dependent vectors, images and audio that can be told to fail, a hook on every embed.
    final class Fake: Embedder, @unchecked Sendable {
        let dim = 16
        private let lock = NSLock()
        private var _texts = 0, _images = 0
        var texts: Int { lock.withLock { _texts } }
        var images: Int { lock.withLock { _images } }
        var onTexts: (([String]) -> Void)?
        var onImage: (() -> Void)?
        var imageFault: (@Sendable ([OmniVisionPreprocess.RawPatches]) -> Bool)?
        func vec(_ text: String) -> [Float] {
            var s: UInt64 = 0xcbf2_9ce4_8422_2325
            for b in text.utf8 { s = (s ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
            var v = [Float](repeating: 0, count: dim)
            for i in 0 ..< dim { s = s &* 6364136223846793005 &+ 1442695040888963407; v[i] = Float(s >> 40) / Float(1 << 24) - 0.5 }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return v.map { $0 / n }
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { lock.withLock { _texts += 1 }; return vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            onTexts?(texts)
            lock.withLock { _texts += texts.count }
            return texts.map(vec)
        }
        func embedImage(_ image: CGImage) -> [Float]? { vec("image") }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
            onImage?()
            if imageFault?(raws) == true { return nil }
            lock.withLock { _images += raws.count }
            return raws.map { _ in vec("image") }
        }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func folder(_ name: String) throws -> (root: URL, db: URL) {
        var r = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: r, withIntermediateDirectories: true)
        if let rp = realpath(r.path, nil) { r = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        let db = r.deletingLastPathComponent().appendingPathComponent("\(name)-db-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: db, withIntermediateDirectories: true)
        let root = r
        addTeardownBlock { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: db) }
        return (root, db)
    }

    private let words = ["harbor", "engine", "anchor", "compass", "lantern", "rigging", "ballast", "keel", "mast", "tide"]

    private func paragraphs(_ tag: String, _ count: Int) -> [String] {
        (0 ..< count).map { k in
            var s = "Section \(k) of \(tag): "
            var i = k
            while s.count < 600 { s += words[i % words.count] + "\(k % 37) "; i += 7 }
            return s + "."
        }
    }

    /// A document with this much text: .rtf or .docx (a zip).
    private func document(_ url: URL, tag: String, paragraphs count: Int = 400) throws {
        let paras = paragraphs(tag, count)
        if url.pathExtension == "rtf" {
            try ("{\\rtf1\\ansi " + paras.map { $0 + "\\par\n" }.joined() + "}").write(to: url, atomically: true, encoding: .utf8)
        } else {
            let s = NSAttributedString(string: paras.joined(separator: "\n"))
            let d = try s.data(from: NSRange(location: 0, length: s.length),
                               documentAttributes: [.documentType: NSAttributedString.DocumentType.officeOpenXML])
            try d.write(to: url)
        }
    }

    private func pdf(_ pages: [String], at url: URL) {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 9, nil)
        for page in pages {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 770
            var line = ""
            func flush() {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: y); CTLineDraw(l, ctx); y -= 11; line = ""
            }
            for word in page.split(separator: " ") {
                if line.count + word.count > 110 { flush() }
                line += (line.isEmpty ? "" : " ") + word
            }
            if !line.isEmpty { flush() }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    private func longPDFPages(_ tag: String, _ count: Int = 40) -> [String] {
        (0 ..< count).map { k in
            var s = "\(tag)\(k)zz "
            var i = k
            while s.count < 3000 { s += words[i % words.count] + "\(k) "; i += 3 }
            return s
        }
    }

    private func pass(_ indexer: Indexer, _ roots: [URL], _ settings: IndexSettings = IndexSettings(enabledKinds: [.text]),
                      timeout: TimeInterval = 180) -> IndexProgress {
        let done = expectation(description: "pass")
        nonisolated(unsafe) var last = IndexProgress()
        indexer.index(roots: roots, settings: settings) { p in if p.done { last = p; done.fulfill() } }
        wait(for: [done], timeout: timeout)
        return last
    }

    private func scratchFiles(_ indexer: Indexer) -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: indexer.turnsDir.path)) ?? []
    }

    // MARK: - 1. A new long converted document from the watcher

    /// Reproduced in the app first: a watcher file got 4 chunks with snippets starting `PK` where a
    /// control indexed by a full pass got 91 of real text.
    func testANewLongDocumentFromTheWatcherIsIndexedFromItsText() throws {
        for ext in ["rtf", "docx"] {
            let (root, db) = try folder("watch-\(ext)")
            let file = root.appendingPathComponent("report.\(ext)")
            try document(file, tag: "report")
            let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
            defer { store.close() }
            let indexer = Indexer(store: store, embedder: Fake())
            let settings = IndexSettings(enabledKinds: [.text])
            guard case .text(let whole) = try FileExtractor.extract(file) else { return XCTFail("\(ext) did not convert") }
            XCTAssertGreaterThan(whole.utf8.count, 3 * Indexer.firstWindowBytes, "too short to take turns")
            let expected = indexer.chunk(whole, settings: settings, origin: .opaque).map { indexer.chunkKey($0.text, settings: settings) }

            XCTAssertTrue(indexer.update(paths: [file.path], settings: settings, roots: [root.path]))

            let chunks = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
            XCTAssertEqual(chunks.map(\.chunkKey), expected, "\(ext): indexed from something other than its text; first snippet: \(chunks.first?.snippet.prefix(40) ?? "")")
            XCTAssertEqual(chunks.map(\.chunkIndex), Array(0 ..< chunks.count), "\(ext): chunk numbers have a gap or a repeat")
            XCTAssertNotNil(store.contentKeys(paths: [file.path])[file.path], "\(ext): the finished document's key was not recorded")
            XCTAssertEqual(indexer.streams.count, 0, "\(ext): a cursor was left behind")
            XCTAssertEqual(scratchFiles(indexer), [], "\(ext): the scratch text outlived the document's turns")
        }
    }

    // MARK: - 2. A new long PDF from the watcher, cut off by a cancel

    func testACancelledLongPDFFromTheWatcherIsNotCutOff() throws {
        let (root, db) = try folder("watch-pdf")
        let file = root.appendingPathComponent("log.pdf")
        pdf(longPDFPages("marker"), at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let inner = EnrichmentTests.Fake()
        let pausing = PageChunkTests.PausingEmbedder(inner: inner, store: store, path: file.path)   // in the second turn
        let indexer = Indexer(store: store, embedder: pausing)
        pausing.indexer = indexer
        let settings = IndexSettings(enabledKinds: [.text])

        indexer.update(paths: [file.path], settings: settings, roots: [root.path])
        XCTAssertTrue(indexer.isCancelled)
        let partial = try XCTUnwrap(store.storedChunks(path: file.path))
        XCTAssertFalse(partial.isEmpty, "the first turn was not stored")
        XCTAssertNil(store.contentKeys(paths: [file.path])[file.path], "a part-read PDF's key was recorded: a copy would be served the part")
        XCTAssertEqual(indexer.streams.count, 1, "nothing marks the rows unfinished")

        // The app puts a cancelled batch back and runs it again (AppModel.drainPendingFSChanges).
        indexer.resetCancelled()
        indexer.update(paths: [file.path], settings: settings, roots: [root.path])
        let whole = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertGreaterThan(whole.count, partial.count, "the file stayed cut off")
        XCTAssertEqual(whole.map(\.chunkIndex), Array(0 ..< whole.count), "chunk numbers have a gap or a repeat")
        let pages = whole.compactMap { Int($0.locator.dropFirst(5)) }
        XCTAssertGreaterThanOrEqual(pages.last ?? 0, 39, "the last pages were not indexed")
        XCTAssertNotNil(store.contentKeys(paths: [file.path])[file.path], "the finished PDF's key was not recorded")
        XCTAssertEqual(indexer.streams.count, 0)
    }

    /// The same cut, found by a full pass instead of the rerun (a batch the app dropped at quit).
    func testAFullPassFinishesALongPDFTheWatcherLeftPartWay() throws {
        let (root, db) = try folder("watch-pdf-pass")
        let file = root.appendingPathComponent("log.pdf")
        pdf(longPDFPages("marker"), at: file)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let inner = EnrichmentTests.Fake()
        let pausing = PageChunkTests.PausingEmbedder(inner: inner, store: store, path: file.path)
        var indexer = Indexer(store: store, embedder: pausing)
        pausing.indexer = indexer
        indexer.update(paths: [file.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        let partial = store.chunkCount(path: file.path)
        XCTAssertGreaterThan(partial, 0)
        indexer = Indexer(store: store, embedder: inner)   // a new launch
        _ = pass(indexer, [root])
        XCTAssertGreaterThan(store.chunkCount(path: file.path), partial, "the file stayed cut off")
        XCTAssertEqual(indexer.streams.count, 0)
    }

    // MARK: - 7. A kill between a long file's first store and its cursor

    /// Nothing may mark a long file's rows complete at the moment the first ones can land, so the
    /// cursor is on disk before the first write: asked for on each file's first embed, which comes
    /// before its first store.
    func testALongFilesCursorIsWrittenBeforeItsFirstRow() throws {
        let (root, db) = try folder("cursor-first")
        let text = root.appendingPathComponent("long.log")
        try paragraphs("thelog", 400).joined(separator: "\n").write(to: text, atomically: true, encoding: .utf8)
        let doc = root.appendingPathComponent("doc.rtf")
        try document(doc, tag: "thedoc")
        let paged = root.appendingPathComponent("paged.pdf")
        pdf(longPDFPages("thepage"), at: paged)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let fake = Fake()
        let indexer = Indexer(store: store, embedder: fake)
        nonisolated(unsafe) var asked = Set<String>(), unmarked = Set<String>()
        let streams = indexer.streams
        fake.onTexts = { texts in
            for (tag, path) in [("thelog", text.path), ("thedoc", doc.path), ("thepage", paged.path)] where texts.contains(where: { $0.contains(tag) }) {
                asked.insert(path)
                if !streams.inProgress(path) { unmarked.insert(path) }
            }
        }
        _ = pass(indexer, [root])
        XCTAssertEqual(asked, [text.path, doc.path, paged.path], "a long file did not take turns")
        XCTAssertEqual(unmarked, [], "embedded, about to be stored, with nothing marking the rows unfinished")
        XCTAssertEqual(indexer.streams.count, 0)
    }

    /// What a kill right after the first store leaves: rows under the real date and a cursor at the
    /// start. The file is read again from the top and comes out whole.
    func testAKillAfterTheFirstStoreRedoesTheFile() throws {
        let (root, db) = try folder("cursor-kill")
        let (_, db2) = try folder("cursor-kill-whole")
        let file = root.appendingPathComponent("long.log")
        try paragraphs("thelog", 400).joined(separator: "\n").write(to: file, atomically: true, encoding: .utf8)
        let wholeStore = try VectorStore(dbURL: db2.appendingPathComponent("index.sqlite"))
        defer { wholeStore.close() }
        _ = pass(Indexer(store: wholeStore, embedder: Fake()), [root])
        let whole = try XCTUnwrap(wholeStore.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertGreaterThan(whole.count, 10)

        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        try store.replaceMany([(file.path, Array(whole.prefix(3)))])
        let f = try XCTUnwrap(store.storedFiles(paths: [file.path])[file.path])
        StreamCursors(directory: db).set(file.path, StreamCursors.Cursor(modified: f.modified, size: f.size, offset: 0, undecoded: Data(),
                                                                         carry: "", lineBase: 0, written: 0))
        let indexer = Indexer(store: store, embedder: Fake())   // the next launch
        _ = pass(indexer, [root])
        let again = try XCTUnwrap(store.storedChunks(path: file.path)).sorted { $0.chunkIndex < $1.chunkIndex }
        XCTAssertEqual(again.map(\.chunkKey), whole.map(\.chunkKey))
        XCTAssertEqual(indexer.streams.count, 0)
    }

    // MARK: - 6. Watcher deletes under a root whose volume has gone

    func testVanishedPathsUnderAMissingRootKeepTheirRows() throws {
        let (base, db) = try folder("vanish-root")
        let root = base.appendingPathComponent("Volume", isDirectory: true)
        let sub = root.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let a = root.appendingPathComponent("a.txt"), b = sub.appendingPathComponent("b.txt")
        try "a note about the harbor and its tides".write(to: a, atomically: true, encoding: .utf8)
        try "a note about the engine and its keel".write(to: b, atomically: true, encoding: .utf8)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Fake())
        _ = pass(indexer, [root])
        XCTAssertEqual(store.chunkCount(path: a.path), 1)
        // The drive goes: the root is not there, and events queued for what was under it arrive.
        let away = base.appendingPathComponent("Unmounted", isDirectory: true)
        try FileManager.default.moveItem(at: root, to: away)
        XCTAssertEqual(indexer.removeVanished([a.path, sub.path], roots: [root.path]), 0, "rows deleted under a root that is not there")
        indexer.update(paths: [a.path, sub.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertEqual(store.chunkCount(path: a.path), 1, "a file's rows went with the drive")
        XCTAssertEqual(store.chunkCount(path: b.path), 1, "a folder's rows went with the drive")
        // With the root back, a file truly gone still leaves.
        try FileManager.default.moveItem(at: away, to: root)
        try FileManager.default.removeItem(at: a)
        XCTAssertEqual(indexer.removeVanished([a.path], roots: [root.path]), 1)
        XCTAssertEqual(store.chunkCount(path: a.path), 0)
    }

    // MARK: - 14. A decode that could not read a known file

    func testAKnownFileThatCouldNotBeReadKeepsItsRowsOnTheWatcherPath() throws {
        let (root, db) = try folder("unreadable")
        let note = root.appendingPathComponent("note.txt")
        try "a note about the harbor and its tides".write(to: note, atomically: true, encoding: .utf8)
        let paged = root.appendingPathComponent("paged.pdf")
        pdf(longPDFPages("page", 2), at: paged)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: EnrichmentTests.Fake())
        _ = pass(indexer, [root])
        XCTAssertGreaterThan(store.chunkCount(path: note.path), 0)
        let before = store.chunkCount(path: paged.path)
        XCTAssertGreaterThan(before, 0)

        // Overwritten in place and read mid-write: a PDF that does not open yet, a file with no
        // read permission for the moment.
        try Data("%PDF-1.7 not finished".utf8).write(to: paged)
        try "a new version being saved".write(to: note, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: note.path)
        addTeardownBlock { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: note.path) }
        indexer.update(paths: [note.path, paged.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertGreaterThan(store.chunkCount(path: note.path), 0, "a file that could not be read lost its rows")
        XCTAssertEqual(store.chunkCount(path: paged.path), before, "a PDF read mid-write lost its rows")

        // A file that really has nothing in it any more still leaves.
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: note.path)
        try "".write(to: note, atomically: true, encoding: .utf8)
        indexer.update(paths: [note.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertEqual(store.chunkCount(path: note.path), 0, "an emptied file kept its rows")
    }

    // MARK: - 18. A discard stops a batch's stores

    func testADiscardDropsWhatTheWatcherBatchHasEmbedded() throws {
        let (root, db) = try folder("discard")
        var files: [URL] = []
        for i in 0 ..< 3 {
            let u = root.appendingPathComponent("p\(i).png")
            let ctx = CGContext(data: nil, width: 96, height: 96, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.setFillColor(CGColor(red: CGFloat(i) / 3, green: 0.5, blue: 0.2, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 96, height: 96))
            let dst = CGImageDestinationCreateWithURL(u as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
            XCTAssertTrue(CGImageDestinationFinalize(dst))
            files.append(u)
        }
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let fake = Fake()
        let indexer = Indexer(store: store, embedder: fake)
        fake.onImage = { indexer.cancel(.discard) }
        indexer.update(paths: files.map(\.path), settings: IndexSettings(enabledKinds: [.image]), roots: [root.path])
        XCTAssertEqual(fake.images, 3, "the batch did not reach the tower")
        for u in files { XCTAssertEqual(store.chunkCount(path: u.path), 0, "stored after a discard: \(u.lastPathComponent)") }

        // A pause keeps them: the tower has already run and they are as good as any others.
        let (root2, db2) = try folder("pause")
        try FileManager.default.copyItem(at: files[0], to: root2.appendingPathComponent("p.png"))
        let store2 = try VectorStore(dbURL: db2.appendingPathComponent("index.sqlite"))
        defer { store2.close() }
        let fake2 = Fake()
        let indexer2 = Indexer(store: store2, embedder: fake2)
        fake2.onImage = { indexer2.cancel(.pause) }
        indexer2.update(paths: [root2.appendingPathComponent("p.png").path], settings: IndexSettings(enabledKinds: [.image]), roots: [root2.path])
        XCTAssertEqual(store2.chunkCount(path: root2.appendingPathComponent("p.png").path), 1)
    }

    // MARK: - 19. Costs

    /// A file that hangs its read holds a reconcile for the whole stall limit, per event.
    func testTheWatcherPathRespectsQuarantine() throws {
        let (root, db) = try folder("quarantine-event")
        let file = root.appendingPathComponent("hang.txt")
        try "a file whose read never finished".write(to: file, atomically: true, encoding: .utf8)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Fake())
        var st = stat()
        XCTAssertEqual(stat(file.path, &st), 0)
        let modified = Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9
        indexer.quarantine.quarantine(path: file.path, modified: modified, size: Int(st.st_size), reason: "test")
        let reads = Indexer.Counter()
        Indexer.testReadDelay = { p in if p == file.path { reads.add(1) }; return nil }
        addTeardownBlock { Indexer.testReadDelay = nil }
        indexer.update(paths: [file.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertEqual(reads.value, 0, "an event read a quarantined file")
        XCTAssertEqual(store.chunkCount(path: file.path), 0)
    }

    /// A long file part-way, then shrunk to a size that does not take turns: reads as `inProgress`
    /// for good and never records its key. And one deleted part-way leaves its cursor and scratch.
    func testStaleCursorsGoAway() throws {
        let (root, db) = try folder("stale-cursor")
        let shrunk = root.appendingPathComponent("shrunk.log")
        try paragraphs("shrunk", 400).joined(separator: "\n").write(to: shrunk, atomically: true, encoding: .utf8)
        let doc = root.appendingPathComponent("gone.rtf")
        try document(doc, tag: "gone")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let pausing = PageChunkTests.PausingEmbedder(inner: EnrichmentTests.Fake(), store: store, path: doc.path)
        var indexer = Indexer(store: store, embedder: pausing)
        pausing.indexer = indexer
        _ = pass(indexer, [root])
        // Whichever got that far, give both a cursor and the document its scratch.
        func cursor(_ u: URL) throws -> StreamCursors.Cursor {
            let f = try XCTUnwrap(store.storedFiles(paths: [u.path])[u.path])
            return StreamCursors.Cursor(modified: f.modified, size: f.size, offset: 0, undecoded: Data(), carry: "", lineBase: 0, written: 0)
        }
        XCTAssertGreaterThan(store.chunkCount(path: doc.path), 0)
        try? indexer.streams.set(doc.path, cursor(doc))
        indexer.streams.set(shrunk.path, StreamCursors.Cursor(modified: 1, size: 1, offset: 0, undecoded: Data(), carry: "", lineBase: 0, written: 0))
        let docCursor = try XCTUnwrap(indexer.streams.all.first { $0.path == doc.path }).cursor
        try FileManager.default.createDirectory(at: indexer.turnsDir, withIntermediateDirectories: true)
        try Data("x\n".utf8).write(to: Indexer.turnTextURL(dir: indexer.turnsDir, path: doc.path, modified: docCursor.modified, size: docCursor.size))
        // Deleted while part-way, and shrunk under the threshold.
        try FileManager.default.removeItem(at: doc)
        try "a short note now".write(to: shrunk, atomically: true, encoding: .utf8)

        indexer = Indexer(store: store, embedder: EnrichmentTests.Fake())
        _ = pass(indexer, [root])
        XCTAssertFalse(indexer.streams.inProgress(doc.path), "a deleted file's cursor stayed")
        XCTAssertFalse(indexer.streams.inProgress(shrunk.path), "a shrunk file reads as in progress for good")
        XCTAssertNotNil(store.contentKeys(paths: [shrunk.path])[shrunk.path], "a shrunk file never recorded its key")
        XCTAssertEqual(scratchFiles(indexer), [], "a deleted file's scratch stayed")
    }

    /// `update()` never prepared the ledger: its first touch opened it under the default recipe and
    /// wiped what a pass under other settings had recorded.
    func testTheWatcherPathDoesNotWipeTheLedger() throws {
        let (root, db) = try folder("ledger-touch")
        let file = root.appendingPathComponent("note.txt")
        try "a note about the harbor".write(to: file, atomically: true, encoding: .utf8)
        let url = db.appendingPathComponent("index.sqlite")
        var settings = IndexSettings(enabledKinds: [.text])
        settings.screenText = false
        settings.soundtrack = false
        do {
            let store = try VectorStore(dbURL: url)
            store.enrichLedger.prepare(settings)
            store.enrichLedger.record(["d|/some/file"])
            store.enrichLedger.backfilled = true
            store.close()
        }
        let store = try VectorStore(dbURL: url)   // the next launch, with a power hold: no pass first
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: Fake())
        indexer.update(paths: [file.path], settings: settings, roots: [root.path])
        XCTAssertEqual(store.enrichLedger.done(["d|/some/file"]), ["d|/some/file"], "the ledger was wiped by its first touch")
        XCTAssertTrue(store.enrichLedger.backfilled)
    }

    /// A copy landing as a directory event: its names are known from the moment the crawl finds
    /// them, or the delete of its original, arriving meanwhile, goes first and it pays its re-embed.
    func testAnEarlyDeleteDuringACrawlLeavesTheCopyItsVectors() throws {
        let (base, db) = try folder("arrival")
        let root = base.appendingPathComponent("root", isDirectory: true)
        let there = root.appendingPathComponent("there", isDirectory: true)
        let sub = there.appendingPathComponent("sub", isDirectory: true)
        let here = root.appendingPathComponent("here", isDirectory: true)
        for d in [sub, here] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
        let original = here.appendingPathComponent("report.txt")
        let text = paragraphs("report", 3).joined(separator: "\n")
        try text.write(to: original, atomically: true, encoding: .utf8)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let fake = Fake()
        let indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        let before = fake.texts
        XCTAssertGreaterThan(before, 0)
        // The copy lands, the original goes; the crawl of the copy's folder finds an .omniignore
        // below it, after its files, and the delete arrives there.
        try text.write(to: there.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        try "".write(to: sub.appendingPathComponent(OmniIgnore.fileName), atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: original)
        nonisolated(unsafe) var removed = -1
        let originalPath = original.path, rootPath = root.path
        indexer.onPolicyFile = { _ in removed = indexer.removeVanished([originalPath], roots: [rootPath]) }
        indexer.update(paths: [there.path, original.path], settings: IndexSettings(enabledKinds: [.text]), roots: [root.path])
        XCTAssertEqual(removed, 0, "the delete went first")
        XCTAssertEqual(fake.texts, before, "the copy went back through the model")
        XCTAssertEqual(store.chunkCount(path: original.path), 0, "the original stayed")
    }

    /// Not a check: `OMNI_ANE_PROBE=1` prints what the cursor written before a long file's first row costs.
    func testProbeCursorWrite() throws {
        guard ProcessInfo.processInfo.environment["OMNI_ANE_PROBE"] == "1" else { throw XCTSkip("a measurement") }
        let (_, db) = try folder("cursor-cost")
        let streams = StreamCursors(directory: db)
        let carry = String(repeating: "harbor engine ", count: 130)   // a held-back piece, ~1.8 KB
        for n in [10, 300, 1000, 3000] {
            for i in streams.count ..< n {
                streams.set("/some/folder/of/files/long\(i).txt", StreamCursors.Cursor(modified: 1, size: 100_000, offset: 65_536, undecoded: Data(), carry: carry, lineBase: 900, written: 40))
            }
            let t0 = Date()
            for k in 0 ..< 20 {
                streams.set("/some/folder/of/files/long0.txt", StreamCursors.Cursor(modified: 1, size: 100_000, offset: UInt64(k), undecoded: Data(), carry: carry, lineBase: 900, written: 40))
            }
            print(String(format: "PROBE cursors=%d one write %.2f ms", n, -t0.timeIntervalSinceNow / 20 * 1000))
        }
    }

    // MARK: - 4. One image a tower refuses

    private func png(_ url: URL, side: Int, shade: CGFloat, wide: Bool = false) {
        let w = wide ? side * 2 : side
        let ctx = CGContext(data: nil, width: w, height: side, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: shade, green: 0.5, blue: 0.2, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: w, height: side))
        let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
    }

    /// A tower program that will not take one size: one nil for the batch, every file of the flush
    /// out, and the walk's order rebuilds the same flush next pass.
    func testOneImageATowerRefusesDoesNotHoldBackItsFlush() throws {
        for path in ["pass", "watcher"] {
            let (root, db) = try folder("refuse-\(path)")
            var good: [URL] = []
            for i in 0 ..< 4 {
                let u = root.appendingPathComponent("good\(i).png")
                png(u, side: 96, shade: CGFloat(i) / 4)
                good.append(u)
            }
            let bad = root.appendingPathComponent("bad.png")
            png(bad, side: 512, shade: 0.9, wide: true)
            let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
            defer { store.close() }
            let fake = Fake()
            fake.imageFault = { raws in raws.contains { raw in raw.gridTHW.first.map { $0.1 != $0.2 } ?? false } }   // the wide one
            let settings = IndexSettings(enabledKinds: [.image])
            let indexer = Indexer(store: store, embedder: fake)
            if path == "pass" { _ = pass(indexer, [root], settings) }
            else { indexer.update(paths: (good + [bad]).map(\.path), settings: settings, roots: [root.path]) }
            for u in good { XCTAssertEqual(store.chunkCount(path: u.path), 1, "\(path): \(u.lastPathComponent) was held back by its neighbour") }
            XCTAssertEqual(store.chunkCount(path: bad.path), 0)
            // The refusal lifts: only it is left to do.
            fake.imageFault = nil
            let before = fake.images
            _ = pass(Indexer(store: store, embedder: fake), [root], settings)
            XCTAssertEqual(store.chunkCount(path: bad.path), 1, "\(path): the refused file did not come back")
            XCTAssertEqual(fake.images - before, 1)
        }
    }

    // MARK: - 5. A read error is not the end of the file

    /// A real volume, pulled for real: `hdiutil detach -force` with a handle open makes the next read fail with EIO.
    final class DiskImage: @unchecked Sendable {
        let image: URL, mount: URL
        init(in dir: URL, name: String, megabytes: Int = 64) throws {
            image = dir.appendingPathComponent(name + ".dmg")
            mount = dir.appendingPathComponent(name + "-mount", isDirectory: true)
            try FileManager.default.createDirectory(at: mount, withIntermediateDirectories: true)
            try Self.hdiutil(["create", "-size", "\(megabytes)m", "-fs", "APFS", "-volname", name, "-quiet", image.path])
            attach()
        }
        func attach() { try? Self.hdiutil(["attach", "-nobrowse", "-mountpoint", mount.path, "-quiet", image.path]) }
        func pull() { try? Self.hdiutil(["detach", "-force", "-quiet", mount.path]) }
        static func hdiutil(_ args: [String]) throws {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/hdiutil")
            p.arguments = args
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus != 0 { throw OmniError.extraction("hdiutil \(args[0]) failed") }
        }
    }

    func testAReadErrorDuringAWholeFileStreamLeavesTheFileToBeRedone() throws {
        let (base, db) = try folder("pulled")
        let disk = try DiskImage(in: base, name: "pulled")
        addTeardownBlock { disk.pull() }
        let file = disk.mount.appendingPathComponent("long.log")
        let old = paragraphs("old", 4000).joined(separator: "\n")   // past the 2 MB one read takes
        XCTAssertGreaterThan(old.utf8.count, FileExtractor.maxTextBytes)
        try old.write(to: file, atomically: true, encoding: .utf8)
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        var settings = IndexSettings(enabledKinds: [.text])
        settings.readLongDataFiles = true
        let root = URL(fileURLWithPath: disk.mount.path, isDirectory: true)
        _ = pass(Indexer(store: store, embedder: Fake()), [root], settings)
        let whole = store.chunkCount(path: file.path)
        XCTAssertGreaterThan(whole, 100)

        // Edited at its head, so the first window reaches the model: the drive goes during that embed,
        // with the stream's handle open and its next read still to come.
        try (paragraphs("edited", 20).joined(separator: "\n") + "\n" + old).write(to: file, atomically: true, encoding: .utf8)
        let fake = Fake()
        nonisolated(unsafe) var pulled = false
        fake.onTexts = { _ in if !pulled { pulled = true; disk.pull() } }
        let p = pass(Indexer(store: store, embedder: fake), [root], settings)
        XCTAssertTrue(pulled, "the first window did not reach the model")
        XCTAssertGreaterThanOrEqual(p.failed, 1, "a read error was taken for the end of the file")
        XCTAssertEqual(store.chunkCount(path: file.path), whole, "the old rows should stand until the new ones are all there")

        // The drive is back. The file is read to its end.
        disk.attach()
        let again = Fake()
        _ = pass(Indexer(store: store, embedder: again), [root], settings)
        XCTAssertGreaterThan(store.chunkCount(path: file.path), whole, "the file stayed cut off at the read error")
    }

    /// A small text file's read goes through the extractor: its read raised an exception there, a crash.
    func testAPulledDriveDuringAWholeReadIsNotAnEmptyFile() throws {
        let (base, db) = try folder("pulled-small")
        let disk = try DiskImage(in: base, name: "small")
        addTeardownBlock { disk.pull() }
        let file = disk.mount.appendingPathComponent("note.txt")
        try "a note about the harbor".write(to: file, atomically: true, encoding: .utf8)
        let handle = try FileHandle(forReadingFrom: file)
        disk.pull()
        XCTAssertThrowsError(try handle.read(upToCount: 16), "the volume did not go")
        try? handle.close()
        XCTAssertThrowsError(try FileExtractor.extract(file), "a file that is not there read as something")
        _ = db
    }

    // MARK: - 11. A deck whose drawing did not finish

    private func helper(_ body: String, in dir: URL) throws -> URL {
        let u = dir.appendingPathComponent("fake-pages-\(UUID().uuidString).sh")
        try ("#!/bin/sh\n" + body + "\n").write(to: u, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: u.path)
        return u
    }

    func testADeckWhoseDrawingTimedOutIsDrawnLater() throws {
        let office = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Tools/eval/office", isDirectory: true)
        let (root, db) = try folder("deck")
        let deck = root.appendingPathComponent("deck.pptx")
        try FileManager.default.copyItem(at: office.appendingPathComponent("volcanoes.pptx"), to: deck)
        let slide = db.appendingPathComponent("slide.png")
        png(slide, side: 96, shade: 0.3)
        let hang = try helper("exec sleep 300", in: db)
        let draws = db.appendingPathComponent("draws")
        let works = try helper("""
        echo x >> '\(draws.path)'
        mkdir -p "$2"; cp '\(slide.path)' "$2/s1.png"
        printf '[{"label":"Slide 1","image":"s1.png","text":""}]' > "$2/pages.json"
        """, in: db)
        let wasHelper = DocumentPictures.helper, wasTimeout = DocumentPictures.timeout
        addTeardownBlock { DocumentPictures.helper = wasHelper; DocumentPictures.timeout = wasTimeout }
        DocumentPictures.timeout = 0.5

        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let fake = Fake()
        let settings = IndexSettings(enabledKinds: [.text])
        DocumentPictures.helper = hang
        _ = pass(Indexer(store: store, embedder: fake), [root], settings)
        let words = try XCTUnwrap(store.storedChunks(path: deck.path))
        XCTAssertFalse(words.isEmpty, "its words wait for nothing")
        XCTAssertFalse(words.contains { $0.kind == FileKind.scan.rawValue })
        XCTAssertNil(store.contentKeys(paths: [deck.path])[deck.path], "a copy would be served the deck without its pictures")

        DocumentPictures.helper = works
        let indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root], settings)
        let rows = try XCTUnwrap(store.storedChunks(path: deck.path))
        XCTAssertTrue(rows.contains { $0.kind == FileKind.scan.rawValue }, "the deck stayed without its pictures")
        XCTAssertEqual(indexer.streams.count, 0)
        XCTAssertNotNil(store.contentKeys(paths: [deck.path])[deck.path])
        // Finished: not drawn again.
        let drew = (try? String(contentsOf: draws, encoding: .utf8))?.count
        _ = pass(Indexer(store: store, embedder: fake), [root], settings)
        XCTAssertEqual((try? String(contentsOf: draws, encoding: .utf8))?.count, drew)
    }

    /// A deck that never finishes costs its timeout on a few passes, not on every pass for good.
    func testADeckThatNeverDrawsStopsBeingTried() throws {
        let office = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Tools/eval/office", isDirectory: true)
        let (root, db) = try folder("deck-never")
        let deck = root.appendingPathComponent("deck.pptx")
        try FileManager.default.copyItem(at: office.appendingPathComponent("volcanoes.pptx"), to: deck)
        let draws = db.appendingPathComponent("draws")
        let hang = try helper("echo x >> '\(draws.path)'\nexec sleep 300", in: db)
        let wasHelper = DocumentPictures.helper, wasTimeout = DocumentPictures.timeout
        addTeardownBlock { DocumentPictures.helper = wasHelper; DocumentPictures.timeout = wasTimeout }
        DocumentPictures.timeout = 0.5
        DocumentPictures.helper = hang
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let fake = Fake()
        for _ in 0 ..< 5 { _ = pass(Indexer(store: store, embedder: fake), [root], IndexSettings(enabledKinds: [.text])) }
        let tries = (try String(contentsOf: draws, encoding: .utf8)).filter { $0 == "x" }.count
        XCTAssertEqual(tries, Indexer.pictureTries)
        XCTAssertEqual(Indexer(store: store, embedder: fake).streams.count, 0)
        XCTAssertFalse(try XCTUnwrap(store.storedChunks(path: deck.path)).isEmpty)
    }

    // MARK: - 1. The one-time repair, and 10. what a stamp may not lose

    /// What the unfixed watcher path left: rows of the file's raw bytes under its real date, and the
    /// content key it recorded for them, in the key's old form (no "x1").
    private func damage(_ file: URL, store: VectorStore, indexer: Indexer, settings: IndexSettings) throws {
        let raw = try Data(contentsOf: file)
        let text = String(data: raw, encoding: .isoLatin1) ?? ""
        let pieces = indexer.chunk(text, settings: settings, origin: .plain, filterOpaque: false)
        var st = stat()
        XCTAssertEqual(stat(file.path, &st), 0)
        let modified = Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9
        let fake = Fake()
        let rows = pieces.enumerated().map { j, p in
            IndexedChunk(path: file.path, modified: modified, size: Int(st.st_size), kind: "text", chunkIndex: j,
                         snippet: String(p.text.prefix(220)), embedding: fake.vec(p.text), locator: p.locator,
                         chunkKey: indexer.chunkKey(p.text, settings: settings))
        }
        try store.replaceMany([(file.path, rows)])
        let key = try XCTUnwrap(Indexer.fileContentKey(CrawledFile(path: file.path, modified: modified, size: Int(st.st_size)), category: .text, dim: 16,
                                                       chunkOverlap: indexer.chunkOverlap, settings: settings))
        store.recordContentKeys([(file.path, key.replacingOccurrences(of: "|x1", with: ""), modified, Int(st.st_size))])
    }

    private func keysOfText(_ file: URL, indexer: Indexer, settings: IndexSettings) throws -> [String] {
        guard case .text(let whole) = try FileExtractor.extract(file) else { XCTFail("did not convert"); return [] }
        return indexer.chunk(whole, settings: settings, origin: .opaque).map { indexer.chunkKey($0.text, settings: settings) }
    }

    func testTheOneTimeRepairReadsDamagedDocumentsAndEmbedsOnlyThem() throws {
        let (root, db) = try folder("repair")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let settings = IndexSettings(enabledKinds: [.text])
        var healthy: [URL] = []
        for i in 0 ..< 6 {
            let u = root.appendingPathComponent("healthy\(i).\(i % 2 == 0 ? "rtf" : "docx")")
            try document(u, tag: "healthy\(i)", paragraphs: i < 3 ? 400 : 12)
            healthy.append(u)
        }
        let fake = Fake()
        var indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        Indexer.stampRereads(store: store, settings: settings, progress: IndexProgress(), notWalked: [])
        let healthyKeys = try healthy.map { try XCTUnwrap(store.storedChunks(path: $0.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey) }

        // New long documents came through the unfixed watcher path.
        var damaged: [URL] = []
        for (i, ext) in ["rtf", "docx"].enumerated() {
            let u = root.appendingPathComponent("damaged\(i).\(ext)")
            try document(u, tag: "damaged\(i)")
            try damage(u, store: store, indexer: indexer, settings: settings)
            damaged.append(u)
        }
        let damagedRows = try damaged.map { try XCTUnwrap(store.storedChunks(path: $0.path)).map(\.chunkKey) }
        let want = try damaged.map { try keysOfText($0, indexer: indexer, settings: settings) }
        XCTAssertNotEqual(damagedRows, want, "the fixture is not damaged")

        // NEGATIVE CONTROL: once stamped, nothing visits them.
        indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        XCTAssertEqual(try damaged.map { try XCTUnwrap(store.storedChunks(path: $0.path)).map(\.chunkKey) }, damagedRows,
                       "a pass read them with the stamp set")

        // An index from before the repair: no stamp.
        store.metaSet(Indexer.documentsReadMetaKey, "")
        let before = fake.texts
        indexer = Indexer(store: store, embedder: fake)
        let p = pass(indexer, [root])
        XCTAssertFalse(p.cancelled)
        for (u, keys) in zip(damaged, want) {
            let rows = try XCTUnwrap(store.storedChunks(path: u.path)).sorted { $0.chunkIndex < $1.chunkIndex }
            XCTAssertEqual(rows.map(\.chunkKey), keys, "\(u.lastPathComponent) was not repaired")
            XCTAssertFalse(rows.contains { $0.snippet.hasPrefix("PK") || $0.snippet.hasPrefix("{\\rtf") })
        }
        // Only what it read differently reached the model: the healthy documents' vectors came back.
        XCTAssertEqual(fake.texts - before, Set(want.flatMap { $0 }).count, "something other than the damaged documents was embedded")
        for (u, keys) in zip(healthy, healthyKeys) {
            XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: u.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey), keys, "\(u.lastPathComponent) moved")
        }
        // Stamped, and a second pass has nothing to do.
        Indexer.stampRereads(store: store, settings: settings, progress: p, notWalked: [])
        let after = fake.texts
        indexer = Indexer(store: store, embedder: fake)
        let q = pass(indexer, [root])
        XCTAssertEqual(fake.texts, after)
        XCTAssertEqual(q.embedded, 0)
    }

    /// A repair cancelled part-way starts again, and what it already repaired answers by its key:
    /// not converted again, nothing embedded.
    func testACancelledRepairDoesNotPayTwice() throws {
        let (root, db) = try folder("repair-resume")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let settings = IndexSettings(enabledKinds: [.text])
        for i in 0 ..< 4 { try document(root.appendingPathComponent("d\(i).rtf"), tag: "d\(i)", paragraphs: 20) }
        let fake = Fake()
        var indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        // A repair pass, not stamped: as if it stopped before its end.
        store.metaSet(Indexer.documentsReadMetaKey, "")
        indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        let before = fake.texts
        let keys = store.contentKeys(paths: [root.appendingPathComponent("d0.rtf").path])
        indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        XCTAssertEqual(fake.texts, before)
        XCTAssertEqual(store.contentKeys(paths: [root.appendingPathComponent("d0.rtf").path]).map(\.value.key), keys.map(\.value.key))
    }

    /// Problem 10: a pass that left a folder out does not lose its files' repair. Not with a folder it
    /// could not read, and not with a paused root; and it does not read everything else again.
    func testAStampDoesNotCoverWhatThePassDidNotRead() throws {
        let (base, db) = try folder("owed")
        let root = base.appendingPathComponent("root", isDirectory: true)
        let locked = root.appendingPathComponent("locked", isDirectory: true)
        let paused = base.appendingPathComponent("paused", isDirectory: true)
        for d in [locked, paused] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
        addTeardownBlock { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path) }
        let open = root.appendingPathComponent("open.rtf"), inLocked = locked.appendingPathComponent("inlocked.rtf"), inPaused = paused.appendingPathComponent("inpaused.rtf")
        for u in [open, inLocked, inPaused] { try document(u, tag: u.lastPathComponent) }
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let settings = IndexSettings(enabledKinds: [.text])
        let fake = Fake()
        let settled = Indexer(store: store, embedder: fake)
        _ = pass(settled, [root, paused])
        for u in [open, inLocked, inPaused] {
            try damage(u, store: store, indexer: settled, settings: settings)
        }
        let want = try [open, inLocked, inPaused].map { try keysOfText($0, indexer: settled, settings: settings) }
        store.metaSet(Indexer.documentsReadMetaKey, "")

        // The repair's pass: one subfolder cannot be opened, one root is paused and not given.
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: locked.path)
        var indexer = Indexer(store: store, embedder: fake)
        let p = pass(indexer, [root])
        XCTAssertFalse(p.cancelled)
        XCTAssertEqual(p.notRead, [locked.path], "the pass did not say what it could not read")
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: open.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey), want[0], "the file it could read was not repaired")
        Indexer.stampRereads(store: store, settings: settings, progress: p, notWalked: [paused.path])
        XCTAssertEqual(store.metaGet(Indexer.documentsReadMetaKey), "1")

        // Both back. Their files are read, once; what was already read is not.
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path)
        let firstRows = try XCTUnwrap(store.fileStatus(paths: [open.path])[open.path])
        indexer = Indexer(store: store, embedder: fake)
        let q = pass(indexer, [root, paused])
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: inLocked.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey), want[1], "the folder it could not read lost its repair")
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: inPaused.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey), want[2], "the paused folder lost its repair")
        XCTAssertEqual(store.fileStatus(paths: [open.path])[open.path]?.modified, firstRows.modified)
        XCTAssertEqual(q.unchanged, 1, "the file already repaired was read again")
        Indexer.stampRereads(store: store, settings: settings, progress: q, notWalked: [])
        XCTAssertEqual(store.metaGet(Indexer.rereadsOwedMetaKey), "[]")
        let done = fake.texts
        indexer = Indexer(store: store, embedder: fake)
        let r = pass(indexer, [root, paused])
        XCTAssertEqual(r.unchanged, 3)
        XCTAssertEqual(fake.texts, done)

        // Settled now: a folder paused over a pass owes nothing, and nothing is read on its resume.
        Indexer.stampRereads(store: store, settings: settings, progress: r, notWalked: [paused.path])
        XCTAssertEqual(store.metaGet(Indexer.rereadsOwedMetaKey), "[]")
        indexer = Indexer(store: store, embedder: fake)
        XCTAssertEqual(pass(indexer, [root, paused]).unchanged, 3)
    }

    /// A document that could not be read during the repair (no permission for the moment, a save
    /// under way) owes it still, though its folder could be listed.
    func testADocumentNotReadDuringTheRepairIsTriedAgain() throws {
        let (root, db) = try folder("owed-file")
        let doc = root.appendingPathComponent("d.rtf")
        try document(doc, tag: "d")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let settings = IndexSettings(enabledKinds: [.text])
        let fake = Fake()
        var indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        try damage(doc, store: store, indexer: indexer, settings: settings)
        let want = try keysOfText(doc, indexer: indexer, settings: settings)
        store.metaSet(Indexer.documentsReadMetaKey, "")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: doc.path)
        addTeardownBlock { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: doc.path) }
        indexer = Indexer(store: store, embedder: fake)
        let p = pass(indexer, [root])
        XCTAssertEqual(p.notRead, [root.path])
        Indexer.stampRereads(store: store, settings: settings, progress: p, notWalked: [])
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: doc.path)
        indexer = Indexer(store: store, embedder: fake)
        _ = pass(indexer, [root])
        XCTAssertEqual(try XCTUnwrap(store.storedChunks(path: doc.path)).sorted { $0.chunkIndex < $1.chunkIndex }.map(\.chunkKey), want, "it lost its repair")
    }

    /// A pass with the text kind off, or one where a file failed, proved nothing about text.
    func testAStampNeedsThePassToHaveReadText() throws {
        let (_, db) = try folder("stamp-kinds")
        let store = try VectorStore(dbURL: db.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        Indexer.stampRereads(store: store, settings: IndexSettings(enabledKinds: [.image]), progress: IndexProgress(), notWalked: [])
        XCTAssertNil(store.metaGet(Indexer.documentsReadMetaKey))
        var failed = IndexProgress(); failed.failed = 1
        Indexer.stampRereads(store: store, settings: IndexSettings(enabledKinds: [.text]), progress: failed, notWalked: [])
        XCTAssertNil(store.metaGet(Indexer.documentsReadMetaKey))
        Indexer.stampRereads(store: store, settings: IndexSettings(enabledKinds: [.text]), progress: IndexProgress(), notWalked: [])
        XCTAssertEqual(store.metaGet(Indexer.documentsReadMetaKey), "1")
    }
}
