import Foundation
import AppKit
import AVFoundation
import CoreText
import ImageIO
import UniformTypeIdentifiers
import OmniKit

// The labeled retrieval eval: what search QUALITY a change buys or costs, measured through the
// app's own path (index with the app's settings and tagger, `store.search` with the query text, the
// filename channel), on a testbed whose answers are known.
//
//   omni-verify evalbed <dir> [spec.json]     generate the testbed (Tools/eval/bed.json by default)
//   omni-verify eval <modelDir> <dir> [--json out.json] [--rebuild] [--tag t]
//
// THE TESTBED is generated on the Mac from the spec and from what every Mac has: the built-in user
// pictures, the offline voices (`say`), and pages, screenshots and video drawn here. Documents in ten
// languages with confusable near-duplicates (INV-2041 / INV-2014, OPS-0417 / OPS-0471), PDFs with a
// fact per page and a scanned receipt, a 120-page log, screenshots marked as screen captures, voice
// memos in nine languages, a ten-minute lecture and a nine-minute video whose topics change on
// segment boundaries, a 3.6 MB log with one line that matters, and distractor notes. Photos are
// named like a camera names them: a filename must not give an answer away.
//
// THE TREE is `<dir>/tree`; the spec and the manifest sit beside it, outside the crawl. The index is
// kept in `<dir>/index-<model>-<hash>`, keyed by the tree's bytes, so A/B runs of the search reuse
// it and a changed testbed or model builds a new one.
//
// SCORING. A query lists acceptable answers; the rank is the first hit that is one of them. An
// answer may name a place: `@p2` the page, `@t240` the media segment starting there, `@L100-200` a
// chunk starting on one of those lines - a hit on the right file at the wrong place does not count.
// R@1, R@5, R@10 and MRR per tag and overall.

struct EvalSpec: Decodable {
    struct Doc: Decodable { let path: String; let text: String }
    struct PDF: Decodable {
        let path: String
        let pages: [String]?
        let scanned: Bool?
        let longPages: Int?
        let filler: String?
        let facts: [String: String]?
    }
    struct Shot: Decodable { let path: String; let title: String; let lines: [String] }
    struct Speech: Decodable { let path: String; let voice: String; let text: String }
    struct LongAudio: Decodable {
        struct Segment: Decodable { let start: Double; let text: String }
        let path: String; let duration: Double; let voice: String; let segments: [Segment]
    }
    struct Photo: Decodable { let src: String; let path: String }
    struct Video: Decodable {
        struct Scene: Decodable { let start: Double; let src: String }
        let path: String; let duration: Double; let scenes: [Scene]
    }
    struct LongText: Decodable { let path: String; let lines: Int; let filler: String; let needles: [String: String] }
    struct Distractors: Decodable { let count: Int; let folders: [String]; let topics: [[String]] }
    struct Query: Decodable { let q: String; let tag: String; let expect: [String] }

    let version: String
    let docs: [Doc]
    let pdfs: [PDF]
    let screenshots: [Shot]
    let speech: [Speech]
    let longAudio: [LongAudio]
    let photos: [Photo]
    let videos: [Video]
    let longText: [LongText]
    let distractors: Distractors
    let queries: [Query]
}

enum EvalCommand {
    static func run(_ args: [String]) async {
        do {
            if args[1] == "evalbed" { try generate(args) } else { try await evaluate(args) }
        } catch {
            print("eval: \(error)")
            exit(1)
        }
    }

    struct Failure: Error, CustomStringConvertible { let description: String; init(_ s: String) { description = s } }

    // MARK: - Generate

    static func generate(_ args: [String]) throws {
        guard args.count >= 3 else { throw Failure("usage: omni-verify evalbed <dir> [spec.json]") }
        let out = URL(fileURLWithPath: (args[2] as NSString).expandingTildeInPath, isDirectory: true)
        let specURL = URL(fileURLWithPath: args.count >= 4 ? args[3] : "Tools/eval/bed.json")
        let specData = try Data(contentsOf: specURL)
        let spec = try JSONDecoder().decode(EvalSpec.self, from: specData)
        let tree = out.appendingPathComponent("tree", isDirectory: true)
        let fm = FileManager.default
        if fm.fileExists(atPath: tree.path) {
            // Only a tree this tool wrote: the manifest beside it says so.
            guard fm.fileExists(atPath: out.appendingPathComponent("manifest.json").path) else {
                throw Failure("\(tree.path) exists and was not written by evalbed; refusing to replace it")
            }
            try fm.removeItem(at: tree)
        }
        try fm.createDirectory(at: tree, withIntermediateDirectories: true)
        let t0 = Date()
        func dest(_ rel: String) throws -> URL {
            let u = tree.appendingPathComponent(rel)
            try fm.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            return u
        }

        for d in spec.docs { try writeDocument(d.text, to: try dest(d.path)) }
        for p in spec.pdfs { try writePDF(p, to: try dest(p.path)) }
        for s in spec.screenshots { try writeScreenshot(s, to: try dest(s.path)) }
        let scratch = out.appendingPathComponent("scratch", isDirectory: true)
        try fm.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: scratch) }
        for s in spec.speech { try speak(s.text, voice: s.voice, to: try dest(s.path), scratch: scratch) }
        for a in spec.longAudio { try writeLongAudio(a, to: try dest(a.path), scratch: scratch) }
        // The built-in pictures are symlinks into the system template: copy what they point at.
        for p in spec.photos { try fm.copyItem(at: URL(fileURLWithPath: p.src).resolvingSymlinksInPath(), to: try dest(p.path)) }
        for v in spec.videos { try writeVideo(v, to: try dest(v.path)) }
        for l in spec.longText { try writeLongText(l, to: try dest(l.path)) }
        try writeDistractors(spec.distractors, tree: tree)

        try specData.write(to: out.appendingPathComponent("bed.json"))
        let hash = try treeHash(tree)
        let manifest: [String: Any] = ["version": spec.version, "tree": hash, "files": try fileCount(tree)]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: out.appendingPathComponent("manifest.json"))
        print(String(format: "evalbed: %d files, tree %@, %.1fs -> %@", try fileCount(tree), hash, -t0.timeIntervalSinceNow, tree.path))
    }

    static func writeDocument(_ text: String, to url: URL) throws {
        switch url.pathExtension.lowercased() {
        case "docx", "rtf":
            let attr = NSAttributedString(string: text, attributes: [.font: NSFont.systemFont(ofSize: 12)])
            let type: NSAttributedString.DocumentType = url.pathExtension.lowercased() == "docx" ? .officeOpenXML : .rtf
            let data = try attr.data(from: NSRange(location: 0, length: attr.length), documentAttributes: [.documentType: type])
            try data.write(to: url)
        default:
            try Data(text.utf8).write(to: url)
        }
    }

    /// Text laid out by Core Text in a frame, so any script finds a font (CJK, Arabic, Devanagari).
    static func drawText(_ text: String, in rect: CGRect, ctx: CGContext, size: CGFloat, color: CGColor = .black) {
        let attr = NSAttributedString(string: text, attributes: [
            .font: NSFont.systemFont(ofSize: size), .foregroundColor: NSColor(cgColor: color) ?? .black,
        ])
        let setter = CTFramesetterCreateWithAttributedString(attr)
        let frame = CTFramesetterCreateFrame(setter, CFRange(location: 0, length: 0), CGPath(rect: rect, transform: nil), nil)
        CTFrameDraw(frame, ctx)
    }

    static func writePDF(_ p: EvalSpec.PDF, to url: URL) throws {
        var pages = p.pages ?? []
        if let n = p.longPages {
            pages = (1 ... n).map { day in
                var text = p.facts?[String(day)].map { $0 + "\n\n" } ?? ""
                text += fillerPage(day: day, kind: p.filler ?? "sailing")
                return text
            }
        }
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { throw Failure("pdf context \(url.path)") }
        for text in pages {
            ctx.beginPDFPage(nil)
            if p.scanned == true {
                // A scan: the page is a picture of text, with no text layer for the extractor to read.
                let scale: CGFloat = 150.0 / 72.0
                let w = Int(612 * scale), h = Int(792 * scale)
                guard let bmp = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
                else { throw Failure("scan bitmap") }
                bmp.setFillColor(gray: 0.97, alpha: 1); bmp.fill(CGRect(x: 0, y: 0, width: w, height: h))
                bmp.scaleBy(x: scale, y: scale)
                drawText(text, in: CGRect(x: 150, y: 200, width: 320, height: 500), ctx: bmp, size: 14)
                if let img = bmp.makeImage() { ctx.draw(img, in: box) }
            } else {
                drawText(text, in: box.insetBy(dx: 72, dy: 72), ctx: ctx, size: 12)
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    /// A day of an unremarkable log: wind, course, distance. Deterministic in the day.
    static func fillerPage(day: Int, kind: String) -> String {
        let dirs = ["north", "north east", "east", "south east", "south", "south west", "west", "north west"]
        let seas = ["calm", "slight", "moderate", "rough"]
        let chores = ["Checked the rigging and greased the winches.", "Cooked rice and beans, washed up in salt water.",
                      "Read in the cockpit during the afternoon watch.", "Topped up the water tanks from the watermaker.",
                      "Changed the jib for the genoa in the morning.", "Logged the barometer every four hours, steady."]
        var lines: [String] = []
        for h in stride(from: 0, to: 24, by: 3) {
            let k = (day * 31 + h * 7) % 97
            lines.append(String(format: "Day %d, %02d:00. Wind from the %@ at %d knots, sea %@. Course %03d, %d nautical miles since noon.",
                                day, h, dirs[k % dirs.count], 8 + k % 14, seas[k % seas.count], (k * 37) % 360, 5 + k % 40))
        }
        lines.append(chores[day % chores.count])
        return lines.joined(separator: "\n")
    }

    static func writeScreenshot(_ s: EvalSpec.Shot, to url: URL) throws {
        let w = 1280, h = 800
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw Failure("screenshot bitmap") }
        ctx.setFillColor(CGColor(red: 0.86, green: 0.88, blue: 0.91, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let win = CGRect(x: 140, y: 120, width: 1000, height: 560)
        ctx.setFillColor(.white); ctx.fill(win)
        ctx.setFillColor(CGColor(red: 0.93, green: 0.93, blue: 0.94, alpha: 1))
        ctx.fill(CGRect(x: win.minX, y: win.maxY - 44, width: win.width, height: 44))
        for (i, c) in [CGColor(red: 1, green: 0.37, blue: 0.34, alpha: 1), CGColor(red: 1, green: 0.74, blue: 0.18, alpha: 1),
                       CGColor(red: 0.16, green: 0.78, blue: 0.25, alpha: 1)].enumerated() {
            ctx.setFillColor(c)
            ctx.fillEllipse(in: CGRect(x: win.minX + 18 + CGFloat(i) * 22, y: win.maxY - 29, width: 13, height: 13))
        }
        drawText(s.title, in: CGRect(x: win.midX - 200, y: win.maxY - 40, width: 400, height: 30), ctx: ctx, size: 15,
                 color: CGColor(gray: 0.3, alpha: 1))
        drawText(s.lines.joined(separator: "\n"), in: CGRect(x: win.minX + 48, y: win.minY + 40, width: win.width - 96, height: win.height - 120),
                 ctx: ctx, size: 26)
        guard let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw Failure("png") }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { throw Failure("png write") }
        // Marked as a screen capture, the way macOS marks its own (Indexer.isScreenshot reads it).
        let flag = try PropertyListSerialization.data(fromPropertyList: true, format: .binary, options: 0)
        _ = flag.withUnsafeBytes { setxattr(url.path, "com.apple.metadata:kMDItemIsScreenCapture", $0.baseAddress, flag.count, 0, 0) }
    }

    static func runTool(_ path: String, _ args: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        try p.run(); p.waitUntilExit()
        guard p.terminationStatus == 0 else { throw Failure("\(path) \(args.prefix(3)) exited \(p.terminationStatus)") }
    }

    static func speak(_ text: String, voice: String, to url: URL, scratch: URL) throws {
        let aiff = scratch.appendingPathComponent(UUID().uuidString + ".aiff")
        defer { try? FileManager.default.removeItem(at: aiff) }
        try runTool("/usr/bin/say", ["-v", voice, "-o", aiff.path, text])
        try runTool("/usr/bin/afconvert", ["-f", "m4af", "-d", "aac", aiff.path, url.path])
    }

    /// Spoken segments placed at their start times in silence, so each topic owns a 240 s segment.
    static func writeLongAudio(_ a: EvalSpec.LongAudio, to url: URL, scratch: URL) throws {
        var pieces: [(Double, AVAudioPCMBuffer)] = []
        var format: AVAudioFormat?
        for s in a.segments {
            let aiff = scratch.appendingPathComponent(UUID().uuidString + ".aiff")
            defer { try? FileManager.default.removeItem(at: aiff) }
            try runTool("/usr/bin/say", ["-v", a.voice, "-o", aiff.path, s.text])
            let f = try AVAudioFile(forReading: aiff)
            guard let buf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length)) else { throw Failure("audio buffer") }
            try f.read(into: buf)
            format = f.processingFormat
            pieces.append((s.start, buf))
        }
        guard let format else { throw Failure("no segments") }
        let total = AVAudioFrameCount(a.duration * format.sampleRate)
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: total), let dstData = out.floatChannelData else { throw Failure("mix buffer") }
        out.frameLength = total
        for ch in 0 ..< Int(format.channelCount) { memset(dstData[ch], 0, Int(total) * MemoryLayout<Float>.size) }
        for (start, buf) in pieces {
            guard let src = buf.floatChannelData else { continue }
            let at = Int(start * format.sampleRate)
            let n = min(Int(buf.frameLength), Int(total) - at)
            guard n > 0 else { continue }
            for ch in 0 ..< Int(format.channelCount) { memcpy(dstData[ch] + at, src[ch], n * MemoryLayout<Float>.size) }
        }
        let wav = scratch.appendingPathComponent(UUID().uuidString + ".caf")
        defer { try? FileManager.default.removeItem(at: wav) }
        do {
            let file = try AVAudioFile(forWriting: wav, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            try file.write(from: out)
        }
        try runTool("/usr/bin/afconvert", ["-f", "m4af", "-d", "aac", wav.path, url.path])
    }

    static func loadImage(_ path: String) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw Failure("image \(path)") }
        return img
    }

    /// Stills at one frame a second, each scene from its start until the next.
    static func writeVideo(_ v: EvalSpec.Video, to url: URL) throws {
        let w = 480, h = 360
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        writer.add(input)
        guard writer.startWriting() else { throw Failure("video writer: \(writer.error.map { "\($0)" } ?? "?")") }
        writer.startSession(atSourceTime: .zero)
        let images = try v.scenes.map { try loadImage($0.src) }
        for t in 0 ..< Int(v.duration) {
            let scene = v.scenes.lastIndex { $0.start <= Double(t) } ?? 0
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            var pb: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess, let pb else { throw Failure("pixel buffer") }
            CVPixelBufferLockBaseAddress(pb, [])
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
                ctx.setFillColor(.black); ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
                ctx.draw(images[scene], in: CGRect(x: (w - h) / 2, y: 0, width: h, height: h))
            }
            CVPixelBufferUnlockBaseAddress(pb, [])
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(t), timescale: 1))
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        guard writer.status == .completed else { throw Failure("video: \(writer.error.map { "\($0)" } ?? "?")") }
    }

    static func writeLongText(_ l: EvalSpec.LongText, to url: URL) throws {
        let topics = ["Standup: reviewed the board, no blockers.", "Design sync: agreed to keep the sidebar collapsed by default.",
                      "Hiring: two phone screens scheduled for next week.", "Ops: rotated the on-call schedule.",
                      "Retro: shorter planning meetings, demo every other Friday.", "Budget: travel stays frozen until the review."]
        var out = ""
        out.reserveCapacity(l.lines * 72)
        var seed: UInt64 = 0x5EED
        for n in 1 ... l.lines {
            if let needle = l.needles[String(n)] { out += needle + "\n"; continue }
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let r = Int(seed >> 33)
            if l.filler == "sensor" {
                out += String(format: "2026-03-%02dT%02d:%02d:%02dZ,sensor-%02d,temp=%.1fC,humidity=%d%%,pressure=%.1fkPa\n",
                              1 + (n / 2880) % 28, (n / 120) % 24, (n / 2) % 60, (n * 30) % 60, r % 16,
                              18 + Double(r % 80) / 10, 30 + r % 40, 100 + Double(r % 30) / 10)
            } else {
                out += "- Week \(1 + n / 600): " + topics[r % topics.count] + "\n"
            }
        }
        try Data(out.utf8).write(to: url)
    }

    static func writeDistractors(_ d: EvalSpec.Distractors, tree: URL) throws {
        for i in 0 ..< d.count {
            let pool = d.topics[i % d.topics.count]
            let take = 3 + i % 4
            let start = (i / d.topics.count) % pool.count
            let text = (0 ..< take).map { pool[(start + $0) % pool.count] }.joined(separator: " ")
            let u = tree.appendingPathComponent(d.folders[i % d.folders.count])
                .appendingPathComponent("Note \(i + 1)" + (i % 2 == 0 ? ".txt" : ".md"))
            try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(text.utf8).write(to: u)
        }
    }

    static func files(_ tree: URL) throws -> [URL] {
        guard let en = FileManager.default.enumerator(at: tree, includingPropertiesForKeys: [.isRegularFileKey]) else { return [] }
        return en.compactMap { $0 as? URL }.filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
            .sorted { $0.path < $1.path }
    }
    static func fileCount(_ tree: URL) throws -> Int { try files(tree).count }

    /// FNV-1a over relative paths and bytes: the index is built for exactly this tree.
    static func treeHash(_ tree: URL) throws -> String {
        var h: UInt64 = 0xcbf29ce484222325
        func mix(_ b: UInt8) { h ^= UInt64(b); h = h &* 0x100000001b3 }
        for u in try files(tree) {
            for b in u.path.dropFirst(tree.path.count).utf8 { mix(b) }
            for b in try Data(contentsOf: u) { mix(b) }
        }
        return String(format: "%016llx", h)
    }

    // MARK: - Evaluate

    static func evaluate(_ args: [String]) async throws {
        guard args.count >= 4 else { throw Failure("usage: omni-verify eval <modelDir> <dir> [--json out] [--rebuild] [--tag t]") }
        let modelDir = URL(fileURLWithPath: (args[2] as NSString).expandingTildeInPath)
        let bed = URL(fileURLWithPath: (args[3] as NSString).expandingTildeInPath, isDirectory: true)
        func opt(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        let spec = try JSONDecoder().decode(EvalSpec.self, from: Data(contentsOf: bed.appendingPathComponent("bed.json")))
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: bed.appendingPathComponent("manifest.json"))) as? [String: Any]
        guard let treeHash = manifest?["tree"] as? String else { throw Failure("no manifest: run evalbed first") }
        let tree = bed.appendingPathComponent("tree", isDirectory: true)
        let treePath = URL(fileURLWithPath: String(cString: realpath(tree.path, nil))).path

        let engine = try await OmniEngine(modelDir: modelDir)
        let indexDir = bed.appendingPathComponent("index-\(modelDir.lastPathComponent)-\(engine.dim)-\(treeHash.prefix(8))", isDirectory: true)
        let doneMark = indexDir.appendingPathComponent("built")
        if args.contains("--rebuild") { try? FileManager.default.removeItem(at: indexDir) }
        try FileManager.default.createDirectory(at: indexDir, withIntermediateDirectories: true)

        // The tagger, as the app attaches it (AppModel.ensureTagger): labels gated for this model,
        // a cache built once, the prior seeded before the tagger is used.
        if engine.supportsImages {
            let cache = indexDir.appendingPathComponent("tags.bin")
            if !FileManager.default.fileExists(atPath: cache.path) {
                let labels = OmniTagger.gatedLabels(modelDir: modelDir)
                _ = OmniTagger.buildCache(labels: labels, embedder: engine, to: cache, isCancelled: { false })
            }
            if let tagger = OmniTagger(cacheURL: cache, dim: engine.dim) {
                engine.seedTaggerPrior(tagger)
                engine.tagger = tagger
            }
        }
        let store = try VectorStore(dbURL: indexDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        if !FileManager.default.fileExists(atPath: doneMark.path) {
            let indexer = Indexer(store: store, embedder: engine)
            let t0 = Date()
            let last: IndexProgress = await withCheckedContinuation { cont in
                DispatchQueue.global().async {
                    indexer.index(roots: [URL(fileURLWithPath: treePath)], settings: IndexSettings()) { p in
                        if p.done { cont.resume(returning: p) }
                    }
                }
            }
            store.prepareLexicalIndex()
            print(String(format: "eval: indexed %d files (%d failed, %d skipped) in %.0fs", last.embedded, last.failed, last.skipped, -t0.timeIntervalSinceNow))
            try Data().write(to: doneMark)
        } else {
            store.prepareLexicalIndex()
        }

        let onlyTag = opt("--tag")
        // Two ranks: at the right PLACE (page, line, moment - what opening the hit shows) and at the
        // right FILE. Apart, they tell retrieval from locator precision: a chunk that starts on page
        // 40 and holds page 41's fact is found (file) and opened a page early (place).
        struct Row { var n = 0, at1 = 0, at5 = 0, at10 = 0, file1 = 0, file10 = 0; var mrr = 0.0 }
        var rows: [String: Row] = [:]
        var perQuery: [[String: Any]] = []
        // The weaker-matches fold (VectorStore.weakFold), as the app applies it to this list.
        var shownTotal = 0, hitsTotal = 0, foldedAnswers = 0, foldedTop10 = 0
        for query in spec.queries where onlyTag == nil || query.tag == onlyTag {
            let hits: [SearchHit]
            if let g = engine.queryVectorGraph(query.q) {
                hits = store.search(queryGraph: g, filter: SearchFilter(), topK: VectorStore.shippedTopK, textQuery: query.q).hits
            } else {
                hits = store.search(engine.embedQuery(query.q), filter: SearchFilter(), topK: VectorStore.shippedTopK, textQuery: query.q)
            }
            let rank = hits.firstIndex { hit in query.expect.contains { matches(hit, $0, tree: treePath) } }.map { $0 + 1 }
            let fileRank = hits.firstIndex { hit in
                query.expect.contains { matches(hit, String($0.split(separator: "@")[0]), tree: treePath) }
            }.map { $0 + 1 }
            let weak = VectorStore.weakFold(hits)
            shownTotal += weak.filter { !$0 }.count
            hitsTotal += hits.count
            if let fileRank, weak[fileRank - 1] {
                foldedAnswers += 1
                if fileRank <= 10 { foldedTop10 += 1 }
            }
            for key in [query.tag, "ALL"] {
                var r = rows[key] ?? Row()
                r.n += 1
                if let rank {
                    if rank <= 1 { r.at1 += 1 }
                    if rank <= 5 { r.at5 += 1 }
                    if rank <= 10 { r.at10 += 1 }
                    r.mrr += 1 / Double(rank)
                }
                if let fileRank {
                    if fileRank <= 1 { r.file1 += 1 }
                    if fileRank <= 10 { r.file10 += 1 }
                }
                rows[key] = r
            }
            let top = hits.first.map { String($0.path.dropFirst(treePath.count + 1)) + ($0.locator.isEmpty ? "" : " [\($0.locator)]") } ?? "-"
            let mark = rank.map { String(format: "#%-3d", $0) } ?? "miss"
            print("\(mark) \(query.tag.padding(toLength: 12, withPad: " ", startingAt: 0)) \(query.q)  ->  \(top)")
            perQuery.append(["q": query.q, "tag": query.tag, "rank": rank ?? 0, "fileRank": fileRank ?? 0,
                             "top": hits.prefix(10).map { String($0.path.dropFirst(treePath.count + 1)) + "|" + $0.locator }])
        }
        print("")
        print("tag            n    R@1    R@5   R@10    MRR   file R@1  R@10")
        for (tag, r) in rows.sorted(by: { ($0.key == "ALL" ? "~" : $0.key) < ($1.key == "ALL" ? "~" : $1.key) }) {
            let n = Double(r.n)
            print(tag.padding(toLength: 12, withPad: " ", startingAt: 0)
                  + String(format: " %3d  %5.2f  %5.2f  %5.2f  %5.3f   %5.2f  %5.2f", r.n, Double(r.at1) / n, Double(r.at5) / n,
                                  Double(r.at10) / n, r.mrr / n, Double(r.file1) / n, Double(r.file10) / n))
        }
        let nq = Double(max(1, perQuery.count))
        print(String(format: "weaker-matches fold: %.1f of %.1f results shown per query; %d found answers folded, %d of them in the top 10",
                     Double(shownTotal) / nq, Double(hitsTotal) / nq,
                     foldedAnswers, foldedTop10))
        if let out = opt("--json") {
            let summary = rows.mapValues { r -> [String: Double] in
                let n = Double(r.n)
                return ["n": n, "r1": Double(r.at1) / n, "r5": Double(r.at5) / n, "r10": Double(r.at10) / n, "mrr": r.mrr / n,
                        "file1": Double(r.file1) / n, "file10": Double(r.file10) / n]
            }
            let obj: [String: Any] = ["tree": treeHash, "model": modelDir.lastPathComponent, "summary": summary, "queries": perQuery]
            try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
        }
    }

    /// Is this hit the expected answer, at the expected place?
    static func matches(_ hit: SearchHit, _ expect: String, tree: String) -> Bool {
        let parts = expect.split(separator: "@", maxSplits: 1).map(String.init)
        guard hit.path == tree + "/" + parts[0] else { return false }
        guard parts.count == 2 else { return true }
        let place = parts[1]
        if place.hasPrefix("p"), let page = Int(place.dropFirst()) { return hit.locator == "Page \(page)" }
        if place.hasPrefix("t"), let t = Double(place.dropFirst()) { return seconds(hit.locator) == t }
        if place.hasPrefix("L") {
            let r = place.dropFirst().split(separator: "-").compactMap { Int($0) }
            guard r.count == 2, hit.locator.hasPrefix("Line "), let line = Int(hit.locator.dropFirst(5)) else { return false }
            return (r[0] ... r[1]).contains(line)
        }
        return false
    }

    /// "4:00" or "1:02:03" in seconds; "" is the start.
    static func seconds(_ locator: String) -> Double? {
        if locator.isEmpty { return 0 }
        let n = locator.split(separator: ":").compactMap { Double($0) }
        switch n.count {
        case 2: return n[0] * 60 + n[1]
        case 3: return n[0] * 3600 + n[1] * 60 + n[2]
        default: return nil
        }
    }
}
