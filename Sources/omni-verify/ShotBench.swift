import Foundation
import AppKit
import CoreVideo
import ImageIO
import Vision
import VideoToolbox
import UniformTypeIdentifiers
#if canImport(OmniKit)
import OmniKit
#endif

/// `omni-verify shotbench <source.swift> <prose.md> <photo> [--scale 1|2] [--frames N] [--dump dir]`
///
/// What the screen history costs, on a generated desktop session rather than anybody's real screen:
/// an editor being typed in, scrolled, a browser page with a photo, switching between the two, and a
/// stretch where only the caret blinks. Every frame goes through the app's own encoder
/// (ScreenFrameEncoder) at several qualities and key-frame spacings, and through HEIC and PNG one
/// picture at a time, with the bytes, the time, the decode cost of a jump into the timeline, and
/// what Vision still reads from the decoded frame against what it read from the original.
enum ShotBench {
    // MARK: The generated session

    struct State {
        var app = 0            // 0 editor, 1 browser
        var typed = 0          // characters of the source typed so far
        var top = 0            // first visible line of the editor
        var scrollY = 0.0      // browser scroll, points
        var caret = true
        var minute = 0
    }

    final class Renderer {
        let width: Int, height: Int, scale: CGFloat
        let source: [Character]
        let prose: NSAttributedString
        let photo: CGImage?
        let mono: NSFont, body: NSFont, ui: NSFont
        let palette: [NSColor] = [.init(red: 0.99, green: 0.37, blue: 0.64, alpha: 1), .init(red: 0.40, green: 0.85, blue: 0.94, alpha: 1),
                                  .init(red: 0.65, green: 0.89, blue: 0.18, alpha: 1), .init(red: 0.90, green: 0.86, blue: 0.45, alpha: 1),
                                  .init(red: 0.68, green: 0.51, blue: 1.00, alpha: 1), .init(white: 0.92, alpha: 1), .init(white: 0.92, alpha: 1)]

        init(scale: Int, source: String, prose: String, photo: CGImage?) {
            self.scale = CGFloat(scale); width = 1920 * scale; height = 1080 * scale
            self.source = Array(source); self.photo = photo
            mono = NSFont(name: "Menlo", size: 13)!; body = NSFont.systemFont(ofSize: 15); ui = NSFont.systemFont(ofSize: 13)
            let p = NSMutableParagraphStyle(); p.lineSpacing = 5; p.paragraphSpacing = 10
            self.prose = NSAttributedString(string: prose, attributes: [.font: body, .foregroundColor: NSColor(white: 0.12, alpha: 1), .paragraphStyle: p])
        }

        func render(_ s: State) -> CVPixelBuffer {
            let pb = ScreenCodec.makePixelBuffer(width: width, height: height)!
            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }
            let cg = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: width, height: height, bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
            cg.translateBy(x: 0, y: CGFloat(height)); cg.scaleBy(x: scale, y: -scale)   // points, top-left origin
            let ns = NSGraphicsContext(cgContext: cg, flipped: true)
            NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = ns
            defer { NSGraphicsContext.restoreGraphicsState() }
            if s.app == 0 { editor(s, cg) } else { browser(s, cg) }
            // The menu bar, with a clock that moves once a minute.
            cg.setFillColor(NSColor(white: s.app == 0 ? 0.16 : 0.93, alpha: 1).cgColor); cg.fill(CGRect(x: 0, y: 0, width: 1920, height: 24))
            let fg = s.app == 0 ? NSColor(white: 0.9, alpha: 1) : NSColor(white: 0.1, alpha: 1)
            (s.app == 0 ? "Editor   File   Edit   View   Navigate   Product   Window   Help" : "Browser   File   Edit   View   History   Bookmarks   Window   Help")
                .draw(at: CGPoint(x: 48, y: 4), withAttributes: [.font: ui, .foregroundColor: fg])
            String(format: "Tue 10 Nov  14:%02d", s.minute % 60).draw(at: CGPoint(x: 1780, y: 4), withAttributes: [.font: ui, .foregroundColor: fg])
            return pb
        }

        private func editor(_ s: State, _ cg: CGContext) {
            cg.setFillColor(NSColor(white: 0.12, alpha: 1).cgColor); cg.fill(CGRect(x: 0, y: 0, width: 1920, height: 1080))
            cg.setFillColor(NSColor(white: 0.16, alpha: 1).cgColor); cg.fill(CGRect(x: 0, y: 24, width: 240, height: 1056))
            for (i, name) in ["AppModel.swift", "Sidebar.swift", "ContentView.swift", "Indexer.swift", "VectorStore.swift", "ScreenCodec.swift",
                              "SettingsView.swift", "SearchToolbar.swift", "FolderBrowser.swift", "MCPAdapter.swift"].enumerated() {
                name.draw(at: CGPoint(x: 28, y: 64 + CGFloat(i) * 24), withAttributes: [.font: ui, .foregroundColor: NSColor(white: i == 1 ? 1 : 0.7, alpha: 1)])
            }
            let text = String(source[0 ..< min(s.typed, source.count)])
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            let rows = 56
            let top = min(s.top, max(0, lines.count - 1))
            for r in 0 ..< rows where top + r < lines.count {
                let y = 40 + CGFloat(r) * 18
                String(format: "%4d", top + r + 1).draw(at: CGPoint(x: 252, y: y), withAttributes: [.font: mono, .foregroundColor: NSColor(white: 0.4, alpha: 1)])
                let line = NSMutableAttributedString()
                for word in lines[top + r].split(separator: " ", omittingEmptySubsequences: false) {
                    var h = 5381; for u in word.utf8 { h = (h &* 33) ^ Int(u) }
                    line.append(NSAttributedString(string: word + " ", attributes: [.font: mono, .foregroundColor: palette[abs(h) % palette.count]]))
                }
                line.draw(at: CGPoint(x: 300, y: y))
                if top + r == lines.count - 1, s.caret {
                    cg.setFillColor(NSColor.white.cgColor)
                    cg.fill(CGRect(x: 300 + line.size().width - 7, y: y, width: 2, height: 16))
                }
            }
        }

        private func browser(_ s: State, _ cg: CGContext) {
            cg.setFillColor(NSColor.white.cgColor); cg.fill(CGRect(x: 0, y: 0, width: 1920, height: 1080))
            cg.saveGState()
            cg.clip(to: CGRect(x: 0, y: 76, width: 1920, height: 1004))
            let y0 = 100 - CGFloat(s.scrollY)
            if let photo {
                // Drawn upright: the context is flipped for text.
                cg.saveGState(); cg.translateBy(x: 420, y: y0 + 520); cg.scaleBy(x: 1, y: -1)
                cg.draw(photo, in: CGRect(x: 0, y: 0, width: 924, height: 520)); cg.restoreGState()
            }
            prose.draw(with: CGRect(x: 420, y: y0 + 560, width: 924, height: 100_000), options: [.usesLineFragmentOrigin])
            cg.restoreGState()
            cg.setFillColor(NSColor(white: 0.95, alpha: 1).cgColor); cg.fill(CGRect(x: 0, y: 24, width: 1920, height: 52))
            "omni.jina.ai/docs/clipboard".draw(at: CGPoint(x: 640, y: 42), withAttributes: [.font: ui, .foregroundColor: NSColor(white: 0.25, alpha: 1)])
        }
    }

    /// (phase name, state, the user stopped here) per frame: a shot every five seconds of a working
    /// half hour. The last frame of each stretch is one the user paused on.
    static func session(source: [Character], proseHeight: Double, frames: Int) -> [(String, State, Bool)] {
        var out: [(String, State, Bool)] = []
        var newlines = [Int](repeating: 0, count: source.count + 1)
        for (i, c) in source.enumerated() { newlines[i + 1] = newlines[i] + (c == "\n" ? 1 : 0) }
        var s = State(); s.typed = min(source.count, 1400)
        var rng = SystemRandomNumberGeneratorSeeded(seed: 7)
        func push(_ phase: String) { s.minute = out.count / 12; out.append((phase, s, false)) }
        func pause() { if !out.isEmpty { out[out.count - 1].2 = true } }
        func type() { s.typed = min(source.count, s.typed + Int.random(in: 8 ... 22, using: &rng)); s.top = max(0, newlines[s.typed] - 40) }
        let unit = max(1, frames / 150)
        for _ in 0 ..< 60 * unit { type(); push("type") }
        pause()
        s.typed = min(source.count, 40_000); s.top = 0
        for _ in 0 ..< 15 * unit { s.top = min(s.top + Int.random(in: 6 ... 25, using: &rng), max(0, newlines[s.typed] - 50)); push("scroll") }
        pause()
        s.app = 1
        for _ in 0 ..< 15 * unit {
            s.scrollY += Double(Int.random(in: 120 ... 400, using: &rng))
            if s.scrollY > proseHeight - 300 { s.scrollY = 0 }
            push("browse")
        }
        pause()
        s.app = 0; s.typed = min(source.count, 9000)
        for _ in 0 ..< 30 * unit { type(); push("type2") }
        pause()
        for _ in 0 ..< 20 * unit { s.caret.toggle(); push("caret") }
        for _ in 0 ..< 10 * unit { s.app = 1 - s.app; push("switch") }
        pause()
        return out
    }

    struct SystemRandomNumberGeneratorSeeded: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 { state &+= 0x9E3779B97F4A7C15; var z = state; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) }
    }

    // MARK: Measures

    static func ms(_ t: Date) -> Double { Date().timeIntervalSince(t) * 1000 }

    static func lines(_ pixels: CVPixelBuffer) -> [String] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate; request.usesLanguageCorrection = false; request.automaticallyDetectsLanguage = true
        try? VNImageRequestHandler(cvPixelBuffer: pixels).perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    }

    static func psnr(_ a: CVPixelBuffer, _ b: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(a, .readOnly); CVPixelBufferLockBaseAddress(b, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(a, .readOnly); CVPixelBufferUnlockBaseAddress(b, .readOnly) }
        let w = CVPixelBufferGetWidth(a), h = CVPixelBufferGetHeight(a)
        let pa = CVPixelBufferGetBaseAddress(a)!.assumingMemoryBound(to: UInt8.self), ra = CVPixelBufferGetBytesPerRow(a)
        let pb = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self), rb = CVPixelBufferGetBytesPerRow(b)
        var se = 0.0
        for y in 0 ..< h {
            var row = 0
            for x in 0 ..< w * 4 where x % 4 != 3 { let d = Int(pa[y * ra + x]) - Int(pb[y * rb + x]); row += d * d }
            se += Double(row)
        }
        let mse = se / Double(w * h * 3)
        return mse == 0 ? .infinity : 10 * log10(255 * 255 / mse)
    }

    /// `hardware`: ImageIO's own choice of encoder, and not the one the history writes with.
    static func still(_ image: CGImage, type: UTType, quality: Double?, hardware: Bool = false) -> Data {
        let data = NSMutableData()
        let dest = CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, quality.map { hardware ? [kCGImageDestinationLossyCompressionQuality: $0] as CFDictionary
                                                                      : ScreenCodec.stillOptions(quality: $0) })
        CGImageDestinationFinalize(dest)
        return data as Data
    }

    static func words(_ lines: [String]) -> Set<String> {
        Set(lines.flatMap { $0.split(whereSeparator: { $0 == " " }).map(String.init) })
    }

    /// The frame as the index reads it: no wider than the index's picture size.
    static func scaled(_ pixels: CVPixelBuffer, maxDimension: Int) -> CGImage? {
        guard let image = ScreenCodec.cgImage(pixels) else { return nil }
        let longest = max(image.width, image.height)
        guard longest > maxDimension else { return image }
        let k = Double(maxDimension) / Double(longest)
        let w = Int(Double(image.width) * k), h = Int(Double(image.height) * k)
        guard let cg = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                 bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue) else { return nil }
        cg.interpolationQuality = .high
        cg.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return cg.makeImage()
    }

    /// Which shots the gate sends to the index at each threshold, and how many of the words that
    /// were ever on screen an indexed shot still holds.
    static func gate(shots: [CVPixelBuffer], script: [(String, State, Bool)], order: [String], indexWidth: Int) {
        let t0 = Date()
        let signatures = shots.map { ScreenSignature($0)! }
        print(String(format: "signature: %.2f ms a frame", ms(t0) / Double(shots.count)))
        let tc = Date()
        var steps: [String: [Double]] = [:]
        for i in 1 ..< shots.count { steps[script[i].0, default: []].append(signatures[i].changedShare(from: signatures[i - 1])) }
        print(String(format: "compare: %.2f ms a pair", ms(tc) / Double(shots.count - 1)))
        let showing = signatures.map(\.detailShare).sorted()
        print(String(format: "showing something: median %.1f%% of tiles (%.1f%% to %.1f%%)", showing[showing.count / 2] * 100, showing[0] * 100, showing[showing.count - 1] * 100))
        print("changed against the shot before, median by stretch: " + order.map { p in
            let v = (steps[p] ?? []).sorted(); return "\(p)=\(String(format: "%.1f%%", (v.isEmpty ? 0 : v[v.count / 2]) * 100))"
        }.joined(separator: " "))
        let tr = Date()
        let read: [Set<String>] = shots.map { shot in
            guard let image = scaled(shot, maxDimension: indexWidth) else { return [] }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate; request.usesLanguageCorrection = false; request.automaticallyDetectsLanguage = true
            try? VNImageRequestHandler(cgImage: image).perform([request])
            return words((request.results ?? []).compactMap { $0.topCandidates(1).first?.string })
        }
        let all = read.reduce(into: Set<String>()) { $0.formUnion($1) }
        print(String(format: "read at %d wide: %.0f ms a frame, %d distinct words over the session", indexWidth, ms(tr) / Double(shots.count), all.count))
        for (working, settle) in [(0.02, true), (0.05, true), (0.10, true), (0.10, false), (0.20, true), (0.35, true), (0.50, true)] {
            var g = ScreenGate()
            var kept: [Int] = []
            for i in shots.indices where g.admit(signatures[i], settled: settle && script[i].2, working: working) { kept.append(i) }
            let held = kept.reduce(into: Set<String>()) { $0.formUnion(read[$1]) }
            let byPhase = order.map { p in "\(p)=\(kept.filter { script[$0].0 == p }.count)" }.joined(separator: " ")
            print(String(format: "gate %.0f%%%@  indexed %d of %d  words held %.1f%% | %@", working * 100, settle ? " +pause" : "       ",
                         kept.count, shots.count, Double(held.count) / Double(max(all.count, 1)) * 100, byPhase))
        }
    }

    /// What the hint on a served screenshot costs: the session around a shot, in a history of whole
    /// working days (a shot every 5 s, 8 h unbroken: the longest walk `session(around:)` makes).
    /// Cold is a history just opened, its segment tables not read yet; warm is every one after.
    /// shotbench hints <new scratch dir> [days]
    static func hints(_ args: [String]) {
        guard let dir = args.first else { print("usage: shotbench hints <new scratch dir> [days]"); return }
        let days = args.count > 1 ? Int(args[1]) ?? 5 : 5
        let root = URL(fileURLWithPath: dir)
        let segments = root.appendingPathComponent("history"), stills = root.appendingPathComponent("stills")
        try? FileManager.default.createDirectory(at: segments, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: stills, withIntermediateDirectories: true)
        let pixels = ScreenCodec.makePixelBuffer(width: 320, height: 200)!
        let day0 = Calendar.current.startOfDay(for: Date()).timeIntervalSince1970 - Double(days) * 86_400 + 9 * 3600
        let perDay = 8 * 3600 / 5
        let tw = Date()
        do {
            let writer = ScreenHistory(history: segments, stills: stills)
            for d in 0 ..< days {
                for k in 0 ..< perDay {
                    // A shot the screen did not change for is not stored: move a block each time.
                    CVPixelBufferLockBaseAddress(pixels, [])
                    let base = CVPixelBufferGetBaseAddress(pixels)!, stride = CVPixelBufferGetBytesPerRow(pixels)
                    memset(base, 255, stride * 200)
                    let x = (k * 37) % 280, y = (k * 53) % 160
                    for row in y ..< y + 40 { memset(base + row * stride + x * 4, Int32(k % 200), 160) }
                    CVPixelBufferUnlockBaseAddress(pixels, [])
                    let stored = writer.record(pixels, time: day0 + Double(d) * 86_400 + Double(k) * 5, display: 1,
                                      app: ["Editor", "Browser", "Mail"][(k / 40) % 3], title: "window \((k / 40) % 9)", settled: false)
                    if stored == nil { print("shot \(k) of day \(d) was not stored"); return }
                }
            }
            writer.stop()
        }
        let files = (try? FileManager.default.contentsOfDirectory(atPath: segments.path))?.count ?? 0
        print(String(format: "history: %d days, %d shots, %d segments, written in %.1f s", days, days * perDay, files, ms(tw) / 1000))
        func ask(_ history: ScreenHistory, _ n: Int) -> [Double] {
            var seed: UInt64 = 42
            return (0 ..< n).map { _ in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                let d = Int(seed >> 33) % days, k = Int(seed >> 13) % perDay
                let t = Date()
                let s = history.session(around: day0 + Double(d) * 86_400 + Double(k) * 5)
                precondition(s?.shots == perDay, "the session is the whole day")
                return ms(t)
            }
        }
        func line(_ name: String, _ v: [Double]) {
            let o = v.sorted()
            print(String(format: "%@  n=%d  p50 %.2f ms  p90 %.2f ms  max %.2f ms", name, o.count, o[o.count / 2], o[min(o.count - 1, o.count * 9 / 10)], o.last!))
        }
        // Cold: a new history for each, so each pays its day's segment scans.
        line("session cold", (0 ..< min(days, 8)).flatMap { _ in ask(ScreenHistory(history: segments, stills: stills), 1) })
        let warm = ScreenHistory(history: segments, stills: stills)
        _ = ask(warm, days * 4)
        line("session warm", ask(warm, 200))
        // Thirty screenshot hits of one answer, as ScreenHints.moments asks: one after another.
        let t30 = Date(); _ = ask(warm, 30)
        print(String(format: "30 hits, each walked: %.1f ms", ms(t30)))
        // The same thirty as the served hint asks them (ScreenHints.moments): a stretch walked once.
        var seed: UInt64 = 7
        let times = (0 ..< 30).map { _ -> Double in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return day0 + Double(Int(seed >> 33) % min(days, 3)) * 86_400 + Double(Int(seed >> 13) % perDay) * 5
        }
        let tb = Date(); let found = warm.sessions(around: times)
        print(String(format: "30 hits over %d days, a stretch walked once: %.2f ms", min(days, 3), ms(tb)))
        precondition(zip(times, found).allSatisfy { $1?.shots == perDay && $1?.nearest.time == $0 }, "each hit keeps its own shot")
    }

    /// `search` with a time range and no query, on a real index: VectorStore.listMatching with the
    /// served filter, by the width of the range. The walk is every resident row whatever the range.
    /// shotbench listrange <index.sqlite> [reps]
    static func listRange(_ args: [String]) throws {
        guard let path = args.first else { print("usage: shotbench listrange <index.sqlite of a CLONE> [reps]"); return }
        let reps = args.count > 1 ? Int(args[1]) ?? 20 : 20
        let to = Date()
        let store = try VectorStore(dbURL: URL(fileURLWithPath: path))
        defer { store.close() }
        print(String(format: "open %.0f ms  rows %d", ms(to), store.count))
        let home = ProcessInfo.processInfo.environment["HOME"] ?? ""
        var served = SearchFilter()
        served.excludeFolders = [home + "/Library/Application Support/Omni/Clipboard", home + "/Library/Application Support/Omni/Screenshots"]
        guard let newest = store.listMatching(filter: SearchFilter(), topK: 1).first?.modified else { print("empty index"); return }
        func time(_ name: String, _ f: SearchFilter, topK: Int) {
            var v: [Double] = [], n = 0
            for _ in 0 ..< reps { let t = Date(); n = store.listMatching(filter: f, topK: topK).count; v.append(ms(t)) }
            let first = v[0]; v.sort()
            print(String(format: "%@  hits %3d  first %.1f ms  p50 %.1f ms  max %.1f ms", name.padding(toLength: 34, withPad: " ", startingAt: 0),
                         n, first, v[v.count / 2], v.last!))
        }
        time("no filter, top 30 (control)", SearchFilter(), topK: 30)
        time("served filter, no range, top 30", served, topK: 30)
        for (name, span) in [("13 min", 780.0), ("1 hour", 3600), ("1 day", 86_400), ("30 days", 30 * 86_400)] {
            var f = served
            f.since = newest - span; f.until = newest + 1
            time("range \(name), top 30", f, topK: 30)
            if span == 86_400 { time("range \(name), top 190", f, topK: 190) }
        }
        var old = served
        old.since = newest - 400 * 86_400; old.until = newest - 399 * 86_400
        time("a day, 400 days back, top 30", old, topK: 30)
    }

    /// WHAT COSTS THE VIDEO ITS REFERENCE (docs/screenshots.md, "A real session"). A shot of an
    /// unchanged screen came out the size of a key frame after the timeline was scrubbed and while
    /// another process encoded. Each arm: six frames of a typing stretch, the thing under test, two
    /// more frames; a frame that is a key or the size of one says the reference was lost.
    /// `shotbench reference <dir of PNG frames> [--child]`.
    static func reference(_ args: [String]) {
        guard let dir = args.first else { print("usage: shotbench reference <frames dir>"); return }
        let frames = pngFrames(dir, 40)
        guard frames.count >= 12 else { print("reference: \(frames.count) frames in \(dir)"); return }
        let w = CVPixelBufferGetWidth(frames[0]), h = CVPixelBufferGetHeight(frames[0])
        if let at = args.firstIndex(of: "--child") {
            // Another process that encodes: what a call or a recording beside Omni is.
            let codec = at + 1 < args.count && args[at + 1] == "h264" ? kCMVideoCodecType_H264 : kCMVideoCodecType_HEVC
            guard let other = OtherEncoder(codec: codec, width: w, height: h) else { return }
            for f in frames.prefix(8) { _ = other.encode(f) }
            return
        }
        func arm(_ name: String, _ between: ([ScreenFrameEncoder.Frame]) -> Void) {
            guard let encoder = ScreenFrameEncoder(width: w, height: h) else { print("\(name): no encoder"); return }
            var out: [ScreenFrameEncoder.Frame] = []
            var times: [Double] = []
            for f in frames[0 ..< 6] { let t = Date(); if let e = encoder.encode(f) { out.append(e) }; times.append(ms(t)) }
            between(out)
            for f in frames[6 ..< 8] { let t = Date(); if let e = encoder.encode(f) { out.append(e) }; times.append(ms(t)) }
            let cells = out.enumerated().map { i, e in "\(e.isKey ? "K" : "d")\(e.data.count)" + (i == 5 ? " |" : "") }
            print(String(format: "%@  %@   encode p50 %.1f ms", name.padding(toLength: 34, withPad: " ", startingAt: 0),
                         cells.joined(separator: " "), times.sorted()[times.count / 2]))
        }
        print("\(w)x\(h), frames 0-5 | the arm | frames 6-7 (K key, d delta, bytes)")
        arm("nothing") { _ in }
        arm("decode what was written") { out in
            let decoder = ScreenFrameDecoder()
            for e in out { _ = decoder.decode(e.data, parameterSets: e.parameterSets) }
        }
        arm("a second encoder, same process") { _ in
            let other = ScreenFrameEncoder(width: w, height: h)
            for f in frames[20 ..< 23] { _ = other?.encode(f) }
        }
        arm("an encoder in another process") { _ in
            let child = Process()
            child.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
            child.arguments = ["shotbench", "reference", dir, "--child"]
            child.standardOutput = FileHandle.nullDevice
            try? child.run(); child.waitUntilExit()
        }
        arm("an H.264 encoder, other process") { _ in
            let child = Process()
            child.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
            child.arguments = ["shotbench", "reference", dir, "--child", "h264"]
            child.standardOutput = FileHandle.nullDevice
            try? child.run(); child.waitUntilExit()
        }
        arm("a HEIC still, software (shipped)") { _ in
            if let image = ScreenCodec.cgImage(frames[5]) { _ = still(image, type: .heic, quality: 0.8) }
        }
        arm("a HEIC still, hardware") { _ in
            if let image = ScreenCodec.cgImage(frames[5]) { _ = still(image, type: .heic, quality: 0.8, hardware: true) }
        }
        arm("30 s with nothing") { _ in Thread.sleep(forTimeInterval: 30) }
    }

    /// An encoder as another app holds one (a call, a recording): hardware, real time.
    final class OtherEncoder {
        private let session: VTCompressionSession
        private var count: Int64 = 0
        init?(codec: CMVideoCodecType, width: Int, height: Int) {
            var s: VTCompressionSession?
            let spec: [CFString: Any] = [kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true]
            guard VTCompressionSessionCreate(allocator: nil, width: Int32(width), height: Int32(height), codecType: codec,
                                             encoderSpecification: spec as CFDictionary, imageBufferAttributes: nil,
                                             compressedDataAllocator: nil, outputCallback: nil, refcon: nil,
                                             compressionSessionOut: &s) == noErr, let s else { return nil }
            session = s
            VTSessionSetProperty(s, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
            VTSessionSetProperty(s, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
            VTSessionSetProperty(s, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, value: 600 as CFNumber)
            VTCompressionSessionPrepareToEncodeFrames(s)
        }
        deinit { VTCompressionSessionInvalidate(session) }

        final class Out: @unchecked Sendable { var key = false; var bytes = 0 }

        func encode(_ pixels: CVPixelBuffer) -> (isKey: Bool, bytes: Int)? {
            let pts = CMTime(value: count, timescale: 30)
            count += 1
            let out = Out()
            let status = VTCompressionSessionEncodeFrame(session, imageBuffer: pixels, presentationTimeStamp: pts,
                                                         duration: CMTime(value: 1, timescale: 30), frameProperties: nil,
                                                         infoFlagsOut: nil) { status, _, sample in
                guard status == noErr, let sample, let block = CMSampleBufferGetDataBuffer(sample) else { return }
                let attachments = (CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false) as? [[CFString: Any]])?.first
                out.key = !((attachments?[kCMSampleAttachmentKey_NotSync] as? Bool) ?? false)
                out.bytes = CMBlockBufferGetDataLength(block)
            }
            guard status == noErr else { return nil }
            VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: pts)
            return out.bytes > 0 ? (out.key, out.bytes) : nil
        }
    }

    /// The other way round: what a shot of Omni's costs an encoder another app is running.
    /// `shotbench neighbour <dir of PNG frames>`.
    static func neighbour(_ args: [String]) {
        guard let dir = args.first else { print("usage: shotbench neighbour <frames dir>"); return }
        let frames = pngFrames(dir, 40)
        guard frames.count >= 30 else { print("neighbour: \(frames.count) frames in \(dir)"); return }
        let w = CVPixelBufferGetWidth(frames[0]), h = CVPixelBufferGetHeight(frames[0])
        print("\(w)x\(h): the other app's 12 frames (K key, d delta, bytes)")
        for (name, codec) in [("HEVC", kCMVideoCodecType_HEVC), ("H.264", kCMVideoCodecType_H264)] {
            for (mine, shooting) in [("no shot between", false), ("a shot between", true)] {
                guard let other = OtherEncoder(codec: codec, width: w, height: h) else { print("\(name): no encoder"); continue }
                var theirs: [String] = [], ours: [String] = []
                // The other app is already running when Omni's recorder starts, and goes on.
                for k in 0 ..< 4 { if let e = other.encode(frames[k]) { theirs.append("\(e.isKey ? "K" : "d")\(e.bytes)") } }
                let shots = shooting ? ScreenFrameEncoder(width: w, height: h) : nil
                theirs.append("|")
                for k in 4 ..< 12 {
                    if let e = shots?.encode(frames[16 + k]) { ours.append("\(e.isKey ? "K" : "d")\(e.data.count)") }
                    if let e = other.encode(frames[k]) { theirs.append("\(e.isKey ? "K" : "d")\(e.bytes)") }
                }
                print("\(name), \(mine)".padding(toLength: 34, withPad: " ", startingAt: 0) + "  theirs " + theirs.joined(separator: " "))
                if !ours.isEmpty { print("".padding(toLength: 34, withPad: " ", startingAt: 0) + "  ours   " + ours.joined(separator: " ")) }
            }
        }
        // Omni's frames by themselves, for the sizes above.
        if let alone = ScreenFrameEncoder(width: w, height: h) {
            print("no other app".padding(toLength: 34, withPad: " ", startingAt: 0) + "  ours   "
                  + (4 ..< 12).compactMap { alone.encode(frames[16 + $0]) }.map { "\($0.isKey ? "K" : "d")\($0.data.count)" }.joined(separator: " "))
        }
    }

    static func pngFrames(_ dir: String, _ limit: Int) -> [CVPixelBuffer] {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix(".png") }.sorted().prefix(limit)
        return names.compactMap { name in
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: dir + "/" + name) as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let pixels = ScreenCodec.makePixelBuffer(width: image.width, height: image.height) else { return nil }
            CVPixelBufferLockBaseAddress(pixels, [])
            defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
            CGContext(data: CVPixelBufferGetBaseAddress(pixels), width: image.width, height: image.height, bitsPerComponent: 8,
                      bytesPerRow: CVPixelBufferGetBytesPerRow(pixels), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)?
                .draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return pixels
        }
    }

    static func run(_ args: [String]) {
        if args.first == "reference" { reference(Array(args.dropFirst())); return }
        if args.first == "neighbour" { neighbour(Array(args.dropFirst())); return }
        if args.first == "hints" { hints(Array(args.dropFirst())); return }
        if args.first == "listrange" { do { try listRange(Array(args.dropFirst())) } catch { print("listrange: \(error)") }; return }
        guard args.count >= 3 else { print("usage: shotbench <source> <prose> <photo> [--scale 1|2] [--frames N] [--dump dir]"); return }
        var scale = 1, frames = 150, dump: String?, feed: String?, only = "codec,gate"
        var i = 3
        while i + 1 < args.count {
            switch args[i] {
            case "--scale": scale = Int(args[i + 1]) ?? 1
            case "--frames": frames = Int(args[i + 1]) ?? 150
            case "--dump": dump = args[i + 1]
            case "--feed": feed = args[i + 1]
            case "--only": only = args[i + 1]
            default: break
            }
            i += 2
        }
        let source = (try? String(contentsOfFile: args[0], encoding: .utf8)) ?? ""
        let prose = (try? String(contentsOfFile: args[1], encoding: .utf8)) ?? ""
        let photo = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        let renderer = Renderer(scale: scale, source: source, prose: prose, photo: photo)
        let proseHeight = renderer.prose.boundingRect(with: CGSize(width: 924, height: 1e7), options: [.usesLineFragmentOrigin]).height + 560
        let script = session(source: renderer.source, proseHeight: proseHeight, frames: frames)
        let t0 = Date()
        let shots = script.map { renderer.render($0.1) }
        let phases = script.map { $0.0 }
        var order: [String] = []
        for p in phases where !order.contains(p) { order.append(p) }
        print("session: \(shots.count) frames \(renderer.width)x\(renderer.height), rendered in \(Int(ms(t0))) ms")
        let sample = stride(from: 0, to: shots.count, by: 10).map { $0 }
        let truth = sample.map { lines(shots[$0]) }
        print("text on the originals: \(truth.map(\.count).reduce(0, +)) lines over \(sample.count) sampled frames")
        // The session as files an isolated app instance records in place of the screen (PerfScript's
        // `shotfeed`): "<n>[!] <App> - <Window>.png", "!" where the user paused.
        if let feed {
            try? FileManager.default.createDirectory(atPath: feed, withIntermediateDirectories: true)
            for (k, step) in script.enumerated() {
                let app = step.1.app == 0 ? "Editor" : "Browser", window = step.1.app == 0 ? "source.swift" : "Field notes"
                let name = String(format: "%04d%@ %@ - %@.png", k, step.2 ? "!" : "", app, window)
                try? still(ScreenCodec.cgImage(shots[k])!, type: .png, quality: nil).write(to: URL(fileURLWithPath: "\(feed)/\(name)"))
            }
            print("feed: \(shots.count) frames in \(feed)")
        }
        if let dump {
            try? FileManager.default.createDirectory(atPath: dump, withIntermediateDirectories: true)
            for k in [0, 60, 80, 95] where k < shots.count {
                try? still(ScreenCodec.cgImage(shots[k])!, type: .png, quality: nil).write(to: URL(fileURLWithPath: "\(dump)/orig-\(k).png"))
            }
        }

        func report(_ name: String, sizes: [Int], encodeMs: Double, decodeMs: Double, jumpMs: Double, psnr: Double, read: Double) {
            var byPhase = ""
            for p in order {
                let idx = phases.indices.filter { phases[$0] == p }
                byPhase += " \(p)=\(idx.map { sizes[$0] }.reduce(0, +) / max(idx.count, 1) / 1024)K"
            }
            print(String(format: "%@  total %.2f MB  %.1f KB/frame  enc %.1f ms  dec %.1f ms  jump %.0f ms  PSNR %.1f dB  text %.1f%% |%@",
                         name.padding(toLength: 24, withPad: " ", startingAt: 0), Double(sizes.reduce(0, +)) / 1e6,
                         Double(sizes.reduce(0, +)) / Double(sizes.count) / 1024, encodeMs, decodeMs, jumpMs, psnr, read * 100, byPhase))
        }
        func readShare(_ decoded: [CVPixelBuffer]) -> Double {
            var same = 0, all = 0
            for (k, pixels) in decoded.enumerated() {
                let got = Set(lines(pixels))
                all += truth[k].count; same += truth[k].filter { got.contains($0) }.count
            }
            return all == 0 ? 1 : Double(same) / Double(all)
        }

        if only.contains("gate") { gate(shots: shots, script: script, order: order, indexWidth: 1568) }
        guard only.contains("codec") else { return }

        // The video, at each quality and key-frame spacing.
        for (quality, keyInterval) in [(0.5, 30), (0.6, 30), (0.7, 30), (0.8, 30), (0.7, 10), (0.7, 60), (0.7, 100_000)] as [(Float, Int)] {
            guard let encoder = ScreenFrameEncoder(width: renderer.width, height: renderer.height, quality: quality, keyInterval: keyInterval) else {
                print("no encoder for \(renderer.width)x\(renderer.height)"); return
            }
            var encoded: [ScreenFrameEncoder.Frame] = []
            let te = Date()
            var slowest = 0.0
            for shot in shots {
                let t = Date()
                guard let f = encoder.encode(shot) else { print("encode failed"); return }
                encoded.append(f); slowest = max(slowest, ms(t))
            }
            let encodeMs = ms(te) / Double(shots.count)
            let decoder = ScreenFrameDecoder()
            var decoded: [Int: CVPixelBuffer] = [:]
            let td = Date()
            for (k, f) in encoded.enumerated() {
                guard let pixels = decoder.decode(f.data, parameterSets: f.parameterSets) else { print("decode failed at \(k)"); return }
                if sample.contains(k) { decoded[k] = pixels }
            }
            let decodeMs = ms(td) / Double(shots.count)
            // A jump into the timeline: the frame just before a key frame, decoded cold from the key before it.
            let keys = encoded.indices.filter { encoded[$0].isKey }
            var jump = 0.0
            if keys.count >= 2 {
                let fresh = ScreenFrameDecoder()
                let tj = Date()
                for k in keys[0] ..< keys[1] { _ = fresh.decode(encoded[k].data, parameterSets: encoded[k].parameterSets) }
                jump = ms(tj)
            } else { jump = decodeMs * Double(shots.count) }
            let quality10 = sample.map { psnr(shots[$0], decoded[$0]!) }
            report(String(format: "hevc q%.1f key %@", quality, keyInterval > 1000 ? "none" : "\(keyInterval)"),
                   sizes: encoded.map(\.data.count), encodeMs: encodeMs, decodeMs: decodeMs, jumpMs: jump,
                   psnr: quality10.reduce(0, +) / Double(quality10.count), read: readShare(sample.map { decoded[$0]! }))
            print(String(format: "    key frames: \(keys.count), mean key \(keys.map { encoded[$0].data.count }.reduce(0, +) / max(keys.count, 1) / 1024) KB, slowest encode %.0f ms", slowest))
            if let dump, quality == 0.7, keyInterval == 30 {
                for k in [0, 60, 80] where decoded[k] != nil {
                    try? still(ScreenCodec.cgImage(decoded[k]!)!, type: .png, quality: nil).write(to: URL(fileURLWithPath: "\(dump)/hevc-\(k).png"))
                }
            }
        }

        // One picture at a time.
        for quality in [0.6, 0.8] {
            var sizes: [Int] = [], decoded: [CVPixelBuffer] = []
            let te = Date()
            var datas: [Data] = []
            for shot in shots { let d = still(ScreenCodec.cgImage(shot)!, type: .heic, quality: quality); sizes.append(d.count); datas.append(d) }
            let encodeMs = ms(te) / Double(shots.count)
            let td = Date()
            var psnrs: [Double] = []
            for k in sample {
                guard let src = CGImageSourceCreateWithData(datas[k] as CFData, nil), let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
                      let pb = ScreenCodec.makePixelBuffer(width: renderer.width, height: renderer.height) else { continue }
                CVPixelBufferLockBaseAddress(pb, [])
                let cg = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: renderer.width, height: renderer.height, bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
                cg.draw(image, in: CGRect(x: 0, y: 0, width: renderer.width, height: renderer.height))
                CVPixelBufferUnlockBaseAddress(pb, [])
                decoded.append(pb); psnrs.append(psnr(shots[k], pb))
            }
            let decodeMs = ms(td) / Double(sample.count)
            report(String(format: "heic q%.1f each", quality), sizes: sizes, encodeMs: encodeMs, decodeMs: decodeMs, jumpMs: decodeMs,
                   psnr: psnrs.reduce(0, +) / Double(max(psnrs.count, 1)), read: readShare(decoded))
        }
        let tp = Date()
        let png = sample.map { still(ScreenCodec.cgImage(shots[$0])!, type: .png, quality: nil).count }
        print(String(format: "png each (sampled)        %.1f KB/frame  enc %.1f ms", Double(png.reduce(0, +)) / Double(png.count) / 1024, ms(tp) / Double(png.count)))
    }
}
