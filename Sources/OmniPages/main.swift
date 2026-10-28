import AppKit
import PDFKit
import WebKit

// omni-pages <file> <outDir> [--max <px>]
//
// Every slide of a PowerPoint deck, or every sheet of a Numbers document, as Quick Look previews it,
// drawn to outDir as PNGs, and outDir/pages.json: [{"label": ..., "image": ..., "text": ...}] in
// page order. A slide is labelled "Slide N", a sheet by its name; `text` is a Numbers sheet's words
// (the preview draws a sheet as PDF, which carries them), empty for a slide. Exit 0 with the pages
// written, 2 when the document has no preview.
//
// Its own process because a deck's preview is HTML, drawn by WebKit, which wants a main thread with
// a run loop: Omni's main thread is the UI's, and the indexer and the command line tools have none
// to lend. `qlmanage -p -o` is the macOS tool that writes a preview out; the only public Quick Look
// API that draws (QLThumbnailGenerator) stops at the first page.

struct Page: Encodable { let label: String; let image: String; let text: String }

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: omni-pages <file> <outDir> [--max <px>]\n".utf8))
    exit(64)
}
let file = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2], isDirectory: true)
let maxSide: CGFloat = args.firstIndex(of: "--max").flatMap { $0 + 1 < args.count ? Double(args[$0 + 1]) : nil }.map { CGFloat($0) } ?? 1568

func finish(_ pages: [Page], code: Int32) -> Never {
    if let data = try? JSONEncoder().encode(pages) { try? data.write(to: outDir.appendingPathComponent("pages.json")) }
    exit(code)
}

/// The preview written by qlmanage: its folder, or nil when the document has none.
func writePreview() -> URL? {
    let ql = outDir.appendingPathComponent("ql", isDirectory: true)
    try? FileManager.default.createDirectory(at: ql, withIntermediateDirectories: true)
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/qlmanage")
    p.arguments = ["-p", "-o", ql.path, file.path]
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    guard (try? p.run()) != nil else { return nil }
    let deadline = Date().addingTimeInterval(60)
    while p.isRunning, Date() < deadline { usleep(20_000) }
    if p.isRunning { p.terminate(); return nil }
    let names = (try? FileManager.default.contentsOfDirectory(atPath: ql.path)) ?? []
    guard let dir = names.first(where: { $0.hasSuffix(".qlpreview") }) else { return nil }
    let url = ql.appendingPathComponent(dir, isDirectory: true)
    return FileManager.default.fileExists(atPath: url.appendingPathComponent("Preview.html").path) ? url : nil
}

func savePNG(_ image: CGImage, _ name: String) -> Bool {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else { return false }
    return (try? data.write(to: outDir.appendingPathComponent(name))) != nil
}

/// A PDF page drawn on white, its long side at most `maxSide`.
func drawPDF(_ page: PDFPage) -> CGImage? {
    let b = page.bounds(for: .mediaBox)
    let scale = min(2, maxSide / max(b.width, b.height, 1))
    let w = Int(b.width * scale), h = Int(b.height * scale)
    guard w > 0, h > 0, let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                           space: CGColorSpaceCreateDeviceRGB(),
                                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: ctx)
    return ctx.makeImage()
}

/// A PDF page's words row by row, a row's pieces left to right and " | " between them. PDFKit's own
/// string reads a table column by column, which parts every name from its number; the lines it
/// finds are put back where the page draws them.
func pdfText(_ page: PDFPage) -> String {
    guard let all = page.selection(for: page.bounds(for: .mediaBox)) else { return page.string ?? "" }
    let lines: [(rect: CGRect, text: String)] = all.selectionsByLine().compactMap { l in
        guard let s = l.string?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return nil }
        return (l.bounds(for: page), s)
    }.sorted { $0.rect.midY != $1.rect.midY ? $0.rect.midY > $1.rect.midY : $0.rect.minX < $1.rect.minX }
    var rows: [[(rect: CGRect, text: String)]] = []
    for line in lines {
        // The same row: its middle within half a line of the row's first piece (PDF y runs upward).
        if let first = rows.last?.first, abs(first.rect.midY - line.rect.midY) < min(first.rect.height, line.rect.height) / 2 {
            rows[rows.count - 1].append(line)
        } else {
            rows.append([line])
        }
    }
    return rows.map { $0.sorted { $0.rect.minX < $1.rect.minX }.map(\.text).joined(separator: " | ") }.joined(separator: "\n")
}

@MainActor
final class Drawer: NSObject, WKNavigationDelegate {
    let web: WKWebView
    let window: NSWindow
    var loaded: CheckedContinuation<Bool, Never>?

    init(rules: WKContentRuleList?) {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        // The preview's own scripts (sheet tabs) are not needed to draw it; ours run regardless.
        config.defaultWebpagePreferences.allowsContentJavaScript = false
        if let rules { config.userContentController.add(rules) }
        web = WKWebView(frame: CGRect(x: 0, y: 0, width: 1024, height: 768), configuration: config)
        window = NSWindow(contentRect: CGRect(x: -20_000, y: -20_000, width: 1024, height: 768),
                          styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = web
        super.init()
        web.navigationDelegate = self
    }

    func load(_ url: URL, root: URL) async -> Bool {
        await withCheckedContinuation { c in
            loaded = c
            web.loadFileURL(url, allowingReadAccessTo: root)
        }
    }
    func webView(_ w: WKWebView, didFinish n: WKNavigation!) { resume(true) }
    func webView(_ w: WKWebView, didFail n: WKNavigation!, withError e: Error) { resume(false) }
    func webView(_ w: WKWebView, didFailProvisionalNavigation n: WKNavigation!, withError e: Error) { resume(false) }
    /// Nothing but the preview's own files.
    func webView(_ w: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        action.request.url?.isFileURL == true ? .allow : .cancel
    }
    private func resume(_ ok: Bool) { loaded?.resume(returning: ok); loaded = nil }

    func json(_ script: String) async -> Any? {
        guard let s = try? await web.evaluateJavaScript(script) as? String else { return nil }
        return try? JSONSerialization.jsonObject(with: Data(s.utf8))
    }

    /// The window is one slide's size, at most this many points a side, and scrolls to each slide in
    /// turn. A window as tall as the deck (110,000 points for 200 slides) is past the largest texture
    /// WindowServer can make (16,384), and a screenshot tool that captures every window takes
    /// WindowServer down on one: a 30,000-point window did, 2026-10-27 (index.md, "Decks and workbooks").
    static let windowCap: CGFloat = 4096

    func resize(_ w: CGFloat, _ h: CGFloat) {
        window.setContentSize(CGSize(width: w, height: h))
        web.frame = CGRect(x: 0, y: 0, width: w, height: h)
    }

    func snapshot(_ rect: CGRect) async -> CGImage? {
        let c = WKSnapshotConfiguration()
        c.rect = rect
        c.afterScreenUpdates = true
        let scale = min(2, maxSide / max(rect.width, rect.height, 1))
        c.snapshotWidth = NSNumber(value: Double(rect.width * scale))
        guard let img = try? await web.takeSnapshot(configuration: c) else { return nil }
        return img.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    func run(preview: URL) async -> [Page] {
        guard await load(preview.appendingPathComponent("Preview.html"), root: preview) else { return [] }
        var pages: [Page] = []
        // A deck: one div.slide per slide, all on the one page.
        let found = await json("""
            JSON.stringify((() => {
              const slides = [...document.querySelectorAll('div.slide')];
              if (slides.length) return {slides: slides.length};
              return {navpane: !!document.querySelector('.navpane-sheet')};
            })())
            """) as? [String: Any]
        if found?["slides"] != nil {
            let rects = (await json("JSON.stringify([...document.querySelectorAll('div.slide')].map(e => { const r = e.getBoundingClientRect(); return [r.left + scrollX, r.top + scrollY, r.width, r.height]; }))") as? [[Double]] ?? [])
                .filter { $0.count == 4 && $0[2] > 0 && $0[3] > 0 }
            let w = min(Self.windowCap, max(320, CGFloat(rects.map { $0[0] + $0[2] }.max() ?? 1024)))
            let h = min(Self.windowCap, max(240, CGFloat(rects.map { $0[3] }.max() ?? 768) + 16))
            resize(w, h)
            for (i, r) in rects.enumerated() {
                // Scrolled so the slide's top is the window's, where the page lets it be.
                let at = await json("window.scrollTo(0, \(r[1])); JSON.stringify([scrollX, scrollY])") as? [Double] ?? [0, 0]
                let rect = CGRect(x: r[0] - at[0], y: r[1] - at[1], width: min(r[2], w), height: min(r[3], h))
                let name = String(format: "page-%04d.png", i + 1)
                if let img = await snapshot(rect), savePNG(img, name) {
                    pages.append(Page(label: "Slide \(i + 1)", image: name, text: ""))
                }
            }
            return pages
        }
        // A Numbers document: a page per sheet, each showing the sheet as a PDF.
        guard found?["navpane"] as? Bool == true else { return pages }
        for (i, tab) in Self.navpaneSheets(preview.appendingPathComponent("Preview.html")).enumerated() {
            let label = (tab.first as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard tab.count > 1, let href = tab[1] as? String, !href.isEmpty else { continue }
            let sheet = preview.appendingPathComponent(href)
            guard sheet.standardizedFileURL.path.hasPrefix(preview.standardizedFileURL.path),
                  await load(sheet, root: preview) else { continue }
            let name = String(format: "page-%04d.png", i + 1)
            // Drawn here from the PDF itself, which carries the sheet's text.
            guard let pdf = await json("JSON.stringify([...document.querySelectorAll('img, embed, object')].map(e => e.getAttribute('src') || e.getAttribute('data')).filter(s => s && /\\.pdf$/i.test(s)))") as? [String],
                  let first = pdf.first,
                  let doc = PDFDocument(url: sheet.deletingLastPathComponent().appendingPathComponent(first)),
                  let page = doc.page(at: 0) else { continue }
            let text = (0 ..< doc.pageCount).compactMap { doc.page(at: $0).map(pdfText) }.joined(separator: "\n")
            if let img = drawPDF(page), savePNG(img, name) { pages.append(Page(label: label, image: name, text: text)) }
        }
        return pages
    }

    /// Numbers' sheet tabs, (name, page) in order. Each tab opens its page from an `onclick`, which
    /// WebKit drops with the preview's scripts turned off, so the tabs are read from the file.
    static func navpaneSheets(_ html: URL) -> [[Any]] {
        guard let s = try? String(contentsOf: html, encoding: .utf8),
              let tag = try? NSRegularExpression(pattern: #"<div[^>]*navpane-sheet[^>]*>"#),
              let link = try? NSRegularExpression(pattern: #"SelectSheet\(\s*\d+\s*,\s*'([^']+)'"#),
              let title = try? NSRegularExpression(pattern: #"title="([^"]*)""#) else { return [] }
        let ns = s as NSString
        return tag.matches(in: s, range: NSRange(location: 0, length: ns.length)).compactMap { m in
            let t = ns.substring(with: m.range)
            let tn = t as NSString, all = NSRange(location: 0, length: tn.length)
            guard let l = link.firstMatch(in: t, range: all) else { return nil }
            let name = title.firstMatch(in: t, range: all).map { tn.substring(with: $0.range(at: 1)) } ?? ""
            return [name, tn.substring(with: l.range(at: 1))]
        }
    }
}

try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
guard let preview = writePreview() else { finish([], code: 2) }
let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
Task { @MainActor in
    // Only the preview's own files load: a document cannot make this process reach the network.
    let store = WKContentRuleListStore(url: outDir.appendingPathComponent("rules", isDirectory: true))
    let rules = try? await store?.compileContentRuleList(forIdentifier: "omni-pages", encodedContentRuleList: """
        [{"trigger": {"url-filter": ".*"}, "action": {"type": "block"}},
         {"trigger": {"url-filter": "^file:"}, "action": {"type": "ignore-previous-rules"}}]
        """)
    let pages = await Drawer(rules: rules).run(preview: preview)
    for d in ["ql", "rules"] { try? FileManager.default.removeItem(at: outDir.appendingPathComponent(d)) }
    finish(pages, code: pages.isEmpty ? 2 : 0)
}
app.run()
