// DropProbe: a drag destination for FileDragUITests. A small floating window at the top right of the
// screen that accepts file URLs and file promises from another app and reports what arrived - the
// files, the promises it received, and the operations the SOURCE offered - as JSON in the
// accessibility value of a label (`probe.report`). The test runner is sandboxed, so it reads that
// label rather than a file. Build with Scripts/drop-probe.sh.
import AppKit

final class DropView: NSView {
    let report = NSTextField(labelWithString: "{}")
    private let inbox = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("dropprobe-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
    private var offered: NSDragOperation = []

    override init(frame: NSRect) {
        super.init(frame: frame)
        registerForDraggedTypes([.fileURL] + NSFilePromiseReceiver.readableDraggedTypes.map { .init($0) })
        report.setAccessibilityIdentifier("probe.report")
        report.frame = bounds.insetBy(dx: 8, dy: 8)
        report.autoresizingMask = [.width, .height]
        report.lineBreakMode = .byCharWrapping
        report.maximumNumberOfLines = 0
        addSubview(report)
        publish(["state": "ready"])
        try? FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.systemYellow.withAlphaComponent(0.25).setFill()
        bounds.fill()
    }

    private func publish(_ obj: [String: Any]) {
        let data = (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])) ?? Data()
        let s = String(data: data, encoding: .utf8) ?? "{}"
        report.stringValue = s
        report.setAccessibilityValue(s)
        report.setAccessibilityLabel(s)
    }

    /// Asks for a copy, the most a well-behaved destination asks of a foreign source. What matters
    /// for the test is what the source OFFERED, recorded separately.
    private func choose(_ s: NSDraggingInfo) -> NSDragOperation {
        offered = s.draggingSourceOperationMask
        if offered.contains(.copy) { return .copy }
        if offered.contains(.generic) { return .generic }
        return []
    }
    override func draggingEntered(_ s: NSDraggingInfo) -> NSDragOperation { choose(s) }
    override func draggingUpdated(_ s: NSDraggingInfo) -> NSDragOperation { choose(s) }

    override func performDragOperation(_ s: NSDraggingInfo) -> Bool {
        let pb = s.draggingPasteboard
        let urls = (pb.readObjects(forClasses: [NSURL.self],
                                   options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        let promises = (pb.readObjects(forClasses: [NSFilePromiseReceiver.self]) as? [NSFilePromiseReceiver]) ?? []
        var base: [String: Any] = [
            "state": "dropped",
            "files": urls.map(\.path),
            "items": pb.pasteboardItems?.count ?? 0,
            "promises": promises.count,
            // A copy leaves the source where it was; a move would not.
            "sourcesStillExist": urls.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
            "offered": [
                "copy": offered.contains(.copy), "move": offered.contains(.move),
                "link": offered.contains(.link), "generic": offered.contains(.generic),
            ],
        ]
        publish(base)
        guard !promises.isEmpty else { return true }
        var written: [String] = []
        let group = DispatchGroup()
        let q = OperationQueue()
        for p in promises {
            group.enter()
            p.receivePromisedFiles(atDestination: inbox, options: [:], operationQueue: q) { url, err in
                if err == nil { written.append(url.lastPathComponent) }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            base["promisedWritten"] = written
            self.publish(base)
        }
        return true
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    func applicationDidFinishLaunching(_ n: Notification) {
        let vis = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let size = NSSize(width: 360, height: 260)
        window = NSWindow(contentRect: NSRect(x: vis.maxX - size.width - 20, y: vis.maxY - size.height - 20,
                                              width: size.width, height: size.height),
                          styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "DropProbe"
        window.level = .floating   // above the app under test wherever it opens
        window.contentView = DropView(frame: NSRect(origin: .zero, size: size))
        window.makeKeyAndOrderFront(nil)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
