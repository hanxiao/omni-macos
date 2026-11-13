import SwiftUI
import AppKit
import CoreVideo
import QuartzCore
import OmniKit

/// What the Screenshots place shows: one day of the screen history as points on a track, and the
/// shot under the playhead.
///
/// ITS OWN OBJECT, NOT AppModel's. Dragging the playhead moves `selected` once per pointer event;
/// in AppModel that would re-run whatever else reads the model in the same body. Only the
/// timeline's own views read this. The picture is not observed state at all: a decoded shot goes
/// straight into a layer (`ShotSurfaceView`), so a drag draws frames without a SwiftUI render.
@MainActor @Observable
final class ScreenTimelineState {
    /// The one on screen, for the perf script's steps.
    @ObservationIgnored static weak var current: ScreenTimelineState?

    /// The local days that have shots, oldest first.
    private(set) var days: [Date] = []
    private(set) var day: Date?
    /// The displays that have shots in `day`, and the one shown.
    private(set) var displays: [UInt32] = []
    private(set) var display: UInt32?
    /// The day's shots on `display`, oldest first.
    private(set) var shots: [ScreenHistory.Shot] = []
    private(set) var selected: Int?
    /// The shown shot's pixel size, for the frame it sits in.
    private(set) var pixelSize = CGSize(width: 16, height: 10)
    /// False until the first listing: the empty state is not shown over a history not read yet.
    private(set) var loaded = false
    /// The stretch of the day the track shows when it is zoomed in; nil is the whole day.
    private(set) var window: ClosedRange<Double>?

    @ObservationIgnored private let history: ScreenHistory
    @ObservationIgnored private weak var surface: ShotSurfaceView?
    /// Held while it is on screen, so the decoder's pool does not hand its surface out again.
    /// By id AND time: after a delete rewrites a segment, the same id is another shot.
    @ObservationIgnored private var shown: (id: String, time: Double, pixels: CVPixelBuffer)?
    @ObservationIgnored private var wanted: ScreenHistory.Shot?
    @ObservationIgnored private var decoding = false
    @ObservationIgnored private var generation = 0
    /// Decode times since the last `takeDecodeTimes`, in ms (the `shotscrub` step).
    @ObservationIgnored private var decodeTimes: [Double] = []

    /// Listings and decodes, one at a time and off the main thread.
    private nonisolated static let queue = DispatchQueue(label: "omni.screen.timeline", qos: .userInitiated)

    init(history: ScreenHistory) {
        self.history = history
        Self.current = self
    }

    var shot: ScreenHistory.Shot? { selected.flatMap { shots.indices.contains($0) ? shots[$0] : nil } }

    // MARK: The track's span

    /// The day on the track: first shot to last, never under ten minutes.
    var daySpan: ClosedRange<Double> {
        guard let first = shots.first?.time, let last = shots.last?.time else { return 0 ... 1 }
        let pad = max(0, 600 - (last - first)) / 2
        return (first - pad) ... (last + pad)
    }

    /// What the track covers now.
    var span: ClosedRange<Double> { window ?? daySpan }

    /// The closest the track zooms: two minutes across, a shot every 5 s well apart.
    static let shortestWindow = 120.0

    /// Zoom by `factor` (over 1: in), the moment at `fraction` of the track's width staying under
    /// the pointer.
    func zoom(by factor: Double, around fraction: Double) {
        let day = daySpan, now = span
        let dayLength = day.upperBound - day.lowerBound, length = now.upperBound - now.lowerBound
        let next = min(dayLength, max(min(Self.shortestWindow, dayLength), length / factor))
        guard abs(next - length) > 0.001 else { return }
        if next >= dayLength { showWholeDay(); return }
        setWindow(start: now.lowerBound + fraction * (length - next), length: next)
    }

    /// Zoom about the playhead, or the middle when it is off the track.
    func zoomAboutSelection(by factor: Double) {
        let now = span
        let f = shot.flatMap { now.contains($0.time) ? ($0.time - now.lowerBound) / (now.upperBound - now.lowerBound) : nil } ?? 0.5
        zoom(by: factor, around: f)
    }

    /// Move a zoomed track along the day by a fraction of its width.
    func pan(by fraction: Double) {
        guard let w = window else { return }
        let length = w.upperBound - w.lowerBound
        setWindow(start: w.lowerBound + fraction * length, length: length)
    }

    func showWholeDay() { if window != nil { window = nil } }

    private func setWindow(start: Double, length: Double) {
        let day = daySpan
        guard length < day.upperBound - day.lowerBound else { showWholeDay(); return }
        let lo = min(max(start, day.lowerBound), day.upperBound - length)
        let next = lo ... (lo + length)
        if window != next { window = next }
    }

    /// A zoomed track follows the playhead when a key or a new shot moves it off either end.
    private func reveal(_ time: Double) {
        guard let w = window, !w.contains(time) else { return }
        let length = w.upperBound - w.lowerBound
        setWindow(start: time - length * (time < w.lowerBound ? 0.1 : 0.9), length: length)
    }

    /// The first shot at or after a moment; `shots.count` when there is none.
    private func firstIndex(atOrAfter time: Double) -> Int {
        var lo = 0, hi = shots.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if shots[mid].time < time { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    // MARK: Listing

    private struct Listing: Sendable {
        var days: [Date] = []
        var day: Date?
        var displays: [UInt32] = []
        var display: UInt32?
        var shots: [ScreenHistory.Shot] = []
    }

    /// Read the history again. The day and the shot stay where they are while they still exist; a
    /// timeline standing on the newest shot follows the next one in.
    func reload(day requested: Date? = nil, display requestedDisplay: UInt32? = nil) {
        generation += 1
        let generation = generation
        let wantDay = requested ?? day
        let wantDisplay = requestedDisplay ?? display
        let follows = requested == nil && requestedDisplay == nil && day == days.last
            && (selected == nil || selected == shots.count - 1)
        // Another display opens at the moment the playhead is on.
        let keep = requested == nil ? shot : nil
        Self.list(history, day: follows ? nil : wantDay, display: wantDisplay) { [weak self] listing in
            guard let self, self.generation == generation else { return }
            self.apply(listing, keep: follows ? nil : keep)
        }
    }

    /// Open at a shot: its day, its display, the playhead on it.
    func go(to target: ScreenHistory.Shot) {
        generation += 1
        let generation = generation
        let day = Calendar.current.startOfDay(for: Date(timeIntervalSince1970: target.time))
        Self.list(history, day: day, display: target.display) { [weak self] listing in
            guard let self, self.generation == generation else { return }
            self.apply(listing, keep: target)
        }
    }

    private nonisolated static func list(_ history: ScreenHistory, day: Date?, display: UInt32?,
                                         done: @escaping @MainActor @Sendable (Listing) -> Void) {
        queue.async {
            var out = Listing()
            out.days = history.days()
            // A day that is gone (deleted, or past retention) gives way to the one after it.
            out.day = day.flatMap { d in out.days.contains(d) ? d : out.days.first { $0 > d } } ?? out.days.last
            if let d = out.day, let next = Calendar.current.date(byAdding: .day, value: 1, to: d) {
                let all = history.shots(from: d.timeIntervalSince1970, to: next.timeIntervalSince1970 - 0.001)
                var counts: [UInt32: Int] = [:]
                for s in all { counts[s.display, default: 0] += 1 }
                out.displays = counts.keys.sorted()
                out.display = display.flatMap { counts[$0] == nil ? nil : $0 }
                    ?? counts.max { $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key }?.key
                out.shots = all.filter { $0.display == out.display }
            }
            let listing = out
            DispatchQueue.main.async { MainActor.assumeIsolated { done(listing) } }
        }
    }

    private func apply(_ listing: Listing, keep: ScreenHistory.Shot?) {
        if days != listing.days { days = listing.days }
        if day != listing.day || display != listing.display { showWholeDay() }
        if day != listing.day { day = listing.day }
        if displays != listing.displays { displays = listing.displays }
        if display != listing.display { display = listing.display }
        if shots != listing.shots { shots = listing.shots }
        if !loaded { loaded = true }
        if let w = window { setWindow(start: w.lowerBound, length: w.upperBound - w.lowerBound) }
        // The shot kept is found by its time: a delete renumbers the shots after it in their
        // segment. One that is gone leaves the playhead on its nearest neighbour.
        var index = shots.isEmpty ? nil : shots.count - 1
        if let keep, !shots.isEmpty {
            var i = min(firstIndex(atOrAfter: keep.time), shots.count - 1)
            if i > 0, keep.time - shots[i - 1].time < shots[i].time - keep.time { i -= 1 }
            index = i
        }
        if selected != index { selected = index }
        if let shot { reveal(shot.time); show(shot) } else { clearPicture() }
    }

    // MARK: Moving

    func select(_ index: Int) {
        guard shots.indices.contains(index), index != selected else { return }
        selected = index
        reveal(shots[index].time)
        show(shots[index])
    }

    /// The shot nearest a moment, among those on the track: a drag never leaves what it shows.
    func select(nearest time: Double) {
        let on = span
        let lo = firstIndex(atOrAfter: on.lowerBound), hi = firstIndex(atOrAfter: on.upperBound.nextUp) - 1
        guard lo <= hi else { return }
        var i = min(max(firstIndex(atOrAfter: time), lo), hi)
        if i > lo, time - shots[i - 1].time <= shots[i].time - time { i -= 1 }
        select(i)
    }

    func step(_ by: Int) {
        guard let selected else { return }
        select(min(max(selected + by, 0), shots.count - 1))
    }

    /// To the next shot the index holds, in either direction.
    func stepIndexed(forward: Bool) {
        guard let selected else { return }
        var i = selected + (forward ? 1 : -1)
        while shots.indices.contains(i) {
            if shots[i].indexed { select(i); return }
            i += forward ? 1 : -1
        }
    }

    func selectEnd(last: Bool) {
        guard !shots.isEmpty else { return }
        select(last ? shots.count - 1 : 0)
    }

    func stepDay(_ by: Int) {
        guard let day, let i = days.firstIndex(of: day), days.indices.contains(i + by) else { return }
        reload(day: days[i + by])
    }

    // MARK: The picture

    func attach(_ view: ShotSurfaceView) {
        surface = view
        view.show(shown?.pixels, time: shown?.time ?? 0)
    }

    /// LATEST WINS. A drag asks for a shot per pointer event and a decode takes a few ms: one
    /// decode runs at a time, and when it lands the newest request is the next one, never a queue
    /// of shots the playhead already left.
    private func show(_ shot: ScreenHistory.Shot) {
        wanted = shot
        if !decoding { decodeNext() }
    }

    private func decodeNext() {
        guard let shot = wanted else { return }
        wanted = nil
        guard shown?.id != shot.id || shown?.time != shot.time else { return }
        decoding = true
        Self.decode(shot, from: history) { [weak self] box, ms in
            guard let self else { return }
            self.decoding = false
            self.decodeTimes.append(ms)
            if let pixels = box.pixels, !self.shots.isEmpty {
                self.shown = (shot.id, shot.time, pixels)
                let size = CGSize(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels))
                if size != self.pixelSize { self.pixelSize = size }
                self.surface?.show(pixels, time: shot.time)
            }
            self.decodeNext()
        }
    }

    private func clearPicture() {
        wanted = nil
        shown = nil
        surface?.show(nil, time: 0)
    }

    private struct PixelBox: @unchecked Sendable { let pixels: CVPixelBuffer? }

    private nonisolated static func decode(_ shot: ScreenHistory.Shot, from history: ScreenHistory,
                                           done: @escaping @MainActor @Sendable (PixelBox, Double) -> Void) {
        queue.async {
            let t0 = CACurrentMediaTime()
            let box = PixelBox(pixels: history.pixels(shot))
            let ms = (CACurrentMediaTime() - t0) * 1000
            DispatchQueue.main.async { MainActor.assumeIsolated { done(box, ms) } }
        }
    }

    /// The shown shot as PNG, for Copy.
    func copyShown() {
        guard let pixels = shown?.pixels, let image = ScreenCodec.cgImage(pixels) else { return }
        let rep = NSBitmapImageRep(cgImage: image)
        if let png = rep.representation(using: .png, properties: [:]) { OmniPasteboard.copyImage(png: png) }
    }

    /// Where the index holds the shown shot, when it does.
    func pathOfShown() -> String? { shot.flatMap { history.path(of: $0) } }

    // MARK: Deleting

    enum DeleteScope: String, Sendable { case shot, session, day }

    /// What a delete would remove: the span, the display (nil: every display), and how many shots.
    struct Deletion: Sendable {
        let scope: DeleteScope
        let from: Double, to: Double
        let display: UInt32?
        let shots: Int
    }

    /// The span a delete around the selected shot covers. One shot is that shot on its display; a
    /// session (the stretch of work around it, shots no more than five minutes apart) and a day
    /// go on every display: what was on the other screen then is part of the same work.
    func deletion(_ scope: DeleteScope, done: @escaping @MainActor @Sendable (Deletion?) -> Void) {
        guard let shot, let day else { done(nil); return }
        if scope == .shot { done(Deletion(scope: scope, from: shot.time, to: shot.time, display: shot.display, shots: 1)); return }
        let history = history
        Self.queue.async {
            var found: Deletion?
            if scope == .session, let s = history.session(around: shot.time) {
                found = Deletion(scope: scope, from: s.start, to: s.end, display: nil, shots: s.shots)
            } else if scope == .day, let next = Calendar.current.date(byAdding: .day, value: 1, to: day) {
                let from = day.timeIntervalSince1970, to = next.timeIntervalSince1970 - 0.001
                found = Deletion(scope: scope, from: from, to: to, display: nil, shots: history.shots(from: from, to: to).count)
            }
            let out = found
            DispatchQueue.main.async { MainActor.assumeIsolated { done(out) } }
        }
    }

    // MARK: The perf script

    var perfSummary: [String: Any] {
        ["day": day.map { $0.timeIntervalSince1970 } ?? 0, "days": days.count, "shots": shots.count,
         "display": display ?? 0, "displays": displays.count,
         "window": window.map { [$0.lowerBound, $0.upperBound] } ?? [],
         "indexed": shots.reduce(0) { $0 + ($1.indexed ? 1 : 0) }, "selected": selected ?? -1,
         "shown": shown?.id ?? "", "time": shot?.time ?? 0, "app": shot?.app ?? "", "title": shot?.title ?? ""]
    }

    /// `shotdragfile`: write the file a drag of the shown shot would carry, and log what it cost.
    func perfDragFile() {
        guard let shown else { omniPerfLog("shot-dragfile nothing shown"); return }
        let t0 = CACurrentMediaTime()
        let file = ShotDragFile(shown.pixels, time: shown.time)
        DispatchQueue.global(qos: .userInitiated).async {
            let url = file.wait()
            let bytes = url.flatMap { (try? $0.resourceValues(forKeys: [.fileSizeKey]))?.fileSize } ?? 0
            omniPerfLog(String(format: "shot-dragfile ms=%.0f bytes=%d path=%@", (CACurrentMediaTime() - t0) * 1000, bytes, url?.path ?? ""))
        }
    }

    /// `shotscrub`: walk `count` shots back from the newest and forward again, each asked for as
    /// the one before lands, and log what a decode cost.
    func perfScrub(_ count: Int, done: @escaping @MainActor () -> Void) {
        guard !shots.isEmpty else { done(); return }
        decodeTimes = []
        let n = min(count, shots.count - 1)
        let path = (0 ... n).map { shots.count - 1 - $0 } + (0 ..< n).reversed().map { shots.count - 1 - $0 }
            + stride(from: shots.count - 1, through: 0, by: -max(1, shots.count / 40)).map { $0 }
        Task { @MainActor in
            let t0 = CACurrentMediaTime()
            for i in path {
                select(i)
                while decoding || wanted != nil { try? await Task.sleep(nanoseconds: 500_000) }
            }
            let times = decodeTimes.sorted()
            if !times.isEmpty {
                omniPerfLog(String(format: "shotscrub steps=%d decodes=%d p50_ms=%.1f p90_ms=%.1f max_ms=%.1f wall_ms=%.0f",
                                   path.count, times.count, times[times.count / 2], times[times.count * 9 / 10],
                                   times[times.count - 1], (CACurrentMediaTime() - t0) * 1000))
            }
            done()
        }
    }
}

/// The file a dragged shot carries: a PNG of the whole frame, named as macOS names a screenshot,
/// in a temporary folder of Omni's.
///
/// A REAL FILE, NOT A PROMISE. A promised file is written when it is dropped, and only where the
/// receiver asks for one: the Finder, Mail and Notes do, a Dock icon, a browser's upload area and
/// most chat windows take a file that already exists. So the file is written when the button goes
/// down on the shot, off the main thread, and a drag that starts before it is done waits for it.
final class ShotDragFile: @unchecked Sendable {
    private static let folder = FileManager.default.temporaryDirectory.appendingPathComponent("omni-shots", isDirectory: true)
    private let group = DispatchGroup()
    /// Written before the group is left, read after a wait on it.
    private var url: URL?

    private struct Pixels: @unchecked Sendable { let buffer: CVPixelBuffer }

    init(_ pixels: CVPixelBuffer, time: Double) {
        let held = Pixels(buffer: pixels)
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            url = Self.write(held.buffer, time: time)
            group.leave()
        }
    }

    func wait() -> URL? {
        group.wait()
        return url
    }

    private static func write(_ pixels: CVPixelBuffer, time: Double) -> URL? {
        let fm = FileManager.default
        try? fm.createDirectory(at: folder, withIntermediateDirectories: true)
        // A day keeps a file past any drop that still reads it.
        let old = Date().addingTimeInterval(-86_400)
        for file in (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [] {
            if let date = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate, date < old {
                try? fm.removeItem(at: file)
            }
        }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        let url = folder.appendingPathComponent("Screenshot " + f.string(from: Date(timeIntervalSince1970: time)) + ".png")
        guard let image = ScreenCodec.cgImage(pixels) else { return nil }
        let rep = NSBitmapImageRep(cgImage: image)
        guard let png = rep.representation(using: .png, properties: [:]), (try? png.write(to: url, options: .atomic)) != nil else { return nil }
        return url
    }
}

/// The decoded shot, drawn by a layer straight from the decoder's surface: no copy and no image
/// conversion per frame. Dragged, it leaves as a file.
final class ShotSurfaceView: NSView, NSDraggingSource {
    private let picture = CALayer()
    private var pixels: CVPixelBuffer?
    private var time = 0.0
    private var pressed: (at: NSPoint, file: ShotDragFile)?

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        picture.contentsGravity = .resizeAspect
        picture.minificationFilter = .trilinear
        picture.actions = ["contents": NSNull(), "bounds": NSNull(), "position": NSNull()]
        layer?.addSublayer(picture)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layout() {
        super.layout()
        picture.frame = bounds
    }

    func show(_ pixels: CVPixelBuffer?, time: Double) {
        self.pixels = pixels
        self.time = time
        guard let pixels else { picture.contents = nil; return }
        if let surface = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() {
            picture.contents = surface
        } else {
            picture.contents = ScreenCodec.cgImage(pixels)
        }
    }

    // MARK: Dragging the shot out

    override func mouseDown(with event: NSEvent) {
        guard let pixels else { return }
        pressed = (event.locationInWindow, ShotDragFile(pixels, time: time))
    }

    override func mouseUp(with event: NSEvent) { pressed = nil }

    override func mouseDragged(with event: NSEvent) {
        guard let press = pressed else { return }
        let p = event.locationInWindow
        guard hypot(p.x - press.at.x, p.y - press.at.y) >= 4 else { return }
        pressed = nil
        guard let url = press.file.wait(), let pixels, let image = ScreenCodec.cgImage(pixels) else { return }
        // A small picture of the shot under the pointer, as the Finder drags an image file.
        let width = min(bounds.width, 220), height = width * CGFloat(image.height) / CGFloat(max(1, image.width))
        let at = convert(p, from: nil)
        let item = NSDraggingItem(pasteboardWriter: url as NSURL)
        item.setDraggingFrame(NSRect(x: at.x - width / 2, y: at.y - height / 2, width: width, height: height),
                              contents: NSImage(cgImage: image, size: NSSize(width: width, height: height)))
        beginDraggingSession(with: [item], event: event, source: self)
    }

    func draggingSession(_ session: NSDraggingSession, sourceOperationMaskFor context: NSDraggingContext) -> NSDragOperation { .copy }
}

private struct ShotSurface: NSViewRepresentable {
    let state: ScreenTimelineState
    func makeNSView(context: Context) -> ShotSurfaceView {
        let view = ShotSurfaceView(frame: .zero)
        state.attach(view)
        return view
    }
    func updateNSView(_ view: ShotSurfaceView, context: Context) {}
}

/// The Screenshots place with no query: the shot, and under it the day as a track.
struct ScreenTimeline: View {
    @Environment(AppModel.self) private var model: AppModel
    /// Made on first appearance, never as the property's initial value (ui.md, "Renders per step").
    @State private var state: ScreenTimelineState?

    var body: some View {
        let _ = UIProbe.count("body.ScreenTimeline")
        Group {
            if let state {
                if state.loaded, state.shots.isEmpty {
                    CenteredStatus(symbol: CenteredStatus.moleEmpty, title: "No screenshots yet",
                                   subtitle: model.screenshotsEnabled ? "Omni takes one when you use the Mac." : "")
                } else {
                    ShotPane(state: state)
                        .safeAreaInset(edge: .bottom, spacing: 0) { TimelineBar(state: state) }
                        .background(TimelineKeys(state: state) { ScreenshotsDelete.confirm(model, state, .shot) })
                }
            } else {
                Color.clear
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            if state == nil { state = ScreenTimelineState(history: AppModel.screenHistory) }
            if let shot = model.takeScreenshotTarget() { state?.go(to: shot) } else { state?.reload() }
        }
        .onChange(of: model.screenshotsRevision) { _, _ in state?.reload() }
        .onChange(of: model.screenshotTargetTick) { _, _ in
            if let shot = model.takeScreenshotTarget() { state?.go(to: shot) }
        }
        .onReceive(NotificationCenter.default.publisher(for: .omniPerfShot)) { note in
            guard let state, let text = note.object as? String else { return }
            let parts = text.split(separator: ":", maxSplits: 1).map(String.init)
            switch parts.first {
            case "select":
                let n = Int(parts.last ?? "") ?? 0
                state.select(n < 0 ? state.shots.count + n : n)
            case "scrub": state.perfScrub(Int(parts.last ?? "") ?? 60) { omniPerfLog("shotscrub done") }
            case "day": state.stepDay(Int(parts.last ?? "") ?? 0)
            case "display":
                let n = Int(parts.last ?? "") ?? 0
                if state.displays.indices.contains(n) { state.reload(display: state.displays[n]) }
            case "zoom":
                if parts.last == "reset" { state.showWholeDay() } else { state.zoomAboutSelection(by: Double(parts.last ?? "") ?? 2) }
            case "pan": state.pan(by: Double(parts.last ?? "") ?? 0)
            case "dragfile": state.perfDragFile()
            case "delete":   // shot | session | day, without the confirmation
                guard let scope = ScreenTimelineState.DeleteScope(rawValue: parts.last ?? "") else { break }
                state.deletion(scope) { d in
                    omniPerfLog("shot-delete asked scope=\(scope.rawValue) shots=\(d?.shots ?? 0)")
                    if let d { model.deleteScreenshots(from: d.from, to: d.to, display: d.display) }
                }
            default: break
            }
        }
    }
}

/// The shot, fitted, with the corners and the lift of a Quick Look preview.
private struct ShotPane: View {
    let state: ScreenTimelineState

    var body: some View {
        let _ = UIProbe.count("body.ShotPane")
        ShotSurface(state: state)
            .aspectRatio(state.pixelSize, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(.separator, lineWidth: 0.5))
            .shadow(color: .black.opacity(0.18), radius: 14, y: 5)
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .contextMenu { ShotMenu(state: state) }
            .accessibilityElement()
            .accessibilityLabel("Screenshot")
    }
}

private struct ShotMenu: View {
    @Environment(AppModel.self) private var model: AppModel
    let state: ScreenTimelineState

    var body: some View {
        Button { state.copyShown() } label: { Label("Copy", systemImage: "doc.on.doc") }
        if state.shot?.indexed == true {
            Button { if let path = state.pathOfShown() { model.searchBySimilar(to: path) } } label: {
                Label("Find Similar", systemImage: "sparkle.magnifyingglass")
            }
        }
        Divider()
        Menu {
            Button("This Screenshot\u{2026}") { ScreenshotsDelete.confirm(model, state, .shot) }
            Button("This Session\u{2026}") { ScreenshotsDelete.confirm(model, state, .session) }
            Button("This Day\u{2026}") { ScreenshotsDelete.confirm(model, state, .day) }
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }
}

private enum ShotCaption {
    static func subject(_ shot: ScreenHistory.Shot) -> String {
        [shot.app, shot.title].filter { !$0.isEmpty }.joined(separator: " - ")
    }
    static func time(_ shot: ScreenHistory.Shot) -> String {
        Date(timeIntervalSince1970: shot.time).formatted(date: .omitted, time: .standard)
    }
    static func spoken(_ shot: ScreenHistory.Shot?) -> String {
        guard let shot else { return "" }
        return [time(shot), subject(shot)].filter { !$0.isEmpty }.joined(separator: ", ")
    }
    static func day(_ day: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(day) { return "Today" }
        if calendar.isDateInYesterday(day) { return "Yesterday" }
        return day.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day())
    }
}

/// The controls under the shot: the day, what the shot shows, and the track. One Liquid Glass
/// surface, the controls inside it plain (ui.md, "Liquid Glass").
private struct TimelineBar: View {
    let state: ScreenTimelineState

    var body: some View {
        let _ = UIProbe.count("body.TimelineBar")
        GlassGroup {
            VStack(spacing: 6) {
                HStack(spacing: 10) {
                    DayPicker(state: state)
                    Spacer(minLength: 8)
                    ShotLabel(state: state)
                    Spacer(minLength: 8)
                    ShotCount(state: state)
                }
                TimelineTrack(state: state)
            }
            .padding(.horizontal, 16)
            .padding(.top, 9)
            .padding(.bottom, 8)
            .glassPanel(cornerRadius: 18)
            .chipShadow()
        }
        .frame(maxWidth: 860)
        .padding(.horizontal, 20)
        .padding(.bottom, 14)
    }
}

private struct DayPicker: View {
    let state: ScreenTimelineState

    private var index: Int? { state.day.flatMap { state.days.firstIndex(of: $0) } }

    var body: some View {
        let _ = UIProbe.count("body.DayPicker")
        HStack(spacing: 2) {
            Button { state.stepDay(-1) } label: { Image(systemName: "chevron.left") }
                .disabled((index ?? 0) == 0)
                .help("Previous day")
                .accessibilityLabel("Previous day")
            Menu {
                ForEach(state.days.reversed(), id: \.self) { day in
                    Toggle(ShotCaption.day(day), isOn: Binding(get: { state.day == day }, set: { _ in state.reload(day: day) }))
                }
                if state.displays.count > 1 {
                    Divider()
                    ForEach(Array(state.displays.enumerated()), id: \.element) { n, display in
                        Toggle("Display \(n + 1)", isOn: Binding(get: { state.display == display },
                                                                 set: { _ in state.reload(display: display) }))
                    }
                }
            } label: {
                Text(state.day.map(ShotCaption.day) ?? "").font(.callout.weight(.medium))
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            Button { state.stepDay(1) } label: { Image(systemName: "chevron.right") }
                .disabled(index.map { $0 >= state.days.count - 1 } ?? true)
                .help("Next day")
                .accessibilityLabel("Next day")
        }
        .buttonStyle(.borderless)
        .frame(minWidth: 150, alignment: .leading)
    }
}

private struct ShotLabel: View {
    let state: ScreenTimelineState

    var body: some View {
        let _ = UIProbe.count("body.ShotLabel")
        if let shot = state.shot {
            HStack(spacing: 8) {
                Text(ShotCaption.time(shot)).font(.callout.monospacedDigit().weight(.medium))
                let subject = ShotCaption.subject(shot)
                if !subject.isEmpty {
                    Text(subject).font(.callout).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            .help(ShotCaption.subject(shot))
        }
    }
}

private struct ShotCount: View {
    let state: ScreenTimelineState

    var body: some View {
        let _ = UIProbe.count("body.ShotCount")
        if let selected = state.selected {
            Text("\((selected + 1).formatted()) of \(state.shots.count.formatted())")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 150, alignment: .trailing)
        }
    }
}

/// The day, true to time: a point per shot (the shots the index holds drawn larger), the hours
/// under them, and the playhead. Drag anywhere on it; pinch or scroll up and down to zoom into a
/// stretch of the day, scroll sideways to move along it.
private struct TimelineTrack: View {
    let state: ScreenTimelineState

    static let inset: CGFloat = 6

    var body: some View {
        let _ = UIProbe.count("body.TimelineTrack")
        let range = state.span
        let zoomed = state.window != nil
        GeometryReader { geo in
            let inset = Self.inset
            let width = max(1, geo.size.width - inset * 2)
            ZStack(alignment: .topLeading) {
                TrackPoints(shots: state.shots, span: range, day: zoomed ? state.daySpan : nil, inset: inset).equatable()
                Playhead(state: state, span: range, inset: inset, width: width)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                let f = min(max((value.location.x - inset) / width, 0), 1)
                state.select(nearest: range.lowerBound + Double(f) * (range.upperBound - range.lowerBound))
            })
        }
        .frame(height: 38)
        .background(TrackZoom(state: state))
        .contextMenu {
            Button("Zoom In") { state.zoomAboutSelection(by: 2) }.keyboardShortcut("=", modifiers: .command)
            Button("Zoom Out") { state.zoomAboutSelection(by: 0.5) }.keyboardShortcut("-", modifiers: .command).disabled(!zoomed)
            Button("Show Whole Day") { state.showWholeDay() }.keyboardShortcut("0", modifiers: .command).disabled(!zoomed)
        }
        .accessibilityElement()
        .accessibilityLabel("Timeline")
        .accessibilityAdjustableAction { direction in
            state.step(direction == .increment ? 1 : -1)
        }
    }
}

/// Pinch and scroll over the track. A local monitor, as the arrow keys have: the track is drawn
/// by SwiftUI, which hands a view neither a scroll wheel nor where a pinch is.
private struct TrackZoom: NSViewRepresentable {
    let state: ScreenTimelineState

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        context.coordinator.install(state, over: view)
        return view
    }
    func updateNSView(_ view: NSView, context: Context) {}
    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) { coordinator.uninstall() }
    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor final class Coordinator {
        private var monitor: Any?

        func install(_ state: ScreenTimelineState, over view: NSView) {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.scrollWheel, .magnify]) { [weak state, weak view] event in
                // Local monitors run on the main thread, where this closure is isolated.
                guard let state, let view, event.window === view.window else { return event }
                let p = view.convert(event.locationInWindow, from: nil)
                guard view.bounds.contains(p) else { return event }
                let width = max(1, view.bounds.width - TimelineTrack.inset * 2)
                let f = Double(min(max((p.x - TimelineTrack.inset) / width, 0), 1))
                if event.type == .magnify {
                    state.zoom(by: 1 + Double(event.magnification), around: f)
                    return nil
                }
                // A wheel turns in lines, a trackpad in points.
                let unit: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
                let dx = event.scrollingDeltaX * unit, dy = event.scrollingDeltaY * unit
                if abs(dx) > abs(dy) { state.pan(by: Double(-dx / width)) }
                else if dy != 0 { state.zoom(by: exp(Double(dy) * 0.01), around: f) }
                return nil
            }
        }

        func uninstall() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }
    }
}

/// Drawn once per listing, not per drag: it reads the shots and never the selection.
private struct TrackPoints: View, Equatable {
    let shots: [ScreenHistory.Shot]
    let span: ClosedRange<Double>
    /// The whole day when the track is zoomed into part of it: drawn as a bar over the points,
    /// the part shown marked on it.
    let day: ClosedRange<Double>?
    let inset: CGFloat

    static let rowHeight: CGFloat = 22

    var body: some View {
        let _ = UIProbe.count("body.TrackPoints")
        Canvas { context, size in
            let width = max(1, size.width - inset * 2)
            let length = span.upperBound - span.lowerBound
            let y = Self.rowHeight / 2
            var rail = Path()
            rail.addRoundedRect(in: CGRect(x: inset, y: y - 0.5, width: width, height: 1), cornerSize: CGSize(width: 0.5, height: 0.5))
            context.fill(rail, with: .style(.quaternary))

            if let day {
                let whole = day.upperBound - day.lowerBound
                let from = inset + CGFloat((span.lowerBound - day.lowerBound) / whole) * width
                let shown = max(8, CGFloat(length / whole) * width)
                var all = Path(), part = Path()
                all.addRoundedRect(in: CGRect(x: inset, y: 0, width: width, height: 2), cornerSize: CGSize(width: 1, height: 1))
                part.addRoundedRect(in: CGRect(x: min(from, inset + width - shown), y: 0, width: shown, height: 2), cornerSize: CGSize(width: 1, height: 1))
                context.fill(all, with: .style(.quaternary))
                context.fill(part, with: .style(.secondary))
            }

            // One mark per point of width: a day of shots 5 s apart is several to a point.
            let columns = Int(width.rounded(.up)) + 1
            var any = [Bool](repeating: false, count: columns), indexed = [Bool](repeating: false, count: columns)
            var first = 0, end = shots.count
            while first < end {
                let mid = (first + end) / 2
                if shots[mid].time < span.lowerBound { first = mid + 1 } else { end = mid }
            }
            for shot in shots[first...] {
                if shot.time > span.upperBound { break }
                let c = min(columns - 1, max(0, Int(((shot.time - span.lowerBound) / length * Double(width)).rounded())))
                any[c] = true
                if shot.indexed { indexed[c] = true }
            }
            var plain = Path(), marked = Path()
            for c in 0 ..< columns where any[c] {
                let x = inset + CGFloat(c)
                if indexed[c] { marked.addEllipse(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6)) }
                else { plain.addEllipse(in: CGRect(x: x - 1.5, y: y - 1.5, width: 3, height: 3)) }
            }
            context.fill(plain, with: .style(.tertiary))
            context.fill(marked, with: .style(.secondary))

            // The hours, or parts of one on a short day.
            let step: Double = length <= 360 ? 60 : length <= 1800 ? 300 : length <= 2 * 3600 ? 900 : length <= 8 * 3600 ? 3600 : 2 * 3600
            let zone = Double(TimeZone.current.secondsFromGMT(for: Date(timeIntervalSince1970: span.lowerBound)))
            var t = ((span.lowerBound + zone) / step).rounded(.up) * step - zone
            var lastLabelEnd = -CGFloat.infinity
            while t <= span.upperBound {
                let x = inset + CGFloat((t - span.lowerBound) / length) * width
                var tick = Path()
                tick.addRect(CGRect(x: x - 0.5, y: Self.rowHeight - 3, width: 1, height: 3))
                context.fill(tick, with: .style(.quaternary))
                let date = Date(timeIntervalSince1970: t)
                let text = context.resolve(Text(date.formatted(date: .omitted, time: .shortened))
                    .font(.caption2.monospacedDigit()).foregroundStyle(.tertiary))
                let w = text.measure(in: size).width
                let left = min(max(x - w / 2, 0), size.width - w)
                if left >= lastLabelEnd + 8 {
                    context.draw(text, at: CGPoint(x: left, y: Self.rowHeight + 1), anchor: .topLeading)
                    lastLabelEnd = left + w
                }
                t += step
            }
        }
    }
}

private struct Playhead: View {
    let state: ScreenTimelineState
    let span: ClosedRange<Double>
    let inset: CGFloat
    let width: CGFloat

    var body: some View {
        let _ = UIProbe.count("body.Playhead")
        if let shot = state.shot, span.contains(shot.time) {
            let x = inset + CGFloat((shot.time - span.lowerBound) / (span.upperBound - span.lowerBound)) * width
            Capsule()
                .fill(Color.accentColor)
                .frame(width: 4, height: TrackPoints.rowHeight - 2)
                .shadow(color: .black.opacity(0.2), radius: 1.5, y: 0.5)
                .offset(x: x - 2, y: 1)
                .allowsHitTesting(false)
        }
    }
}

/// Left and right step a shot, with Option to the next one the index holds and with Command to
/// either end of the day; Command with = - 0 zooms the track; Command-Delete deletes the shot:
/// whichever part of the window has focus, except a text field.
private struct TimelineKeys: NSViewRepresentable {
    let state: ScreenTimelineState
    let delete: @MainActor () -> Void

    func makeNSView(context: Context) -> NSView {
        context.coordinator.install(state, delete: delete)
        return NSView(frame: .zero)
    }
    func updateNSView(_ view: NSView, context: Context) {}
    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) { coordinator.uninstall() }
    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor final class Coordinator {
        private var monitor: Any?

        func install(_ state: ScreenTimelineState, delete: @escaping @MainActor () -> Void) {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak state] event in
                // Local monitors run on the main thread, where this closure is isolated.
                guard let state, [123, 124, 24, 27, 29, 51].contains(event.keyCode),
                      let window = NSApp.keyWindow, window.identifier?.rawValue.hasPrefix("main") ?? false,
                      !(window.firstResponder is NSText) else { return event }
                let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
                if event.keyCode != 123, event.keyCode != 124 {
                    // = (with or without Shift, as Zoom In is everywhere), -, 0 and Delete.
                    guard flags == .command || (event.keyCode == 24 && flags == [.command, .shift]), state.shot != nil else { return event }
                    switch event.keyCode {
                    case 24: state.zoomAboutSelection(by: 2)
                    case 27: state.zoomAboutSelection(by: 0.5)
                    case 29: state.showWholeDay()
                    default: delete()
                    }
                    return nil
                }
                let forward = event.keyCode == 124
                if flags == .command { state.selectEnd(last: forward) }
                else if flags == .option { state.stepIndexed(forward: forward) }
                else if flags.isEmpty { state.step(forward ? 1 : -1) }
                else { return event }
                return nil
            }
        }

        func uninstall() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }
    }
}

/// Confirmation for deleting every screenshot, shared by the sidebar and Settings.
@MainActor
enum ScreenshotsClear {
    static func confirm(_ model: AppModel) {
        let alert = NSAlert()
        alert.messageText = "Clear all screenshots?"
        alert.addButton(withTitle: "Clear")
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { model.clearScreenshots() }
    }
}

/// Confirmation for deleting one shot, the session around it, or its day.
@MainActor
enum ScreenshotsDelete {
    static func confirm(_ model: AppModel, _ state: ScreenTimelineState, _ scope: ScreenTimelineState.DeleteScope) {
        state.deletion(scope) { deletion in
            guard let d = deletion, d.shots > 0 else { return }
            let alert = NSAlert()
            let clock = Date.FormatStyle(date: .omitted, time: .shortened)
            let from = Date(timeIntervalSince1970: d.from), to = Date(timeIntervalSince1970: d.to)
            switch d.scope {
            case .shot:
                alert.messageText = "Delete this screenshot?"
                alert.informativeText = from.formatted(date: .omitted, time: .standard)
            case .session:
                alert.messageText = "Delete this session?"
                alert.informativeText = d.shots == 1 ? "1 screenshot, \(from.formatted(clock))"
                    : "\(d.shots.formatted()) screenshots, \(from.formatted(clock)) to \(to.formatted(clock))"
            case .day:
                let calendar = Calendar.current
                let day = calendar.isDateInToday(from) ? "Today" : calendar.isDateInYesterday(from) ? "Yesterday"
                    : from.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day())
                alert.messageText = "Delete this day?"
                alert.informativeText = "\(day), \(d.shots.formatted()) screenshot\(d.shots == 1 ? "" : "s")"
            }
            alert.addButton(withTitle: "Delete")
            alert.addButton(withTitle: "Cancel")
            alert.buttons.first?.hasDestructiveAction = true
            if alert.runModal() == .alertFirstButtonReturn { model.deleteScreenshots(from: d.from, to: d.to, display: d.display) }
        }
    }
}

extension Notification.Name {
    static let omniPerfShot = Notification.Name("omni.perf.shot")
}
