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

    @ObservationIgnored private let history: ScreenHistory
    @ObservationIgnored private weak var surface: ShotSurfaceView?
    /// Held while it is on screen, so the decoder's pool does not hand its surface out again.
    @ObservationIgnored private var shown: (id: String, pixels: CVPixelBuffer)?
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
        let keep = requested == nil && requestedDisplay == nil ? shot : nil
        Self.list(history, day: follows ? nil : wantDay, display: wantDisplay) { [weak self] listing in
            guard let self, self.generation == generation else { return }
            self.apply(listing, keep: follows ? nil : keep)
        }
    }

    private nonisolated static func list(_ history: ScreenHistory, day: Date?, display: UInt32?,
                                         done: @escaping @MainActor @Sendable (Listing) -> Void) {
        queue.async {
            var out = Listing()
            out.days = history.days()
            out.day = day.flatMap { out.days.contains($0) ? $0 : nil } ?? out.days.last
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
        if day != listing.day { day = listing.day }
        if displays != listing.displays { displays = listing.displays }
        if display != listing.display { display = listing.display }
        if shots != listing.shots { shots = listing.shots }
        if !loaded { loaded = true }
        let index = keep.flatMap { k in shots.firstIndex { $0.index == k.index && $0.segment == k.segment } }
            ?? (shots.isEmpty ? nil : shots.count - 1)
        if selected != index { selected = index }
        if let shot { show(shot) } else { clearPicture() }
    }

    // MARK: Moving

    func select(_ index: Int) {
        guard shots.indices.contains(index), index != selected else { return }
        selected = index
        show(shots[index])
    }

    /// The shot nearest a moment of the day.
    func select(nearest time: Double) {
        guard !shots.isEmpty else { return }
        var lo = 0, hi = shots.count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if shots[mid].time < time { lo = mid + 1 } else { hi = mid }
        }
        if lo > 0, time - shots[lo - 1].time <= shots[lo].time - time { lo -= 1 }
        select(lo)
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
        view.show(shown?.pixels)
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
        guard shown?.id != shot.id else { return }
        decoding = true
        Self.decode(shot, from: history) { [weak self] box, ms in
            guard let self else { return }
            self.decoding = false
            self.decodeTimes.append(ms)
            if let pixels = box.pixels, !self.shots.isEmpty {
                self.shown = (shot.id, pixels)
                let size = CGSize(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels))
                if size != self.pixelSize { self.pixelSize = size }
                self.surface?.show(pixels)
            }
            self.decodeNext()
        }
    }

    private func clearPicture() {
        wanted = nil
        shown = nil
        surface?.show(nil)
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

    /// The shown shot's file in the index, when it has one.
    func stillOfShown() -> URL? { shot.flatMap { history.still(of: $0) } }

    // MARK: The perf script

    var perfSummary: [String: Any] {
        ["day": day.map { $0.timeIntervalSince1970 } ?? 0, "days": days.count, "shots": shots.count,
         "indexed": shots.reduce(0) { $0 + ($1.indexed ? 1 : 0) }, "selected": selected ?? -1,
         "shown": shown?.id ?? "", "time": shot?.time ?? 0, "app": shot?.app ?? "", "title": shot?.title ?? ""]
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

/// The decoded shot, drawn by a layer straight from the decoder's surface: no copy and no image
/// conversion per frame.
final class ShotSurfaceView: NSView {
    private let picture = CALayer()

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

    func show(_ pixels: CVPixelBuffer?) {
        guard let pixels else { picture.contents = nil; return }
        if let surface = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() {
            picture.contents = surface
        } else {
            picture.contents = ScreenCodec.cgImage(pixels)
        }
    }
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
        Group {
            if let state {
                if state.loaded, state.shots.isEmpty {
                    CenteredStatus(symbol: CenteredStatus.moleEmpty, title: "No screenshots yet",
                                   subtitle: model.screenshotsEnabled ? "Omni takes one when you use the Mac." : "")
                } else {
                    ShotPane(state: state)
                        .safeAreaInset(edge: .bottom, spacing: 0) { TimelineBar(state: state) }
                        .background(TimelineKeys(state: state))
                }
            } else {
                Color.clear
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            if state == nil { state = ScreenTimelineState(history: AppModel.screenHistory) }
            state?.reload()
        }
        .onChange(of: model.screenshotsRevision) { _, _ in state?.reload() }
        .onReceive(NotificationCenter.default.publisher(for: .omniPerfShot)) { note in
            guard let state, let text = note.object as? String else { return }
            let parts = text.split(separator: ":", maxSplits: 1).map(String.init)
            switch parts.first {
            case "select":
                let n = Int(parts.last ?? "") ?? 0
                state.select(n < 0 ? state.shots.count + n : n)
            case "scrub": state.perfScrub(Int(parts.last ?? "") ?? 60) { omniPerfLog("shotscrub done") }
            case "day": state.stepDay(Int(parts.last ?? "") ?? 0)
            default: break
            }
        }
    }
}

/// The shot, fitted, with the corners and the lift of a Quick Look preview.
private struct ShotPane: View {
    let state: ScreenTimelineState

    var body: some View {
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
            Button { if let url = state.stillOfShown() { model.searchBySimilar(to: url.path) } } label: {
                Label("Find Similar", systemImage: "sparkle.magnifyingglass")
            }
            Button { if let url = state.stillOfShown() { NSWorkspace.shared.activateFileViewerSelecting([url]) } } label: {
                Label("Show in Finder", systemImage: "folder")
            }
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
        if let selected = state.selected {
            Text("\((selected + 1).formatted()) of \(state.shots.count.formatted())")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 150, alignment: .trailing)
        }
    }
}

/// The day, true to time: a point per shot (the shots the index holds drawn larger), the hours
/// under them, and the playhead. Drag anywhere on it.
private struct TimelineTrack: View {
    let state: ScreenTimelineState

    /// The stretch of the day the track covers: first shot to last, never under ten minutes.
    private var span: ClosedRange<Double> {
        guard let first = state.shots.first?.time, let last = state.shots.last?.time else { return 0 ... 1 }
        let pad = max(0, 600 - (last - first)) / 2
        return (first - pad) ... (last + pad)
    }

    var body: some View {
        let range = span
        GeometryReader { geo in
            let inset: CGFloat = 6
            let width = max(1, geo.size.width - inset * 2)
            ZStack(alignment: .topLeading) {
                TrackPoints(shots: state.shots, span: range, inset: inset).equatable()
                Playhead(state: state, span: range, inset: inset, width: width)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                let f = min(max((value.location.x - inset) / width, 0), 1)
                state.select(nearest: range.lowerBound + Double(f) * (range.upperBound - range.lowerBound))
            })
        }
        .frame(height: 38)
        .accessibilityElement()
        .accessibilityLabel("Timeline")
        .accessibilityAdjustableAction { direction in
            state.step(direction == .increment ? 1 : -1)
        }
    }
}

/// Drawn once per listing, not per drag: it reads the shots and never the selection.
private struct TrackPoints: View, Equatable {
    let shots: [ScreenHistory.Shot]
    let span: ClosedRange<Double>
    let inset: CGFloat

    static let rowHeight: CGFloat = 22

    var body: some View {
        Canvas { context, size in
            let width = max(1, size.width - inset * 2)
            let length = span.upperBound - span.lowerBound
            let y = Self.rowHeight / 2
            var rail = Path()
            rail.addRoundedRect(in: CGRect(x: inset, y: y - 0.5, width: width, height: 1), cornerSize: CGSize(width: 0.5, height: 0.5))
            context.fill(rail, with: .style(.quaternary))

            // One mark per point of width: a day of shots 5 s apart is several to a point.
            let columns = Int(width.rounded(.up)) + 1
            var any = [Bool](repeating: false, count: columns), indexed = [Bool](repeating: false, count: columns)
            for shot in shots {
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
            let step: Double = length <= 1800 ? 300 : length <= 2 * 3600 ? 900 : length <= 8 * 3600 ? 3600 : 2 * 3600
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
        if let shot = state.shot {
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
/// either end of the day: whichever part of the window has focus, except a text field.
private struct TimelineKeys: NSViewRepresentable {
    let state: ScreenTimelineState

    func makeNSView(context: Context) -> NSView {
        context.coordinator.install(state)
        return NSView(frame: .zero)
    }
    func updateNSView(_ view: NSView, context: Context) {}
    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) { coordinator.uninstall() }
    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor final class Coordinator {
        private var monitor: Any?

        func install(_ state: ScreenTimelineState) {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak state] event in
                // Local monitors run on the main thread, where this closure is isolated.
                guard let state, event.keyCode == 123 || event.keyCode == 124,
                      let window = NSApp.keyWindow, window.identifier?.rawValue.hasPrefix("main") ?? false,
                      !(window.firstResponder is NSText) else { return event }
                let forward = event.keyCode == 124
                let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
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
        alert.messageText = "Clear screenshots?"
        alert.informativeText = "Every screenshot Omni recorded will be deleted."
        alert.addButton(withTitle: "Clear")
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { model.clearScreenshots() }
    }
}

extension Notification.Name {
    static let omniPerfShot = Notification.Name("omni.perf.shot")
}
