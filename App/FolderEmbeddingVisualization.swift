import SwiftUI
import AppKit
import OmniKit

/// The folder embedding map: a 2D scatter of every indexed file under the selected folder, laid out
/// by `ProjectionEngine` so semantically similar files cluster together. It only ever reads the
/// embeddings already in the store - it never indexes or embeds, so un-indexed files simply don't
/// appear. Dots are colored by file type (one main hue per kind) and shaded per extension; they are
/// drawn semi-transparent so overlapping dots reveal cluster density.
///
/// Rendering is done by `MetalScatterView` (GPU point sprites), not SwiftUI `Canvas`: at tens of
/// thousands of files a CoreGraphics per-point loop dominates and re-runs every pan/zoom frame.
/// Here positions+colors upload to a GPU buffer once, and pan/zoom are a uniform update. Hover is a
/// separate SwiftUI overlay (a ring + filename), so moving the mouse never re-renders the cloud.
/// Shown only by ContentView precedence (a folder is selected and no query is active).
struct FolderEmbeddingVisualization: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(\.colorScheme) private var colorScheme   // re-shade the palette when light/dark flips
    let folderName: String

    @State private var hovered: ProjectionPoint?
    @State private var hoverLocation: CGPoint = .zero
    @State private var selectedIndex: Int?                // clicked point; its kNN stay lit, rest dimmed
    @State private var litNeighbors: [Int] = []           // the selected point's kNN row (for the overlay)
    @State private var positions: [SIMD2<Float>] = []     // model-space, row-aligned with colors/folderProjection
    @State private var baseColors: [SIMD4<Float>] = []    // full per-point RGBA (pre-dimming)
    @State private var colors: [SIMD4<Float>] = []        // displayed RGBA (dimmed when a point is selected)
    @State private var bbox = SIMD4<Float>(0, 0, 1, 1)    // cached (cx, cy, extX, extY) for O(1) hit-test + ring
    @State private var dataVersion = 0
    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = .zero
    @GestureState private var pinch: CGFloat = 1
    @GestureState private var dragOffset: CGSize = .zero
    @State private var scroller = ScrollZoomCatcher()   // mouse-wheel / two-finger-scroll -> zoom

    private static let inset: CGFloat = 24      // padding from the canvas edges (the fitted view)
    private static let hitRadius: CGFloat = 10  // hover hit-test tolerance
    private static let dotAlpha: Float = 0.55   // translucent so overlap shows density
    private static let zoomRange: ClosedRange<CGFloat> = 0.4 ... 40
    private static let zoomStep: CGFloat = 1.35

    private var effectiveZoom: CGFloat { (zoom * pinch).clamped(to: Self.zoomRange) }
    private var effectivePan: CGSize { CGSize(width: pan.width + dragOffset.width, height: pan.height + dragOffset.height) }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // GPU point cloud. Redraws only when data/zoom/pan change (not on hover).
                MetalScatterView(points: positions, colors: colors, dataVersion: dataVersion,
                                 zoom: effectiveZoom, pan: effectivePan,
                                 dotRadius: Self.radius(for: positions.count), inset: Self.inset)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)   // the GPU canvas isn't a VoiceOver target; see the container label

                // Empty state: folder selected but nothing under it is indexed (and not mid-fit).
                if positions.isEmpty && !model.folderProjectionFitting {
                    ContentUnavailableView {
                        Label("No files to map", systemImage: "circle.grid.cross")
                    } description: {
                        Text("Nothing under \(folderName) is indexed yet.")
                    }
                    .allowsHitTesting(false)
                }

                // Spotlight overlay over the grey cloud: thin lines from the selected file to each of
                // its nearest neighbors, then a thumbnail at every lit point so you see what each file
                // IS without hovering. Thumbnails don't capture clicks, so clicking one re-selects that
                // neighbor (the dot underneath) - letting you walk the neighbor graph.
                if let sel = selectedIndex, sel < model.folderProjection.count {
                    let pts = model.folderProjection
                    ZStack {
                        Canvas { ctx, size in
                            let map = screenMap(in: size)
                            let selP = map(pts[sel].position)
                            for nb in litNeighbors where nb < pts.count {
                                var path = Path(); path.move(to: selP); path.addLine(to: map(pts[nb].position))
                                ctx.stroke(path, with: .color(.primary.opacity(0.22)), lineWidth: 1)
                            }
                        }
                        // Neighbors first, the selected file LAST so it (larger, accent-ringed) sits on top.
                        ForEach((litNeighbors + [sel]).filter { $0 < pts.count }, id: \.self) { i in
                            let isSel = (i == sel)
                            Thumbnail(path: pts[i].path, side: isSel ? 52 : 38, corner: 6)
                                .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous)
                                    .strokeBorder(isSel ? Color.accentColor : .primary.opacity(0.2), lineWidth: isSel ? 2.5 : 1))
                                .shadow(color: .black.opacity(0.25), radius: 3, y: 1)
                                .position(screenPoint(pts[i].position, in: geo.size))
                        }
                    }
                    .allowsHitTesting(false)
                    .transition(.opacity)
                }

                // Hover overlay: a ring on the dot + a thumbnail-and-name chip near the cursor.
                if let h = hovered {
                    let s = screenPoint(h.position, in: geo.size)
                    let d = max(Self.radius(for: positions.count), 3) * 2 + 5
                    Circle().stroke(.primary, lineWidth: 1.5)
                        .frame(width: d, height: d).position(s).allowsHitTesting(false)
                    hoverChip(for: h, in: geo.size)
                }

                caption
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(Design.gapLarge)
                    .allowsHitTesting(false)

                zoomControls
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                    .padding(Design.gapLarge)

                legend
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                    .padding(Design.gapLarge)
                    .allowsHitTesting(false)
            }
            // Interaction on the ZStack itself (the Metal view is hit-testing-disabled): drag pans,
            // pinch zooms, hover picks the nearest point. This is the structure the Canvas version
            // used (hover on the container), which reliably receives the hover stream.
            .contentShape(Rectangle())
            // One drag gesture handles BOTH pan and click (a separate TapGesture gets swallowed by the
            // drag). minimumDistance 0 means a plain click ends with ~0 translation -> treat as a click
            // that toggles the spotlight on the nearest point; a real drag pans. The spotlight reuses
            // the kNN graph computed during the fit - no recompute.
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($dragOffset) { v, s, _ in s = v.translation }
                    .onEnded { v in
                        if hypot(v.translation.width, v.translation.height) < 4 {
                            let idx = nearestIndex(to: v.location, in: geo.size)
                            withAnimation(.easeOut(duration: 0.16)) {
                                selectedIndex = (idx != nil && idx == selectedIndex) ? nil : idx
                            }
                            applyHighlight()
                        } else {
                            pan.width += v.translation.width; pan.height += v.translation.height
                        }
                    }
            )
            // Two-finger pinch to zoom (alongside the +/-/reset controls and ⌘+ / ⌘- / ⌘0).
            .simultaneousGesture(
                MagnifyGesture()
                    .updating($pinch) { v, s, _ in s = v.magnification }
                    .onEnded { v in zoom = (zoom * v.magnification).clamped(to: Self.zoomRange) }
            )
            .onContinuousHover { phase in
                switch phase {
                case .active(let loc):
                    hoverLocation = loc
                    hovered = nearestIndex(to: loc, in: geo.size).map { model.folderProjection[$0] }
                case .ended:
                    hovered = nil
                }
            }
            // Right-click a dot for the same file actions as a search result. The target is the dot
            // under the cursor (hover tracks it); over empty space there's nothing to act on.
            .contextMenu { if let h = hovered { dotMenu(h.path) } }
            // Mouse-wheel / two-finger-scroll zoom (anchored at the cursor), gated to the map's frame.
            .onAppear {
                scroller.vizFrame = geo.frame(in: .global)
                scroller.onScroll = { loc, f, sz in zoomAt(loc, factor: f, size: sz) }
                scroller.install()
            }
            .onChange(of: geo.frame(in: .global)) { scroller.vizFrame = $0 }
            .onDisappear { scroller.remove() }
            // The cloud itself isn't individually navigable, but expose a container summary + the
            // interaction model so VoiceOver users aren't met with an opaque canvas.
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Folder map for \(folderName). \(model.folderProjection.count) files laid out by similarity.")
            .accessibilityHint("Drag to pan, scroll or pinch to zoom, click a file to highlight its nearest neighbors.")
        }
        // A new folder resets the view and clears the cloud (folderProjection is empty until the fit
        // lands, so this blanks the old map under the "Mapping..." spinner instead of leaving it
        // stale). The GPU buffer then rebuilds when a new layout lands (keyed on the generation, since
        // two folders can share a file count) or the appearance flips.
        .onChange(of: model.selectedFolderForViz) { selectedIndex = nil; resetView(); rebuildPoints() }
        .onChange(of: model.projectionGeneration) { rebuildPoints() }
        .onChange(of: colorScheme) { rebuildPoints() }
        .onAppear { rebuildPoints() }
    }

    // MARK: - Zoom

    private func zoomBy(_ factor: CGFloat) {
        withAnimation(.easeOut(duration: 0.15)) { zoom = (zoom * factor).clamped(to: Self.zoomRange) }
    }
    private func resetView() {
        withAnimation(.easeOut(duration: 0.18)) { zoom = 1; pan = .zero }
    }

    /// Zoom by `factor` while keeping the model point under the cursor fixed on screen (anchored zoom).
    /// Mirrors the transform: screen = mid + (model-center)*baseScale*zoom + pan, so to hold the cursor
    /// point put pan' = pan - (cursor - mid - pan)*(f-1).
    private func zoomAt(_ loc: CGPoint, factor: CGFloat, size: CGSize) {
        let newZoom = (zoom * factor).clamped(to: Self.zoomRange)
        let f = newZoom / zoom
        guard abs(f - 1) > 1e-4 else { return }
        pan.width  -= (loc.x - size.width  / 2 - pan.width)  * (f - 1)
        pan.height -= (loc.y - size.height / 2 - pan.height) * (f - 1)
        zoom = newZoom
    }

    @ViewBuilder private var zoomControls: some View {
        HStack(spacing: 2) {
            Button { zoomBy(1 / Self.zoomStep) } label: { Image(systemName: "minus") }
                .help("Zoom out (\u{2318}\u{2212})")
                .accessibilityLabel("Zoom out")
                .keyboardShortcut("-", modifiers: .command)
                .disabled(zoom <= Self.zoomRange.lowerBound)
            Button { resetView() } label: { Image(systemName: "arrow.counterclockwise") }
                .help("Reset zoom (\u{2318}0)")
                .accessibilityLabel("Reset zoom")
                .keyboardShortcut("0", modifiers: .command)
                .disabled(zoom == 1 && pan == .zero)
            Button { zoomBy(Self.zoomStep) } label: { Image(systemName: "plus") }
                .help("Zoom in (\u{2318}+)")
                .accessibilityLabel("Zoom in")
                .keyboardShortcut("=", modifiers: .command)
                .disabled(zoom >= Self.zoomRange.upperBound)
        }
        .buttonStyle(.borderless)
        .labelStyle(.iconOnly)
        .font(.system(size: 12, weight: .medium))
        .padding(.horizontal, 8).padding(.vertical, 5)
        .glassChip(interactive: true)
    }

    // MARK: - Overlays

    @ViewBuilder private var caption: some View {
        let count = model.folderProjection.count
        HStack(spacing: 6) {
            if model.folderProjectionFitting {
                ProgressView().controlSize(.small).scaleEffect(0.7)
                Text("Mapping \(folderName)\u{2026}")
            } else {
                Image(systemName: "circle.grid.cross").foregroundStyle(.secondary)
                Text(folderName).fontWeight(.medium)
                    .lineLimit(1).truncationMode(.middle).frame(maxWidth: 220)   // cap only the name, not the chip
                if count > 0 {
                    Text("\(count) file\(count == 1 ? "" : "s")").foregroundStyle(.secondary).lineLimit(1)
                }
            }
        }
        .font(.callout)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .glassChip()
    }

    @ViewBuilder private var legend: some View {
        let present = FileKind.allCases.filter { kind in
            model.folderProjection.contains { $0.kind == kind.rawValue }
        }
        if !present.isEmpty {
            HStack(spacing: 12) {
                ForEach(present, id: \.self) { kind in
                    HStack(spacing: 4) {
                        Circle().fill(kind.vizColor).frame(width: 8, height: 8)
                        Text(kind.title).font(.caption)
                    }
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            .glassChip()
        }
    }

    // MARK: - Data

    /// Build the GPU-ready positions + colors and cache the bounding box. Resolve each kind's dynamic
    /// system color to HSB ONCE for the active appearance (inside `performAsCurrentDrawingAppearance`,
    /// which is what makes the palette adapt to light/dark), then shade per extension to straight RGBA
    /// inline - no per-point NSColor/Color allocation, so it stays cheap for 50k+ files.
    private func rebuildPoints() {
        let pts = model.folderProjection
        var baseHSB: [String: (h: CGFloat, s: CGFloat, b: CGFloat)] = [:]
        let resolve = {
            for k in FileKind.allCases {
                var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                (k.vizNSColor.usingColorSpace(.sRGB) ?? k.vizNSColor).getHue(&h, saturation: &s, brightness: &b, alpha: &a)
                baseHSB[k.rawValue] = (h, s, b)
            }
        }
        if let appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua) {
            appearance.performAsCurrentDrawingAppearance(resolve)
        } else {
            resolve()
        }

        var pos = [SIMD2<Float>](); pos.reserveCapacity(pts.count)
        var col = [SIMD4<Float>](); col.reserveCapacity(pts.count)
        var mn = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var mx = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        let fallback = SIMD4<Float>(0.5, 0.5, 0.5, Self.dotAlpha)
        for p in pts {
            pos.append(p.position)
            if p.position.x.isFinite, p.position.y.isFinite {
                mn = pointwiseMin(mn, p.position); mx = pointwiseMax(mx, p.position)
            }
            if let base = baseHSB[p.kind] {
                col.append(FileKind.vizShadeRGBA(base: base, ext: (p.path as NSString).pathExtension, alpha: Self.dotAlpha))
            } else {
                col.append(fallback)
            }
        }
        if pts.isEmpty { mn = .zero; mx = .zero }
        let ext = pointwiseMax(mx - mn, SIMD2<Float>(1e-5, 1e-5))
        positions = pos
        baseColors = col
        bbox = SIMD4<Float>((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, ext.x, ext.y)
        applyHighlight()
    }

    /// Produce the displayed colors from `baseColors`: with no selection, show them as-is; with a
    /// selection, spotlight the clicked point + its (already-computed) 10 nearest neighbors and dim
    /// everyone else. Cheap - it only modulates alpha, no re-shading - then re-uploads the buffer.
    private func applyHighlight() {
        guard let sel = selectedIndex, sel >= 0, sel < baseColors.count else {
            colors = baseColors
            litNeighbors = []
            dataVersion &+= 1
            return
        }
        var nbrs: [Int] = []
        let k = model.folderKNNk, knn = model.folderKNN, take = min(10, k)
        if take > 0, sel * k + take <= knn.count {
            for j in 0 ..< take { nbrs.append(Int(knn[sel * k + j])) }
        }
        litNeighbors = nbrs
        // The cloud goes neutral grey (no color); the selected point + its neighbors are redrawn in
        // their real colors, larger and connected, by the SwiftUI overlay on top.
        let grey = SIMD4<Float>(0.55, 0.55, 0.55, 0.10)
        colors = baseColors.map { _ in grey }
        dataVersion &+= 1
    }

    /// File actions for a dot - the same set (and shortcuts) as a search result's context menu.
    /// "Find Similar" reuses `setFileQuery`, which runs a file-as-query search: that activates a query,
    /// so ContentView precedence swaps the map out for the live results (clearing it returns to the map).
    @ViewBuilder private func dotMenu(_ path: String) -> some View {
        Button("Open") { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
            .keyboardShortcut("o", modifiers: .command)
        Button("Quick Look") { model.previewURL = URL(fileURLWithPath: path) }
            .keyboardShortcut("y", modifiers: .command)
        Divider()
        Button("Find Similar") { model.setFileQuery(URL(fileURLWithPath: path), similar: true) }
            .keyboardShortcut("f", modifiers: [.command, .option])
        Divider()
        Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        Divider()
        Button("Copy Path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
        .keyboardShortcut("c", modifiers: [.command, .option])
    }

    @ViewBuilder private func hoverChip(for h: ProjectionPoint, in size: CGSize) -> some View {
        VStack(spacing: 5) {
            Thumbnail(path: h.path, side: 72)
            Text((h.path as NSString).lastPathComponent)
                .font(.caption).lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: 132)
        }
        .padding(7)
        .glassChip()
        .allowsHitTesting(false)
        .position(x: min(max(hoverLocation.x, 80), size.width - 80),
                  y: max(64, hoverLocation.y - 66))
    }

    /// Dot radius shrinks for very large folders so a dense cloud stays legible. Passed to the GPU.
    private static func radius(for n: Int) -> CGFloat {
        switch n {
        case ..<2_000:  return 3.6
        case ..<8_000:  return 2.7
        case ..<20_000: return 2.0
        default:        return 1.4
        }
    }

    // MARK: - Geometry (shared with the Metal shader's transform; uses the cached bbox, O(1) setup)

    /// Aspect-preserving fit of the cached bbox into the inset rect, then user zoom + pan. Mirrors the
    /// vertex shader so the hover hit-test and ring line up exactly with the rendered dots.
    private func screenMap(in size: CGSize) -> (SIMD2<Float>) -> CGPoint {
        let cx = bbox.x, cy = bbox.y, extX = bbox.z, extY = bbox.w
        let rect = CGRect(origin: .zero, size: size).insetBy(dx: Self.inset, dy: Self.inset)
        let scale = min(rect.width / CGFloat(extX), rect.height / CGFloat(extY)) * effectiveZoom
        let pan = effectivePan
        return { v in
            let x = v.x.isFinite ? v.x : cx, y = v.y.isFinite ? v.y : cy
            return CGPoint(x: rect.midX + CGFloat(x - cx) * scale + pan.width,
                           y: rect.midY + CGFloat(y - cy) * scale + pan.height)
        }
    }

    private func screenPoint(_ v: SIMD2<Float>, in size: CGSize) -> CGPoint { screenMap(in: size)(v) }

    /// Index of the nearest point to `loc` within `hitRadius` (linear scan - no rendering, so cheap
    /// even at 50k+). Returns the index so callers can look up the file and its kNN row.
    private func nearestIndex(to loc: CGPoint, in size: CGSize) -> Int? {
        let pts = model.folderProjection
        guard pts.count > 0 else { return nil }
        let map = screenMap(in: size)
        var best: Int?
        var bestD = Self.hitRadius * Self.hitRadius
        for i in 0 ..< pts.count {
            let s = map(pts[i].position)
            let dx = s.x - loc.x, dy = s.y - loc.y
            let d = dx * dx + dy * dy
            if d < bestD { bestD = d; best = i }
        }
        return best
    }
}

private func pointwiseMin(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> SIMD2<Float> { SIMD2(min(a.x, b.x), min(a.y, b.y)) }
private func pointwiseMax(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> SIMD2<Float> { SIMD2(max(a.x, b.x), max(a.y, b.y)) }

/// Captures scroll-wheel / two-finger-scroll over the map and reports a zoom factor + cursor location.
/// SwiftUI has no scroll hook for a custom view, so we use a local NSEvent monitor and gate it to the
/// map's frame (so scrolling the sidebar or anywhere else still scrolls normally). Held in @State so
/// the monitor survives view updates; the closures read live @State through their captured wrappers.
final class ScrollZoomCatcher {
    var onScroll: ((_ location: CGPoint, _ factor: CGFloat, _ size: CGSize) -> Void)?
    var vizFrame: CGRect = .zero          // the map's frame in SwiftUI global coords
    private var monitor: Any?

    func install() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
            guard let self, let content = event.window?.contentView else { return event }
            let p = event.locationInWindow
            let sp = CGPoint(x: p.x, y: content.bounds.height - p.y)   // AppKit (bottom-left) -> SwiftUI (top-left)
            guard self.vizFrame.contains(sp) else { return event }     // not over the map: scroll normally
            let d = Double(event.scrollingDeltaY)
            // Trackpad deltas are large+continuous; mouse-wheel notches are small+discrete - tune apart.
            let factor = CGFloat(event.hasPreciseScrollingDeltas ? exp(d * 0.004) : exp(d * 0.12))
            if abs(factor - 1) > 1e-4 {
                self.onScroll?(CGPoint(x: sp.x - self.vizFrame.minX, y: sp.y - self.vizFrame.minY),
                               factor, self.vizFrame.size)
            }
            return nil   // consume so the page doesn't also scroll
        }
    }
    func remove() { if let m = monitor { NSEvent.removeMonitor(m) }; monitor = nil; onScroll = nil }
    deinit { remove() }
}

private extension Comparable {
    func clamped(to r: ClosedRange<Self>) -> Self { min(max(self, r.lowerBound), r.upperBound) }
}
