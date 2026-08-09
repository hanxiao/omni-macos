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
    @State private var hoveredIndex: Int?                     // gates the publish above to real changes
    @State private var hoverDwellPath: String?                // hovered path once the cursor settles
    @State private var hoverDwellTask: Task<Void, Never>?
    @State private var hoverLocation: CGPoint = .zero
    @State private var lastHoverResolveLoc: CGPoint = .zero   // last cursor pos we ran the hit-test for (movement gate)
    @State private var rebuildTask: Task<Void, Never>?        // off-main point-cloud build, cancelled on rapid re-fit
    @State private var selectedIndex: Int?                // clicked point; its kNN stay lit, rest dimmed
    @State private var litNeighbors: [Int] = []           // the selected point's kNN row (for the overlay)
    @State private var positions: [SIMD2<Float>] = []     // model-space, row-aligned with colors/folderProjection
    @State private var baseColors: [SIMD4<Float>] = []    // full per-point RGBA (pre-dimming)
    @State private var colors: [SIMD4<Float>] = []        // displayed RGBA (dimmed when a point is selected)
    @State private var bbox = SIMD4<Float>(0, 0, 1, 1)    // cached (cx, cy, extX, extY) for O(1) hit-test + ring
    @State private var presentKinds: [FileKind] = []
    /// Columns of the grid when the layout is gridified (0 = not gridified). The dot size is
    /// derived from this: in grid mode the cell PITCH is what a dot must fit inside.
    @State private var gridCols = 0      // legend entries, computed once per projection (not per frame)
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
                                 dotRadius: dotRadius(in: geo.size), inset: Self.inset)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)   // the GPU canvas isn't a VoiceOver target; see the container label

                // Empty state: folder selected but nothing under it is indexed (and not mid-fit).
                if positions.isEmpty && !model.folderProjectionFitting {
                    // Same empty-state treatment as every search state in this pane
                    // (CenteredStatus) - the system ContentUnavailableView used different
                    // typography, so the pane's empty states visibly changed style.
                    CenteredStatus(symbol: "circle.grid.cross", title: "No files to map",
                                   subtitle: "Nothing under \(folderName) is indexed yet.")
                        .allowsHitTesting(false)
                }

                // Spotlight overlay over the grey cloud: thin lines from the selected file to each of
                // its nearest neighbors, then a thumbnail at every lit point so you see what each file
                // IS without hovering. Thumbnails don't capture clicks, so clicking one re-selects that
                // neighbor (the dot underneath) - letting you walk the neighbor graph.
                // COORDINATES COME FROM `positions`, NEVER from folderProjection: with "spread dots"
                // on, `positions` is the gridified layout the GPU actually draws (and the one the
                // hit-test scans), while folderProjection still holds the pre-grid fit. Drawing from
                // the latter put every ring, line and thumbnail at the file's OLD location, so the
                // spotlight and the hover ring pointed at a dot that wasn't there. folderProjection
                // is used for the path only.
                if let sel = selectedIndex, sel < positions.count, positions.count == model.folderProjection.count {
                    let pts = model.folderProjection
                    ZStack {
                        Canvas { ctx, size in
                            let map = screenMap(in: size)
                            let selP = map(positions[sel])
                            for nb in litNeighbors where nb < positions.count {
                                var path = Path(); path.move(to: selP); path.addLine(to: map(positions[nb]))
                                ctx.stroke(path, with: .color(.primary.opacity(0.22)), lineWidth: 1)
                            }
                        }
                        // Neighbors first, the selected file LAST so it (larger, accent-ringed) sits on top.
                        ForEach((litNeighbors + [sel]).filter { $0 < positions.count }, id: \.self) { i in
                            let isSel = (i == sel)
                            Thumbnail(path: pts[i].path, side: isSel ? 52 : 38, corner: 6)
                                .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous)
                                    .strokeBorder(isSel ? Color.accentColor : .primary.opacity(0.2), lineWidth: isSel ? 2.5 : 1))
                                .shadow(color: .black.opacity(0.25), radius: 3, y: 1)
                                .position(screenPoint(positions[i], in: geo.size))
                        }
                    }
                    .allowsHitTesting(false)
                    .transition(.opacity)
                }

                // The floating glass layer: hover chip + caption + zoom cluster + legend share ONE
                // GlassEffectContainer on macOS 26, so the up-to-four glass surfaces over the Metal
                // map render in a single effect pass per frame (the hover chip tracks the cursor at
                // event rate - per-element passes would be the expensive way). They sit in corners,
                // so the merge distance never triggers; the container is purely the batching.
                GlassGroup {
                    ZStack {
                        // Hover: a ring on the dot (plain stroke, not glass) + a thumbnail-and-name
                        // chip near the cursor.
                        // Ring position from `positions` (the drawn layout), same reason as the
                        // spotlight above; `hovered` supplies the name and thumbnail only.
                        if let h = hovered, let hi = hoveredIndex, hi < positions.count {
                            let s = screenPoint(positions[hi], in: geo.size)
                            let d = max(Self.radius(for: positions.count), 3) * 2 + 5
                            Circle().stroke(.primary, lineWidth: 1.5)
                                .frame(width: d, height: d).position(s).allowsHitTesting(false)
                            hoverChip(for: h, in: geo.size)
                        }

                        // Interactive (it hosts the layout menu), like the zoom cluster below it.
                        caption
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                            .padding(Design.gapLarge)

                        zoomControls
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                            .padding(Design.gapLarge)

                        legend
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                            .padding(Design.gapLarge)
                            .allowsHitTesting(false)
                    }
                }
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
                            // The spotlight needs the embedding-space neighbor graph, which only the
                            // UMAP layout computes. In PCA mode (no kNN) a click does nothing.
                            if model.folderKNNk > 0 {
                                let idx = nearestIndex(to: v.location, in: geo.size)
                                withAnimation(.easeOut(duration: 0.16)) {
                                    selectedIndex = (idx != nil && idx == selectedIndex) ? nil : idx
                                }
                                applyHighlight()
                            }
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
                    // onContinuousHover fires on every mouse sample. Publishing hoverLocation each
                    // time recomputes the whole body AND recomposites the Liquid Glass overlays
                    // (caption/legend/zoom/hover-chip share one GlassEffectContainer) at event rate -
                    // 60-120 glass passes/sec on a fast sweep over the live Metal cloud. Gate BOTH the
                    // chip-position publish and the O(N) nearest-point scan to ~3px of movement: the
                    // chip following in 3px steps is imperceptible, and body/glass passes drop to
                    // roughly sweep-distance/3px. (The hover chip only renders when `hovered != nil`,
                    // so a position update finer than the hit-test would not even be visible.)
                    if hypot(loc.x - lastHoverResolveLoc.x, loc.y - lastHoverResolveLoc.y) >= 3 {
                        lastHoverResolveLoc = loc
                        hoverLocation = loc
                        // Publish the hovered POINT only when the index actually changes. Sweeping
                        // within one dot re-published an identical ProjectionPoint every 3px, and
                        // because it carries two Strings that meant retain/release plus a chip
                        // rebuild per sample for no visible difference. The chip still follows the
                        // cursor via hoverLocation.
                        let idx = nearestIndex(to: loc, in: geo.size)
                        if idx != hoveredIndex {
                            hoveredIndex = idx
                            hovered = idx.map { model.folderProjection[$0] }
                            let path = hovered?.path
                            hoverDwellTask?.cancel()
                            hoverDwellTask = Task { @MainActor in
                                try? await Task.sleep(for: .milliseconds(140))
                                if !Task.isCancelled { hoverDwellPath = path }
                            }
                        }
                    }
                case .ended:
                    hovered = nil; hoveredIndex = nil
                    hoverDwellTask?.cancel(); hoverDwellTask = nil; hoverDwellPath = nil
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
            .accessibilityLabel("Folder map for \(folderName). \(model.folderProjection.count) file\(model.folderProjection.count == 1 ? "" : "s") laid out by similarity.")
            .accessibilityHint("Drag to pan, scroll or pinch to zoom, click a file to highlight its nearest neighbors.")
        }
        // A new folder resets the view and clears the cloud (folderProjection is empty until the fit
        // lands, so this blanks the old map under the "Mapping..." spinner instead of leaving it
        // stale). The GPU buffer then rebuilds when a new layout lands (keyed on the generation, since
        // two folders can share a file count) or the appearance flips.
        .onChange(of: model.selectedFolderForViz) { clearCursor(); resetView(); rebuildPoints() }
        // A refit of the SAME folder - a PCA/UMAP switch, a stats reconcile, a post-index refit -
        // bumps the generation without changing the folder, and every row index it produces refers
        // to a different file. selectedIndex/hovered are row indices into the projection they were
        // taken from, so they have to die with it; only the camera survives, because zoom and pan
        // are in layout space and a refit of the same folder should not throw the user's view away.
        .onChange(of: model.projectionGeneration) { clearCursor(); rebuildPoints() }
        .onChange(of: colorScheme) { rebuildPoints() }
        .onAppear { rebuildPoints() }
    }

    // MARK: - Zoom

    private func zoomBy(_ factor: CGFloat) {
        withAnimation(.easeOut(duration: 0.15)) { zoom = (zoom * factor).clamped(to: Self.zoomRange) }
    }
    /// Drop every piece of state that is an index into, or a lookup against, the current
    /// projection. Anything that survives a refit must be in layout space, not row space.
    private func clearCursor() {
        hovered = nil
        hoveredIndex = nil
        hoverDwellPath = nil
        selectedIndex = nil
        litNeighbors = []
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

    /// The top-left chip: what the map is showing, plus the map's own layout control. The two
    /// layout choices that used to live only in Settings > Performance (PCA/UMAP, and the
    /// no-overlap spread) hang off a pop-up at its trailing edge, so switching dense/sparse is one
    /// click on the map itself. One capsule, a hairline separating label from control - the Tahoe
    /// idiom for a map's own mode switch, where Maps puts its layer picker too.
    ///
    /// The live text stays OUTSIDE the Menu's label: a menu label is snapshotted, so the
    /// observation reads inside it are not tracked, and a chip built while the fit was still
    /// running kept showing "Desktop" with no file count for as long as the map was open.
    @ViewBuilder private var caption: some View {
        let count = model.folderProjection.count
        HStack(spacing: 6) {
            if model.folderProjectionFitting {
                ProgressView().controlSize(.mini)   // .mini renders crisp; scaleEffect rasterized fuzzy
                Text("Mapping \(folderName)\u{2026}")
            } else {
                Image(systemName: "circle.grid.cross").foregroundStyle(.secondary)
                Text(folderName).fontWeight(.medium)
                    .lineLimit(1).truncationMode(.middle)   // hug the name; .frame(maxWidth:) would expand it greedily
                if count > 0 {
                    // "N of M" when the folder was subsampled to fit the memory cap (the map is a
                    // representative sample, not every file). M is the pre-sample total for THIS folder.
                    let total = model.folderProjectionTotal
                    Text(total > count ? "\(count) of \(total) files"
                                       : "\(count) file\(count == 1 ? "" : "s")")
                        .foregroundStyle(.secondary).lineLimit(1)
                    Divider().frame(height: 13)
                    layoutMenu
                }
            }
        }
        .font(.callout)
        .padding(.horizontal, 10).padding(.vertical, 5)
        .glassChip(interactive: true)
    }

    /// Layout pop-up inside the caption chip. Icon-only and static by design (see `caption`).
    @ViewBuilder private var layoutMenu: some View {
        Menu {
            Picker("Layout", selection: Binding(get: { model.mapUsesUMAP }, set: { model.mapUsesUMAP = $0 })) {
                Text("Fast \u{00B7} PCA").tag(false)
                Text("Detailed \u{00B7} UMAP").tag(true)
            }
            .pickerStyle(.inline)
            Divider()
            Toggle("Grid layout", isOn: Binding(get: { model.mapNoOverlap }, set: { model.mapNoOverlap = $0 }))
        } label: {
            Image(systemName: "chevron.down")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 16, height: 16)
                .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)   // the chevron IS the label
        .fixedSize()
        .help("Map layout")
        .accessibilityLabel("Map layout")
    }

    @ViewBuilder private var legend: some View {
        if !presentKinds.isEmpty {
            HStack(spacing: 12) {
                ForEach(presentKinds, id: \.self) { kind in
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
        // Resolve the kind->HSB palette on the main actor: performAsCurrentDrawingAppearance is what
        // makes it adapt to light/dark, and it is only ~8 FileKinds, so it stays on @MainActor.
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
        let alpha = Self.dotAlpha
        // The per-point loop (positions + per-ext shading + bbox over up to 60k points, each with an
        // NSString ext alloc + FNV hash + HSB->RGB) is tens of ms - run it OFF the main actor and hop the
        // finished arrays back. Cancel any in-flight build so rapid folder switches / light-dark flips
        // (which can fire selectedFolderForViz AND projectionGeneration in one tick) don't stack loops.
        rebuildTask?.cancel()
        let noOverlap = model.mapNoOverlap
        rebuildTask = Task { @MainActor in
            let built = await Task.detached(priority: .userInitiated) { () -> (pos: [SIMD2<Float>], col: [SIMD4<Float>], bbox: SIMD4<Float>, kinds: [FileKind], gridCols: Int)? in
                var gridColsOut = 0
                if Task.isCancelled { return nil }
                var pos = [SIMD2<Float>](); pos.reserveCapacity(pts.count)
                var col = [SIMD4<Float>](); col.reserveCapacity(pts.count)
                var mn = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
                var mx = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
                let fallback = SIMD4<Float>(0.5, 0.5, 0.5, alpha)
                var kindsSeen = Set<String>()
                // Colour is a PURE function of (kind, extension), and a real corpus has a few
                // hundred distinct extensions - not one per file. The per-point form produced 20
                // distinct colours from 260k NSString bridges and 260k HSB conversions. Measured
                // (omni-verify vizbuildbench, 260k points): 75.4 ms -> 12.1 ms, byte-identical.
                //
                // Note WHERE the cost was: memoising the shade alone buys nothing (78.8 ms - the
                // String key costs what the maths did). It is `(path as NSString).pathExtension`
                // that dominates, bridging and allocating once per point, so the extension is read
                // straight out of the UTF-8 and the memo is keyed by an integer.
                var shadeMemo = [Int64: SIMD4<Float>](minimumCapacity: 512)
                for p in pts {
                    pos.append(p.position)
                    kindsSeen.insert(p.kind)
                    if p.position.x.isFinite, p.position.y.isFinite {
                        mn = pointwiseMin(mn, p.position); mx = pointwiseMax(mx, p.position)
                    }
                    guard let base = baseHSB[p.kind] else { col.append(fallback); continue }
                    // Extension without NSString: walk the UTF-8 back to the dot, stopping at the
                    // last path separator so a dotted directory name cannot be read as one.
                    let u = p.path.utf8
                    var extStart = u.endIndex
                    var seenDot = false
                    var idx = u.endIndex
                    while idx > u.startIndex {
                        let prev = u.index(before: idx)
                        let c = u[prev]
                        if c == UInt8(ascii: "/") { break }
                        if c == UInt8(ascii: ".") {
                            // A dot that STARTS the last component is not an extension separator -
                            // NSString gives ".gitignore" an empty pathExtension, and without this
                            // every dotfile would take a different shade than it used to.
                            if prev == u.startIndex || u[u.index(before: prev)] == UInt8(ascii: "/") { break }
                            extStart = idx; seenDot = true; break
                        }
                        idx = prev
                    }
                    var key: Int64 = 1469598103934665603 &* -1
                    for b in p.kind.utf8 { key = (key ^ Int64(b)) &* 16777619 }
                    key = (key ^ 0x5F) &* 16777619
                    if seenDot { for b in u[extStart...] { key = (key ^ Int64(b)) &* 16777619 } }
                    if let c = shadeMemo[key] { col.append(c); continue }
                    let ext = seenDot ? String(decoding: u[extStart...], as: UTF8.self) : ""
                    let c = FileKind.vizShadeRGBA(base: base, ext: ext, alpha: alpha)
                    shadeMemo[key] = c
                    col.append(c)
                }
                if pts.isEmpty { mn = .zero; mx = .zero }
                // Optional overlap removal, applied to the FINISHED layout (display-only, so the
                // cached projection is untouched and the toggle needs no refit). Grid sized ~15%
                // larger than the point count so clusters keep a little slack instead of being
                // packed solid. Measured 9.2 ms at 61.6k points - see `omni-verify gridbench`.
                if noOverlap, pts.count > 1 {
                    var flat = [Float](repeating: 0, count: pos.count * 2)
                    for (i, v) in pos.enumerated() { flat[2*i] = v.x; flat[2*i+1] = v.y }
                    let cells = Int(Double(pos.count) * 1.15)
                    let cols = max(1, Int(Double(cells).squareRoot().rounded(.up)))
                    let rows = max(1, (cells + cols - 1) / cols)
                    let g = ProjectionEngine.gridify(flat, count: pos.count, cols: cols, rows: rows)
                    gridColsOut = cols
                    mn = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
                    mx = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
                    for i in 0 ..< pos.count {
                        let v = SIMD2<Float>(g[2*i], g[2*i+1])
                        pos[i] = v
                        mn = pointwiseMin(mn, v); mx = pointwiseMax(mx, v)
                    }
                }
                let ext = pointwiseMax(mx - mn, SIMD2<Float>(1e-5, 1e-5))
                let present = FileKind.allCases.filter { kindsSeen.contains($0.rawValue) }
                return (pos, col, SIMD4<Float>((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, ext.x, ext.y), present, gridColsOut)
            }.value
            guard !Task.isCancelled, let built else { return }
            positions = built.pos
            baseColors = built.col
            bbox = built.bbox
            presentKinds = built.kinds
            gridCols = built.gridCols
            applyHighlight()
        }
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
    /// "Find similar" reuses `setFileQuery`, which runs a file-as-query search: that activates a query,
    /// so ContentView precedence swaps the map out for the live results (clearing it returns to the map).
    @ViewBuilder private func dotMenu(_ path: String) -> some View {
        Button("Open") { NSWorkspace.shared.openAsync(URL(fileURLWithPath: path)) }
            .keyboardShortcut("o", modifiers: .command)
        Button("Quick Look") { model.previewURL = URL(fileURLWithPath: path) }
            .keyboardShortcut("y", modifiers: .command)
        Divider()
        Button("Find similar") { model.setFileQuery(URL(fileURLWithPath: path), similar: true) }
            .keyboardShortcut("f", modifiers: [.command, .option])
        Divider()
        Button("Reveal in Finder") { NSWorkspace.shared.revealAsync(URL(fileURLWithPath: path)) }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        Divider()
        Button("Copy path") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(path, forType: .string)
        }
        .keyboardShortcut("c", modifiers: [.command, .option])
    }

    @ViewBuilder private func hoverChip(for h: ProjectionPoint, in size: CGSize) -> some View {
        VStack(spacing: 5) {
            // Dwell-gated: sweeping the cursor across a dense cloud crosses dozens of dots, and
            // handing Thumbnail a new path for each one queues a decode (and a QuickLook fallback
            // for non-images) per dot - work whose result is on screen for a few milliseconds. The
            // ring and the filename stay immediate; only the picture waits for the cursor to settle.
            Thumbnail(path: hoverDwellPath ?? h.path, side: 72)
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

    /// Dot radius for the current layout. In GRID mode it is derived from the cell pitch on screen,
    /// not from the point count: a grid whose dots are wider than its cells fuses into a solid
    /// sheet, and then the only thing the eye can pick out is the empty cells - the map reads as
    /// far fewer files than it is drawing. Zooming in grows the cells, so the dots grow with them.
    private func dotRadius(in size: CGSize) -> CGFloat {
        let base = Self.radius(for: positions.count)
        guard gridCols > 0, bbox.z > 0 else { return base }
        let rect = CGRect(origin: .zero, size: size).insetBy(dx: Self.inset, dy: Self.inset)
        let scale = min(rect.width / CGFloat(bbox.z), rect.height / CGFloat(bbox.w)) * effectiveZoom
        let pitch = CGFloat(bbox.z) / CGFloat(gridCols) * scale      // one cell, in points
        return max(0.5, min(base, pitch * 0.38))                      // ~24% gap between neighbours
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
    /// Scans the packed `positions` array in MODEL space rather than `model.folderProjection` in
    /// screen space. Two changes, both about what the loop touches per mouse-move sample:
    ///   - stride drops from a ~40-byte ProjectionPoint (SIMD2 plus two Strings) to an 8-byte
    ///     SIMD2<Float>, so the scan streams 5x fewer bytes - the ~8x-worse axis on a low-end Mac;
    ///   - the per-point call to `screenMap`'s escaping closure disappears. The transform is
    ///     uniform-scale, so inverting it ONCE and comparing in model space is equivalent: a
    ///     screen-space hit circle is a model-space circle of radius hitRadius/scale.
    /// Same index returned, so the ring, chip and kNN spotlight are unchanged.
    private func nearestIndex(to loc: CGPoint, in size: CGSize) -> Int? {
        let pts = positions
        // Row-aligned with folderProjection by construction; if a rebuild is mid-flight and they
        // disagree, resolve nothing this sample rather than index the shorter array.
        guard !pts.isEmpty, pts.count == model.folderProjection.count else { return nil }
        let cx = bbox.x, cy = bbox.y, extX = bbox.z, extY = bbox.w
        let rect = CGRect(origin: .zero, size: size).insetBy(dx: Self.inset, dy: Self.inset)
        let scale = min(rect.width / CGFloat(extX), rect.height / CGFloat(extY)) * effectiveZoom
        guard scale > 0, scale.isFinite else { return nil }
        let panNow = effectivePan
        let mx = Float((loc.x - rect.midX - panNow.width) / scale) + cx
        let my = Float((loc.y - rect.midY - panNow.height) / scale) + cy
        let r = Float(Self.hitRadius / scale)
        var best: Int?
        var bestD = r * r
        pts.withUnsafeBufferPointer { buf in
            for i in 0 ..< buf.count {
                let p = buf[i]
                // Same non-finite clamp screenMap applies, so a bad position stays at the center.
                let dx = (p.x.isFinite ? p.x : cx) - mx
                let dy = (p.y.isFinite ? p.y : cy) - my
                let d = dx * dx + dy * dy
                if d < bestD { bestD = d; best = i }
            }
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
