import SwiftUI
import AppKit
import QuickLook
import OmniKit

/// Kinds whose snippets are generated content tags (media) - the Generate Tags menu item
/// applies to these only; text snippets are real excerpts. (File-scope: ResultsList is
/// generic, which forbids static stored properties.)
private let taggableKinds: Set<String> = [
    FileKind.image.rawValue, FileKind.scan.rawValue, FileKind.video.rawValue
]

struct ResultsList<Footer: View>: View {
    @Environment(AppModel.self) private var model: AppModel
    let results: [SearchHit]
    @ViewBuilder var footer: Footer
    @State private var expanded: Set<String> = []
    @State private var passagesCache: [String: [ChunkHit]] = [:]
    @State private var gridWidth: CGFloat = 0
    /// Grid counterpart of the list's inline expansion: the path whose passages popover is open.
    @State private var passagesPopover: String?
    /// The result-set identity we have already scrolled to the top for: query AND filters, so a
    /// toolbar filter change - which replaces every row without touching the query - resets too. Scrolling on the query
    /// change alone fired before the new results arrived, so it scrolled to the OUTGOING first row
    /// and the view stayed put - visible when replaying a history item from a scrolled list.
    /// Scrolling when that identity is republished, which happens in the same block that assigns
    /// the rows, fires once per new result set and never on a same-query refresh from live indexing.
    @State private var scrolledForQuery: String?
    /// One name shared by the frame reporters, the drag gesture, and the rubber-band overlay. The list
    /// and gallery are never on screen together, so reusing the string is safe. The realized-item frames
    /// themselves live as @State INSIDE the marquee modifier - they refresh on every scroll tick
    /// (viewport-relative), so keeping them out of this view's body avoids re-evaluating the whole
    /// list/gallery on scroll just to track rectangles only a drag ever reads.
    private let marqueeSpace = "omni.results.viewport"

    private func toggle(_ path: String) {
        // Animated: the chevron rotation and the panel's insertion/removal track this mutation.
        withAnimation(.easeOut(duration: 0.18)) {
            if expanded.contains(path) { expanded.remove(path) }
            else { expanded.insert(path); fetchPassages(path) }
        }
    }

    /// Open/close a stack. Selection deliberately stays on the representative: opening a stack
    /// reveals copies, it does not change what the user has chosen.
    private func toggleStack(_ id: String) {
        withAnimation(.easeOut(duration: 0.18)) {
            if model.expandedStacks.contains(id) { model.expandedStacks.remove(id) }
            else { model.expandedStacks.insert(id) }
        }
    }

    private func fetchPassages(_ path: String) {
        guard passagesCache[path] == nil else { return }
        // Re-validate after the await, against the LIVE model. passages(for:) ranks against the
        // query vector that was current when it started, and the rank runs on the store's serial
        // queue behind indexing and search, so a background refresh landing inside that window is
        // ordinary. The cache is cleared by a different handler than the one that fills it, and
        // nothing orders the two: without this guard the task resumes and writes the OLD query's
        // ranking into the just-cleared dictionary, where the `passagesCache[path] == nil`
        // short-circuit above then serves it until the next results change.
        let token = model.resolvedQuery
        Task {
            let ranked = await model.passages(for: path)
            // renderedPaths: an opened COPY is a live row, and testing `results` alone threw its
            // ranking away on arrival - the panel then sat on "Ranking passages..." forever,
            // because the cache entry it waits for was never written.
            guard model.resolvedQuery == token, model.renderedPaths.contains(path) else { return }
            passagesCache[path] = ranked
        }
    }

    var body: some View {
        Group {
            switch model.viewMode {
            case .list: listView
            case .grid: gridView
            }
        }
        // The presenter sits on the view that owns the results, not on the whole pane. Hoisting it
        // to ContentView (so the folder map could share it) put it above the focused view instead
        // of on it, and from there a menu-invoked preview re-entered the panel's own open. The map
        // carries its own presenter for its dot menu; the two views are never mounted together, so
        // there is still exactly one in the hierarchy at any time.
        //
        // The setter DROPS a write that changes nothing. While it opens the panel,
        // _QuickLook_SwiftUI writes the current item back through this binding, and publishing that
        // no-op write re-fires the value action, re-entering -[QLPreviewPanel _openWithEffect:]
        // while the panel's item list is momentarily empty; the shim then indexes that empty list
        // and traps (EXC_BREAKPOINT, macOS 26.5.1).
        .quickLookPreview(Binding(get: { model.previewURL },
                                  set: { if $0 != model.previewURL { model.previewURL = $0 } }))
        // Space toggles Quick Look in both views regardless of focus, and is left alone while
        // editing text (the search field). The selection drives what is previewed.
        .background(QuickLookKeyMonitor(
            onSpace: { model.toggleQuickLook() },
            onPreviewArrow: { vertical, forward in
                // Only hijack arrows while Quick Look is open - then move the selection (which,
                // via its didSet, keeps previewURL on the selected row so the panel updates
                // live). In the gallery, up/down move by visual row, like Finder.
                guard model.previewURL != nil else { return false }
                let grid = model.viewMode == .grid
                let step = (vertical && grid) ? gridColumns : 1
                model.moveSelection(rowDelta: forward ? step : -step, gridColumns: grid ? gridColumns : nil)
                return true
            },
            isPreviewOpen: { model.previewURL != nil }))
        .onKeyPress(.return) { if model.hasSelection { model.openSelected(); return .handled }; return .ignored }
        // Passages are ranked against the CURRENT query vector, so the QUERY is what invalidates
        // them - not the path list, which is a different publish and decoupled in both directions.
        // It changes constantly under an unchanged query (every background index pass, a trashed
        // row, "show N more matches"), and each one wiped a still-valid cache and closed an open
        // panel; and it can come back byte-identical across a genuinely NEW query (a small or
        // filtered index whose hits all fit under topK, in a content-derived sort order), which
        // left the cache holding a ranking computed against the previous query's vector.
        .onChange(of: model.resolvedQuery) { _, _ in
            expanded = []
            passagesCache = [:]
            passagesPopover = nil
        }
        // A row that vanished under the SAME query takes its own state with it, and nothing else's:
        // the remaining panels are still ranked against the query that is still on screen.
        .onChange(of: results.map(\.path)) { _, _ in
            // renderedPaths, not the representative list: a copy inside an OPEN stack is on screen
            // and its passages panel must survive a re-rank exactly like any other row's.
            let live = model.renderedPaths
            expanded.formIntersection(live)
            if let p = passagesPopover, !live.contains(p) { passagesPopover = nil }
        }
    }

    // MARK: - List

    // A plain scroll of rows (not a `List`), so selection, click, double-click, right-click, and
    // arrow-key navigation behave EXACTLY like the gallery. `List(selection:)` showed the system
    // grey highlight (only accent while focused) and would not take arrow keys once the rows had
    // their own tap gestures - this drives all of it explicitly instead.
    private var listView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(model.groups) { group in
                        let hit = group.representative
                        VStack(spacing: 0) {
                            ResultRow(hit: hit,
                                      selected: model.selectedPaths.contains(hit.path),
                                      // Only multi-chunk files (long docs, multi-page PDFs) have a
                                      // per-chunk breakdown; a single-embedding file gets no chevron.
                                      expandable: hit.chunkCount > 1,
                                      expanded: expanded.contains(hit.path),
                                      onToggle: { toggle(hit.path) },
                                      // Stack: the row stands for `count` files. The badge is its own
                                      // control so it never fights the passages chevron next to it.
                                      stack: group.isStack ? (group.count, group.reason) : nil,
                                      stackOpen: model.expandedStacks.contains(group.id),
                                      onToggleStack: { toggleStack(group.id) })
                                // Result rows are intentionally NOT draggable: an in-app row drag was
                                // easy to misclick onto the search drop target. Drag-to-search is for
                                // files coming from OUTSIDE the app (Finder); use Find similar / Reveal
                                // in Finder for a result.
                                .contentShape(Rectangle())
                                .onTapGesture { handleTap(hit.path) }
                                .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                                .contextMenu { menu(hit) }
                                .reportResultFrame(hit.path, in: marqueeSpace)
                            // chunkCount guard: if a reindex turned the file single-chunk while its
                            // path sat in `expanded` (same result set, so the reset below does not
                            // fire), the chevron is gone - don't strand an open expansion either.
                            if expanded.contains(hit.path), hit.chunkCount > 1 {
                                PassagesView(passages: passagesCache[hit.path],
                                             fileName: URL(fileURLWithPath: hit.path).lastPathComponent,
                                             path: hit.path, kind: hit.kind)
                                    .padding(10)
                                    // A flat elevated fill, not vibrancy: blur belongs on sidebars and
                                    // popovers; this excerpt card sits inside the opaque scrolling
                                    // content where text must stay crisp and high-contrast.
                                    .background(Color(.controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                                    .padding(.leading, 52)
                                    .padding(.trailing, 12)
                                    .padding(.bottom, 8)
                                    .transition(.opacity)
                            }
                            // The stack, opened: its other copies as ordinary rows, indented under
                            // the one that represents them. They are full rows on purpose - every
                            // per-file action (open, Quick Look, reveal, trash) works on a copy
                            // exactly as it does on any other result.
                            if group.isStack, model.expandedStacks.contains(group.id) {
                                ForEach(group.members.dropFirst(), id: \.path) { member in
                                    // Level 2. A copy is a whole result, so it keeps its OWN
                                    // per-chunk disclosure: stack open -> copy row -> that copy's
                                    // matching passages. The chevron, the cache and the fetch are
                                    // the same ones the representative uses (all keyed by path),
                                    // so nothing here is a parallel implementation.
                                    VStack(spacing: 0) {
                                        ResultRow(hit: member,
                                                  selected: model.selectedPaths.contains(member.path),
                                                  expandable: member.chunkCount > 1,
                                                  expanded: expanded.contains(member.path),
                                                  onToggle: { toggle(member.path) })
                                            .contentShape(Rectangle())
                                            .onTapGesture { handleTap(member.path) }
                                            .simultaneousGesture(TapGesture(count: 2).onEnded { open(member.path) })
                                            .contextMenu { menu(member) }
                                            .reportResultFrame(member.path, in: marqueeSpace)
                                        if expanded.contains(member.path), member.chunkCount > 1 {
                                            PassagesView(passages: passagesCache[member.path],
                                                         fileName: URL(fileURLWithPath: member.path).lastPathComponent,
                                                         path: member.path, kind: member.kind)
                                                .padding(10)
                                                .background(Color(.controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                                                // Indented one level deeper than the representative's
                                                // panel, so the nesting is legible at a glance.
                                                .padding(.leading, 52)
                                                .padding(.trailing, 12)
                                                .padding(.bottom, 8)
                                                .transition(.opacity)
                                        }
                                    }
                                    .padding(.leading, 28)
                                    .transition(.opacity)
                                }
                            }
                        }
                        .id(hit.path)
                    }
                    footer
                }
                .padding(.horizontal, Design.gapLarge)
                .padding(.vertical, 8)
            }
            // Arrow keys move the selection up/down (Return/Space handled on the body). Same
            // focusable + onMoveCommand wiring the gallery uses. Right/left disclose/collapse the
            // selected row's passages - the Finder list-view convention for expandable rows.
            .focusable()
            .focusEffectDisabled()
            .onMoveCommand { direction in
                switch direction {
                case .up: model.moveSelection(rowDelta: -1)
                case .down: model.moveSelection(rowDelta: 1)
                case .right:
                    // renderedHit, not `results`: right-arrow must disclose the passages of an
                    // opened COPY too, and that row is not in the representative list.
                    if let sel = model.selection, !expanded.contains(sel),
                       (model.renderedHit(sel)?.chunkCount ?? 0) > 1 { toggle(sel) }
                case .left:
                    if let sel = model.selection, expanded.contains(sel) { toggle(sel) }
                @unknown default: break
                }
            }
            // Keep the selected row on screen as it moves (so arrowing past the fold scrolls).
            .onChange(of: model.selection) { _, sel in
                guard let sel else { return }
                // anchor nil = minimal scroll to visible (Finder/Mail behavior); centering on every
                // arrow press made keyboard navigation jumpy.
                withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(sel, anchor: nil) }
            }
            // A NEW query reads top-down: jump back to the best hit once its results exist.
            // Keyed on the result-set identity rather than on the path list, because the paths are
            // not query identity: on a corpus where every hit fits under topK, in a content-derived
            // sort order, two different queries can return a byte-identical array, and then this
            // handler never ran at all - no jump to the best hit, and the gate was left holding the
            // previous query while the model had moved on, so the next unrelated same-query refresh
            // scrolled to the top out of nowhere. resultsToken is published inside the same
            // synchronous block that assigns the rows, so it can never fire before they exist.
            // initial: true seeds the gate at mount. ResultsList is only rendered when results are
            // non-empty, so the view appears at the moment the FIRST result set lands and that
            // landing never fires a plain onChange - the gate would stay nil for the whole first
            // query, and the next same-query row change (a background reindex, "show N more", a
            // trashed row) would yank a scrolled list back to the top.
            .onChange(of: model.resultsToken, initial: true) { _, _ in
                guard scrolledForQuery != model.resultsToken else { return }
                scrolledForQuery = model.resultsToken
                if let first = results.first?.path { proxy.scrollTo(first, anchor: .top) }
            }
            .marqueeSelect(space: marqueeSpace)
        }
    }

    // MARK: - Gallery

    private let gridMin: CGFloat = 172
    private var gridColumns: Int {
        let usable = gridWidth - Design.gapLarge * 2
        return max(1, Int((usable + Design.gapLarge) / (gridMin + Design.gapLarge)))
    }

    private var gridView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: gridMin, maximum: 220), spacing: Design.gapLarge)], spacing: Design.gapLarge) {
                    ForEach(model.groups) { group in
                        let hit = group.representative
                        ResultGridItem(hit: hit, selected: model.selectedPaths.contains(hit.path),
                                       stack: group.isStack ? (group.count, group.reason) : nil,
                                       stackOpen: model.expandedStacks.contains(group.id),
                                       onToggleStack: { toggleStack(group.id) })
                            // Make the whole cell tappable, not just the opaque thumbnail/label - without
                            // this, clicking the transparent padding around a small item did nothing.
                            // (The list row already has this; the grid relied on .draggable's hit area,
                            // which was removed.)
                            .contentShape(Rectangle())
                            .onTapGesture { handleTap(hit.path) }
                            .simultaneousGesture(TapGesture(count: 2).onEnded { open(hit.path) })
                            .contextMenu { menu(hit) }
                            .reportResultFrame(hit.path, in: marqueeSpace)
                            // The grid's counterpart of the list's inline expansion: a popover
                            // anchored to the cell (the Photos/Finder info pattern - cells stay
                            // uniform, the breakdown floats with system vibrancy). Passages are
                            // fetched BEFORE presenting (see the menu action): swapping a loading
                            // placeholder for the loaded view would animate an NSPopover window
                            // resize mid-presentation, which crashes AppKit (NSMoveHelper SEGV).
                            .popover(isPresented: Binding(
                                get: { passagesPopover == hit.path },
                                set: { if !$0 { passagesPopover = nil } }
                            ), arrowEdge: .bottom) {
                                ScrollView {
                                    PassagesView(passages: passagesCache[hit.path],
                                                 fileName: URL(fileURLWithPath: hit.path).lastPathComponent,
                                                 path: hit.path, kind: hit.kind)
                                        .padding(12)
                                }
                                .frame(width: 380)
                                .frame(maxHeight: 320)
                            }
                            .id(hit.path)
                        // An opened stack spills its copies into the grid as ordinary cells right
                        // after the one that represents them - the gallery stays a uniform grid
                        // (no cell grows, nothing reflows around a panel), and every copy is a
                        // full-sized, selectable, openable result.
                        if group.isStack, model.expandedStacks.contains(group.id) {
                            ForEach(group.members.dropFirst(), id: \.path) { member in
                                ResultGridItem(hit: member, selected: model.selectedPaths.contains(member.path))
                                    .contentShape(Rectangle())
                                    .onTapGesture { handleTap(member.path) }
                                    .simultaneousGesture(TapGesture(count: 2).onEnded { open(member.path) })
                                    .contextMenu { menu(member) }
                                    .reportResultFrame(member.path, in: marqueeSpace)
                                    // Level 2 in the gallery: a copy opens its own passages popover,
                                    // anchored to its own cell, exactly like the representative.
                                    .popover(isPresented: Binding(
                                        get: { passagesPopover == member.path },
                                        set: { if !$0 { passagesPopover = nil } }
                                    ), arrowEdge: .bottom) {
                                        ScrollView {
                                            PassagesView(passages: passagesCache[member.path],
                                                         fileName: URL(fileURLWithPath: member.path).lastPathComponent,
                                                         path: member.path, kind: member.kind)
                                                .padding(12)
                                        }
                                        .frame(width: 380)
                                        .frame(maxHeight: 320)
                                    }
                                    .id(member.path)
                                    .transition(.opacity)
                            }
                        }
                    }
                }
                .padding(Design.gapLarge)
                footer
            }
            .background(GeometryReader { g in
                Color.clear
                    .onAppear { gridWidth = g.size.width }
                    .onChange(of: g.size.width) { _, w in gridWidth = w }
            })
            // Make the gallery keyboard-navigable like the list: arrow keys move the selection by
            // column/row, and Return/Space (handled on the body) then open/preview it.
            .focusable()
            .focusEffectDisabled()
            .onMoveCommand { direction in
                switch direction {
                case .up: model.moveSelection(rowDelta: -gridColumns, gridColumns: gridColumns)
                case .down: model.moveSelection(rowDelta: gridColumns, gridColumns: gridColumns)
                case .left: model.moveSelection(rowDelta: -1, gridColumns: gridColumns)
                case .right: model.moveSelection(rowDelta: 1, gridColumns: gridColumns)
                @unknown default: break
                }
            }
            // Keep the selected cell on screen as arrow keys move it (matches the list view).
            .onChange(of: model.selection) { _, sel in
                guard let sel else { return }
                // anchor nil = minimal scroll to visible (Finder/Mail behavior); centering on every
                // arrow press made keyboard navigation jumpy.
                withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(sel, anchor: nil) }
            }
            // A NEW query reads top-down: jump back to the best hit once its results exist.
            // Keyed on the result-set identity rather than on the path list, because the paths are
            // not query identity: on a corpus where every hit fits under topK, in a content-derived
            // sort order, two different queries can return a byte-identical array, and then this
            // handler never ran at all - no jump to the best hit, and the gate was left holding the
            // previous query while the model had moved on, so the next unrelated same-query refresh
            // scrolled to the top out of nowhere. resultsToken is published inside the same
            // synchronous block that assigns the rows, so it can never fire before they exist.
            // initial: true seeds the gate at mount. ResultsList is only rendered when results are
            // non-empty, so the view appears at the moment the FIRST result set lands and that
            // landing never fires a plain onChange - the gate would stay nil for the whole first
            // query, and the next same-query row change (a background reindex, "show N more", a
            // trashed row) would yank a scrolled list back to the top.
            .onChange(of: model.resultsToken, initial: true) { _, _ in
                guard scrolledForQuery != model.resultsToken else { return }
                scrolledForQuery = model.resultsToken
                if let first = results.first?.path { proxy.scrollTo(first, anchor: .top) }
            }
            .marqueeSelect(space: marqueeSpace)
        }
    }

    @ViewBuilder private func menu(_ hit: SearchHit) -> some View {
        // NOTE: no side effects in this builder - macOS evaluates context-menu builders eagerly
        // during row rendering, so a "select on menu open" hack here thrashed the selection on
        // every results render. Instead each ACTION selects the row it acts on. (Shortcut chords are
        // display-only hints: a chord declared inside a context menu never fires on macOS, so the
        // real key handling lives on the Edit/File menus.)
        let path = hit.path
        let count = model.selectedPaths.count
        let selectionHasMedia = model.rawResults.contains {
            model.selectedPaths.contains($0.path) && taggableKinds.contains($0.kind)
        }
        // Right-clicking a row that is part of a multi-selection acts on the WHOLE selection (Finder
        // behavior); only the actions that extend to many are shown - the single-item ones (Quick
        // Look, passages, Find similar, Ignore folder) are hidden so the menu stays coherent.
        // Tahoe (macOS 26) context menus carry a leading SF Symbol per item; we icon EVERY item so the
        // menu reads consistently (a half-iconed menu looks broken). On macOS 14/15 the system renders
        // these Labels text-only, so this degrades cleanly. Symbols track Finder's conventions.
        if count > 1, model.selectedPaths.contains(path) {
            Button { model.openSelected() } label: { Label("Open", systemImage: "arrow.up.forward.app") }
                .keyboardShortcut("o", modifiers: .command)
            Button { model.revealSelected() } label: { Label("Reveal in Finder", systemImage: "folder") }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button { model.copySelectedPaths() } label: { Label("Copy \(count) paths", systemImage: "doc.on.doc") }
                .keyboardShortcut("c", modifiers: .command)
            // Native macOS share picker (AirDrop, Mail, Messages, ...) over the whole selection - the
            // same system sheet Finder's Share opens, anchored to the menu. (ShareLink, no deprecated API.)
            ShareLink(items: model.selectedURLsOrdered) { Label("Share\u{2026}", systemImage: "square.and.arrow.up") }
            // (Re)generate content tags for the selected media - explicit request, HQ quality.
            // Shown only when the selection contains taggable media and the tagger is ready.
            if model.canGenerateTags, selectionHasMedia {
                Button { model.requestTags(Array(model.selectedPaths)) } label: { Label("Generate Tags", systemImage: "tag") }
            }
            Divider()
            Button(role: .destructive) { model.moveSelectedToTrash() } label: { Label("Move \(count) items to Trash", systemImage: "trash") }
                .keyboardShortcut(.delete, modifiers: .command)
            Divider()
            Button { model.selectAllResults() } label: { Label("Select all", systemImage: "checkmark.circle") }
                .keyboardShortcut("a", modifiers: .command)
        } else {
            // Stack actions, above the per-file ones. Everything else in this menu acts on the
            // REPRESENTATIVE only - a collapsed stack is one file as far as opening, previewing and
            // trashing go - so the two stack-wide actions are stated explicitly with their count.
            // Deleting copies you cannot see is exactly the mistake this wording exists to prevent.
            if let group = model.groups.first(where: { $0.id == path }), group.isStack {
                Button { toggleStack(group.id) } label: {
                    Label(model.expandedStacks.contains(group.id) ? "Hide \(group.count - 1) copies" : "Show \(group.count - 1) copies",
                          systemImage: "square.stack.3d.down.right")
                }
                Button { model.selectPaths(group.paths) } label: {
                    Label("Select all \(group.count)", systemImage: "checkmark.circle")
                }
                Divider()
                Button(role: .destructive) { model.moveToTrash(group.paths) } label: {
                    Label("Move all \(group.count) copies to Trash", systemImage: "trash")
                }
                Divider()
            }
            Button { model.selectSingle(path); open(path) } label: { Label("Open", systemImage: "arrow.up.forward.app") }
                .keyboardShortcut("o", modifiers: .command)
            Button { model.selectSingle(path); model.showPreview(URL(fileURLWithPath: path)) } label: { Label("Quick Look", systemImage: "eye") }
                .keyboardShortcut("y", modifiers: .command)
            // Per-chunk breakdown (pages of a PDF, passages of a long doc) - only for files that
            // actually have several chunks. The list expands inline; the grid opens a popover.
            if hit.chunkCount > 1 {
                switch model.viewMode {
                case .list:
                    Button { toggle(path) } label: {
                        Label(expanded.contains(path) ? "Hide matching passages" : "Show matching passages", systemImage: "text.alignleft")
                    }
                case .grid:
                    Button {
                        // Load first, present after: the popover must mount at its final size
                        // (see the crash note at the .popover site). Both writes are re-validated
                        // against the LIVE model after the rank's await, for the same reason
                        // fetchPassages is: the results can be replaced while the store's serial
                        // queue works through it, and presenting then either showed the previous
                        // query's passages or anchored the popover to a cell that no longer exists,
                        // which presents nothing and leaves the state pointing at a dead path.
                        Task {
                            let token = model.resolvedQuery
                            let ranked = passagesCache[path] == nil ? await model.passages(for: path) : nil
                            guard model.resolvedQuery == token,
                                  model.renderedPaths.contains(path) else { return }
                            if let ranked { passagesCache[path] = ranked }
                            passagesPopover = path
                        }
                    } label: { Label("Show matching passages", systemImage: "text.alignleft") }
                }
            }
            Divider()
            // Use this file itself as the query - doc-vs-doc "more like this" across all modalities.
            Button { model.setFileQuery(URL(fileURLWithPath: path), similar: true) } label: { Label("Find similar", systemImage: "sparkle.magnifyingglass") }
                .keyboardShortcut("f", modifiers: [.command, .option])
            // (Re)generate this file's content tags - explicit request, HQ quality. Media only:
            // a text file's snippet is a real excerpt, tags would be a downgrade.
            if model.canGenerateTags, taggableKinds.contains(hit.kind) {
                Button { model.selectSingle(path); model.requestTags([path]) } label: { Label("Generate Tags", systemImage: "tag") }
            }
            Button { model.selectSingle(path); reveal(path) } label: { Label("Reveal in Finder", systemImage: "folder") }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(path, forType: .string)
            } label: { Label("Copy path", systemImage: "doc.on.doc") }
            .keyboardShortcut("c", modifiers: .command)
            // Native macOS share picker for this file - the same system sheet Finder's Share opens.
            ShareLink(item: URL(fileURLWithPath: path)) { Label("Share\u{2026}", systemImage: "square.and.arrow.up") }
            Divider()
            Button(role: .destructive) { model.moveToTrash([path]) } label: { Label("Move to Trash", systemImage: "trash") }
                .keyboardShortcut(.delete, modifiers: .command)
            Button { model.selectAllResults() } label: { Label("Select all", systemImage: "checkmark.circle") }
                .keyboardShortcut("a", modifiers: .command)
            // Exclude this result's folder from indexing - the "stop showing me this build/cache
            // noise" action. Routes through the same apply path as the Settings ignore editor (backed
            // up, pruned, persisted, visible there). Hidden when the folder is an indexed root:
            // removing a whole root belongs to the sidebar, with its confirmation.
            if model.canIgnoreEnclosingFolder(ofPath: path) {
                Divider()
                Button { model.ignoreEnclosingFolder(ofPath: path) } label: {
                    Label("Ignore folder \u{201C}\((path as NSString).deletingLastPathComponent.components(separatedBy: "/").last ?? "")\u{201D}", systemImage: "eye.slash")
                }
            }
        }
    }

    private func open(_ path: String) { NSWorkspace.shared.openAsync(URL(fileURLWithPath: path)) }
    private func reveal(_ path: String) { NSWorkspace.shared.revealAsync(URL(fileURLWithPath: path)) }

    /// Click selection with Finder modifiers: Cmd toggles a row, Shift extends the range from the
    /// anchor, plain click replaces the selection.
    private func handleTap(_ path: String) {
        let m = NSEvent.modifierFlags
        if m.contains(.command) { model.toggleSelection(path) }
        else if m.contains(.shift) { model.extendSelection(to: path) }
        else { model.selectSingle(path) }
        // A click is a deliberate navigation - record it as a back/forward stop. Arrow-key moves
        // (moveSelection) deliberately don't, so browsing the list doesn't flood the trail.
        model.captureNavStop()
    }
}

struct ResultRow: View {
    let hit: SearchHit
    var selected: Bool = false
    @Environment(\.controlActiveState) private var controlActive
    var expandable: Bool = false
    var expanded: Bool = false
    var onToggle: (() -> Void)? = nil
    /// Non-nil when this row stands for a stack of duplicates: (member count, why they grouped).
    var stack: (count: Int, reason: ResultGroup.Reason)? = nil
    var stackOpen: Bool = false
    var onToggleStack: (() -> Void)? = nil
    private var url: URL { URL(fileURLWithPath: hit.path) }
    /// Emphasized = selected AND the window is key: the state that earns the solid accent fill and
    /// white text. A non-key window falls back to the unemphasized grey, like every native list.
    private var emphasized: Bool { selected && controlActive == .key }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            StackedThumbnail(path: hit.path, side: 40, corner: 6, depth: stack.map { min(2, $0.count - 1) } ?? 0)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(url.lastPathComponent).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                    if let stack { stackBadge(stack) }
                }
                if !hit.snippet.isEmpty, hit.snippet != url.lastPathComponent {
                    Text(hit.snippet).font(.body).foregroundStyle(.secondary).lineLimit(1)
                }
                HStack(spacing: 5) {
                    KindGlyph(kind: hit.kind)
                    MediaInfoLabel(path: hit.path, kind: hit.kind, width: hit.width, height: hit.height, duration: hit.duration, separator: true)
                    if !hit.locator.isEmpty {
                        // Where in the file the best-matching chunk sits ("Page 3", "Line 1240").
                        Text(hit.locator)
                        Text("\u{00B7}")
                    }
                    Text(prettyDir(url)).lineLimit(1).truncationMode(.middle)
                    if hit.modified > 0 {
                        Text("·")
                        Text(Date(timeIntervalSince1970: hit.modified), format: .relative(presentation: .named))
                    }
                }
                .font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer()
            Text(scoreText(hit.score)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            // The disclosure occupies a slot of FIXED width that exists on every row, present or
            // not. Making the whole control conditional shifted the score column by 24pt between
            // neighbouring rows - the eye reads a results list down its right edge, and a number
            // that jumps left and right by whether that file happens to have several chunks is
            // noise the user cannot act on. Same reason Finder reserves its disclosure column.
            Group {
                if expandable {
                    Button { onToggle?() } label: {
                        Image(systemName: "chevron.right")
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                            .foregroundStyle(.tertiary)
                            // A bare glyph is a ~16px target; give it a comfortable hit area.
                            .frame(width: 24, height: 24)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Show matching passages (right arrow)")
                    .accessibilityLabel(expanded ? "Hide matching passages" : "Show matching passages")
                } else {
                    // An explicit filler, not an empty branch: a Group whose condition is false
                    // yields no view at all and .frame() then has nothing to size, so the slot
                    // collapsed and the score still jumped. Color.clear occupies the slot.
                    Color.clear
                }
            }
            .frame(width: 24, height: 24)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        // Native list selection (Finder, Mail): the SOLID emphasized accent fill with white text when
        // the window is key, the system's unemphasized grey (with normal label text) when it is not -
        // the cue for where keyboard input lands. White text comes from driving the whole row's
        // foreground to the selection text color, so the secondary/tertiary metadata derive their
        // translucent-white tints from it automatically. Radius concentric with the 6pt thumbnail
        // corners across the 6pt padding.
        .foregroundStyle(emphasized ? AnyShapeStyle(Color(nsColor: .alternateSelectedControlTextColor)) : AnyShapeStyle(.primary))
        .background(
            selected ? (controlActive == .key
                ? Color(nsColor: .selectedContentBackgroundColor)
                : Color(nsColor: .unemphasizedSelectedContentBackgroundColor)) : .clear,
            in: RoundedRectangle(cornerRadius: Design.cornerSmall + 6, style: .continuous)
        )
    }
}

/// The matching passages (chunks) of a file, each shown as an excerpt with a top/bottom
/// alpha fade to signal there is more text before and after it in the file. A chunk's
/// locator ("Page 3", "Line 1240") leads the excerpt; for scanned-PDF pages the snippet is
/// just the file name, so the locator + score carry the row alone.
/// Chrome-free (rows only): the list wraps it in an inline card, the grid in a popover.
/// `passages == nil` means STILL LOADING (the rank runs async on the store queue) - render a quiet
/// placeholder, never "No passages": conflating the two flashed the empty state for the fetch's
/// duration before the real rows swapped in.
struct PassagesView: View {
    let passages: [ChunkHit]?
    var fileName: String = ""
    var path: String = ""
    var kind: String = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(passages ?? []) { p in
                HStack(alignment: .top, spacing: 8) {
                    RoundedRectangle(cornerRadius: 1.5).fill(.quaternary).frame(width: 3)
                    // Visual match evidence where the file has it: the video frame at the
                    // segment's stored timestamp, the rendered page for a PDF chunk.
                    if !path.isEmpty, ChunkPreview.expects(path: path, kind: kind, locator: p.locator) {
                        ChunkThumb(path: path, kind: kind, locator: p.locator)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        if !p.locator.isEmpty {
                            Text(p.locator).font(.caption2.weight(.medium)).foregroundStyle(.tertiary)
                        }
                        if !p.snippet.isEmpty, p.snippet != fileName {
                            Text(p.snippet)
                                .font(.body).foregroundStyle(.secondary)
                                .lineLimit(3)
                                // The fade signals there is more text before/after the excerpt -
                                // applied to the excerpt only so the locator stays crisp.
                                .mask(LinearGradient(stops: [
                                    .init(color: .clear, location: 0),
                                    .init(color: .black, location: 0.22),
                                    .init(color: .black, location: 0.78),
                                    .init(color: .clear, location: 1),
                                ], startPoint: .top, endPoint: .bottom))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    Text(scoreText(p.score)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
            }
            if let passages, passages.isEmpty {
                Text("No passages").font(.caption).foregroundStyle(.tertiary)
            } else if passages == nil {
                // Loading: keep the card's footprint stable with a quiet placeholder row.
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Ranking passages\u{2026}").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
    }
}

struct ResultGridItem: View {
    let hit: SearchHit
    let selected: Bool
    /// Non-nil when this cell stands for a stack of duplicates.
    var stack: (count: Int, reason: ResultGroup.Reason)? = nil
    var stackOpen: Bool = false
    var onToggleStack: (() -> Void)? = nil
    @Environment(\.controlActiveState) private var controlActive
    private var url: URL { URL(fileURLWithPath: hit.path) }

    var body: some View {
        VStack(spacing: 6) {
            StackedThumbnail(path: hit.path, side: 128, corner: Design.corner,
                             depth: stack.map { min(2, $0.count - 1) } ?? 0)
                .overlay {
                    // Glass chips over imagery (the one legitimate in-content use of vibrancy):
                    // legible over bright and dark thumbnails, appearance-adaptive. The pair shares
                    // one GlassEffectContainer per cell, so a visible grid renders one glass pass
                    // per cell instead of two - and on a narrow cell where a long locator nears the
                    // score, the effects blend instead of seaming.
                    GlassGroup(spacing: 10) {
                        ZStack {
                            Text(scoreText(hit.score)).font(.caption2.monospacedDigit()).foregroundStyle(.primary)
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .glassChip().padding(5)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                            // Match position inside the file (page/line), mirroring the score chip.
                            if !hit.locator.isEmpty {
                                Text(hit.locator).font(.caption2.monospacedDigit()).foregroundStyle(.primary)
                                    .padding(.horizontal, 5).padding(.vertical, 2)
                                    .glassChip().padding(5)
                                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                            }
                            // Stack count, bottom-trailing so it never collides with the locator or
                            // the score. Pressing it opens the stack, exactly like the list badge.
                            if let stack {
                                Button { onToggleStack?() } label: {
                                    HStack(spacing: 3) {
                                        Image(systemName: stackOpen ? "chevron.up" : "square.stack.3d.down.right.fill")
                                        Text("\(stack.count)").monospacedDigit()
                                    }
                                    .font(.caption2)
                                    .foregroundStyle(.primary)
                                    .padding(.horizontal, 5).padding(.vertical, 2)
                                    .glassChip(interactive: true)
                                    .contentShape(Capsule())
                                }
                                .buttonStyle(.plain)
                                .padding(5)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                                .help(stack.reason == .exact
                                      ? "\(stack.count) byte-identical copies"
                                      : "\(stack.count) near-identical files")
                            }
                        }
                    }
                }
            // STRICT GRID ALIGNMENT (Finder convention): every cell reserves the same label
            // footprint - exactly two caption lines for the name and one caption2 line for the info -
            // whether or not the content fills it. Without this, 1-line names and caption-less text
            // files made cells shorter, and LazyVGrid centers short cells in the row slot, so
            // thumbnails floated at different heights across a row. The hidden template sets the slot
            // height (layout-robust, no hardcoded points) while the visible name keeps hugging its
            // selection capsule - reservesSpace on the Text itself would stretch the capsule over the
            // empty reserved line. Top-aligned, like Finder: the name starts right under the icon.
            ZStack(alignment: .top) {
                Text(verbatim: "X\nX").font(.caption).padding(.vertical, 1).hidden()
                Text(url.lastPathComponent).font(.caption).lineLimit(2)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(selected && controlActive == .key ? .white : .primary)
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(selected ? (controlActive == .key
                        ? Color(nsColor: .selectedContentBackgroundColor)
                        : Color(nsColor: .unemphasizedSelectedContentBackgroundColor)) : .clear, in: Capsule())
                    .frame(maxWidth: 150)
            }
            // The snippet - for media, the content tags ("cat, couch, crib"); for text, the matching
            // excerpt. The list row shows it, so the grid does too; without it a gallery of photos
            // was the one view where tagging produced nothing visible. Same reserve-the-line trick as
            // the name above, so cells without a snippet keep the row's thumbnails aligned.
            ZStack {
                Text(verbatim: "0").font(.caption2).hidden()
                if !hit.snippet.isEmpty, hit.snippet != url.lastPathComponent {
                    Text(hit.snippet).font(.caption2).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.tail)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 150)
                }
            }
            ZStack {
                Text(verbatim: "0").font(.caption2).hidden()
                MediaInfoLabel(path: hit.path, kind: hit.kind, width: hit.width, height: hit.height, duration: hit.duration, separator: false)
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        // Native selection: a translucent accent fill behind the whole cell (thumbnail + label),
        // the way Finder and Photos indicate selection - not a hard ring hugging the image.
        // Unemphasized grey when the window is not key; radius concentric with the 8pt thumbnail
        // corners across the 8pt padding.
        .background(
            selected ? (controlActive == .key
                ? Color(nsColor: .selectedContentBackgroundColor).opacity(0.18)
                : Color(nsColor: .unemphasizedSelectedContentBackgroundColor).opacity(0.8)) : .clear,
            in: RoundedRectangle(cornerRadius: Design.corner + 8, style: .continuous)
        )
    }
}

/// The count control on a stacked row: "3 copies" / "3 similar", clickable to open the stack.
/// A button rather than a label - the count is the ONLY affordance that says other files are here,
/// so it has to be the thing you press. Kept out of the passages chevron's way on the trailing edge.
extension ResultRow {
    @ViewBuilder func stackBadge(_ stack: (count: Int, reason: ResultGroup.Reason)) -> some View {
        Button { onToggleStack?() } label: {
            HStack(spacing: 3) {
                Image(systemName: stackOpen ? "chevron.down" : "square.stack.3d.down.right.fill")
                    .font(.caption2)
                Text(stack.reason == .exact ? "\(stack.count) copies" : "\(stack.count) similar")
                    .font(.caption).monospacedDigit()
            }
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(.quaternary, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(stack.reason == .exact
              ? "\(stack.count) byte-identical copies - click to show them"
              : "\(stack.count) near-identical files - click to show them")
    }
}

/// A thumbnail that reads as a pile when it stands for more than one file: `depth` cards peeking
/// out behind the real one, offset down-right like a Dock stack. Purely decorative - the cards are
/// the same rounded shape as the thumbnail so the pile keeps its silhouette at any size.
struct StackedThumbnail: View {
    let path: String
    let side: CGFloat
    var corner: CGFloat = 6
    /// 0 = a plain thumbnail (no pile drawn at all).
    var depth: Int = 0

    var body: some View {
        ZStack(alignment: .topLeading) {
            ForEach(Array((0 ..< depth).reversed()), id: \.self) { i in
                let step = CGFloat(i + 1) * (side > 60 ? 5 : 3)
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .fill(Color(nsColor: .windowBackgroundColor))
                    .overlay(RoundedRectangle(cornerRadius: corner, style: .continuous)
                        .strokeBorder(.primary.opacity(0.14), lineWidth: 1))
                    .frame(width: side, height: side)
                    .offset(x: step, y: step)
                    .shadow(color: .black.opacity(0.10), radius: 1, y: 0.5)
            }
            Thumbnail(path: path, side: side, corner: corner)
        }
        // Reserve the pile's overhang so a stacked row is not wider than its neighbours.
        .padding(.trailing, depth > 0 ? CGFloat(depth) * (side > 60 ? 5 : 3) : 0)
    }
}

/// In-memory cache for the on-disk fallback below, so re-scrolling a result list never re-reads the
/// same file's header. Only legacy indexes (created before metadata was stored) ever hit this.
final class MediaInfoCache: @unchecked Sendable {
    static let shared = MediaInfoCache()
    private let cache = NSCache<NSString, NSString>()
    init() { cache.countLimit = 4096 }
    func get(_ key: String) -> String? { cache.object(forKey: key as NSString) as String? }
    func set(_ key: String, _ value: String) { cache.setObject(value as NSString, forKey: key as NSString) }
}

/// Original resolution (images) or duration (audio/video). Prefers the value captured at index time
/// (zero disk access); only older indexes that predate stored metadata fall back to reading the file
/// once, off the main thread and cached, so scrolling stays smooth.
struct MediaInfoLabel: View {
    let path: String
    let kind: String
    var width: Int = 0
    var height: Int = 0
    var duration: Double = 0
    var separator: Bool
    @State private var loaded: String?

    private var stored: String? {
        switch FileKind(rawValue: kind) {
        case .image: return (width > 0 && height > 0) ? "\(width)\u{00D7}\(height)" : nil
        case .video, .audio: return duration > 0 ? formatDuration(duration) : nil
        default: return nil
        }
    }
    // scan excluded like text: a scanned-PDF row has no media header to read, so the legacy
    // disk fallback would spawn a wasted task per row.
    private var isMedia: Bool { FileKind(rawValue: kind).map { $0 != .text && $0 != .scan } ?? false }

    var body: some View {
        if let text = stored ?? loaded {
            HStack(spacing: 5) {
                Text(text)
                if separator { Text("\u{00B7}") }
            }
        } else if isMedia {
            // Legacy row with no stored metadata: read the header once, cached.
            Color.clear.frame(width: 0, height: 0).task(id: path) { loaded = await Self.load(path: path, kind: kind) }
        }
    }

    private static func load(path: String, kind: String) async -> String? {
        if let cached = MediaInfoCache.shared.get(path) { return cached }
        let result = await Task.detached(priority: .utility) { () -> String? in
            let url = URL(fileURLWithPath: path)
            switch FileKind(rawValue: kind) {
            case .image:
                if let s = FileExtractor.imagePixelSize(url) { return "\(s.width)\u{00D7}\(s.height)" }
            case .video, .audio:
                if let d = FileExtractor.mediaDuration(url) { return formatDuration(d) }
            default:
                return nil
            }
            return nil
        }.value
        if let result { MediaInfoCache.shared.set(path, result) }
        return result
    }
}

private func formatDuration(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec) : String(format: "%d:%02d", m, sec)
}

struct KindGlyph: View {
    let kind: String
    var body: some View {
        if let k = FileKind(rawValue: kind), k != .text {
            Image(systemName: k.symbol)
        }
    }
}

private func scoreText(_ score: Float) -> String { String(format: "%.0f%%", max(0, min(1, score)) * 100) }

private func prettyDir(_ url: URL) -> String {
    let dir = url.deletingLastPathComponent().path
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    return dir.hasPrefix(home) ? "~" + dir.dropFirst(home.count) : dir
}

// MARK: - Marquee (rubber-band) selection

/// Frames of the realized result rows/cells, keyed by path, gathered in the scroll viewport's
/// coordinate space. Each item publishes its own frame; the modifier reduces them into one map.
private struct ResultItemFramesKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { $1 })
    }
}

private extension View {
    /// Publish this item's frame (in the named viewport space) for marquee hit-testing. A transparent
    /// GeometryReader background measures without affecting layout or hit area.
    func reportResultFrame(_ path: String, in space: String) -> some View {
        background(GeometryReader { g in
            Color.clear.preference(key: ResultItemFramesKey.self, value: [path: g.frame(in: .named(space))])
        })
    }

    func marqueeSelect(space: String) -> some View {
        modifier(MarqueeSelect(space: space))
    }
}

/// Finder-style rubber-band selection: a left-button click-drag over the results draws a rectangle and
/// selects every item it touches. On macOS a click-drag does NOT scroll (the wheel/trackpad do), so the
/// drag is free to mean "marquee" without fighting the scroll view. minimumDistance keeps a plain click
/// a click (the row's own tap still fires). Holding Shift or Command adds to the existing selection.
private struct MarqueeSelect: ViewModifier {
    @Environment(AppModel.self) private var model
    let space: String
    // Owned here, not in ResultsList: the item frames refresh on every scroll tick, so confining them to
    // this modifier means a scroll re-evaluates only this overlay, not the whole list/gallery body.
    @State private var frames: [String: CGRect] = [:]
    @State private var origin: CGPoint?
    @State private var rect: CGRect?
    @State private var base: Set<String> = []

    func body(content: Content) -> some View {
        content
            .coordinateSpace(name: space)
            .onPreferenceChange(ResultItemFramesKey.self) { frames = $0 }
            .overlay(alignment: .topLeading) {
                if let rect {
                    // The native macOS rubber-band is a NEUTRAL translucent grey, not the accent: a
                    // Finder marquee sampled on white measured fill = grey 230 (label @ 0.10) and border
                    // = grey 170 (label @ 0.33). Driving both off labelColor reproduces those exactly in
                    // light mode and inverts to the same faint-white rectangle in dark, like the system.
                    Rectangle()
                        .fill(Color(nsColor: .labelColor).opacity(0.1))
                        .overlay(Rectangle().strokeBorder(Color(nsColor: .labelColor).opacity(0.33), lineWidth: 1))
                        .frame(width: rect.width, height: rect.height)
                        .offset(x: rect.minX, y: rect.minY)
                        .allowsHitTesting(false)
                }
            }
            .gesture(
                DragGesture(minimumDistance: 6, coordinateSpace: .named(space))
                    .onChanged { v in
                        if origin == nil {
                            origin = v.startLocation
                            let m = NSEvent.modifierFlags
                            base = (m.contains(.shift) || m.contains(.command)) ? model.selectedPaths : []
                        }
                        let o = origin ?? v.startLocation
                        let r = CGRect(x: min(o.x, v.location.x), y: min(o.y, v.location.y),
                                       width: abs(v.location.x - o.x), height: abs(v.location.y - o.y))
                        rect = r
                        let hit = Set(frames.compactMap { $0.value.intersects(r) ? $0.key : nil })
                        model.applyMarqueeSelection(base.union(hit))
                    }
                    .onEnded { _ in origin = nil; rect = nil }
            )
    }
}
