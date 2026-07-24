import SwiftUI
import AppKit
import OmniKit
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var debounce: Task<Void, Never>?
    @State private var historyDebounce: Task<Void, Never>?
    @State private var fileDropTargeted = false

    // Progressive disclosure: only offer search once there is something to search. During model
    // loading, onboarding, and the no-folders state the search field stays hidden (not dimmed).
    private var showsSearch: Bool { model.phase == .ready && !model.roots.isEmpty }

    /// Apply a user edit of the search box: parse it into the semantic query + qualifiers, apply the
    /// filters, clear a file query if real text was typed, and schedule the (debounced) search. The
    /// box binds to the RAW typed string; `set` (user edits only) routes here.
    private func handleQueryEdit(_ raw: String) {
        model.applyParsedQuery(raw)
        model.suggestionsAllowed = true   // this fires only on real keystrokes (the .searchable set:), so arm the dropdown
        if !model.query.isEmpty, model.fileQuery != nil { model.fileQuery = nil; model.queryError = nil }
        // Instant search off: typing still parses filters and pops suggestions, but the search
        // itself waits for Return (.onSubmit) - kinder to low-end GPUs. Clearing the box always
        // runs (its empty-query path clears the stale results); auto-history records only what
        // actually searched (Return records via onSubmit).
        let cleared = raw.trimmingCharacters(in: .whitespaces).isEmpty
        if model.fileQuery == nil, model.instantSearchEnabled || cleared { scheduleSearch() }
        if model.instantSearchEnabled { scheduleHistoryRecord() }
    }

    var body: some View {
        Group {
            if showsSearch {
                split
                    .searchable(text: Binding(get: { model.rawQuery }, set: { handleQueryEdit($0) }),
                                placement: .toolbar, prompt: "Search by meaning") {
                        // Typeahead: keys (ty -> type:), values (type: -> image/...), and matching past
                        // queries as instant (cached) shortcuts. Navigate with arrows + Return. Only while
                        // the user is typing - a programmatic box change (history replay, filter menu) keeps
                        // the dropdown closed (suggestionsAllowed is false unless handleQueryEdit armed it).
                        ForEach(model.suggestionsAllowed ? searchSuggestions(model.rawQuery) : [], id: \.completion) { sug in
                            Label(sug.label, systemImage: sug.icon).searchCompletion(sug.completion)
                        }
                    }
                    .onSubmit(of: .search) { model.search(); model.recordCurrentSearchToHistory(viaSubmit: true) }
            } else {
                split
            }
        }
        // Spotlight-style: put the caret in the search field as soon as the app can search.
        .onChange(of: showsSearch, initial: true) { _, shows in if shows { focusSearchField() } }
        // Profiling progress as a native sheet on the main window (not a stray floating panel).
        .sheet(isPresented: Binding(get: { model.isProfilingRunning }, set: { _ in })) {
            ProfilingSheet()
        }
    }

    private func focusSearchField() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            guard let window = NSApp.windows.first(where: { $0.isVisible && $0.toolbar != nil }),
                  let item = window.toolbar?.items.compactMap({ $0 as? NSSearchToolbarItem }).first
            else { return }
            window.makeFirstResponder(item.searchField)
        }
    }

    private var split: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 230, ideal: 260, max: 320)
        } detail: {
            detail
                // No navigationTitle/navigationSubtitle: either one claims the leading toolbar slot
                // and pushes back/forward to its right. The WINDOW title ("Omni", from the Window
                // scene) stays visible - stock titlebar chrome on Sequoia is load-bearing (system
                // sidebar toggle, tracking separator, divider-drag rendering), and Tahoe shows the
                // title too.
                .toolbar { toolbar }
                .background(WindowTitleHider(onSearchByFile: { model.searchByFilePanel() }))
        }
    }

    // MARK: - Detail

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            // No subtitle: it repeated the title ("Starting the on-device model..." under "Loading
            // the Omni model"), and the determinate bar already communicates the wait.
            CenteredStatus(symbol: "brain", title: "Loading the Omni model", subtitle: "", showSpinner: true, progress: model.loadingProgress)
        case .noModel:
            OnboardingView()
        case .failed(let msg):
            EngineFailedView(message: msg)
        case .ready:
            ready
        }
    }

    @ViewBuilder private var ready: some View {
        content
    }

    /// The folder embedding map is shown ONLY in the empty-result region and ONLY when nothing
    /// search-related is active: a folder is selected, the query box is empty (typed AND file), no
    /// raw results, no query error, and nothing resolving. Active queries/results always win - this
    /// flips false the instant the user types, hiding the viz purely by precedence (the selected
    /// folder is not cleared, so clearing the query brings the cached map back instantly).
    private var showsFolderViz: Bool {
        model.selectedFolderForViz != nil && !model.hasQuery && model.fileQuery == nil
            && model.rawResults.isEmpty && model.queryError == nil && !model.isResolving
    }

    @ViewBuilder private var content: some View {
        VStack(spacing: 0) {
            if let fq = model.fileQuery { FileQueryChip(fileQuery: fq) }
            else if !model.activeQualifiers.isEmpty || model.literalQuery { QualifierBar() }
            if !model.results.isEmpty {
                ResultsList(results: model.results) { belowThresholdFooter }
            } else if showsFolderViz {
                FolderEmbeddingVisualization(folderName: model.selectedFolderForViz!.lastPathComponent)
            } else {
                emptyState
            }
        }
        // Drag an image, file, or text from anywhere (Finder, a browser, another app) - or paste one
        // (Cmd-V) - to search by it. SwiftUI's .onDrop gives us reachability over the results list and
        // the empty state alike, but a web image dragged from Chrome/Safari arrives as inline encoded
        // bytes, a file promise, or a remote URL - none of which loadObject(NSImage/NSURL) can resolve.
        // So we read the raw drag pasteboard for the full flavor set, and fall back to the providers.
        .onDrop(of: [.image, .fileURL, .url, .text, .plainText], isTargeted: $fileDropTargeted) { providers in
            handleSearchDrop(providers)
        }
        .overlay {
            if fileDropTargeted {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: 2).padding(6).allowsHitTesting(false)
            }
        }
    }

    // MARK: - Drag to search

    /// Route a drop onto the search surface. Reachability is SwiftUI's (the closure fires over the
    /// results list and the empty state alike); for the payload we read the raw drag pasteboard,
    /// which carries the full flavor set a browser provides, then fall back to the item providers.
    private func handleSearchDrop(_ providers: [NSItemProvider]) -> Bool {
        if handleDragPasteboard(NSPasteboard(name: .drag)) { return true }
        return handleDropProviders(providers)
    }

    /// Read a drag/paste pasteboard in fidelity order: a local file, inline image bytes (Chrome's
    /// public.jpeg/png, Safari/Firefox public.tiff), a file promise (Safari/Chrome), a remote image
    /// URL it downloads (Firefox / URL-only), then any other bitmap, then text. Model calls run on
    /// the main actor; the async promise/download paths hop back to it.
    private func handleDragPasteboard(_ pb: NSPasteboard) -> Bool {
        // 1) Local file (Finder, Mail attachment): the existing file-query path. Take the first
        //    SUPPORTED file, so a mixed multi-file drag still finds the one Omni can search.
        if let url = (pb.readObjects(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) as? [URL])?
            .first(where: { FileExtractor.kind(for: $0) != nil }) {
            Task { @MainActor in model.setFileQuery(url) }
            return true
        }
        // 2) Inline image bytes - synchronous, no network; preferred (Omni re-encodes to PNG anyway).
        if let (data, ext) = Self.pasteboardImageBytes(pb) {
            Task { @MainActor in model.searchByImage(data: data, suggestedExtension: ext) }
            return true
        }
        // 3) File promise (Safari, Chrome): the browser writes the file into our temp dir, async.
        //    Honor a single promise (the first) so one drag is one search - a multi-image drag must
        //    not fire N staggered searches with a nondeterministic winner. The omni-drop- prefix lets
        //    the launch sweep reclaim the dir. (FileManager.temporaryDirectory is per-user-writable.)
        if let promise = (pb.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil)
            as? [NSFilePromiseReceiver])?.first {
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("omni-drop-promise-\(UUID().uuidString)", isDirectory: true)
            try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            promise.receivePromisedFiles(atDestination: dest, options: [:],
                                         operationQueue: OperationQueue()) { url, error in
                guard error == nil, FileExtractor.kind(for: url) != nil else { return }
                Task { @MainActor in model.setFileQuery(url) }
            }
            return true
        }
        // 4) Remote image URL (Firefox / URL-only): download, then confirm it decodes as an image
        //    (a linked <img> can put the link href on public.url instead of the image src).
        if let remote = (pb.readObjects(forClasses: [NSURL.self], options: nil) as? [URL])?
            .first(where: { !$0.isFileURL && ($0.scheme == "http" || $0.scheme == "https") }) {
            downloadImage(remote, textFallback: remote.absoluteString)   // a bare link -> text search
            return true
        }
        // 5) Any other bitmap the pasteboard can vend.
        if let img = NSImage(pasteboard: pb) {
            Task { @MainActor in model.searchByImage(img) }
            return true
        }
        // 6) Text.
        if let s = pb.string(forType: .string), !s.isEmpty {
            Task { @MainActor in model.searchByText(s) }
            return true
        }
        return false
    }

    /// Fallback when the drag pasteboard is unavailable: resolve through the SwiftUI item providers.
    /// Covers the common cases (Finder file, a data-backed image, a Chrome web image whose remote URL
    /// surfaces as a provider), and only text-searches a dropped URL when it is not an image.
    private func handleDropProviders(_ providers: [NSItemProvider]) -> Bool {
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, url.isFileURL, FileExtractor.kind(for: url) != nil else { return }
                Task { @MainActor in model.setFileQuery(url) }
            }
            return true
        }
        if let p = providers.first(where: { $0.canLoadObject(ofClass: NSImage.self) }) {
            _ = p.loadObject(ofClass: NSImage.self) { obj, _ in
                guard let img = obj as? NSImage else { return }
                Task { @MainActor in model.searchByImage(img) }
            }
            return true
        }
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, !url.isFileURL,
                      url.scheme == "http" || url.scheme == "https" else { return }
                downloadImage(url, textFallback: url.absoluteString)
            }
            return true
        }
        if let p = providers.first(where: { $0.canLoadObject(ofClass: NSString.self) }) {
            _ = p.loadObject(ofClass: NSString.self) { obj, _ in
                guard let s = obj as? String else { return }
                Task { @MainActor in model.searchByText(s) }
            }
            return true
        }
        return false
    }

    /// Download a remote URL and search by it if the bytes decode as an image; otherwise, if a text
    /// fallback was given (a bare hyperlink), run a text search on it.
    private func downloadImage(_ url: URL, textFallback: String? = nil) {
        URLSession.shared.dataTask(with: url) { data, response, _ in
            guard let data, NSImage(data: data) != nil else {
                if let textFallback { Task { @MainActor in model.searchByText(textFallback) } }
                return
            }
            let ext = response?.mimeType
                .flatMap { UTType(mimeType: $0)?.preferredFilenameExtension } ?? "png"
            Task { @MainActor in model.searchByImage(data: data, suggestedExtension: ext) }
        }.resume()
    }

    /// Encoded image bytes off a pasteboard: a named bitmap type first, then any image UTI
    /// (catches Chrome's public.jpeg / public.gif / public.webp).
    static func pasteboardImageBytes(_ pb: NSPasteboard) -> (Data, String)? {
        for t in [NSPasteboard.PasteboardType.png, .tiff] {
            if let d = pb.data(forType: t), let e = UTType(t.rawValue)?.preferredFilenameExtension {
                return (d, e)
            }
        }
        for t in pb.types ?? [] {
            guard let ut = UTType(t.rawValue), ut.conforms(to: .image),
                  let d = pb.data(forType: t) else { continue }
            return (d, ut.preferredFilenameExtension ?? "png")
        }
        return nil
    }

    @ViewBuilder private var emptyState: some View {
        // Indexing is invisible here - the sidebar's per-folder progress is the only cue, and
        // search works while it runs. The user just adds folders and searches.
        if model.roots.isEmpty {
            CenteredStatus(symbol: "folder.badge.plus", title: "Add a folder to search",
                           subtitle: "Choose the folders you want to search. Omni indexes them automatically and keeps them up to date.",
                           showSpinner: false, action: ("Add folder\u{2026}", { pickFolder() }))
        } else if let err = model.queryError {
            CenteredStatus(symbol: "exclamationmark.triangle", title: "Couldn't search by that file",
                           subtitle: err, showSpinner: false)
        } else if model.indexObsolete && model.hasQuery {
            // A dim/model mismatch makes every search return nothing; explain it and offer both the
            // cheap fix (switch back to the model the index was built with) and the rebuild.
            let built = model.indexBuiltVariant
            CenteredStatus(symbol: "arrow.triangle.2.circlepath",
                           title: built != nil ? "Switch to \(built!.title) or reindex" : "Reindex to search",
                           subtitle: built != nil
                               ? "This index was built with \(built!.title), but \(model.modelVariant.title) is loaded. Switch back to keep your index, or reindex with the current model."
                               : "This index was built with a different model than the one loaded. Reindex to search again.",
                           showSpinner: false,
                           action: built.map { v in ("Switch to \(v.title)", { model.selectVariant(v) }) },
                           secondary: ("Reindex", { model.startIndexing() }))
        } else if !model.hasQuery || model.isResolving {
            // Idle prompt, and the in-flight search state. They share one calm placeholder so a
            // pending search only fades a small spinner in under the same prompt - it never flashes
            // "No matches" while the debounce/search for what you just typed is still running.
            SearchWaysPrompt(
                title: model.indexedFiles > 0 ? "Search \(model.indexedFiles.formatted()) file\(model.indexedFiles == 1 ? "" : "s")" : "Search your files",
                showSpinner: model.isResolving)
        } else if model.hiddenByThreshold > 0 {
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle",
                           title: "No results above \(Int(model.minScore * 100))%",
                           subtitle: "\(model.hiddenByThreshold) weaker \(model.hiddenByThreshold == 1 ? "match is" : "matches are") hidden by the relevance threshold.",
                           showSpinner: false, action: ("Show all matches", { model.showAllBelowThreshold() }))
        } else if model.filtersActive {
            // Filters can hide every result; the empty state is the only place left to escape them.
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle", title: "No matches",
                           subtitle: "Filters are hiding every result.", showSpinner: false,
                           action: ("Clear filters", { model.clearFilters() }))
        } else {
            CenteredStatus(symbol: "magnifyingglass", title: "No matches", subtitle: "Try a different phrase.", showSpinner: false)
        }
    }

    @ViewBuilder private var belowThresholdFooter: some View {
        if model.hiddenByThreshold > 0 {
            Button { model.showAllBelowThreshold() } label: {
                Label("Show \(model.hiddenByThreshold) more \(model.hiddenByThreshold == 1 ? "match" : "matches")", systemImage: "chevron.down")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
    }

    // MARK: - Toolbar

    // On Tahoe, place the filter with sort/view (trailing) so the three result controls share one
    // Liquid Glass pill; on earlier macOS keep it leading so the existing toolbar layout is untouched.
    private var filterPlacement: ToolbarItemPlacement {
        if #available(macOS 26.0, *) { return .primaryAction } else { return .automatic }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        // No explicit sidebar toggle: Sequoia's system toggle (next to the traffic lights, like
        // Finder) lives and dies with the split-view TRACKING SEPARATOR item, which this app keeps
        // (transparent - see the tuner). Hiding the title does NOT collapse it; only removing the
        // separator item does. An explicit button here therefore always duplicates the system one.
        // Finder-style back/forward at the leading edge of the content toolbar. The window's title TEXT
        // is hidden (WindowTitleHider), so the chevrons own the leading edge with no "Omni" label. The
        // toolbar ITEM is unconditional and wraps the chevrons in an always-present HStack: a directly
        // conditional .navigation item reorders unpredictably, whereas the always-present HStack holds a
        // stable leading slot and the conditional chevrons inside it just appear/vanish. Progressive
        // disclosure: the chevrons show only once there's somewhere to go, so the idle state is empty
        // here. Cmd-[ / Cmd-] match Finder and Safari; each chevron disables independently at the end of
        // its trail. Grouped so on Tahoe they share one Liquid Glass pill.
        ToolbarItem(placement: .navigation) {
            HStack(spacing: 0) {
                // A completely empty toolbar item has zero intrinsic size, and AppKit logs an
                // "ambiguous width/height" warning for it on every toolbar layout pass - measured
                // on macOS 26 too (12 hits at launch), not just 14/15, so the filler is
                // unconditional. 1pt of clear gives the item a size without a visible footprint.
                Color.clear.frame(width: 1, height: 1)
                if model.phase == .ready, model.canGoBack || model.canGoForward {
                    ControlGroup {
                        // The View menu owns Cmd-[ / Cmd-] (single owner, avoids a duplicate-shortcut
                        // conflict); these buttons are click targets that name the same chords.
                        Button { model.goBack() } label: { Image(systemName: "chevron.backward") }
                            .disabled(!model.canGoBack)
                            .help("Back  \u{2318}[")
                            .accessibilityLabel("Back")
                        Button { model.goForward() } label: { Image(systemName: "chevron.forward") }
                            .disabled(!model.canGoForward)
                            .help("Forward  \u{2318}]")
                            .accessibilityLabel("Forward")
                    }
                    .fixedSize()
                }
                // Trailing 1pt filler, same reason as the leading one (all systems).
                Color.clear.frame(width: 1, height: 1)
            }
        }
        // Flexible space after back/forward pushes every other control to the trailing edge (chevrons
        // own the left, everything else is right-aligned), and on Tahoe it's also the correct separator
        // between Liquid Glass toolbar groups - the leading chevron pill and the trailing control pills
        // read as distinct glass surfaces.
        if #available(macOS 26.0, *) {
            ToolbarSpacer(.flexible)
        }
        // Pre-Tahoe gets the equivalent flexible space from WindowTitleHider's tuner, which inserts
        // AppKit's native .flexibleSpace NSToolbarItem after the chevrons: a SwiftUI
        // `ToolbarItem { Spacer() }` is silently DROPPED on macOS 14/15 (verified via the live
        // NSToolbar's item list), so without the AppKit item nothing separates the leading chevrons
        // from the trailing cluster and the stretchy search field parks every control center-left.
        // Search by a file lives INSIDE the search field (trailing upload glyph, installed by
        // WindowTitleHider's tuner - magnifier left, upload right), not as a separate toolbar
        // button. The File menu owns the Shift-Cmd-O shortcut; the in-field button is the click
        // target naming the same chord.
        // Bookmark the current search. The only way into History when recording is set to "Only when
        // I bookmark", and a quick save otherwise. Appears once there's a search to keep.
        if model.phase == .ready, model.hasActiveSearch {
            ToolbarItem(placement: .primaryAction) {
                Button { model.toggleBookmarkCurrentSearch() } label: {
                    // No explicit color in the unbookmarked state, so the toolbar can dim it like every
                    // other button when the window resigns key (e.g. while Settings is open). Yellow is
                    // applied only when bookmarked, where the lit status color is intentional.
                    if model.currentSearchIsBookmarked {
                        Image(systemName: "star.fill").foregroundStyle(.yellow)
                    } else {
                        Image(systemName: "star")
                    }
                }
                // Cmd-D is owned by the File-menu "Bookmark search" command (single owner, avoids a
                // duplicate-shortcut conflict); the tooltip names it, and accessibilityLabel is what
                // VoiceOver reads and what the toolbar-overflow menu shows for this icon-only button.
                .help(model.currentSearchIsBookmarked ? "Remove bookmark  \u{2318}D" : "Bookmark this search  \u{2318}D")
                .accessibilityLabel(model.currentSearchIsBookmarked ? "Remove bookmark" : "Bookmark search")
            }
        }
        // Progressive disclosure: the filter/sort/view chrome appears only once there are results
        // to act on - hidden, not greyed out, during onboarding and the idle/empty states.
        // Exception: keep the filter menu reachable whenever a filter is active, so a filter that
        // hides every result can still be cleared (otherwise the menu vanishes with the results).
        if model.phase == .ready, !model.rawResults.isEmpty || model.filtersActive {
        // Filter joins sort/view in the trailing placement so on Tahoe the three result controls
        // share ONE Liquid Glass pill (search-by-file + bookmark form the other). filterPlacement
        // keeps filter leading on pre-26 so the Sequoia toolbar layout is unchanged.
        ToolbarItem(placement: filterPlacement) {
            filterMenu.disabled(model.indexedFiles == 0)
        }
        }
        // Result presentation - sort + view. Only meaningful with results.
        if model.phase == .ready, !model.rawResults.isEmpty {
        ToolbarItem(placement: .primaryAction) {
            if #available(macOS 26.0, *) {
                // Tahoe: the inline sort menu + segmented view toggle render and overflow cleanly.
                ControlGroup {
                    Menu {
                        Picker("Sort by", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                            ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                        }
                    } label: { Image(systemName: "arrow.up.arrow.down") }
                    .help("Sort by \(model.sortOrder.title)")
                    .accessibilityLabel("Sort results")

                    Picker("View", selection: Binding(get: { model.viewMode }, set: { model.viewMode = $0 })) {
                        Image(systemName: "list.bullet").accessibilityLabel("List view").tag(ResultViewMode.list)
                        Image(systemName: "square.grid.2x2").accessibilityLabel("Gallery view").tag(ResultViewMode.grid)
                    }
                    .pickerStyle(.segmented)
                    .help("Switch between list and gallery")
                }
            } else {
                // Sequoia and earlier: a ControlGroup of a menu + segmented picker overflows into an
                // empty, icon-less toolbar dropdown. Use one compact labeled menu instead so it always
                // shows its icon and survives overflow.
                Menu {
                    Picker("View", selection: Binding(get: { model.viewMode }, set: { model.viewMode = $0 })) {
                        Label("as Gallery", systemImage: "square.grid.2x2").tag(ResultViewMode.grid)
                        Label("as List", systemImage: "list.bullet").tag(ResultViewMode.list)
                    }
                    Divider()
                    Picker("Sort by", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                        ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                    }
                } label: {
                    Label("View options", systemImage: "slider.horizontal.3")
                }
                .help("Sort and view")
            }
        }
        }
    }

    private var filterKinds: [FileKind] {
        // Show indexed kinds, plus any kind currently being filtered on - otherwise a filter for a
        // kind that is not (yet) in the index would be invisible and impossible to untoggle.
        let present = FileKind.indexable.filter { model.indexedKinds.contains($0.rawValue) || model.filterKinds.contains($0) }
        // Scanned is ALWAYS offered (unlike the four detection kinds): it is an extraction-time
        // sub-kind users may not know they have until they filter for it.
        return (present.isEmpty ? [.image, .video, .audio] : present) + [.scan]
    }

    private var filterMenu: some View {
        Menu {
            Section("Show") {
                ForEach(filterKinds, id: \.self) { kind in
                    Toggle(isOn: Binding(
                        get: { model.filterKinds.contains(kind) },
                        // Text carries its sub-kind along (scanned PDFs are documents too), so
                        // toggling Text never silently drops scans from the results. The Scanned
                        // PDFs toggle stays independent for narrowing within text.
                        set: { on in
                            if on {
                                model.filterKinds.insert(kind)
                                if kind == .text { model.filterKinds.insert(.scan) }
                            } else {
                                model.filterKinds.remove(kind)
                                if kind == .text { model.filterKinds.remove(.scan) }
                            }
                        }
                    )) { Label(kind.title, systemImage: kind.symbol) }
                }
            }
            Picker("Folder", selection: Binding(
                get: { model.filterFolder?.path ?? "" },
                set: { model.filterFolder = $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
            )) {
                Text("All folders").tag("")
                ForEach(model.roots, id: \.self) { Text($0.lastPathComponent).tag($0.path) }
            }
            Picker("Extension", selection: Binding(get: { model.filterExt }, set: { model.filterExt = $0 })) {
                Text("Any extension").tag("")
                ForEach(model.indexedExts, id: \.self) { Text(".\($0)").tag($0) }
            }
            Picker("Date", selection: Binding(get: { model.dateRange }, set: { model.dateRange = $0 })) {
                ForEach(DateRange.allCases) { Text($0.title).tag($0) }
            }
            Picker("Relevance", selection: Binding(get: { model.minScore }, set: { model.minScore = $0 })) {
                Text("Any").tag(0.0); Text("25%").tag(0.25); Text("50%").tag(0.5); Text("70%").tag(0.7)
            }
            Divider()
            Button("Clear filters") { model.clearFilters() }.disabled(!model.filtersActive)
        } label: {
            Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .help("Filter results")
        .accessibilityLabel("Filter results")
    }

    private func scheduleSearch() {
        debounce?.cancel()
        debounce = Task {
            try? await Task.sleep(nanoseconds: 180_000_000)
            if !Task.isCancelled { model.search() }
        }
    }

    // Auto-record (history mode .auto) only after the query has been settled for 3s, so a search has
    // to be one the user actually dwelled on - quick type-and-click-through queries aren't stored.
    // Cancelled on every keystroke, so it only fires once typing stops. (No effect in .onSubmit /
    // .manual modes, which record on Return / the bookmark button instead.)
    private func scheduleHistoryRecord() {
        historyDebounce?.cancel()
        historyDebounce = Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if !Task.isCancelled { model.recordCurrentSearchToHistory() }
        }
    }

    // MARK: - Query-language autocomplete

    struct Suggestion: Hashable { let label: String; let completion: String; let icon: String }

    /// Typeahead for the search box: complete a partial qualifier key (`ty` -> `type:`) or a key's
    /// values (`type:` -> image/video/...). Returns full-string completions - the text before the
    /// active token is preserved, so selecting one keeps the rest of the query intact.
    private func searchSuggestions(_ raw: String) -> [Suggestion] {
        guard !model.literalQuery else { return [] }
        var out: [Suggestion] = []
        let prefix: String, tok: String
        if let sp = raw.lastIndex(of: " ") {
            prefix = String(raw[...sp]); tok = String(raw[raw.index(after: sp)...])
        } else {
            prefix = ""; tok = raw
        }
        if !tok.isEmpty {
            if let colon = tok.firstIndex(of: ":") {                   // value completion: key:partial
                let keyTyped = String(tok[..<colon])
                if let canon = SearchQueryParser.canonicalKey(keyTyped.lowercased()) {
                    let partial = String(tok[tok.index(after: colon)...]).lowercased()
                    out += valueSuggestions(canon).filter { $0.lowercased().hasPrefix(partial) }.prefix(8).map {
                        let v = $0.contains(" ") ? "\"\($0)\"" : $0
                        return Suggestion(label: "\(keyTyped):\($0)", completion: "\(prefix)\(keyTyped):\(v)", icon: "tag")
                    }
                }
            } else {                                                   // key completion: bare prefix
                let neg = tok.hasPrefix("-") ? "-" : ""
                let low = (neg.isEmpty ? tok : String(tok.dropFirst())).lowercased()
                if !low.isEmpty {
                    for k in ["type:", "tag:", "ext:", "in:", "date:", "after:", "score:", "sort:"] where k.hasPrefix(low) {
                        out.append(Suggestion(label: neg + k, completion: "\(prefix)\(neg)\(k)", icon: "line.3.horizontal.decrease.circle"))
                    }
                }
            }
        }
        // Past queries as quick shortcuts: already query-side embedded (cached), so picking one
        // searches instantly without a trip to the sidebar.
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.count >= 1 {
            let needle = trimmed.lowercased()
            let hist = model.searchHistory
                .filter { !$0.isFile && $0.displayText.lowercased().contains(needle) && $0.displayText.lowercased() != needle }
                .sorted { a, b in a.bookmarked != b.bookmarked ? a.bookmarked : a.lastUsed > b.lastUsed }
                .prefix(5)
            out += hist.map { Suggestion(label: $0.displayText, completion: $0.displayText, icon: $0.bookmarked ? "star.fill" : "clock") }
        }
        return Array(out.prefix(10))
    }

    private func valueSuggestions(_ key: String) -> [String] {
        switch key {
        case "type": return ["image", "video", "audio", "text", "scanned"]
        case "date": return ["any", "week", "month", "year"]
        case "after": return ["week", "month", "year", "7d", "30d", "1y"]
        case "score": return ["25%", "50%", "70%"]
        case "sort": return ["relevance", "name", "date"]
        case "ext": return model.indexedExts
        case "in": return model.roots.map { ($0.path as NSString).abbreviatingWithTildeInPath }
        default: return []
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { model.addRoots(panel.urls) }
    }

}

/// A thin bar under the search field showing the qualifiers Omni parsed from the box (or the
/// literal-mode state), with a one-click toggle to treat the box as plain text instead of filters.
private struct QualifierBar: View {
    @Environment(AppModel.self) private var model: AppModel
    var body: some View {
        HStack(spacing: 6) {
            if model.literalQuery {
                Image(systemName: "textformat").foregroundStyle(.secondary).frame(width: 18)
                Text("Plain-text search").foregroundStyle(.secondary)
                Text("- qualifiers ignored").font(.caption).foregroundStyle(.tertiary)
            } else {
                Image(systemName: "line.3.horizontal.decrease.circle").foregroundStyle(.secondary).frame(width: 18)
                ForEach(Array(model.activeQualifiers.enumerated()), id: \.offset) { _, q in
                    HStack(spacing: 3) {
                        if q.negated { Text("not").font(.caption2).foregroundStyle(.tertiary) }
                        Text(q.key).fontWeight(.medium).foregroundStyle(.tint)
                        Text(q.value).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    }
                    .font(.caption)
                    .padding(.horizontal, 7).padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
                }
            }
            Spacer(minLength: 8)
            Button { model.toggleLiteralQuery() } label: {
                Label(model.literalQuery ? "Use as query" : "Plain text",
                      systemImage: model.literalQuery ? "line.3.horizontal.decrease.circle" : "textformat")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .help(model.literalQuery
                  ? "Interpret key:value as filters again"
                  : "Embed the box text as-is, ignoring key:value qualifiers")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 6)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }
}

/// A thin bar above the results showing the active file query (a file used as the search subject),
/// with a clear button. Reuses Thumbnail and a native .bar material.
private struct FileQueryChip: View {
    @Environment(AppModel.self) private var model: AppModel
    let fileQuery: AppModel.FileQuery
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: fileQuery.similar ? "square.on.square" : "photo.badge.magnifyingglass")
                .foregroundStyle(.secondary).frame(width: 18)
            Thumbnail(path: fileQuery.url.path, side: 18, corner: 4)
            Text(fileQuery.similar ? "Similar to" : "Searching by").foregroundStyle(.secondary)
            Text(fileQuery.url.lastPathComponent).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 8)
            Button { model.clearFileQuery() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary).help("Clear file query")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }
}

struct CenteredStatus: View {
    let symbol: String
    let title: String
    let subtitle: String
    var showSpinner: Bool = false
    /// Determinate 0...1 -> a native linear progress bar replaces the indeterminate spinner.
    var progress: Double? = nil
    var action: (String, () -> Void)? = nil
    var secondary: (String, () -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title)
            if !subtitle.isEmpty {
                Text(subtitle).font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 400)
            }
            if let progress {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 280)
                    .padding(.top, 4)
            } else if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
            if action != nil || secondary != nil {
                HStack(spacing: 10) {
                    if let action { Button(action.0, action: action.1).buttonStyle(.borderedProminent) }
                    if let secondary { Button(secondary.0, action: secondary.1) }
                }
                .controlSize(.large).padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// The idle search prompt. Same icon + title as CenteredStatus, but instead of one sentence it lists
/// every way to start a search as bullets - typing, dragging, pasting, picking a file, Find Similar.
/// Icons match the real controls (photo.badge.magnifyingglass = the search-by-file button,
/// square.on.square = the Find Similar / file-query chip) so the list maps onto the actual UI.
struct SearchWaysPrompt: View {
    let title: String
    var showSpinner: Bool = false

    // (icon, text). Icons mirror the toolbar/menu/chip controls they describe.
    private let ways: [(icon: String, text: String)] = [
        ("character.cursor.ibeam", "Type a phrase, ranked by meaning"),
        ("arrow.down.doc", "Drag in an image, file, or text"),
        ("doc.on.clipboard", "Paste an image or text  \u{2318}V"),
        ("photo.badge.magnifyingglass", "Search by a file  \u{21E7}\u{2318}O"),
        ("square.on.square", "Right-click a result for Find Similar"),
    ]

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "sparkle.magnifyingglass").font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title)
            VStack(alignment: .leading, spacing: 10) {
                ForEach(ways, id: \.icon) { w in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Image(systemName: w.icon).foregroundStyle(.tertiary).frame(width: 20)
                        Text(w.text)
                    }
                }
            }
            .font(.callout).foregroundStyle(.secondary)   // content-width block; the outer VStack centers it
            if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct EngineFailedView: View {
    @Environment(AppModel.self) private var model: AppModel
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text("Omni can't load its model").font(.title)
            HStack {
                Button("Retry") { model.retryBootstrap() }.buttonStyle(.borderedProminent)
                Button("Choose model folder\u{2026}") { pickModel() }
            }
            .controlSize(.large)
            DisclosureGroup("Details") {
                Text(message).font(.caption.monospaced()).foregroundStyle(.secondary)
                    .textSelection(.enabled).frame(maxWidth: 460, alignment: .leading)
            }
            .frame(maxWidth: 460)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
}

/// Hides the window's TITLE TEXT (not the title bar) so the toolbar's leading slot is free for the
/// back/forward chevrons and there's no redundant "Omni" label - while the title bar (and its Liquid
/// Glass toolbar material) stays intact, unlike `.hiddenTitleBar`. Re-applied on every update because
/// SwiftUI re-asserts `.visible` from the Window scene's title; the window keeps its "Omni" title for
/// the Window menu, Mission Control, and Stage Manager.
private struct WindowTitleHider: NSViewRepresentable {
    /// Called when the in-field search-by-file button is clicked.
    var onSearchByFile: () -> Void

    /// Toolbar tuner (an invisible background view holding coalesced observers; despite the
    /// legacy name it no longer touches the window title - stock Sequoia titlebar chrome, i.e.
    /// the visible "Omni" title, the system sidebar toggle, and the split tracking separator,
    /// proved load-bearing: every attempt to hide any of them broke another state - the toggle
    /// vanished, section layout collapsed, or divider drags misrendered). What it does do:
    /// - pre-Tahoe: caps the search field at Tahoe's ~300pt and installs the flexible-space
    ///   width constraint on the chevrons item so the trailing cluster hugs the right edge.
    /// - all systems: keeps the search-by-file button installed INSIDE the search field's
    ///   trailing edge (magnifier left, upload right), docking left of the clear (x) button
    ///   whenever the field has text, and shrinks the stock magnifier to match.
    /// All work runs in a coalesced main.async pass - never synchronously inside a window or
    /// toolbar notification, where mutations mid-SwiftUI-commit are unsafe.
    final class TunerView: NSView {
        var onSearchByFile: (() -> Void)?
        // nonisolated(unsafe): deinit is nonisolated under strict concurrency; the view lives and
        // dies on the main thread, so the unregistration is race-free in practice.
        nonisolated(unsafe) private var observers: [NSObjectProtocol] = []
        private static let accessoryTag = 0xF17E

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard observers.isEmpty, window != nil else { return }
            scheduleApply()
            let nc = NotificationCenter.default
            observers.append(nc.addObserver(forName: NSWindow.didUpdateNotification, object: window, queue: nil) { [weak self] _ in
                self?.scheduleApply()
            })
            for name in [NSToolbar.didRemoveItemNotification, NSToolbar.willAddItemNotification] {
                observers.append(nc.addObserver(forName: name, object: nil, queue: nil) { [weak self] _ in
                    self?.scheduleApply()
                })
            }
        }
        deinit {
            for o in observers { NotificationCenter.default.removeObserver(o) }
        }

        // nonisolated(unsafe): touched from notification closures that are main-thread in practice
        // (window updates, toolbar mutations); a stale read only coalesces one extra pass.
        nonisolated(unsafe) private var applyScheduled = false
        private nonisolated func scheduleApply() {
            guard !applyScheduled else { return }
            applyScheduled = true
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.applyScheduled = false
                if let w = self.window { self.apply(w) }
            }
        }

        private func apply(_ w: NSWindow) {
            guard let toolbar = w.toolbar else { return }
            for item in toolbar.items {
                guard let s = item as? NSSearchToolbarItem else { continue }
                installAccessory(in: s.searchField)
            }
        }

        /// The upload button lives as a subview of the NSSearchField, frame-pinned to the pill's
        /// trailing edge; when the field has text it steps left so the system clear (x) button
        /// stays fully clickable. Defensive by construction: if any expectation fails the button
        /// simply does not appear - the field itself is never altered.
        private func installAccessory(in field: NSSearchField) {
            let side: CGFloat = 20   // hit target + hover-highlight capsule; the glyph inside is 11pt
            let hasText = !field.stringValue.isEmpty
            // Anchor to the CELL's cancel-button rect - the pill's true inner trailing edge. The
            // NSSearchField view can extend past the drawn pill (trailing padding), so math off
            // bounds.width parked the button visually outside the field.
            let cancelRect = (field.cell as? NSSearchFieldCell)?.cancelButtonRect(forBounds: field.bounds)
                ?? NSRect(x: field.bounds.width - 24, y: 0, width: 16, height: field.bounds.height)
            let x = hasText ? cancelRect.minX - side - 2 : cancelRect.maxX - side
            let y = (field.bounds.height - side) / 2
            if let b = field.viewWithTag(Self.accessoryTag) as? NSButton {
                let want = NSRect(x: x, y: y, width: side, height: side)
                if b.frame != want { b.frame = want }
                return
            }
            // photo.badge.magnifyingglass: the SAME symbol this action already wears in the
            // empty-state hint row and the file-query chip. square.and.arrow.up was tried first
            // and read as Share - which it literally is elsewhere in this app (context menus).
            guard let icon = NSImage(systemSymbolName: "photo.badge.magnifyingglass",
                                     accessibilityDescription: "Search by a file") else { return }
            // Shrink the field's own magnifier to the same 11pt so the two glyphs read as one
            // family (the stock loupe is drawn noticeably larger). Idempotent via the cell tag.
            if let cell = field.cell as? NSSearchFieldCell, let loupeCell = cell.searchButtonCell,
               loupeCell.tag != Self.accessoryTag,
               let loupe = NSImage(systemSymbolName: "magnifyingglass", accessibilityDescription: "Search") {
                let img = loupe.withSymbolConfiguration(.init(pointSize: 11, weight: .regular))
                loupeCell.image = img
                loupeCell.alternateImage = img
                loupeCell.tag = Self.accessoryTag
            }
            let b = NSButton(frame: NSRect(x: x, y: y, width: side, height: side))
            b.tag = Self.accessoryTag
            // 11pt glyph inside the 20pt hover target: the accessoryBarAction highlight capsule
            // gets visible breathing room around the icon, matching the (shrunk) magnifier.
            // accessoryBarAction + border-on-hover is the native in-field button treatment
            // (Spotlight's mic): a soft rounded highlight on hover, darker while pressed.
            b.image = icon.withSymbolConfiguration(.init(pointSize: 11, weight: .regular))
            b.bezelStyle = .accessoryBarAction
            b.isBordered = true
            b.showsBorderOnlyWhileMouseInside = true
            b.setButtonType(.momentaryPushIn)
            b.imagePosition = .imageOnly
            b.contentTintColor = .secondaryLabelColor
            b.target = self
            b.action = #selector(fireSearchByFile)
            b.toolTip = "Search by a file (image, audio, video, or text)  \u{21E7}\u{2318}O"
            b.setAccessibilityLabel("Search by a file")
            b.autoresizingMask = [.minXMargin]   // stay pinned to the trailing edge on resize
            field.addSubview(b)
        }

        @objc private func fireSearchByFile() { onSearchByFile?() }
    }

    func makeNSView(context: Context) -> NSView {
        let v = TunerView()
        v.onSearchByFile = onSearchByFile
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? TunerView)?.onSearchByFile = onSearchByFile
    }
}
