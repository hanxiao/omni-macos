import SwiftUI
import AppKit
import OmniKit

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
        if !model.query.isEmpty, model.fileQuery != nil { model.fileQuery = nil; model.queryError = nil }
        if model.fileQuery == nil { scheduleSearch() }
        scheduleHistoryRecord()
    }

    var body: some View {
        Group {
            if showsSearch {
                split
                    .searchable(text: Binding(get: { model.rawQuery }, set: { handleQueryEdit($0) }),
                                placement: .toolbar, prompt: "Search by meaning") {
                        // Typeahead: keys (ty -> type:), values (type: -> image/...), and matching past
                        // queries as instant (cached) shortcuts. Navigate with arrows + Return.
                        ForEach(searchSuggestions(model.rawQuery), id: \.completion) { sug in
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
                .navigationTitle("Omni")
                .navigationSubtitle(subtitle)
                .toolbar { toolbar }
        }
    }

    private var subtitle: String {
        guard model.phase == .ready, model.hasQuery, !model.isResolving else { return "" }
        let n = model.results.count
        if n == 0 { return "" }
        return n >= 60 ? "Top \(n) results" : "\(n) result\(n == 1 ? "" : "s")"
    }

    // MARK: - Detail

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            CenteredStatus(symbol: "brain", title: "Loading the Omni model", subtitle: "Starting the on-device model\u{2026}", showSpinner: true)
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
        // Drop a supported file from Finder anywhere on the content to search by it.
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first(where: { FileExtractor.kind(for: $0) != nil }) else { return false }
            model.setFileQuery(url)
            return true
        } isTargeted: { fileDropTargeted = $0 }
        .overlay {
            if fileDropTargeted {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: 2).padding(6).allowsHitTesting(false)
            }
        }
    }

    @ViewBuilder private var emptyState: some View {
        // Indexing is invisible here - the sidebar's per-folder progress is the only cue, and
        // search works while it runs. The user just adds folders and searches.
        if model.roots.isEmpty {
            CenteredStatus(symbol: "folder.badge.plus", title: "Add a folder to search",
                           subtitle: "Choose the folders you want to search. Omni indexes them automatically and keeps them up to date.",
                           showSpinner: false, action: ("Add Folder\u{2026}", { pickFolder() }))
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
                               ? "This index was built with \(built!.title) (\(model.indexStoredDim)-dim) but \(model.modelVariant.title) is loaded. Switch back to keep your index, or reindex with the current model."
                               : "This index was built with a different model than the one loaded. Reindex to search again.",
                           showSpinner: false,
                           action: built.map { v in ("Switch to \(v.title)", { model.selectVariant(v) }) },
                           secondary: ("Reindex", { model.startIndexing() }))
        } else if !model.hasQuery || model.isResolving {
            // Idle prompt, and the in-flight search state. They share one calm placeholder so a
            // pending search only fades a small spinner in under the same prompt - it never flashes
            // "No matches" while the debounce/search for what you just typed is still running.
            CenteredStatus(symbol: "sparkle.magnifyingglass",
                           title: model.indexedFiles > 0 ? "Search \(model.indexedFiles.formatted()) files" : "Search your files",
                           subtitle: "Type a phrase. Results are ranked by meaning, across images, video, audio, and text.",
                           showSpinner: model.isResolving)
        } else if model.hiddenByThreshold > 0 {
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle",
                           title: "No results above \(Int(model.minScore * 100))%",
                           subtitle: "\(model.hiddenByThreshold) weaker \(model.hiddenByThreshold == 1 ? "match is" : "matches are") hidden by the relevance threshold.",
                           showSpinner: false, action: ("Show All Matches", { model.showAllBelowThreshold() }))
        } else if model.filtersActive {
            // Filters can hide every result; the empty state is the only place left to escape them.
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle", title: "No matches",
                           subtitle: "Filters are hiding every result.", showSpinner: false,
                           action: ("Clear Filters", { model.clearFilters() }))
        } else {
            CenteredStatus(symbol: "magnifyingglass", title: "No matches", subtitle: "Try a different phrase.", showSpinner: false)
        }
    }

    @ViewBuilder private var belowThresholdFooter: some View {
        if model.hiddenByThreshold > 0 {
            Button { model.showAllBelowThreshold() } label: {
                Label("Show \(model.hiddenByThreshold) More Matches", systemImage: "chevron.down")
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
        // macOS 26 (Tahoe) shows the NavigationSplitView sidebar toggle automatically; macOS 15 and
        // earlier don't, so add an explicit one there (toggleSidebar: travels the responder chain to
        // the split view controller backing NavigationSplitView).
        if #unavailable(macOS 26.0) {
            ToolbarItem(placement: .navigation) {
                Button { NSApp.sendAction(Selector(("toggleSidebar:")), to: nil, from: nil) } label: {
                    Image(systemName: "sidebar.left")
                }
                .help("Show or hide the sidebar")
            }
        }
        // Search by a file (any modality - the embedding space is shared). Available whenever the
        // app can search, since it can start a query from the empty state too.
        if model.phase == .ready {
            ToolbarItem(placement: .automatic) {
                Button { pickFile() } label: { Image(systemName: "photo.badge.magnifyingglass") }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    .help("Search by a file (image, audio, video, or text)  \u{21E7}\u{2318}O")
                    .accessibilityLabel("Search by a File")
            }
        }
        // Bookmark the current search. The only way into History when recording is set to "Only when
        // I bookmark", and a quick save otherwise. Appears once there's a search to keep.
        if model.phase == .ready, model.hasActiveSearch {
            ToolbarItem(placement: .automatic) {
                Button { model.toggleBookmarkCurrentSearch() } label: {
                    Image(systemName: model.currentSearchIsBookmarked ? "star.fill" : "star")
                        .foregroundStyle(model.currentSearchIsBookmarked ? Color.yellow : Color.primary)
                }
                // Cmd-D is owned by the File-menu "Bookmark Search" command (single owner, avoids a
                // duplicate-shortcut conflict); the tooltip names it, and accessibilityLabel is what
                // VoiceOver reads and what the toolbar-overflow menu shows for this icon-only button.
                .help(model.currentSearchIsBookmarked ? "Remove bookmark  \u{2318}D" : "Bookmark this search  \u{2318}D")
                .accessibilityLabel(model.currentSearchIsBookmarked ? "Remove Bookmark" : "Bookmark Search")
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
                        Picker("Sort By", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                            ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                        }
                    } label: { Image(systemName: "arrow.up.arrow.down") }
                    .help("Sort by \(model.sortOrder.title)")

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
                    Picker("Sort By", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                        ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                    }
                } label: {
                    Label("View Options", systemImage: "slider.horizontal.3")
                }
                .help("Sort and view")
            }
        }
        }
    }

    private var filterKinds: [FileKind] {
        // Show indexed kinds, plus any kind currently being filtered on - otherwise a filter for a
        // kind that is not (yet) in the index would be invisible and impossible to untoggle.
        let present = FileKind.allCases.filter { model.indexedKinds.contains($0.rawValue) || model.filterKinds.contains($0) }
        return present.isEmpty ? [.image, .video, .audio] : present
    }

    private var filterMenu: some View {
        Menu {
            Section("Show") {
                ForEach(filterKinds, id: \.self) { kind in
                    Toggle(isOn: Binding(
                        get: { model.filterKinds.contains(kind) },
                        set: { on in if on { model.filterKinds.insert(kind) } else { model.filterKinds.remove(kind) } }
                    )) { Label(kind.title, systemImage: kind.symbol) }
                }
            }
            Picker("Folder", selection: Binding(
                get: { model.filterFolder?.path ?? "" },
                set: { model.filterFolder = $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
            )) {
                Text("All Folders").tag("")
                ForEach(model.roots, id: \.self) { Text($0.lastPathComponent).tag($0.path) }
            }
            Picker("Extension", selection: Binding(get: { model.filterExt }, set: { model.filterExt = $0 })) {
                Text("Any Extension").tag("")
                ForEach(model.indexedExts, id: \.self) { Text(".\($0)").tag($0) }
            }
            Picker("Date", selection: Binding(get: { model.dateRange }, set: { model.dateRange = $0 })) {
                ForEach(DateRange.allCases) { Text($0.title).tag($0) }
            }
            Picker("Relevance", selection: Binding(get: { model.minScore }, set: { model.minScore = $0 })) {
                Text("Any").tag(0.0); Text("25%").tag(0.25); Text("50%").tag(0.5); Text("70%").tag(0.7)
            }
            Divider()
            Button("Clear Filters") { model.clearFilters() }.disabled(!model.filtersActive)
        } label: {
            Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .help("Filter results")
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
                    for k in ["type:", "ext:", "in:", "date:", "after:", "score:", "sort:"] where k.hasPrefix(low) {
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
        case "type": return ["image", "video", "audio", "text"]
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
        if panel.runModal() == .OK { for url in panel.urls { model.addRoot(url) } }
    }

    private func pickFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Search"
        panel.message = "Choose an image, audio, video, or text file to search by"
        if panel.runModal() == .OK, let url = panel.url { model.setFileQuery(url) }
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
                Text("Literal search").foregroundStyle(.secondary)
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
    var action: (String, () -> Void)? = nil
    var secondary: (String, () -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title)
            Text(subtitle).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 400)
            if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
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

struct EngineFailedView: View {
    @Environment(AppModel.self) private var model: AppModel
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 40, weight: .light)).foregroundStyle(.tertiary)
            Text("Engine failed to load").font(.title2).fontWeight(.semibold)
            HStack {
                Button("Retry") { model.retryBootstrap() }.buttonStyle(.borderedProminent)
                Button("Choose Model Folder\u{2026}") { pickModel() }
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
