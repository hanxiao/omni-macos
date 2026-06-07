import SwiftUI
import AppKit
import OmniKit

struct ContentView: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var debounce: Task<Void, Never>?
    @State private var historyDebounce: Task<Void, Never>?

    // Progressive disclosure: only offer search once there is something to search. During model
    // loading, onboarding, and the no-folders state the search field stays hidden (not dimmed).
    private var showsSearch: Bool { model.phase == .ready && !model.roots.isEmpty }

    var body: some View {
        Group {
            if showsSearch {
                split
                    .searchable(text: Binding(get: { model.query }, set: { model.query = $0 }), placement: .toolbar, prompt: "Search your files by meaning")
                    .onChange(of: model.query) { _, _ in scheduleSearch(); scheduleHistoryRecord() }
                    .onSubmit(of: .search) { model.search() }
            } else {
                split
            }
        }
        // Spotlight-style: put the caret in the search field as soon as the app can search. The
        // macOS 15 .searchFocused API is unavailable on our 14 target, so focus the toolbar's
        // NSSearchField directly once it exists.
        .onChange(of: showsSearch, initial: true) { _, shows in if shows { focusSearchField() } }
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
        guard model.phase == .ready, !model.query.isEmpty, !model.isResolving else { return "" }
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

    @ViewBuilder private var content: some View {
        if !model.results.isEmpty {
            ResultsList(results: model.results) { belowThresholdFooter }
        } else {
            emptyState
        }
    }

    @ViewBuilder private var emptyState: some View {
        // Indexing is invisible here - the sidebar's per-folder progress is the only cue, and
        // search works while it runs. The user just adds folders and searches.
        if model.roots.isEmpty {
            CenteredStatus(symbol: "folder.badge.plus", title: "Add a folder to search",
                           subtitle: "Choose the folders you want to search. Omni indexes them automatically and keeps them up to date.",
                           showSpinner: false, action: ("Add Folder\u{2026}", { pickFolder() }))
        } else if model.query.isEmpty || model.isResolving {
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
        // Progressive disclosure: the filter/sort/view chrome appears only once there are results
        // to act on - hidden, not greyed out, during onboarding and the idle/empty states.
        // Exception: keep the filter menu reachable whenever a filter is active, so a filter that
        // hides every result can still be cleared (otherwise the menu vanishes with the results).
        if model.phase == .ready, !model.rawResults.isEmpty || model.filtersActive {
        // Filtering - one home.
        ToolbarItem(placement: .automatic) {
            filterMenu.disabled(model.indexedFiles == 0)
        }
        }
        // Result presentation - sort + view grouped together. Only meaningful with results.
        if model.phase == .ready, !model.rawResults.isEmpty {
        ToolbarItem(placement: .primaryAction) {
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

    // History records on a longer (2x) debounce than the search itself, so only a query the user
    // actually settled on is stored - not every transient keystroke.
    private func scheduleHistoryRecord() {
        historyDebounce?.cancel()
        historyDebounce = Task {
            try? await Task.sleep(nanoseconds: 360_000_000)
            if !Task.isCancelled { model.recordCurrentSearchToHistory() }
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { for url in panel.urls { model.addRoot(url) } }
    }
}

struct CenteredStatus: View {
    let symbol: String
    let title: String
    let subtitle: String
    var showSpinner: Bool = false
    var action: (String, () -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title)
            Text(subtitle).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 400)
            if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
            if let action {
                Button(action.0, action: action.1).controlSize(.large).buttonStyle(.borderedProminent).padding(.top, 4)
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
