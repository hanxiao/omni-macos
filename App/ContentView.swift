import SwiftUI
import OmniKit

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var debounce: Task<Void, Never>?

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 230, ideal: 260, max: 320)
        } detail: {
            detail
                .navigationTitle("Omni")
                .navigationSubtitle(subtitle)
                .toolbar { toolbar }
        }
        .searchable(text: $model.query, placement: .toolbar, prompt: "Search your files by meaning")
        .onChange(of: model.query) { _, _ in scheduleSearch() }
        .onSubmit(of: .search) { model.search() }
    }

    private var subtitle: String {
        guard model.phase == .ready, !model.query.isEmpty, !model.searching else { return "" }
        let n = model.results.count
        if n == 0 { return "" }
        return n >= 60 ? "Top \(n) results" : "\(n) result\(n == 1 ? "" : "s")"
    }

    // MARK: - Detail

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            CenteredStatus(symbol: "brain", title: "Loading omni model", subtitle: "Bringing up the MLX engine...", showSpinner: true)
        case .noModel:
            OnboardingView()
        case .failed(let msg):
            EngineFailedView(message: msg)
        case .ready:
            ready
        }
    }

    @ViewBuilder private var ready: some View {
        VStack(spacing: 0) {
            if model.indexedFiles > 0 { FilterBar(); Divider() }
            content
        }
    }

    @ViewBuilder private var content: some View {
        if !model.results.isEmpty {
            ResultsList(results: model.results) { belowThresholdFooter }
        } else {
            emptyState
        }
    }

    @ViewBuilder private var emptyState: some View {
        if model.isIndexing {
            CenteredStatus(symbol: "circle.dotted", title: "Indexing\u{2026}",
                           subtitle: "\(model.progress.embedded) files added so far.", showSpinner: true)
        } else if model.indexedFiles == 0 {
            CenteredStatus(symbol: "square.stack.3d.up", title: "Nothing indexed yet",
                           subtitle: "Index your folders to start searching.", showSpinner: false,
                           action: ("Index", { model.startIndexing() }))
        } else if model.query.isEmpty {
            CenteredStatus(symbol: "sparkle.magnifyingglass", title: "Search \(model.indexedFiles) files",
                           subtitle: "Type a phrase. Results are ranked by meaning, across images, video, audio, and text.", showSpinner: false)
        } else if model.hiddenByThreshold > 0 {
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle",
                           title: "No results above \(Int(model.minScore * 100))%",
                           subtitle: "\(model.hiddenByThreshold) weaker match\(model.hiddenByThreshold == 1 ? "" : "es") are hidden by the relevance threshold.",
                           showSpinner: false, action: ("Show All Matches", { model.showAllBelowThreshold() }))
        } else {
            CenteredStatus(symbol: "magnifyingglass", title: "No matches", subtitle: "Try a different phrase.", showSpinner: false)
        }
    }

    @ViewBuilder private var belowThresholdFooter: some View {
        if model.hiddenByThreshold > 0 {
            Button { model.showAllBelowThreshold() } label: {
                Label("Show \(model.hiddenByThreshold) more below \(Int(model.minScore * 100))%", systemImage: "chevron.down")
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
        ToolbarItem(placement: .primaryAction) {
            SettingsLink { Image(systemName: "gearshape") }
                .help("Settings (\u{2318},)")
        }
        ToolbarItem(placement: .primaryAction) {
            Menu {
                Picker("Sort By", selection: $model.sortOrder) {
                    ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                }
            } label: { Image(systemName: "arrow.up.arrow.down") }
            .help("Sort results")
            .disabled(model.rawResults.isEmpty)
        }
        ToolbarItem(placement: .primaryAction) {
            Picker("View", selection: $model.viewMode) {
                Image(systemName: "list.bullet").tag(ResultViewMode.list)
                Image(systemName: "square.grid.2x2").tag(ResultViewMode.grid)
            }
            .pickerStyle(.segmented)
            .help("View as list or gallery")
        }
        ToolbarItem(placement: .primaryAction) {
            switch model.indexState {
            case .indexing:
                Button { model.pauseIndexing() } label: { Image(systemName: "pause.circle") }
                    .help("Pause indexing")
            case .paused:
                Button { model.startIndexing() } label: { Image(systemName: "play.circle") }
                    .help("Resume indexing")
            case .idle:
                Button { model.startIndexing() } label: { Image(systemName: "arrow.clockwise") }
                    .help("Index")
                    .disabled(!model.canIndex)
            }
        }
    }

    private func scheduleSearch() {
        debounce?.cancel()
        debounce = Task {
            try? await Task.sleep(nanoseconds: 180_000_000)
            if !Task.isCancelled { model.search() }
        }
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
            Image(systemName: symbol).font(.system(size: 40, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title2).fontWeight(.semibold)
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
    @EnvironmentObject var model: AppModel
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 40, weight: .light)).foregroundStyle(.tertiary)
            Text("Engine failed to load").font(.title2).fontWeight(.semibold)
            HStack {
                Button("Retry") { model.retryBootstrap() }.buttonStyle(.borderedProminent)
                Button("Choose Model Folder...") { pickModel() }
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
