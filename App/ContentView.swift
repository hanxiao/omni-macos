import SwiftUI
import OmniKit

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var debounce: Task<Void, Never>?

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 240, ideal: 270, max: 340)
        } detail: {
            detail
        }
    }

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            CenteredStatus(symbol: "brain", title: "Loading omni model", subtitle: "Bringing up the MLX engine...", showSpinner: true)
        case .noModel:
            OnboardingView()
        case .failed(let msg):
            CenteredStatus(symbol: "exclamationmark.triangle", title: "Engine failed to load", subtitle: msg, showSpinner: false)
        case .ready:
            searchSurface
        }
    }

    private var searchSurface: some View {
        VStack(spacing: 0) {
            searchField
            if model.indexedFiles > 0 { FilterBar() }
            Divider()
            if model.results.isEmpty {
                emptyResults
            } else {
                ResultsList(results: model.results)
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.title3)
            TextField("Search your files by meaning", text: $model.query)
                .textFieldStyle(.plain)
                .font(.title3)
                .onSubmit { model.search() }
                .onChange(of: model.query) { _, _ in scheduleSearch() }
            if model.searching {
                ProgressView().controlSize(.small)
            } else if !model.query.isEmpty {
                Button { model.query = ""; model.results = [] } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func scheduleSearch() {
        debounce?.cancel()
        debounce = Task {
            try? await Task.sleep(nanoseconds: 180_000_000)
            if !Task.isCancelled { model.search() }
        }
    }

    private var emptyResults: some View {
        Group {
            if model.indexedFiles == 0 {
                CenteredStatus(
                    symbol: "square.stack.3d.up",
                    title: "Nothing indexed yet",
                    subtitle: "Index your folders to start searching.",
                    showSpinner: false,
                    action: ("Index now", { model.startIndexing() }))
            } else if model.query.isEmpty {
                CenteredStatus(
                    symbol: "sparkle.magnifyingglass",
                    title: "Search \(model.indexedFiles) files",
                    subtitle: "Type a phrase. Results are ranked by meaning, not keywords.",
                    showSpinner: false)
            } else {
                CenteredStatus(symbol: "magnifyingglass", title: "No matches", subtitle: "Try a different phrase.", showSpinner: false)
            }
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
            Image(systemName: symbol)
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.tertiary)
            Text(title).font(.title2).fontWeight(.semibold)
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
            if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
            if let action {
                Button(action.0, action: action.1)
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
