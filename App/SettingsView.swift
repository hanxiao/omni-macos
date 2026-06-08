import SwiftUI
import AppKit
import OmniKit

struct SettingsView: View {
    var body: some View {
        TabView {
            ActivityTab().tabItem { Label("Indexing", systemImage: "arrow.triangle.2.circlepath") }
            ContentTypesTab().tabItem { Label("Content", systemImage: "square.grid.2x2") }
            PerformanceTab().tabItem { Label("Performance", systemImage: "speedometer") }
            IndexTab().tabItem { Label("Storage", systemImage: "externaldrive") }
            HistoryTab().tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
            ServingTab().tabItem { Label("Serving", systemImage: "network") }
        }
        // Size to the selected tab rather than forcing one height across five differently sized
        // panes (the Storage tab can show an out-of-date banner plus a Model section). Keeps the
        // first section header clear of the tab strip and removes dead space on short tabs.
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// Live indexing status and the manual Index / Reindex / Pause controls. This is the
/// single home for the detail that used to clutter the sidebar.
private struct ActivityTab: View {
    @Environment(AppModel.self) private var model: AppModel

    private var overall: Double {
        let rs = model.progress.perRoot.values
        let total = rs.reduce(0) { $0 + $1.total }
        guard total > 0 else { return 0 }
        return Double(rs.reduce(0) { $0 + $1.done }) / Double(total)
    }

    /// Aggregate done/total across the roots being indexed (a full pass or one or more folder-adds).
    private var activeCounts: (done: Int, total: Int) {
        let rs = model.progress.perRoot.values
        return (rs.reduce(0) { $0 + $1.done }, rs.reduce(0) { $0 + $1.total })
    }

    /// "12.3 files/sec · 45k tok/s" during a full pass, or "45k tok/s" during a background reconcile
    /// where there is no per-file count. nil when nothing is being embedded.
    private var rateLabel: String? {
        guard model.tokensPerSec > 0 else { return nil }
        let tok = model.tokensPerSec >= 1000 ? String(format: "%.1fk", model.tokensPerSec / 1000) : String(format: "%.0f", model.tokensPerSec)
        return model.filesPerSec > 0
            ? String(format: "%.1f files/sec \u{00B7} %@ tok/s", model.filesPerSec, tok)
            : "\(tok) tok/s"
    }

    var body: some View {
        Form {
            Section {
                switch model.indexState {
                case .indexing:
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            ProgressView().controlSize(.small)
                            Text(model.isPreparing ? "Preparing\u{2026}" : "Indexing\u{2026}").fontWeight(.medium)
                            Spacer()
                            if !model.isPreparing, let rateLabel {
                                Text(rateLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            Button("Pause") { model.pauseIndexing() }.controlSize(.small)
                        }
                        if model.isPreparing {
                            // No file processed yet: scanning folders / warming up the model. Show an
                            // explanation rather than a 0% bar that looks frozen.
                            Text("Scanning your folders and warming up the model\u{2026}")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            ProgressView(value: overall)
                            HStack {
                                Text("\(model.progress.embedded) added")
                                if model.progress.unchanged > 0 { Text("\u{00B7} \(model.progress.unchanged) up to date") }
                                if model.progress.skipped > 0 { Text("\u{00B7} \(model.progress.skipped) skipped") }
                                if model.progress.failed > 0 { Text("\u{00B7} \(model.progress.failed) failed") }
                            }
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            Text(URL(fileURLWithPath: model.progress.currentPath).lastPathComponent)
                                .font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
                        }
                    }
                case .paused:
                    HStack(spacing: 8) {
                        Image(systemName: "pause.circle.fill").foregroundStyle(.orange)
                        Text("Paused \u{00B7} \(model.indexedFiles.formatted()) files indexed")
                        Spacer()
                        Button("Resume") { model.startIndexing() }.controlSize(.small)
                    }
                case .idle:
                    if !model.activeRoots.isEmpty {
                        // A newly added folder (or a background reconcile) is embedding right now.
                        // It tracks per-root totals just like a full pass, so show the same progress.
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                ProgressView().controlSize(.small)
                                Text("Updating\u{2026}").fontWeight(.medium)
                                Spacer()
                                if let rateLabel {
                                    Text(rateLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                }
                            }
                            if activeCounts.total > 0 {
                                ProgressView(value: overall)
                                Text("\(activeCounts.done.formatted()) / \(activeCounts.total.formatted()) files")
                                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                            Text(model.indexedFiles == 0 ? "Nothing indexed yet" : "Up to date \u{00B7} \(model.indexedFiles.formatted()) files")
                            Spacer()
                            Button(model.indexedFiles == 0 ? "Index" : "Update") { model.startIndexing() }
                                .controlSize(.small).disabled(!model.canIndex)
                        }
                    }
                }
            } header: {
                Text("Status")
            } footer: {
                Text("Stays current automatically as files change. Update checks now.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    let rp = model.progress.perRoot[url.path]
                    HStack {
                        Image(systemName: "folder").foregroundStyle(.secondary)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if let rp, rp.total > 0, rp.done < rp.total,
                           model.isIndexing || model.activeRoots.contains(url.path) {
                            Text("\(rp.done.formatted()) / \(rp.total.formatted())")
                                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        } else if model.activeRoots.contains(url.path) {
                            Text("Updating\u{2026}").font(.caption).foregroundStyle(.secondary)
                        } else if let c = model.folderFileCounts[url.path] {
                            Text("\(c.formatted()) files").font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}

private struct ContentTypesTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var selectedKind: FileKind = .image
    @State private var extFilter = ""

    private var visibleExtensions: [String] {
        let all = FileExtractor.extensions(for: selectedKind)
        let q = extFilter.trimmingCharacters(in: .whitespaces).lowercased()
        return q.isEmpty ? all : all.filter { $0.contains(q) }
    }

    var body: some View {
        // Same grouped Form as every other Settings tab (bold section titles, rounded inset cards),
        // height-capped so the long Text extension set scrolls inside the pane instead of growing it.
        let kindOff = !model.settings.contains(selectedKind)
        Form {
            Section {
                ForEach(model.kindOrder, id: \.self) { kind in
                    kindToggle(kind, kind.title)
                        .draggable(kind.rawValue)
                        .dropDestination(for: String.self) { items, _ in
                            guard let raw = items.first, let dragged = FileKind(rawValue: raw) else { return false }
                            model.moveKind(dragged, before: kind)
                            return true
                        }
                }
            } header: {
                Text("What to Index")
            } footer: {
                Text("Drag to set which types are indexed first.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                Picker("Minimum image size", selection: Binding(get: { model.minImageDimension }, set: { model.minImageDimension = $0 })) {
                    Text("No minimum").tag(0)
                    Text("64 px").tag(64)
                    Text("128 px").tag(128)
                    Text("256 px").tag(256)
                    Text("512 px").tag(512)
                }
                Picker("Minimum audio length", selection: Binding(get: { model.minAudioSeconds }, set: { model.minAudioSeconds = $0 })) {
                    Text("No minimum").tag(0.0)
                    Text("1 second").tag(1.0)
                    Text("3 seconds").tag(3.0)
                    Text("5 seconds").tag(5.0)
                    Text("10 seconds").tag(10.0)
                }
                Picker("Minimum video length", selection: Binding(get: { model.minVideoSeconds }, set: { model.minVideoSeconds = $0 })) {
                    Text("No minimum").tag(0.0)
                    Text("1 second").tag(1.0)
                    Text("3 seconds").tag(3.0)
                    Text("5 seconds").tag(5.0)
                    Text("10 seconds").tag(10.0)
                }
                Picker("Minimum text length", selection: Binding(get: { model.minTextChars }, set: { model.minTextChars = $0 })) {
                    Text("No minimum").tag(0)
                    Text("16 characters").tag(16)
                    Text("64 characters").tag(64)
                    Text("256 characters").tag(256)
                }
            } header: {
                Text("Skip Small Files")
            } footer: {
                Text("Skips files below these sizes: icons, thumbnails, very short clips.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                Picker("Type", selection: $selectedKind) {
                    Text("Images").tag(FileKind.image)
                    Text("Video").tag(FileKind.video)
                    Text("Audio").tag(FileKind.audio)
                    Text("Text").tag(FileKind.text)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Filter", text: $extFilter).textFieldStyle(.plain)
                    if !extFilter.isEmpty {
                        Button { extFilter = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).foregroundStyle(.tertiary)
                    }
                    // Scoped to the selected type only, so it lives with that type's list.
                    let allOn = !visibleExtensions.isEmpty && visibleExtensions.allSatisfy { model.isExtensionEnabled($0) }
                    Button(allOn ? "Disable All" : "Enable All") {
                        model.setExtensionsEnabled(visibleExtensions, !allOn)
                    }
                    .buttonStyle(.link)
                    .disabled(kindOff || visibleExtensions.isEmpty)
                }

                ForEach(visibleExtensions, id: \.self) { ext in
                    Toggle(isOn: Binding(
                        get: { model.isExtensionEnabled(ext) },
                        set: { model.setExtensionEnabled(ext, $0) }
                    )) {
                        Text(".\(ext)").font(.body.monospaced())
                    }
                    .toggleStyle(.checkbox)
                    .disabled(kindOff)
                }
            } header: {
                Text("Extensions")
            } footer: {
                Text("Turning an extension off removes those files from the index.")
            }
        }
        .formStyle(.grouped)
        .frame(height: 520)   // matches the Serving tab so switching tall tabs doesn't jump
    }

    @ViewBuilder private func kindToggle(_ k: FileKind, _ label: String) -> some View {
        let off = (k == .audio && !model.audioSupported)
        Toggle(isOn: Binding(get: { model.settings.contains(k) }, set: { model.setIndexKind(k, $0) })) {
            HStack(spacing: 8) {
                // Drag-handle affordance: signals the row can be dragged to reorder.
                Image(systemName: "line.3.horizontal").foregroundStyle(.tertiary).font(.callout)
                Label(label, systemImage: k.symbol)
            }
        }
        .toggleStyle(.switch)
        .disabled(off)
    }
}

private struct PerformanceTab: View {
    @Environment(AppModel.self) private var model: AppModel
    private var memoryCeiling: Double { max(8, min(model.physicalMemoryGB.rounded(), 128)) }
    var body: some View {
        Form {
            Section {
                Picker("Max image size", selection: Binding(get: { model.maxImageDimension }, set: { model.maxImageDimension = $0 })) {
                    Text("1024 px").tag(1024)
                    Text("1280 px").tag(1280)
                    Text("1568 px \u{00B7} recommended").tag(1568)
                    Text("2048 px").tag(2048)
                }
                Picker("Max frames per video", selection: Binding(get: { model.maxVideoFrames }, set: { model.maxVideoFrames = $0 })) {
                    Text("3").tag(3)
                    Text("6 \u{00B7} recommended").tag(6)
                    Text("9").tag(9)
                    Text("18").tag(18)
                }
                Picker("Max characters per chunk", selection: Binding(get: { model.maxTextChunkChars }, set: { model.maxTextChunkChars = $0 })) {
                    Text("1200").tag(1200)
                    Text("1800").tag(1800)
                    Text("2400").tag(2400)
                    Text("3600").tag(3600)
                }
            } header: {
                Text("Throughput")
            } footer: {
                Text("Smaller caps trade some detail for faster indexing.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Maximum memory")
                        Spacer()
                        Text(model.maxMemoryGB == 0 ? "Unlimited" : "\(Int(model.maxMemoryGB)) GB")
                            .foregroundStyle(.secondary).monospacedDigit()
                    }
                    Slider(value: Binding(
                        get: { model.maxMemoryGB },
                        set: { model.maxMemoryGB = $0.rounded() }
                    ), in: 0 ... memoryCeiling) {
                        Text("Maximum memory")
                    } minimumValueLabel: {
                        Text("Off").font(.caption2).foregroundStyle(.tertiary)
                    } maximumValueLabel: {
                        Text("\(Int(memoryCeiling))").font(.caption2).foregroundStyle(.tertiary)
                    }
                    .labelsHidden()
                }
            } header: {
                Text("Memory")
            } footer: {
                Text("Caps memory the model may use. Keep it above ~4 GB; 0 means unlimited.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                LabeledContent("Benchmark this Mac") {
                    Button("Run Profiling\u{2026}") { Task { await model.runProfiling() } }
                        .controlSize(.small)
                        .disabled(model.isProfilingRunning || !model.canIndex)
                }
                Toggle(isOn: Binding(get: { model.shareProfilingResults }, set: { model.shareProfilingResults = $0 })) {
                    Text("Share results")
                }
                if let r = model.lastProfilingReport {
                    LabeledContent("Last run") {
                        Text(String(format: "%.0f files/sec \u{00B7} %.1f GB peak VRAM",
                                    r.metrics.filesPerSec, Double(r.metrics.peakVramDeltaBytes) / 1_073_741_824))
                            .foregroundStyle(.secondary).monospacedDigit()
                    }
                }
            } header: {
                Text("Profiling")
            } footer: {
                Text("Benchmarks indexing on a fixed 1,000-file dataset. Sharing sends hardware and timing only - never your files - to the public results on hanxiao.io/omni.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

/// Search History preferences - what gets remembered, for how long, and a way to clear it.
/// Mirrors how macOS surfaces recents/Smart Folders: an explicit recording mode, a time window,
/// and a destructive clear that spares the user's explicit bookmarks.
private struct HistoryTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var confirmClear = false
    var body: some View {
        Form {
            Section {
                Picker("Add searches to History", selection: Binding(get: { model.historyMode }, set: { model.historyMode = $0 })) {
                    ForEach(HistoryMode.allCases) { Text($0.title).tag($0) }
                }
                .help("Choose when a search is remembered in the sidebar")
            } footer: {
                Text(model.historyMode.detail).font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Picker("Keep history for", selection: Binding(get: { model.historyRetentionDays }, set: { model.historyRetentionDays = $0 })) {
                    Text("3 days").tag(3)
                    Text("7 days").tag(7)
                    Text("14 days").tag(14)
                    Text("31 days").tag(31)
                }
                .help("How long recent searches are kept before they are removed")
            } footer: {
                Text("Recent searches older than this are removed automatically. Bookmarked searches are always kept.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                LabeledContent("Saved searches") {
                    Text("\(model.recentHistoryCount) recent \u{00B7} \(model.bookmarkCount) bookmarked")
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                Button("Clear Search History\u{2026}", role: .destructive) { confirmClear = true }
                    .disabled(model.recentHistoryCount == 0)
                    .help("Remove all recent searches. Your bookmarks are kept.")
            } footer: {
                Text("Clearing removes recent searches from the sidebar. Your bookmarks stay.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("Clear all recent searches?", isPresented: $confirmClear) {
            Button("Clear Search History", role: .destructive) { model.clearSearchHistory() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your bookmarked searches will be kept.")
        }
    }
}

private struct IndexTab: View {
    @Environment(AppModel.self) private var model: AppModel
    var body: some View {
        Form {
            if model.indexObsolete {
                Section {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Index doesn't match the loaded model").fontWeight(.medium)
                            if let v = model.indexBuiltVariant {
                                Text("Built with \(v.title) (\(model.indexStoredDim)-dim); \(model.modelVariant.title) is loaded. Switch back to keep this index, or reindex with the current model.")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else {
                                Text("It was built with an older embedding version. Reindex so results stay accurate.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                if let v = model.indexBuiltVariant {
                                    Button("Switch to \(v.title)") { model.selectVariant(v) }.disabled(model.isDownloading)
                                }
                                Button("Reindex") { model.startIndexing() }
                                    .disabled(model.isIndexing || !model.canIndex)
                            }
                            .controlSize(.small)
                        }
                    }
                }
            }
            Section("Index") {
                LabeledContent("Indexed files", value: "\(model.indexedFiles)")
                LabeledContent("Embeddings", value: "\(model.indexedChunks)")
                LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: model.dbSizeBytes, countStyle: .file))
                if let last = model.lastIndexed {
                    LabeledContent("Last indexed", value: last.formatted(.relative(presentation: .named)))
                }
            }
            Section {
                // Selecting a variant switches to it if installed, or downloads it if not - no
                // separate download button.
                Picker("Model", selection: Binding(
                    get: { model.modelVariant },
                    set: { model.selectVariant($0) }
                )) {
                    ForEach(ModelVariant.allCases, id: \.self) { v in
                        Text(model.installedVariants[v] != nil ? v.title : "\(v.title) \u{00B7} download")
                            .tag(v)
                    }
                }
                .disabled(model.isDownloading || model.isIndexing)

                if model.isDownloading {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: model.downloadFraction)
                        Text(model.downloadLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                } else if !model.modelPath.isEmpty {
                    Text(model.modelPath).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(3).truncationMode(.middle).textSelection(.enabled)
                    HStack {
                        Button("Change\u{2026}") { pickModel() }
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.modelPath)])
                        }
                    }
                    .controlSize(.small)
                }
            } header: {
                Text("Model")
            } footer: {
                Text("Pick a variant to switch to it or download it. Switching rebuilds the index - the two models use different embeddings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                if !model.dbPath.isEmpty {
                    Text(model.dbPath).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(3).truncationMode(.middle).textSelection(.enabled)
                }
                HStack {
                    Button("Change\u{2026}") { pickDatabase() }
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dbPath)])
                    }
                    .disabled(model.dbPath.isEmpty)
                }
                .controlSize(.small)
            } header: {
                Text("Database Location")
            } footer: {
                Text("Where the search index is stored. Changing the folder loads the index from there.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "Choose the model folder (model.safetensors, config.json, tokenizer.json)"
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
    private func pickDatabase() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "Choose a folder to store the search index"
        if panel.runModal() == .OK, let url = panel.url { model.setDatabaseDir(url) }
    }
}
