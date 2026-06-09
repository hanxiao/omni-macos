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
    @State private var draft = ""
    @State private var loaded = false
    @State private var showSamples = false
    @State private var previewTask: Task<Void, Never>?

    private var dirty: Bool { model.ignoreTextIsDirty(draft) }

    var body: some View {
        Form {
            Section {
                ForEach(model.kindOrder, id: \.self) { kind in
                    orderRow(kind)
                        .draggable(kind.rawValue)
                        .dropDestination(for: String.self) { items, _ in
                            guard let raw = items.first, let dragged = FileKind(rawValue: raw) else { return false }
                            model.moveKind(dragged, before: kind)
                            return true
                        }
                }
            } header: {
                Text("File Types")
            } footer: {
                Text("Turn a type off to stop indexing it and free its model from memory. Drag to set which types index first.")
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
                Text("Skips files below these sizes, like icons, thumbnails, and very short clips.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                IgnoreEditor(text: $draft)
                    .frame(minHeight: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))

                previewBar
            } header: {
                Text("Ignore Rules")
            } footer: {
                Text("Applied after the file-type switches above. One .gitignore pattern per line: a leading ! re-includes, a trailing / matches folders.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onAppear { if !loaded { draft = model.ignoreText; loaded = true } }
        .onChange(of: draft) { _, newValue in schedulePreview(newValue) }
        .confirmationDialog(
            model.pendingDisable.map { "Stop indexing \($0.kind.title.lowercased())?" } ?? "",
            isPresented: Binding(get: { model.pendingDisable != nil }, set: { if !$0 { model.pendingDisable = nil } }),
            presenting: model.pendingDisable
        ) { pd in
            Button("Remove \(pd.count) from index", role: .destructive) { model.applyKind(pd.kind, on: false, purge: true) }
            Button("Keep in index") { model.applyKind(pd.kind, on: false, purge: false) }
            Button("Cancel", role: .cancel) { model.pendingDisable = nil }
        } message: { pd in
            Text("\(pd.count) \(pd.kind.title.lowercased()) \(pd.count == 1 ? "file is" : "files are") already indexed. Remove them now, or keep them searchable and just stop indexing new ones.")
        }
    }

    /// One modality: drag handle (reorder = index priority) + an on/off switch. Off skips the kind
    /// AND unloads its model from memory; the ignore rules below filter further within what stays on.
    @ViewBuilder private func orderRow(_ k: FileKind) -> some View {
        let on = model.kindEnabled(k)
        HStack(spacing: 8) {
            Image(systemName: "line.3.horizontal").foregroundStyle(.tertiary).font(.callout)
            Label(k.title, systemImage: k.symbol)
            Spacer()
            Toggle("", isOn: Binding(get: { model.kindEnabled(k) }, set: { model.toggleKind(k, on: $0) }))
                .labelsHidden().toggleStyle(.switch).controlSize(.mini)
        }
        .opacity(on ? 1 : 0.55)
    }

    @ViewBuilder private var previewBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let d = model.ignorePreview?.danger {
                Label(d, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 12) {
                if let p = model.ignorePreview {
                    Text("\(p.kept.formatted()) kept")
                        .foregroundStyle(.secondary)
                    Text("\(p.removed.formatted()) removed")
                        .foregroundStyle(p.removed > 0 ? .orange : .secondary)
                    if !p.samples.isEmpty {
                        Button("Show samples") { showSamples = true }
                            .buttonStyle(.link)
                            .popover(isPresented: $showSamples, arrowEdge: .bottom) { samplePopover(p.samples) }
                    }
                } else if dirty {
                    Text("Calculating...").foregroundStyle(.secondary)
                } else {
                    Text("These rules are applied.").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Import\u{2026}") { importIgnoreFile() }
                    .help("Load patterns from a file on disk into the editor.")
                if model.ignoreHasBackup {
                    Button("Revert") {
                        model.revertIgnore()
                        draft = model.ignoreText
                    }
                    .help("Undo the last applied change.")
                }
                Button("Apply") {
                    previewTask?.cancel()
                    model.applyIgnoreText(draft)
                }
                .keyboardShortcut("s", modifiers: .command)
                .buttonStyle(.borderedProminent)
                .disabled(!dirty)
            }
            .font(.callout)
        }
    }

    @ViewBuilder private func samplePopover(_ samples: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Files this removes (sample)")
                .font(.caption.bold()).foregroundStyle(.secondary)
            ForEach(samples, id: \.self) { path in
                Text((path as NSString).abbreviatingWithTildeInPath)
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1).truncationMode(.middle)
            }
        }
        .padding(12)
        .frame(width: 360, alignment: .leading)
    }

    /// Load an ignore file from disk into the editor draft (Apply still commits it).
    private func importIgnoreFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.message = "Choose a .omniignore or text file of ignore patterns"
        if panel.runModal() == .OK, let url = panel.url, let text = try? String(contentsOf: url, encoding: .utf8) {
            draft = text
        }
    }

    /// Debounce the dry-run so we don't query the index on every keystroke.
    private func schedulePreview(_ text: String) {
        previewTask?.cancel()
        previewTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            model.previewIgnore(text)
        }
    }
}

/// Plain-text editor (NSTextView) for the .omniignore: monospaced, with every smart substitution
/// disabled so glob patterns are typed literally (no curly quotes, em-dashes, or autocorrect).
private struct IgnoreEditor: NSViewRepresentable {
    @Binding var text: String

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.borderType = .noBorder
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        guard let tv = scroll.documentView as? NSTextView else { return scroll }
        tv.delegate = context.coordinator
        tv.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        tv.isRichText = false
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.isContinuousSpellCheckingEnabled = false
        tv.isGrammarCheckingEnabled = false
        tv.allowsUndo = true
        tv.textContainerInset = NSSize(width: 6, height: 8)
        tv.drawsBackground = false
        tv.string = text
        return scroll
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let tv = nsView.documentView as? NSTextView, tv.string != text else { return }
        tv.string = text
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, NSTextViewDelegate {
        let parent: IgnoreEditor
        init(_ parent: IgnoreEditor) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            parent.text = tv.string
        }
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
                Picker("Folder map layout", selection: Binding(get: { model.mapUsesUMAP }, set: { model.mapUsesUMAP = $0 })) {
                    Text("Fast \u{00B7} PCA").tag(false)
                    Text("Detailed \u{00B7} UMAP").tag(true)
                }
            } header: {
                Text("Folder map")
            } footer: {
                Text("PCA is instant and light, the safe default. UMAP separates clusters better and enables click-to-spotlight of nearest neighbors. Large folders are sampled to stay within the memory cap below.")
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
                Text("Caps memory for the model and the folder map. Keep it above ~4 GB; 0 means unlimited.")
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
