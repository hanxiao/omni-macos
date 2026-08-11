import SwiftUI
import AppKit
import OmniKit

private enum SettingsTab: Hashable { case files, content, performance, storage, history, serving }

struct SettingsView: View {
    // Selection is BOUND, not left to the TabView, purely so the live memory sampler can be gated
    // on "Performance is the visible tab". A SwiftUI TabView keeps a pane alive once it has been
    // visited, so .onAppear/.task alone would keep sampling forever after one visit.
    @State private var tab: SettingsTab = .files

    var body: some View {
        TabView(selection: $tab) {
            ActivityTab().tabItem { Label("Files", systemImage: "folder") }
                .tag(SettingsTab.files)
            ContentTypesTab().tabItem { Label("Content", systemImage: "square.grid.2x2") }
                .tag(SettingsTab.content)
            PerformanceTab(isVisible: tab == .performance).tabItem { Label("Performance", systemImage: "speedometer") }
                .tag(SettingsTab.performance)
            IndexTab().tabItem { Label("Storage", systemImage: "externaldrive") }
                .tag(SettingsTab.storage)
            HistoryTab().tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
                .tag(SettingsTab.history)
            ServingTab().tabItem { Label("Serving", systemImage: "network") }
                .tag(SettingsTab.serving)
        }
        // Size to the selected tab rather than forcing one height across five differently sized
        // panes (the Storage tab can show an out-of-date banner plus a Model section). Keeps the
        // first section header clear of the tab strip and removes dead space on short tabs.
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// Live indexing status and the manual Index / Pause / Update control. It sits at the top of
/// Storage > Index, next to the file and chunk counts it is changing - one place to see what the
/// index holds and whether it is still growing.
private struct IndexStatusRow: View {
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

    /// "12.3 files/sec · 45k tokens/sec" during a full pass, or "45k tokens/sec" during a background reconcile
    /// where there is no per-file count. nil when nothing is being embedded.
    private var rateLabel: String? {
        guard model.tokensPerSec > 0 else { return nil }
        let tok = model.tokensPerSec >= 1000 ? String(format: "%.1fk", model.tokensPerSec / 1000) : String(format: "%.0f", model.tokensPerSec)
        return model.filesPerSec > 0
            ? String(format: "%.1f files/sec \u{00B7} %@ tokens/sec", model.filesPerSec, tok)
            : "\(tok) tokens/sec"
    }

    var body: some View {
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
                    Text("Scanning folders, warming up the model\u{2026}")
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
                Text("Paused")
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
                    // No count here: the Indexed files row sits directly below.
                    Text(model.indexedFiles == 0 ? "Nothing indexed yet" : "Up to date")
                    Spacer()
                    Button(model.indexedFiles == 0 ? "Index" : "Update") { model.startIndexing() }
                        .controlSize(.small).disabled(!model.canIndex)
                }
            }
        }
    }
}

/// What Omni watches: which file types are indexed, tagging, and the folder list.
private struct ActivityTab: View {
    @Environment(AppModel.self) private var model: AppModel

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
                Text("File types")
            } footer: {
                Text("Turn off to stop indexing a type and free its model. Drag to reorder.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                Toggle("Generate tags", isOn: Binding(
                    get: { model.imageTagsEnabled },
                    set: { model.imageTagsEnabled = $0 }
                ))
                .toggleStyle(.switch)
            } header: {
                Text("Image & video tagging")
            } footer: {
                Text("A few words per photo, video, or scan (\"cat, couch, crib\"), on-device while indexing.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("iCloud") {
                Picker("Files not downloaded", selection: Binding(get: { model.skipDatalessFiles },
                                                                 set: { model.skipDatalessFiles = $0 })) {
                    Text("Skip").tag(true)
                    Text("Download and index").tag(false)
                }
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
                        } else if (model.activeRoots.contains(url.path) || model.isFolderQueued(url))
                                    && !((rp?.total ?? 0) > 0 && (rp?.done ?? 0) >= (rp?.total ?? 0)) {
                            // Counting, or waiting its turn behind another pass. There is no total
                            // to show yet, and the stored count for a folder nothing has crawled is
                            // a truthful "0 files" that reads as "this folder is empty".
                            ProgressView().controlSize(.small)
                                .help(model.isFolderQueued(url) ? "Waiting to be indexed" : "Counting files\u{2026}")
                        } else if let c = model.folderFileCounts[url.path] {
                            Text("\(c.formatted()) file\(c == 1 ? "" : "s")").font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        // The kind on/off toggles live in THIS tab (orderRow). A confirmationDialog only presents
        // while its host view is on screen, so the disable-confirmation must be attached HERE, next
        // to the toggles - when it lived on the Content tab, disabling a kind-with-files from the
        // Files tab set pendingDisable but the dialog never showed, so applyKind was never called
        // and the toggle silently did nothing.
        .confirmationDialog(
            model.pendingDisable.map { "Stop indexing \($0.kind.title.lowercased())?" } ?? "",
            isPresented: Binding(get: { model.pendingDisable != nil }, set: { if !$0 { model.pendingDisable = nil } }),
            presenting: model.pendingDisable
        ) { pd in
            Button("Remove \(pd.count) from index", role: .destructive) { model.applyKind(pd.kind, on: false, purge: true) }
            Button("Keep in index") { model.applyKind(pd.kind, on: false, purge: false) }
            Button("Cancel", role: .cancel) { model.pendingDisable = nil }
        } message: { pd in
            Text("\(pd.count) \(pd.kind.rawValue) \(pd.count == 1 ? "file is" : "files are") already indexed. Remove them, or keep them searchable and stop indexing new ones.")
        }
    }

    @ViewBuilder private func orderRow(_ k: FileKind) -> some View {
        // While a disable is awaiting the purge/keep dialog the kind is still in enabledKinds, so reflect
        // the pending-off state so the switch doesn't snap back to ON under the dialog.
        let on = model.kindEnabled(k) && model.pendingDisable?.kind != k
        HStack(spacing: 8) {
            Image(systemName: "line.3.horizontal").foregroundStyle(.tertiary).font(.callout)
            Label(k.title, systemImage: k.symbol)
            Spacer()
            Toggle("", isOn: Binding(get: { on }, set: { v in Task { await model.toggleKind(k, on: v) } }))
                .labelsHidden().toggleStyle(.switch).controlSize(.mini)
        }
        .opacity(on ? 1 : 0.55)
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
            // "Skip small files" left the direction to the reader: is 300 the floor or the ceiling?
            // The header states it, so each row is just a modality and a number and no footer is
            // needed to explain which way it cuts.
            Section("Skip files smaller than") {
                MinimumField(kind: .image, label: "Images", unit: "px",
                             value: Binding(get: { Double(model.minImageDimension) },
                                            set: { model.minImageDimension = Int($0.rounded()) }))
                MinimumField(kind: .audio, label: "Audio", unit: "sec", decimals: 1,
                             value: Binding(get: { model.minAudioSeconds }, set: { model.minAudioSeconds = $0 }))
                MinimumField(kind: .video, label: "Video", unit: "sec", decimals: 1,
                             value: Binding(get: { model.minVideoSeconds }, set: { model.minVideoSeconds = $0 }))
                MinimumField(kind: .text, label: "Text", unit: "chars",
                             value: Binding(get: { Double(model.minTextChars) },
                                            set: { model.minTextChars = Int($0.rounded()) }))
            }

            Section {
                IgnoreEditor(text: $draft)
                    .frame(minHeight: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))

                previewBar
            } header: {
                Text("Ignore rules")
            } footer: {
                Text("One .gitignore pattern per line: leading ! re-includes, trailing / matches folders.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onAppear { if !loaded { draft = model.ignoreText; loaded = true } }
        .onChange(of: draft) { _, newValue in schedulePreview(newValue) }
    }

    /// The published preview ONLY when it describes the text currently in the editor. The two are
    /// separate publishes: `draft` moves on every keystroke, the preview 350ms and a whole index
    /// scan later, and previewIgnore deliberately leaves the previous result up while it recomputes.
    /// Rendering it unconditionally meant the bar showed the OLD rule's numbers - and the old
    /// rule's danger banner, or no banner at all - while Apply stayed enabled the whole time, so a
    /// rule that prunes the entire index could be committed under the harmless numbers of the rule
    /// before it. Mismatched now falls through to the existing "Calculating..." branch.
    private var preview: AppModel.IgnorePreview? {
        guard let p = model.ignorePreview, p.forText == draft else { return nil }
        return p
    }

    /// Live preview of what the current ignore rules match: a danger warning plus the affected-file
    /// count, so the user sees the blast radius before saving.
    @ViewBuilder private var previewBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let d = preview?.danger {
                Label(d, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 12) {
                if let p = preview {
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
                    Text("Calculating\u{2026}").foregroundStyle(.secondary)
                } else {
                    Text("Rules active.").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Import\u{2026}") { importIgnoreFile() }
                    .help("Load patterns from a file")
                if model.ignoreHasBackup {
                    Button("Revert") {
                        model.revertIgnore()
                        draft = model.ignoreText
                    }
                    .help("Undo the last applied change")
                }
                Button("Apply") {
                    previewTask?.cancel()
                    model.applyIgnoreText(draft)
                }
                .keyboardShortcut("s", modifiers: .command)
                .buttonStyle(.borderedProminent)
                // isPaperRunning: applying rules deletes rows and VACUUMs the user's store from a
                // detached task, which is a write to that store while the suite holds process-wide
                // levers (vacuumSmallCache among them) - and it competes with every measurement.
                // The draft is kept, so Apply works the moment the run ends.
                .disabled(!dirty || model.isPaperRunning)
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
    /// True only while THIS is the selected Settings tab - the live memory sampler's on switch.
    var isVisible: Bool
    @Environment(AppModel.self) private var model: AppModel
    /// Only for the hidden paper run: its sheet lives on the main window, which may be closed.
    @Environment(\.openWindow) private var openWindow
    private var memoryCeiling: Double { max(8, min(model.physicalMemoryGB.rounded(), 128)) }
    var body: some View {
        Form {
            Section {
                Toggle("Search as you type", isOn: Binding(
                    get: { model.instantSearchEnabled },
                    set: { model.instantSearchEnabled = $0 }
                ))
                .toggleStyle(.switch)
                // Byte-identical copies always collapse - that can only ever be right. This governs
                // the NEAR tier, which is a similarity judgement, so it stays switchable.
                Toggle("Stack near-identical results", isOn: Binding(
                    get: { model.groupNearDuplicates },
                    set: { model.groupNearDuplicates = $0 }
                ))
                .toggleStyle(.switch)
                .help("Off: only byte-identical copies are stacked")
            } header: {
                Text("Search")
            } footer: {
                Text("Off: results update on Return. Identical copies always stack; near-identical ones are a judgement call.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                Picker("Max image size", selection: Binding(get: { model.maxImageDimension }, set: { model.maxImageDimension = $0 })) {
                    Text("1024 px").tag(1024)
                    Text("1280 px").tag(1280)
                    Text("1568 px \u{00B7} recommended").tag(1568)
                    Text("2048 px").tag(2048)
                }
                Picker("Max frames per video", selection: Binding(get: { model.maxVideoFrames }, set: { model.maxVideoFrames = $0 })) {
                    Text("6").tag(6)
                    Text("16").tag(16)
                    Text("32 \u{00B7} recommended").tag(32)
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
                Text("Smaller caps index faster, with less detail.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Picker("Folder map layout", selection: Binding(get: { model.mapUsesUMAP }, set: { model.mapUsesUMAP = $0 })) {
                    Text("Fast \u{00B7} PCA").tag(false)
                    Text("Detailed \u{00B7} UMAP").tag(true)
                }
                Toggle("Grid layout", isOn: Binding(get: { model.mapNoOverlap }, set: { model.mapNoOverlap = $0 }))
                    .help("One file per cell, so every dot stays separately visible; hides density")
            } header: {
                Text("Folder map")
            } footer: {
                Text("PCA is instant; UMAP separates clusters better and adds click-to-spotlight.")
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
                    // Locked while the paper run holds the cap. Settings is its own window, so this
                    // pane stays live behind the run's sheet: moving the slider would persist the
                    // new value and apply it, and the run's restore would then put the OLD cap back
                    // on MLX - leaving the effective cap and the one shown here disagreeing until
                    // relaunch. It also silently corrupts the run, which pins the cap as a class.
                    .disabled(model.isPaperRunning)
                }
                MemoryBreakdown(isVisible: isVisible)
            } header: {
                Text("Memory")
            } footer: {
                // Names the two slices the cap actually governs, now that the bar above makes the
                // difference visible: the cap is an MLX limit, so a total above it is normal.
                Text(model.isPaperRunning
                     ? "Locked while the benchmark runs; your cap is restored after."
                     : "The cap covers Model and Cache above, not the whole app. 0 is unlimited.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                LabeledContent("Benchmark This Mac") {
                    HStack(spacing: 8) {
                        Button("Run benchmark") { Task { await model.runProfiling() } }
                            .controlSize(.small)
                            .disabled(model.isProfilingRunning || model.isPaperRunning || !model.canIndex)
                        // Hidden developer control (PaperGate: OMNI_PAPER=1, the omni.paper default,
                        // or Option held). Absent rather than disabled when the gate is closed.
                        // Gated on phase == .ready, NOT canIndex: the paper suite measures a
                        // self-contained synthetic workload and is exactly as valid on a machine
                        // where the user has never picked a folder.
                        PaperGated {
                            // Opens the main window FIRST: the progress sheet - and the only Cancel
                            // button a 25-minute run has - is presented by ContentView, and the main
                            // window is closable while Settings stays open. Started from there with
                            // it closed, the run had no progress, no cancel and no result sheet, and
                            // indexing stayed suppressed until it finished on its own.
                            Button("Paper") {
                                openWindow(id: "main")
                                Task { await model.runPaperBenchmark() }
                            }
                            .controlSize(.small)
                            .disabled(model.isPaperRunning || model.isProfilingRunning || model.phase != .ready)
                            .help("Paper suite - up to 25 min, synthetic data, your index untouched")
                        }
                    }
                }
                Toggle(isOn: Binding(get: { model.shareProfilingResults }, set: { model.shareProfilingResults = $0 })) {
                    Text("Share results")
                }
                if let r = model.lastProfilingReport {
                    LabeledContent("Last run") {
                        Text(String(format: "%.0f files/sec \u{00B7} %.1f GB peak memory",
                                    r.metrics.filesPerSec, Double(r.metrics.peakVramDeltaBytes) / 1_073_741_824))
                            .foregroundStyle(.secondary).monospacedDigit()
                    }
                }
            } header: {
                Text("Profiling")
            } footer: {
                Text("Indexes a fixed 300-file dataset. Sharing sends hardware and timing only - never your files - to hanxiao.io/omni.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

/// Live breakdown of Omni's OWN memory - the same capacity-bar idiom System Settings > Storage
/// uses for a disk, scaled to this process instead of the machine. The whole bar is the app's
/// phys_footprint (what Activity Monitor shows for Omni), and the slices are measured parts of it,
/// so the question it answers is "where did Omni's memory go", never "how full is my Mac".
private struct MemoryBreakdown: View {
    /// Sampling runs ONLY while the Performance tab is the visible one. Not `.onAppear`: a
    /// SwiftUI TabView keeps a visited pane alive, so an appear-driven loop would keep ticking
    /// behind every other tab and after the window is closed. Nothing outside this pane - search,
    /// indexing, the main window - ever pays for the monitor.
    var isVisible: Bool
    @Environment(AppModel.self) private var model: AppModel
    @State private var sample = AppModel.MemorySample()

    /// Order matters: biggest and most stable first, catch-all last, so the bar doesn't reshuffle
    /// as values move. Grey for the remainder mirrors the free-space slice in System Settings.
    private var slices: [(name: String, color: Color, bytes: Int, help: String)] {
        // The folder map is NOT a slice here. It retains tens of MB - a sliver next to Model and
        // Index - so a fifth colour bought a legend row the eye cannot find in the bar. It stays in
        // `Other`, and `sample.viz` still carries the number for the OMNI_MEM_LOG trace.
        [("Model", .blue, sample.model, "Weights and activations held by MLX"),
         ("Cache", .teal, sample.cache, "Freed MLX buffers kept for reuse - reclaimed under memory pressure"),
         ("Index", .purple, sample.index, "Vectors and row table the search reads"),
         ("Other", Color(nsColor: .systemGray), sample.other, "App, thumbnails, database cache, frameworks")]
    }

    private func fmt(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Omni is using")
                Spacer()
                Text(fmt(sample.total)).foregroundStyle(.secondary).monospacedDigit()
            }
            bar
            legend
        }
        // Keyed on isVisible: SwiftUI cancels and restarts the task whenever it flips, so leaving
        // the tab stops the loop at the next await and re-entering starts a fresh one.
        .task(id: isVisible) {
            guard isVisible else { return }
            while !Task.isCancelled {
                sample = await model.sampleMemory()
                if omniMemLogEnabled {
                    FileHandle.standardError.write(Data("[mem-ui] tick\n".utf8))
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    @ViewBuilder private var bar: some View {
        GeometryReader { geo in
            let total = max(1, sample.total)
            HStack(spacing: 0) {
                ForEach(Array(slices.enumerated()), id: \.offset) { i, s in
                    // The last slice takes whatever is left instead of its own rounded width, so
                    // four roundings can never leave a hairline gap at the trailing edge.
                    let w = i == slices.count - 1
                        ? nil
                        : (geo.size.width * CGFloat(s.bytes) / CGFloat(total)).rounded(.down)
                    Rectangle().fill(s.color)
                        .frame(width: w)
                        .frame(maxWidth: w == nil ? .infinity : nil)
                }
            }
        }
        .frame(height: 16)
        .background(Color(nsColor: .quaternaryLabelColor))
        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    }

    @ViewBuilder private var legend: some View {
        LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading),
                            GridItem(.flexible(), alignment: .leading)], spacing: 4) {
            ForEach(Array(slices.enumerated()), id: \.offset) { _, s in
                HStack(spacing: 5) {
                    Circle().fill(s.color).frame(width: 7, height: 7)
                    Text(s.name)
                    Spacer(minLength: 4)
                    Text(fmt(s.bytes)).foregroundStyle(.secondary).monospacedDigit()
                }
                .help(s.help)
            }
        }
        .font(.caption)
    }
}

/// What the index costs on disk, drawn the same way memory is - because the same mistake is
/// available in both places. "Size: 3.27 GB" reads as the whole index, and after the migration the
/// SQLite database is the SMALLEST of the three files that matter: the vectors are another 6.5 GB
/// sitting beside it. A bar makes the proportion obvious at a glance, and the legend says which
/// files would cost a reindex if lost and which the app simply rebuilds.
private struct DiskBreakdown: View {
    let entries: [VectorStore.DiskUse.Entry]

    /// Warm for the files that ARE the index, cool for everything derived from them. The split is
    /// the one fact a user needs here, so it is carried by hue rather than by a footnote.
    private func color(_ e: VectorStore.DiskUse.Entry) -> Color {
        switch e.name {
        case "Vectors":         return .orange
        case "Snippets":        return .pink
        case "Scan codes":      return .teal
        case "Filename index":  return .mint
        default:                return .gray
        }
    }

    private func fmt(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private var total: Int64 { max(1, entries.reduce(0) { $0 + $1.bytes }) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Size")
                Spacer()
                Text(fmt(entries.reduce(0) { $0 + $1.bytes }))
                    .foregroundStyle(.secondary).monospacedDigit()
            }
            GeometryReader { geo in
                HStack(spacing: 0) {
                    ForEach(Array(entries.enumerated()), id: \.offset) { i, e in
                        // Last slice absorbs the rounding, so several roundings cannot leave a
                        // hairline gap at the trailing edge.
                        let w = i == entries.count - 1
                            ? nil
                            : (geo.size.width * CGFloat(e.bytes) / CGFloat(total)).rounded(.down)
                        Rectangle().fill(color(e))
                            .frame(width: w)
                            .frame(maxWidth: w == nil ? .infinity : nil)
                    }
                }
            }
            .frame(height: 16)
            .background(Color(nsColor: .quaternaryLabelColor))
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))

            LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading),
                                GridItem(.flexible(), alignment: .leading)], spacing: 4) {
                ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                    HStack(spacing: 5) {
                        Circle().fill(color(e)).frame(width: 7, height: 7)
                        Text(e.name)
                        Spacer(minLength: 4)
                        Text(fmt(e.bytes)).foregroundStyle(.secondary).monospacedDigit()
                    }
                    .help(e.irreplaceable ? e.detail : "\(e.detail) - rebuilt automatically if deleted")
                }
            }
            .font(.caption)
        }
    }
}

/// A minimum, typed rather than chosen. These were dropdowns of four preset values, which is the
/// wrong control for a threshold: the right number depends on what someone keeps in their folders,
/// and 0 (index everything) has to be reachable in the same place as 300.
///
/// Just the field: a stepper next to it added a control for values nobody arrives at by nudging -
/// these are typed once and forgotten. The bound is one-sided, so the field clamps rather than
/// rejecting - a typed "-5" becomes 0, which is a real setting (index everything) instead of an
/// error nobody can act on.
private struct MinimumField: View {
    let kind: FileKind
    let label: String
    let unit: String
    var decimals: Int = 0
    @Binding var value: Double

    private var clamped: Binding<Double> {
        Binding(get: { Swift.max(0, value) }, set: { value = Swift.max(0, $0) })
    }

    var body: some View {
        // Explicit HStack rather than LabeledContent: the label and the field are centred on each
        // other here, which a label column does not promise once the row holds a bordered control.
        HStack(alignment: .center, spacing: 6) {
            // Same symbols as the File types list, so a modality looks the same wherever it appears.
            Label(label, systemImage: kind.symbol)
            Spacer(minLength: 8)
            // BORDERED, AND THE STANDARD ROW HEIGHT. Both stock styles fail one of those: the
            // default draws no border (the number reads as static text, not something you can
            // type in) and .squareBorder/.roundedBorder are 45pt rows against the 37pt every
            // other row in Settings uses. A plain field in a drawn box is both.
            TextField("", value: clamped, format: .number.precision(.fractionLength(0 ... decimals)))
                .textFieldStyle(.plain)
                .multilineTextAlignment(.trailing)
                .padding(.horizontal, 6)
                .padding(.vertical, 1)
                .frame(width: 56)
                .background(RoundedRectangle(cornerRadius: 5).fill(Color(nsColor: .textBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Color(nsColor: .separatorColor)))
            // Fixed width so the fields line up in a column ("px", "sec" and "chars" are different
            // lengths), and CENTRED in it - a leading-aligned unit sat hard against the box on one
            // row and adrift on the next. Vertical centring comes from the HStack, so the unit, the
            // number and the label all sit on one line.
            Text(unit).foregroundStyle(.secondary)
                .frame(width: 40, alignment: .center)
        }
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
            } footer: {
                Text("Older searches are removed automatically. Bookmarks are kept.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                LabeledContent("Saved searches") {
                    Text("\(model.recentHistoryCount) recent \u{00B7} \(model.bookmarkCount) bookmarked")
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                Button("Clear search history\u{2026}", role: .destructive) { confirmClear = true }
                    .disabled(model.recentHistoryCount == 0)
                    .help("Remove all recent searches; bookmarks are kept")
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("Clear all recent searches?", isPresented: $confirmClear) {
            Button("Clear search history", role: .destructive) { model.clearSearchHistory() }
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
                                Text("Built with \(v.title), now running \(model.modelVariant.title). Switch back to keep it, or reindex.")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else {
                                Text("Built with an older embedding version. Reindex to keep results accurate.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                if let v = model.indexBuiltVariant {
                                    // isPaperRunning: a variant switch tears the engine down and
                                    // rebuilds it, and the paper run is holding that exact engine
                                    // on a detached thread for up to 25 minutes.
                                    Button("Switch to \(v.title)") { model.selectVariant(v) }
                                        .disabled(model.isDownloading || model.isPaperRunning)
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
                IndexStatusRow()
                LabeledContent("Indexed files", value: model.indexedFiles.formatted())
                LabeledContent("Indexed chunks", value: model.indexedChunks.formatted())
                if model.diskUse.isEmpty {
                    LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: model.dbSizeBytes, countStyle: .file))
                } else {
                    DiskBreakdown(entries: model.diskUse)
                }
                // The size above does NOT fall while this runs, and saying so is the whole point of
                // showing it: converting a row rewrites it shorter without freeing a page, so the
                // file holds its size until the conversion finishes and the space is reclaimed in
                // one step. Without this line the number looks stuck and the work looks broken.
                if let m = model.storageMigration, m.total > 0 {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Optimizing storage")
                            Spacer()
                            Text("\(Int(Double(m.done) / Double(m.total) * 100))%")
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        ProgressView(value: Double(m.done), total: Double(m.total))
                            .progressViewStyle(.linear)
                        Text("Frees \(ByteCountFormatter.string(fromByteCount: m.bytesToReclaim, countStyle: .file)) when it finishes.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let last = model.lastIndexed {
                    LabeledContent("Last indexed", value: last.formatted(.relative(presentation: .named)))
                }
                // Manual row instead of LabeledContent: a long path makes LabeledContent
                // wrap the value side under the label. The path gets the whole value side
                // of the label line; the buttons drop to a second line so they never
                // squeeze it into heavy truncation.
                VStack(spacing: 6) {
                    HStack(spacing: 8) {
                        Text("Location")
                        Spacer()
                        if !model.dbPath.isEmpty {
                            Text((model.dbPath as NSString).abbreviatingWithTildeInPath)
                                .font(.caption.monospaced()).foregroundStyle(.secondary)
                                .lineLimit(1).truncationMode(.middle)
                                .help(model.dbPath)
                        }
                    }
                    HStack(spacing: 8) {
                        Spacer()
                        Button("Change\u{2026}") { pickDatabase() }
                            .help("Where the index is stored; changing loads the index from there")
                            // isPaperRunning: the run captured the CURRENT index paths as the ones
                            // its filesystem must refuse to open, and a swap mid-run would move the
                            // index out from under that list.
                            .disabled(model.isPaperRunning)
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dbPath)])
                        }
                        .disabled(model.dbPath.isEmpty)
                    }
                    .controlSize(.small)
                }
            }
            Section {
                // Selecting a variant switches to it if installed, or downloads it if not - no
                // separate download button.
                Picker("Embedding model", selection: Binding(
                    get: { model.modelVariant },
                    set: { model.selectVariant($0) }
                )) {
                    ForEach(ModelVariant.allCases, id: \.self) { v in
                        Text(model.installedVariants[v] != nil ? v.title : "Download \(v.title)\u{2026}")
                            .tag(v)
                    }
                }
                // isPaperRunning for the same reason as the banner's switch button: the run holds
                // the loaded engine, and selectVariant replaces it.
                .disabled(model.isDownloading || model.isIndexing || model.isPaperRunning)

                if model.isDownloading {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: model.downloadFraction)
                        HStack {
                            Text(model.downloadLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            Spacer()
                            Button("Cancel") { model.cancelDownload() }.controlSize(.small)
                        }
                    }
                } else if !model.modelPath.isEmpty {
                    // Same layout as the Index section's Location row: the path gets the whole
                    // value side of the label line, the buttons drop to a second line.
                    VStack(spacing: 6) {
                        HStack(spacing: 8) {
                            Text("Location")
                            Spacer()
                            Text((model.modelPath as NSString).abbreviatingWithTildeInPath)
                                .font(.caption.monospaced()).foregroundStyle(.secondary)
                                .lineLimit(1).truncationMode(.middle)
                                .help(model.modelPath)
                        }
                        HStack(spacing: 8) {
                            Spacer()
                            Button("Change\u{2026}") { pickModel() }
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.modelPath)])
                            }
                        }
                        .controlSize(.small)
                    }
                }
            } header: {
                Text("Model")
            } footer: {
                Text("Picking a variant switches to it, or downloads it. Switching rebuilds the index.")
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
