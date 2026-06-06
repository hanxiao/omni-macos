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
                            Text("Indexing\u{2026}").fontWeight(.medium)
                            Spacer()
                            if let rateLabel {
                                Text(rateLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            Button("Pause") { model.pauseIndexing() }.controlSize(.small)
                        }
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
                Text("Keeps itself current in the background as files change; Update catches up any changes now.")
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
                Text("Drag a type onto another to reorder how they are indexed.")
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
                Text("Files below these thresholds are skipped on the next index - useful for icons, thumbnails, and very short clips.")
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
                Text("Smaller caps trade detail for faster indexing; images resize to about 1.3 MP anyway, so a lower image cap is free.")
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
                Text("Hard cap on the model's memory. Keep it above ~4 GB so the model stays resident; 0 means unlimited.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
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
                            Text("Index is out of date").fontWeight(.medium)
                            Text("It was built with an older embedding version. Reindex so results stay accurate.")
                                .font(.caption).foregroundStyle(.secondary)
                            Button("Reindex") { model.startIndexing() }
                                .controlSize(.small)
                                .disabled(model.isIndexing || !model.canIndex)
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
                Picker("Model Variant", selection: Binding(
                    get: { model.modelVariant },
                    set: { model.switchVariant($0) }
                )) {
                    ForEach(ModelVariant.allCases, id: \.self) { v in
                        Text(model.installedVariants[v] != nil ? v.title : "\(v.title) \u{00B7} not installed")
                            .tag(v)
                            .disabled(model.installedVariants[v] == nil)   // can't switch to an uninstalled variant
                    }
                }
                .disabled(model.isDownloading)

                if model.isDownloading {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: model.downloadFraction)
                        Text(model.downloadLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                } else {
                    ForEach(ModelVariant.allCases.filter { model.installedVariants[$0] == nil }, id: \.self) { v in
                        Button("Download \(v.title)") { model.downloadModel(v) }
                    }
                    Button("Choose Model Folder\u{2026}") { pickModel() }
                }
            } header: {
                Text("Model")
            } footer: {
                Text("Switching variant rebuilds the index. Both variants share one embedding space.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                if !model.dbPath.isEmpty {
                    Text(model.dbPath).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(3).truncationMode(.middle).textSelection(.enabled)
                }
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dbPath)])
                }
                .disabled(model.dbPath.isEmpty)
            } header: {
                Text("Database Location")
            }
        }
        .formStyle(.grouped)
    }
    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
}
