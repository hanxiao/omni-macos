import SwiftUI
import AppKit
import OmniKit

struct SettingsView: View {
    var body: some View {
        TabView {
            ActivityTab().tabItem { Label("Indexing", systemImage: "arrow.triangle.2.circlepath") }
            ContentTypesTab().tabItem { Label("Content", systemImage: "square.grid.2x2") }
            FiltersTab().tabItem { Label("Filters", systemImage: "line.3.horizontal.decrease") }
            PerformanceTab().tabItem { Label("Performance", systemImage: "speedometer") }
            IndexTab().tabItem { Label("Storage", systemImage: "externaldrive") }
        }
        .frame(width: 480, height: 380)
    }
}

/// Live indexing status and the manual Index / Reindex / Pause controls. This is the
/// single home for the detail that used to clutter the sidebar.
private struct ActivityTab: View {
    @EnvironmentObject var model: AppModel

    private var overall: Double {
        let rs = model.progress.perRoot.values
        let total = rs.reduce(0) { $0 + $1.total }
        guard total > 0 else { return 0 }
        return Double(rs.reduce(0) { $0 + $1.done }) / Double(total)
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
                            Button("Pause") { model.pauseIndexing() }.controlSize(.small)
                        }
                        ProgressView(value: overall)
                        HStack {
                            Text("\(model.progress.embedded) added")
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
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        Text(model.indexedFiles == 0 ? "Nothing indexed yet" : "Up to date \u{00B7} \(model.indexedFiles.formatted()) files")
                        Spacer()
                        Button(model.indexedFiles == 0 ? "Index" : "Reindex") { model.startIndexing() }
                            .controlSize(.small).disabled(!model.canIndex)
                    }
                }
            } header: {
                Text("Status")
            } footer: {
                Text("The index keeps itself current in the background as files change. Reindex to rebuild from scratch.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Folders") {
                ForEach(model.roots, id: \.self) { url in
                    let rp = model.progress.perRoot[url.path]
                    HStack {
                        Image(systemName: "folder").foregroundStyle(.secondary)
                        Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if model.isIndexing, let rp, rp.total > 0, rp.done < rp.total {
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
    @EnvironmentObject var model: AppModel
    var body: some View {
        Form {
            Section {
                kindToggle(.image, "Images")
                kindToggle(.video, "Video")
                kindToggle(.audio, "Audio")
                kindToggle(.text, "Text & Documents")
            } header: {
                Text("What to index")
            } footer: {
                Text("Everything is embedded into one space, so a text query finds any modality. Turning a type off removes it right away; turning it on indexes those files in the background.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
    @ViewBuilder private func kindToggle(_ k: FileKind, _ label: String) -> some View {
        let off = (k == .audio && !model.audioSupported)
        Toggle(isOn: Binding(get: { model.settings.contains(k) }, set: { model.setIndexKind(k, $0) })) {
            Label(label, systemImage: k.symbol)
        }
        .disabled(off)
    }
}

private struct FiltersTab: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        Form {
            Section {
                Picker("Minimum image size", selection: $model.minImageDimension) {
                    Text("No minimum").tag(0)
                    Text("64 px").tag(64)
                    Text("128 px").tag(128)
                    Text("256 px").tag(256)
                    Text("512 px").tag(512)
                }
                Picker("Minimum audio length", selection: $model.minAudioSeconds) {
                    Text("No minimum").tag(0.0)
                    Text("1 second").tag(1.0)
                    Text("3 seconds").tag(3.0)
                    Text("5 seconds").tag(5.0)
                    Text("10 seconds").tag(10.0)
                }
                Picker("Minimum video length", selection: $model.minVideoSeconds) {
                    Text("No minimum").tag(0.0)
                    Text("1 second").tag(1.0)
                    Text("3 seconds").tag(3.0)
                    Text("5 seconds").tag(5.0)
                    Text("10 seconds").tag(10.0)
                }
                Picker("Minimum text length", selection: $model.minTextChars) {
                    Text("No minimum").tag(0)
                    Text("16 characters").tag(16)
                    Text("64 characters").tag(64)
                    Text("256 characters").tag(256)
                }
            } header: {
                Text("Skip small files")
            } footer: {
                Text("Files below these thresholds are not indexed - useful for ignoring icons, thumbnails, and very short clips. Applies on the next index.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

private struct PerformanceTab: View {
    @EnvironmentObject var model: AppModel
    private var memoryCeiling: Double { max(8, min(model.physicalMemoryGB.rounded(), 128)) }
    var body: some View {
        Form {
            Section {
                Picker("Max image size", selection: $model.maxImageDimension) {
                    Text("1024 px").tag(1024)
                    Text("1280 px").tag(1280)
                    Text("1568 px (recommended)").tag(1568)
                    Text("2048 px").tag(2048)
                }
                Stepper("Video frames per clip: \(model.maxVideoFrames)", value: $model.maxVideoFrames, in: 1 ... 16)
            } header: {
                Text("Throughput")
            } footer: {
                Text("Large images are downscaled before embedding; the model resizes to about 1.3 MP regardless, so a smaller cap speeds indexing with no quality loss.")
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
                Text("Hard cap on the model's GPU/unified memory, enforced by MLX immediately. 0 = unlimited. The model needs a few GB resident, so keep this above ~4 GB. (Embeddings run one at a time, so there is no batch size to trade off yet.)")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

private struct IndexTab: View {
    @EnvironmentObject var model: AppModel
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
                            Button("Reindex Now") { model.startIndexing() }
                                .controlSize(.small)
                                .disabled(model.isIndexing)
                        }
                    }
                }
            }
            Section("Storage") {
                LabeledContent("Indexed files", value: "\(model.indexedFiles)")
                LabeledContent("Embeddings", value: "\(model.indexedChunks)")
                LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: model.dbSizeBytes, countStyle: .file))
                if let last = model.lastIndexed {
                    LabeledContent("Last indexed", value: last.formatted(.relative(presentation: .named)))
                }
                LabeledContent("Embedding version", value: model.embeddingVersion)
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
                Text("Database location")
            }
            Section {
                Picker("Variant", selection: Binding(
                    get: { model.modelVariant },
                    set: { model.switchVariant($0) }
                )) {
                    ForEach(ModelVariant.allCases, id: \.self) { v in
                        Text(model.installedVariants[v] != nil ? v.title : "\(v.title) (not installed)").tag(v)
                    }
                }
                .disabled(model.isDownloading)
                LabeledContent("Audio", value: model.audioSupported ? "Available" : "Unavailable")

                if model.isDownloading {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: model.downloadFraction)
                        Text(model.downloadLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                } else {
                    ForEach(ModelVariant.allCases.filter { model.installedVariants[$0] == nil }, id: \.self) { v in
                        Button("Download \(v.title)") { model.downloadModel(v) }
                    }
                    Button("Change Model Folder\u{2026}") { pickModel() }
                }
            } header: {
                Text("Model")
            } footer: {
                Text("Switching variant rebuilds the index. Both variants share one embedding space.")
                    .font(.caption).foregroundStyle(.secondary)
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
