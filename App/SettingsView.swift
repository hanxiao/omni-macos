import SwiftUI
import AppKit
import OmniKit

struct SettingsView: View {
    var body: some View {
        TabView {
            ContentTypesTab().tabItem { Label("Content", systemImage: "square.grid.2x2") }
            PerformanceTab().tabItem { Label("Performance", systemImage: "speedometer") }
            ModelTab().tabItem { Label("Model", systemImage: "cpu") }
        }
        .frame(width: 460, height: 320)
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
                Text("Everything is embedded into one space, so a text query finds any modality. Reindex after changing these.")
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

private struct PerformanceTab: View {
    @EnvironmentObject var model: AppModel
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
                Text("Large images are downscaled before embedding; the model resizes to about 1.3 MP regardless, so a smaller cap speeds indexing with no quality loss. Fewer video frames index faster.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Picker("GPU cache limit", selection: $model.gpuCacheMB) {
                    Text("Unlimited").tag(0)
                    Text("2 GB").tag(2048)
                    Text("4 GB").tag(4096)
                    Text("8 GB").tag(8192)
                    Text("16 GB").tag(16384)
                }
            } header: {
                Text("Memory")
            } footer: {
                Text("Caps the MLX buffer cache to bound memory during long indexing runs. Applies after the model reloads (relaunch).")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

private struct ModelTab: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        Form {
            Section("Model") {
                LabeledContent("Engine", value: "jina-embeddings-v5-omni-small")
                LabeledContent("Runtime", value: "MLX-Swift (in-process)")
                LabeledContent("Image embedding", value: model.supportsImages ? "Available" : "Unavailable")
                LabeledContent("Audio embedding", value: model.audioSupported ? "Available" : "Unavailable")
            }
            Section {
                if !model.modelPath.isEmpty {
                    Text(model.modelPath).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(2).truncationMode(.middle).textSelection(.enabled)
                }
                Button("Change Model Folder...") { pick() }
            } header: {
                Text("Location")
            }
        }
        .formStyle(.grouped)
    }
    private func pick() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
}
