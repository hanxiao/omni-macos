import SwiftUI
import AppKit
import OmniKit

struct OnboardingView: View {
    @Environment(AppModel.self) private var model: AppModel

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "square.stack")
                .font(.system(size: 46, weight: .light))
                .foregroundStyle(.tertiary)
            Text("Welcome to Omni")
                .font(.title).fontWeight(.semibold)
            Text("Search images, video, audio, and text by meaning. The model runs on-device - pick one to download.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 440)

            if model.isDownloading {
                VStack(spacing: 8) {
                    ProgressView(value: model.downloadFraction)
                        .frame(width: 360)
                    Text(model.downloadLabel)
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(.top, 4)
            } else {
                VStack(spacing: 10) {
                    variantButton(.nano, size: "~1.9 GB \u{00B7} faster", prominent: true)
                    variantButton(.small, size: "~3.1 GB \u{00B7} higher quality", prominent: false)
                }
                .padding(.top, 4)

                Button("Choose Model Folder\u{2026}") { pick() }
                    .buttonStyle(.plain).font(.callout).foregroundStyle(.secondary).padding(.top, 6)
            }

            if model.downloadFailed {
                Text(model.downloadLabel).font(.caption).foregroundStyle(.red).frame(maxWidth: 440)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    @ViewBuilder private func variantButton(_ v: ModelVariant, size: String, prominent: Bool) -> some View {
        let content = HStack {
            Image(systemName: "arrow.down.circle")
            VStack(alignment: .leading, spacing: 1) {
                Text("Download \(v.title)").fontWeight(.medium)
                Text(size).font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(width: 260, alignment: .leading)

        if prominent {
            Button { model.downloadModel(v) } label: { content }
                .controlSize(.large).buttonStyle(.borderedProminent)
        } else {
            Button { model.downloadModel(v) } label: { content }
                .controlSize(.large).buttonStyle(.bordered)
        }
    }

    private func pick() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Select"
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
}
