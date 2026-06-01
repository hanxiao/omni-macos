import SwiftUI
import AppKit

struct OnboardingView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "shippingbox")
                .font(.system(size: 46, weight: .light))
                .foregroundStyle(.tertiary)
            Text("Locate the omni model")
                .font(.title2).fontWeight(.semibold)
            Text("Omni needs the jina-embeddings-v5-omni-small-mlx model directory (the folder containing model.safetensors, tokenizer.json, and adapters/).")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)

            Button {
                pick()
            } label: {
                Label("Choose Model Folder", systemImage: "folder")
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)

            VStack(alignment: .leading, spacing: 4) {
                Text("Searched automatically:").font(.caption).foregroundStyle(.secondary)
                Text("· $OMNI_MODEL_DIR")
                Text("· ~/Library/Application Support/Omni/model")
                Text("· HuggingFace cache (models--jinaai--jina-embeddings-v5-omni-small-mlx)")
            }
            .font(.caption.monospaced())
            .foregroundStyle(.tertiary)
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func pick() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Select"
        if panel.runModal() == .OK, let url = panel.url {
            model.setModelDir(url)
        }
    }
}
