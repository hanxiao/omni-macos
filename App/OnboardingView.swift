import SwiftUI
import AppKit
import OmniKit

struct OnboardingView: View {
    @Environment(AppModel.self) private var model: AppModel

    // House rules for this screen, so it stays calm as copy changes:
    // - TWO type sizes: `.title` for the one headline, `.callout` for every other string.
    // - TWO text colors: primary for what you act on, secondary for everything that explains it.
    //   No tertiary/quaternary text - stacking greys is what made this read as a wall.
    // - ONE decision: which model to download. Nothing else is offered here.
    var body: some View {
        VStack(spacing: 18) {
            // The app's own icon, read from the running bundle (NOT an asset name or a bundled
            // file): whatever ships as AppIcon is what shows here, so the icon and this screen can
            // never drift apart. It replaces an SF Symbol placeholder - a generic square.stack on
            // the one screen where the app introduces itself.
            Image(nsImage: NSApp.applicationIconImage)
                .resizable().interpolation(.high)
                .frame(width: 84, height: 84)
                .accessibilityHidden(true)   // the headline right below already names the app
            Text("Welcome to Omni")
                .font(.title).fontWeight(.semibold)
            // Says WHAT is downloading and WHY there is a download at all: people who just
            // installed the app read a second download as a mistake or a trick. It also carries the
            // privacy claim, which is why the lock.shield block that used to close this screen is
            // gone rather than reworded.
            Text("Searching by meaning needs an embedding model. It downloads once, then runs on this Mac - your files never leave it.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 440)

            if model.isDownloading || model.isOCRDownloading {
                let embedding = model.isDownloading
                VStack(spacing: 8) {
                    ProgressView(value: embedding ? model.downloadFraction : model.ocrDownloadFraction)
                        .frame(width: 300)
                    HStack(spacing: 10) {
                        Text(embedding ? model.downloadLabel : model.ocrDownloadLabel)
                        Text(embedding ? model.downloadSpeed : model.ocrDownloadSpeed)
                            .foregroundStyle(.secondary)
                    }
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    // A multi-GB download on a slow connection must be escapable (HIG); partial
                    // files are kept and skipped on the next attempt.
                    Button("Cancel") { embedding ? model.cancelDownload() : model.cancelOCRDownload() }
                        .controlSize(.small)
                        .padding(.top, 2)
                }
                .padding(.top, 4)
            } else {
                // No "choose a folder" escape hatch. Omni does not run arbitrary models: these are
                // the two jina-embeddings-v5-omni builds, and loading merges the retrieval LoRA and
                // upcasts the backbone for THIS architecture - point it at anything else and it
                // fails deep in the load. Someone who already has the weights is still covered
                // without a control here: ModelLocator finds an existing HuggingFace snapshot or a
                // staged copy on its own, and Settings > Storage > Model keeps an explicit
                // Change... for the rare case, where the surrounding context makes it honest.
                VStack(spacing: 10) {
                    downloadButton(title: "Download Omni Nano", size: "~1.9 GB",
                                   prominent: true) { model.downloadModel(.nano) }
                    // The second choice is the OCR add-on, not the larger embedding build. Someone
                    // meeting the app for the first time is choosing what it can DO, and a second
                    // embedding variant that is 60% bigger for a quality difference they cannot
                    // see yet is not that choice - it stays in Settings > Storage, where the
                    // surrounding context makes it answerable.
                    downloadButton(title: "Download OCR model", size: "~4.5 GB \u{00B7} recommended",
                                   prominent: false) { model.downloadOCRModel(.balanced) }
                }
                .padding(.top, 4)

                // Frames the macOS permission prompts BEFORE they fire: they arrive right after the
                // download with no other context.
                Text("Next, Omni asks for access to Desktop, Documents, and Downloads.")
                    .font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 420)
                    .padding(.top, 6)
            }

            if model.downloadFailed {
                Text(model.downloadLabel).font(.callout).foregroundStyle(.red).frame(maxWidth: 440)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    @ViewBuilder private func downloadButton(title: String, size: String, prominent: Bool,
                                             action: @escaping () -> Void) -> some View {
        // 160, not 260: the width is here only so the two buttons agree, and the old one left a
        // finger of empty pill past the longest line in either of them.
        let content = HStack {
            Image(systemName: "arrow.down.circle")
            VStack(alignment: .leading, spacing: 1) {
                Text(title).fontWeight(.medium)
                // Smaller than the action it qualifies: a size and a tag are not the decision.
                Text(size).font(.caption).foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .frame(width: 160, alignment: .leading)

        if prominent {
            Button(action: action) { content }
                .controlSize(.large).buttonStyle(.borderedProminent)
        } else {
            Button(action: action) { content }
                .controlSize(.large).buttonStyle(.bordered)
        }
    }
}
