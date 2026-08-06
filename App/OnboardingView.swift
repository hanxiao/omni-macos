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

            if model.isDownloading {
                VStack(spacing: 8) {
                    ProgressView(value: model.downloadFraction)
                        .frame(width: 360)
                    Text(model.downloadLabel)
                        .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    // A multi-GB download on a slow connection must be escapable (HIG); partial
                    // files are kept and skipped on the next attempt.
                    Button("Cancel") { model.cancelDownload() }
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
                    variantButton(.nano, size: "~1.9 GB \u{00B7} recommended", prominent: true)
                    variantButton(.small, size: "~3.1 GB \u{00B7} higher quality", prominent: false)
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

    @ViewBuilder private func variantButton(_ v: ModelVariant, size: String, prominent: Bool) -> some View {
        let content = HStack {
            Image(systemName: "arrow.down.circle")
            VStack(alignment: .leading, spacing: 1) {
                Text("Download \(v.title)").fontWeight(.medium)
                Text(size).foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .frame(width: 260, alignment: .leading)

        if prominent {
            Button { model.downloadModel(v) } label: { content }
                .controlSize(.large).buttonStyle(.borderedProminent)
        } else {
            Button { model.downloadModel(v) } label: { content }
                .controlSize(.large).buttonStyle(.bordered)
        }
    }
}
