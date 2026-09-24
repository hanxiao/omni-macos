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
            // The plain mole, through the same view as every status screen: one size, one colour.
            StatusGlyph(symbol: CenteredStatus.mole)
            Text("Welcome to Omni")
                .font(.title).fontWeight(.semibold)
            // Names what downloads, so a second download right after install is not read as a mistake.
            Text("Omni needs its search model before it can start.")
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
                    downloadButton(0, prominent: true) { model.downloadModel(.nano) }
                    // The second choice is the OCR add-on, not the larger embedding build. Someone
                    // meeting the app for the first time is choosing what it can DO, and a second
                    // embedding variant that is 60% bigger for a quality difference they cannot
                    // see yet is not that choice - it stays in Settings > Storage, where the
                    // surrounding context makes it answerable.
                    downloadButton(1, prominent: false) { model.downloadOCRModel(.balanced) }
                }
                .padding(.top, 4)

                // A line here used to frame the macOS permission prompts before they fired:
                // "Next, Omni asks for access to Desktop, Documents, and Downloads." Nothing is
                // seeded on a first launch any more (AppModel.loadRoots), so no prompt follows the
                // download and there is nothing to frame - the user picks a folder when they are
                // ready and macOS asks then, in the context of their own choice.
            }

            if model.downloadFailed {
                Text(model.downloadLabel).font(.callout).foregroundStyle(.red).frame(maxWidth: 440)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    /// The two choices, in one place: each button lays out a hidden copy of BOTH so the pair
    /// measures to the widest of them. Equal width with no pixel guess - a fixed 260 left a finger
    /// of empty pill past anything either of them said, and every narrower guess clipped whichever
    /// title was longest.
    private static let choices: [(title: String, size: String)] = [
        ("Download embedding model", "~1.9 GB"),
        ("Download OCR model", "Optional, ~4.5 GB"),
    ]

    @ViewBuilder private func downloadButton(_ choice: Int, prominent: Bool,
                                             action: @escaping () -> Void) -> some View {
        let content = ZStack(alignment: .leading) {
            ForEach(Self.choices, id: \.title) { other in
                label(other.title, other.size).hidden()
            }
            label(Self.choices[choice].title, Self.choices[choice].size)
        }

        // NO FOCUS HALO. The window focuses the first button on open, and the ring drawn around it
        // made it read 6pt taller than its twin (47 against 41 measured, same button underneath).
        // The recommended one is the default button instead, so Return still starts it.
        if prominent {
            Button(action: action) { content }
                .controlSize(.large).buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .focusEffectDisabled()
        } else {
            Button(action: action) { content }
                .controlSize(.large).buttonStyle(.bordered)
                .focusEffectDisabled()
        }
    }

    private func label(_ title: String, _ size: String) -> some View {
        HStack {
            Image(systemName: "arrow.down.circle")
            VStack(alignment: .leading, spacing: 1) {
                Text(title).fontWeight(.medium)
                // Smaller than the action it qualifies: a size and a tag are not the decision.
                Text(size).font(.caption).foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .fixedSize(horizontal: true, vertical: false)
    }
}
