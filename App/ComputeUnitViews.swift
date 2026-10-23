import SwiftUI
import OmniKit

/// Neural Engine or GPU: the same facts on the first-run screen and in Settings > Performance.
/// Only what holds on every Mac (index.md, "The Neural Engine, measured again": beside another
/// app's GPU work, that app lost 0.7% with Omni on the Neural Engine and 31% with it on the GPU).
/// Speed is left out: which unit is faster depends on the Mac.
enum ComputeUnitFacts {
    static let rows: [(label: String, neuralEngine: String, gpu: String)] = [
        ("Power", "Low", "Higher"),
        ("Other apps on the GPU", "Unaffected", "Slowed"),
        ("Photos, audio, video", "GPU", "GPU"),
    ]
    static func name(_ u: OmniEngine.ComputeUnit) -> String { u == .neuralEngine ? "Neural Engine" : "GPU" }
    static func symbol(_ u: OmniEngine.ComputeUnit) -> String { u == .neuralEngine ? "brain" : "cpu" }
}

/// The comparison under the picker in Settings: one column per unit, the chosen one in primary.
struct ComputeUnitGrid: View {
    var selected: OmniEngine.ComputeUnit
    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
            ForEach(ComputeUnitFacts.rows, id: \.label) { row in
                GridRow {
                    Text(row.label).foregroundStyle(.secondary)
                    Text(row.neuralEngine).foregroundStyle(selected == .neuralEngine ? .primary : .secondary)
                    Text(row.gpu).foregroundStyle(selected == .gpu ? .primary : .secondary)
                }
            }
        }
        .font(.callout)
    }
}

/// The screen after the first download: where the model runs. One decision, the Neural Engine
/// chosen, Continue. The OnboardingView house rules: `.title` once, `.callout` for the rest,
/// primary for what you act on and secondary for what explains it.
struct ComputeChoiceView: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var choice: OmniEngine.ComputeUnit = .neuralEngine

    var body: some View {
        VStack(spacing: 18) {
            StatusGlyph(symbol: CenteredStatus.mole)
            Text("Where Omni runs")
                .font(.title).fontWeight(.semibold)
            HStack(alignment: .top, spacing: 12) {
                card(.neuralEngine, ["Low power", "Leaves the GPU to other apps"])
                card(.gpu, ["Higher power", "Shares the GPU with other apps"])
            }
            .padding(.top, 4)
            VStack(spacing: 2) {
                Text("Photos, audio and video use the GPU either way.")
                Text("Change it in Settings > Performance.")
            }
            .font(.callout).foregroundStyle(.secondary)
            Button("Continue") { model.finishComputeChoice(choice) }
                .controlSize(.large).buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .focusEffectDisabled()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func card(_ unit: OmniEngine.ComputeUnit, _ lines: [String]) -> some View {
        let on = choice == unit
        return Button { choice = unit } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: ComputeUnitFacts.symbol(unit))
                    Text(ComputeUnitFacts.name(unit)).fontWeight(.medium)
                    Spacer(minLength: 8)
                    Image(systemName: on ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(on ? Color.accentColor : .secondary)
                }
                ForEach(lines, id: \.self) { Text($0).foregroundStyle(.secondary) }
            }
            .font(.callout)
            .padding(12)
            .frame(width: 210, alignment: .leading)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(on ? Color.accentColor : .clear, lineWidth: 2))
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(.plain)
        .focusEffectDisabled()
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}
