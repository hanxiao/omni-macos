import SwiftUI
import OmniKit

/// Neural Engine or GPU, all or nothing: on the Neural Engine every model Omni runs for search and
/// indexing (text, images, video, audio, tags) stays off the GPU; on the GPU it is fastest and shares
/// the GPU with other apps. The same facts on the first-run screen and in Settings > Performance
/// (index.md, "The Neural Engine, all or nothing": beside another app's GPU work that app lost 0.7%
/// with Omni on the Neural Engine and 31% with it on the GPU; a photo 625 ms against 112 ms here).
enum ComputeUnitFacts {
    static let rows: [(label: String, neuralEngine: String, gpu: String)] = [
        ("Other apps on the GPU", "Unaffected", "Slowed"),
        ("Indexing speed", "Slower", "Fastest"),
        ("Power", "Low", "Higher"),
    ]
    static func name(_ u: OmniEngine.ComputeUnit) -> String { u == .neuralEngine ? "Neural Engine" : "GPU" }
    static func symbol(_ u: OmniEngine.ComputeUnit) -> String { u == .neuralEngine ? "brain" : "cpu" }
}

/// The comparison under the switch in Settings: one column per unit, the chosen one in primary.
struct ComputeUnitGrid: View {
    var selected: OmniEngine.ComputeUnit
    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
            GridRow {
                Text("")
                Text(ComputeUnitFacts.name(.neuralEngine)).fontWeight(.medium)
                    .foregroundStyle(selected == .neuralEngine ? .primary : .secondary)
                Text(ComputeUnitFacts.name(.gpu)).fontWeight(.medium)
                    .foregroundStyle(selected == .gpu ? .primary : .secondary)
            }
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
                card(.neuralEngine, ["Never uses the GPU", "Low power", "Slower indexing"])
                card(.gpu, ["Fastest", "Shares the GPU with other apps", "Higher power"])
            }
            .padding(.top, 4)
            Text("Change it in Settings > Performance.")
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

/// One Neural Engine build's cancel flag, read on the build's thread between compiles and set on the
/// main actor when the setting changes (AppModel.applyComputeUnit).
final class AneBuildGate: @unchecked Sendable {
    let generation: Int
    private let lock = NSLock()
    private var _cancelled = false
    init(_ generation: Int) { self.generation = generation }
    var cancelled: Bool {
        get { lock.withLock { _cancelled } }
        set { lock.withLock { _cancelled = newValue } }
    }
}
