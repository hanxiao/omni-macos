import SwiftUI
import OmniKit

/// Filter row under the toolbar search field: native file-kind toggle chips, plus a
/// popover for folder, extension, date, and the relevance threshold.
struct FilterBar: View {
    @EnvironmentObject var model: AppModel
    @State private var showAdvanced = false

    private var chipKinds: [FileKind] {
        let present = FileKind.allCases.filter { model.indexedKinds.contains($0.rawValue) }
        return present.isEmpty ? [.image, .video, .audio] : present
    }

    var body: some View {
        HStack(spacing: Design.gap) {
            ForEach(chipKinds, id: \.self) { kind in
                Toggle(isOn: Binding(
                    get: { model.filterKinds.contains(kind) },
                    set: { on in if on { model.filterKinds.insert(kind) } else { model.filterKinds.remove(kind) } }
                )) {
                    Label(kind.title, systemImage: kind.symbol)
                }
                .toggleStyle(.button)
                .buttonBorderShape(.capsule)
                .controlSize(.small)
                .tint(.accentColor)
            }

            Spacer()

            if model.filtersActive {
                Button("Clear") { model.clearFilters() }
                    .buttonStyle(.plain).font(.callout).foregroundStyle(.secondary)
            }
            Button { showAdvanced.toggle() } label: {
                Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
            }
            .buttonStyle(.plain)
            .foregroundStyle(model.filtersActive ? Color.accentColor : .secondary)
            .help("Filters")
            .popover(isPresented: $showAdvanced, arrowEdge: .bottom) { advanced }
        }
        .padding(.horizontal, Design.gapLarge)
        .padding(.vertical, Design.gap)
    }

    private var advanced: some View {
        Form {
            Picker("Folder", selection: Binding(
                get: { model.filterFolder?.path ?? "" },
                set: { model.filterFolder = $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
            )) {
                Text("All Folders").tag("")
                ForEach(model.roots, id: \.self) { Text($0.lastPathComponent).tag($0.path) }
            }

            Picker("Type", selection: $model.filterExt) {
                Text("Any Extension").tag("")
                ForEach(model.indexedExts, id: \.self) { Text(".\($0)").tag($0) }
            }

            Picker("Date", selection: $model.dateRange) {
                ForEach(DateRange.allCases) { Text($0.title).tag($0) }
            }

            Picker("Relevance", selection: $model.minScore) {
                Text("Any").tag(0.0)
                Text("25%").tag(0.25)
                Text("50%").tag(0.5)
                Text("70%").tag(0.7)
            }

            Button("Reset Filters") { model.clearFilters() }
        }
        .formStyle(.grouped)
        .frame(width: 280)
        .scrollContentBackground(.hidden)
        .padding(.vertical, 4)
    }
}
