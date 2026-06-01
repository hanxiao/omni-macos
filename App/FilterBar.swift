import SwiftUI
import OmniKit

/// Filter row under the search field: file-kind chips inline, plus a popover with
/// folder, extension, and min-score controls.
struct FilterBar: View {
    @EnvironmentObject var model: AppModel
    @State private var showAdvanced = false

    private var chipKinds: [FileKind] {
        let present = FileKind.allCases.filter { model.indexedKinds.contains($0.rawValue) }
        return present.isEmpty ? [.image, .video] : present
    }

    var body: some View {
        HStack(spacing: 8) {
            ForEach(chipKinds, id: \.self) { kind in
                KindChip(kind: kind, selected: model.filterKinds.contains(kind)) {
                    model.toggleFilterKind(kind)
                }
            }
            Spacer()
            if model.filtersActive {
                Button { model.clearFilters() } label: {
                    Label("Clear", systemImage: "xmark.circle.fill").labelStyle(.titleAndIcon)
                }
                .buttonStyle(.plain)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Button { showAdvanced.toggle() } label: {
                Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
            }
            .buttonStyle(.plain)
            .foregroundStyle(model.filtersActive ? Color.accentColor : .secondary)
            .popover(isPresented: $showAdvanced, arrowEdge: .bottom) { advanced }

            Picker("View", selection: $model.viewMode) {
                Image(systemName: "list.bullet").tag(ResultViewMode.list)
                Image(systemName: "square.grid.2x2").tag(ResultViewMode.grid)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }

    private var advanced: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Folder").font(.caption).foregroundStyle(.secondary)
                Picker("Folder", selection: Binding(
                    get: { model.filterFolder?.path ?? "" },
                    set: { newPath in
                        model.filterFolder = newPath.isEmpty ? nil : URL(fileURLWithPath: newPath)
                        model.search()
                    }
                )) {
                    Text("All Folders").tag("")
                    ForEach(model.roots, id: \.self) { url in
                        Text(url.lastPathComponent).tag(url.path)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("File extension").font(.caption).foregroundStyle(.secondary)
                TextField("e.g. png, mp4, pdf", text: $model.filterExt)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.search() }
                    .onChange(of: model.filterExt) { _, _ in model.search() }
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Minimum score").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.0f%%", model.minScore * 100))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Slider(value: $model.minScore, in: 0 ... 0.8) { editing in
                    if !editing { model.search() }
                }
            }

            Button("Clear Filters") { model.clearFilters() }
                .controlSize(.small)
        }
        .padding(16)
        .frame(width: 260)
    }
}

struct KindChip: View {
    let kind: FileKind
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: kind.symbol).font(.caption2)
                Text(kind.title).font(.caption)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(selected ? Color.accentColor : Color(.quaternaryLabelColor).opacity(0.5),
                        in: Capsule())
            .foregroundStyle(selected ? Color.white : Color.primary)
        }
        .buttonStyle(.plain)
    }
}
