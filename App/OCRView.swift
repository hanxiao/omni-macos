import AppKit
import OmniKit
import SwiftUI
import UniformTypeIdentifiers

/// The OCR workspace: drop a document, watch it transcribe, read the Markdown.
///
/// Layout follows the app's existing shape rather than inventing one - content at centre stage,
/// a page rail on the trailing edge, and the only floating element is the readout, which exists
/// because a progress bar pinned to a toolbar tells you nothing while you are reading.
struct OCRView: View {
    @Environment(OCRSession.self) private var session
    @State private var mode: ViewMode = .rendered
    @State private var dropTargeted = false
    /// Per-page edits, keyed by page id. Only pages that have finished decoding can be edited.
    @State private var edits: [Int: String] = [:]

    enum ViewMode: String, CaseIterable, Identifiable {
        case rendered, split, raw
        var id: String { rawValue }
        var label: String {
            switch self {
            case .rendered: return "Formatted"
            case .split: return "Side by Side"
            case .raw: return "Markdown"
            }
        }
        var symbol: String {
            switch self {
            case .rendered: return "doc.richtext"
            case .split: return "rectangle.split.2x1"
            case .raw: return "chevron.left.forwardslash.chevron.right"
            }
        }
    }

    var body: some View {
        Group {
            switch session.phase {
            case .empty:
                DropZone(targeted: dropTargeted) { chooseFiles() }
            case .needsModel:
                ModelMissing()
            case .failed(let message):
                // Explicitly, not via `default`: a failed run used to fall through to the
                // workspace and render as "Preparing" forever, which is the worst possible
                // reading of an error - it looks like patience will fix it.
                Failure(message: message) { session.clear() }
            default:
                workspace
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .dropDestination(for: URL.self) { urls, _ in
            session.open(urls: urls)
            return true
        } isTargeted: { dropTargeted = $0 }
        .overlay(alignment: .top) {
            if dropTargeted && session.phase != .empty { DropOverlay() }
        }
        .toolbar { toolbar }
    }

    // MARK: - Workspace

    private var workspace: some View {
        HStack(spacing: 0) {
            content
            Divider()
            PageRail()
                .frame(width: 156)
        }
        .overlay(alignment: .bottom) {
            if session.isBusy || session.completedPages > 0 { ProgressReadout() }
        }
    }

    @ViewBuilder private var content: some View {
        if let page = session.visiblePage {
            switch mode {
            case .rendered:
                MarkdownPane(text: page.text, page: page)
            case .raw:
                RawPane(page: page, edit: editBinding(for: page))
            case .split:
                // A plain proportional split, not HSplitView. HSplitView propagates its children's
                // minimum widths up as its own, and inside a NavigationSplitView detail pane that
                // squeezes the app's sidebar past its own minimum - the sidebar labels start
                // clipping. Halves that simply divide what is available cannot do that.
                HStack(spacing: 0) {
                    MarkdownPane(text: page.text, page: page)
                        .frame(maxWidth: .infinity)
                    Divider()
                    RawPane(page: page, edit: editBinding(for: page))
                        .frame(maxWidth: .infinity)
                }
            }
        } else {
            CenteredHint(symbol: "text.viewfinder", title: "Preparing",
                         detail: "Rendering pages and loading the model.")
        }
    }

    /// Edits live in the view, not the session: the session owns what the model produced, and
    /// overwriting that with a half-typed buffer would lose the transcription on a re-render.
    private func editBinding(for page: OCRSession.Page) -> Binding<String?> {
        Binding(get: { edits[page.id] }, set: { edits[page.id] = $0 })
    }

    private func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.pdf, .png, .jpeg, .tiff, .heic, .bmp, .gif]
        panel.message = "Choose a PDF or images to transcribe"
        if panel.runModal() == .OK { session.open(urls: panel.urls) }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        if session.phase != .empty && session.phase != .needsModel {
            ToolbarItem {
                Picker("View", selection: $mode) {
                    ForEach(ViewMode.allCases) { m in
                        Label(m.label, systemImage: m.symbol).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .labelStyle(.iconOnly)
                .help("Switch between formatted Markdown, side by side, and raw text")
            }
            ToolbarItem {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.documentMarkdown, forType: .string)
                } label: {
                    Label("Copy Markdown", systemImage: "doc.on.doc")
                }
                .help("Copy the whole document as Markdown")
                .disabled(session.completedPages == 0)
            }
            ToolbarItem {
                Button { chooseFiles() } label: {
                    Label("Open Document", systemImage: "folder")
                }
                .help("Open another document")
            }
            ToolbarItem {
                Button(role: .destructive) { session.clear() } label: {
                    Label("Close Document", systemImage: "xmark.circle")
                }
                .help("Close this document")
            }
        }
    }
}

// MARK: - Page rail

private struct PageRail: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: Design.gap) {
                    ForEach(session.pages) { page in
                        PageThumb(page: page, selected: session.visiblePage?.id == page.id)
                            .id(page.id)
                            .onTapGesture { session.select(page.id) }
                    }
                }
                .padding(Design.gap)
            }
            .onChange(of: session.visiblePage?.id) { _, id in
                guard let id else { return }
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .center) }
            }
        }
        .background(.background.secondary)
    }
}

private struct PageThumb: View {
    let page: OCRSession.Page
    let selected: Bool

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: Design.cornerSmall)
                    .fill(.background.tertiary)
                if let thumbnail = page.thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: Design.cornerSmall))
                }
                if page.state == .running {
                    ProgressView().controlSize(.small)
                }
                if page.state == .failed {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
            }
            .frame(height: 150)
            // Unprocessed pages are dimmed rather than hidden: the rail doubles as the progress
            // display, so the shape of what is left has to stay visible.
            .opacity(page.state == .done ? 1 : (page.state == .running ? 0.85 : 0.4))
            .overlay {
                RoundedRectangle(cornerRadius: Design.cornerSmall)
                    .strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 2)
            }
            .animation(.easeOut(duration: 0.25), value: page.state)

            Text(page.label)
                .font(.caption2)
                .foregroundStyle(selected ? .primary : .secondary)
                .lineLimit(1)
        }
        .contentShape(Rectangle())
        .help(page.state == .done
              ? "\(page.label) - \(page.tokens) tokens"
              : "\(page.label) - not transcribed yet")
    }
}

// MARK: - Panes

private struct MarkdownPane: View {
    let text: String
    let page: OCRSession.Page

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(MarkdownBlock.parse(text).enumerated()), id: \.offset) { _, block in
                    block.view
                }
                if page.state == .running { TypingCaret() }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(.horizontal, 28)
            .padding(.vertical, 24)
            .frame(maxWidth: .infinity, alignment: .center)
            .textSelection(.enabled)
        }
    }
}

/// Raw Markdown. Editable only once the page has finished decoding.
///
/// That is not a policy choice, it is a correctness one. A `TextEditor` bound to a getter that
/// reads streaming text and a setter that writes `@State` mutates state during view update, and
/// SwiftUI responds by wedging the subtree - the whole side-by-side pane stopped refreshing while
/// the formatted view beside it kept streaming. Editing text that the next token is about to
/// overwrite was never meaningful anyway.
private struct RawPane: View {
    let page: OCRSession.Page
    @Binding var edit: String?

    private var editable: Bool { page.state == .done }

    var body: some View {
        Group {
            if editable {
                TextEditor(text: Binding(get: { edit ?? page.text }, set: { edit = $0 }))
                    .font(.system(.body, design: .monospaced))
                    .scrollContentBackground(.hidden)
            } else {
                ScrollView {
                    Text(page.text)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                }
            }
        }
        .background(.background.secondary)
        .overlay(alignment: .topTrailing) {
            if !editable {
                Text("read-only while transcribing")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(6)
            }
        }
    }
}

/// A caret that shows decoding is still going, so a pause reads as "working" rather than "done".
private struct TypingCaret: View {
    @State private var on = true
    var body: some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(Color.accentColor)
            .frame(width: 7, height: 15)
            .opacity(on ? 1 : 0.15)
            .task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(520))
                    withAnimation(.easeInOut(duration: 0.2)) { on.toggle() }
                }
            }
    }
}

// MARK: - Floating readout

private struct ProgressReadout: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        GlassGroup(spacing: 10) {
            HStack(spacing: 12) {
                if session.isBusy {
                    ProgressView(value: session.progress)
                        .progressViewStyle(.circular)
                        .controlSize(.small)
                } else {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text(session.isBusy
                         ? "Page \(min(session.completedPages + 1, session.pages.count)) of \(session.pages.count)"
                         : "\(session.completedPages) of \(session.pages.count) pages")
                        .font(.callout.weight(.medium))
                    Text(detail)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                if session.isBusy {
                    Button { session.cancel() } label: {
                        Image(systemName: "stop.fill")
                    }
                    .buttonStyle(.borderless)
                    .help("Stop transcribing")
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .glassChip(interactive: session.isBusy)
        }
        .shadow(color: .black.opacity(0.16), radius: 10, y: 3)
        .padding(.bottom, 18)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.easeOut(duration: 0.25), value: session.isBusy)
    }

    private var detail: String {
        var parts: [String] = []
        if session.currentTokensPerSecond > 0 {
            parts.append(String(format: "%.0f tok/s", session.currentTokensPerSecond))
        }
        if session.elapsed > 0 {
            parts.append(String(format: "%.0fs", session.elapsed))
        }
        parts.append(session.documentName)
        return parts.joined(separator: "  ·  ")
    }
}

// MARK: - Empty and error states

private struct DropZone: View {
    let targeted: Bool
    var onChoose: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "text.viewfinder")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(targeted ? Color.accentColor : .secondary)
            Text("Drop a document to transcribe")
                .font(.title3.weight(.medium))
            Text("PDFs and images become Markdown on this Mac. Nothing is uploaded.")
                .font(.callout)
                .foregroundStyle(.secondary)
            // Dragging is the fast path, not the only one: a file already open in another app, or
            // one reached through the sidebar, has no window to drag from.
            Button("Choose Document\u{2026}", action: onChoose)
                .buttonStyle(.borderedProminent)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [7, 5]))
                .foregroundStyle(targeted ? Color.accentColor : Color.secondary.opacity(0.35))
                .padding(28)
        }
        .animation(.easeOut(duration: 0.15), value: targeted)
    }
}

private struct DropOverlay: View {
    var body: some View {
        Label("Drop to transcribe", systemImage: "arrow.down.doc")
            .font(.callout.weight(.medium))
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .glassChip()
            .padding(.top, 14)
            .transition(.opacity)
    }
}

private struct Failure: View {
    let message: String
    var onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.orange)
            Text("Could not transcribe that document").font(.title3.weight(.medium))
            ScrollView {
                Text(message)
                    .font(.callout.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: 520, alignment: .leading)
            }
            .frame(maxHeight: 160)
            Button("Start Over", action: onDismiss)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
}

private struct ModelMissing: View {
    var body: some View {
        CenteredHint(symbol: "arrow.down.circle",
                     title: "The OCR model is not downloaded",
                     detail: "Settings > Storage > OCR model. It is about 4.5 GB and runs entirely on this Mac.")
    }
}

private struct CenteredHint: View {
    let symbol: String
    let title: String
    let detail: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.secondary)
            Text(title).font(.title3.weight(.medium))
            Text(detail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
