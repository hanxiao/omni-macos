import AppKit
import OmniKit
import SwiftUI
import UniformTypeIdentifiers

/// The OCR workspace: drop a document, watch it transcribe, read the Markdown.
///
/// Layout follows the platform's shape rather than inventing one. The page navigator is a real
/// `inspector` - the system's trailing utility column - not a hand-rolled fixed-width strip: that
/// is what gives it a draggable divider, a remembered width, the standard show/hide behaviour and
/// a toolbar button in the slot the HIG reserves for exactly this ("the trailing edge contains
/// buttons that open nearby inspectors"). Inside it the pages are a `List` with a selection
/// binding, so arrow keys, the focus ring and VoiceOver all work without writing any of them.
///
/// The only element floating over the content is the progress readout, because a progress bar
/// pinned to a toolbar tells you nothing while you are reading, and it withdraws a few seconds
/// after the run ends rather than sitting on the text forever.
struct OCRView: View {
    @Environment(OCRSession.self) private var session
    @State private var dropTargeted = false
    @State private var copied = false

    var body: some View {
        @Bindable var session = session
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
        .inspector(isPresented: railBinding) {
            PageRail()
                .inspectorColumnWidth(min: 132, ideal: 168, max: 280)
        }
        .dropDestination(for: URL.self) { urls, _ in
            session.open(urls: urls)
            return true
        } isTargeted: { dropTargeted = $0 }
        .overlay(alignment: .top) {
            if dropTargeted && session.phase != .empty { DropOverlay() }
        }
        .overlay(alignment: .top) {
            if let notice = session.notice { NoticeChip(text: notice) }
        }
        .animation(.easeOut(duration: 0.2), value: session.notice)
        // Escape is the system's "stop what you are doing". It reaches here because the workspace
        // is the focused content of the window.
        .onExitCommand { if session.isBusy { session.cancel() } }
        .toolbar { toolbar }
    }

    /// The navigator is only meaningful once there are pages; `railVisible` remembers whether the
    /// user closed it so it stays closed across documents.
    private var railBinding: Binding<Bool> {
        Binding(get: { session.railVisible && !session.pages.isEmpty },
                set: { session.railVisible = $0 })
    }

    // MARK: - Workspace

    private var workspace: some View {
        content
            .overlay(alignment: .bottom) {
                if session.readoutVisible { ProgressReadout() }
            }
            // The readout's own `.transition` can only play if the animation is attached where the
            // condition lives. Declared on the view itself it animated nothing.
            .animation(.easeOut(duration: 0.28), value: session.readoutVisible)
    }

    @ViewBuilder private var content: some View {
        if let index = session.visibleIndex {
            let page = session.pages[index]
            switch session.mode {
            case .rendered:
                MarkdownPane(index: index, streaming: page.state == .running)
            case .raw:
                RawPane(index: index, page: page)
            case .split:
                // A plain proportional split, not HSplitView. HSplitView propagates its children's
                // minimum widths up as its own, and inside a NavigationSplitView detail pane that
                // squeezes the app's sidebar past its own minimum - the sidebar labels start
                // clipping. Halves that simply divide what is available cannot do that.
                HStack(spacing: 0) {
                    MarkdownPane(index: index, streaming: page.state == .running)
                        .frame(maxWidth: .infinity)
                    Divider()
                    RawPane(index: index, page: page)
                        .frame(maxWidth: .infinity)
                }
            }
        } else {
            CenteredHint(symbol: "text.viewfinder", title: "Preparing",
                         detail: "Rendering pages and loading the model.")
        }
    }

    private func chooseFiles() { session.chooseAndOpen() }

    // MARK: - Toolbar

    /// Three groups, which is the HIG's stated maximum: how the text is shown, what to do with it,
    /// and the navigator toggle. `ToolbarSpacer` is what separates them into distinct Liquid Glass
    /// surfaces on Tahoe - without it every OCR control shares one pill with the mode toggle that
    /// belongs to the window, and the picker reads as part of the same control.
    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        if hasDocument {
            if #available(macOS 26.0, *) { ToolbarSpacer(.fixed) }
            ToolbarItem(id: "ocr.view", placement: .primaryAction) {
                Picker("View", selection: Binding(get: { session.mode }, set: { session.mode = $0 })) {
                    ForEach(OCRSession.ViewMode.allCases) { m in
                        Label(m.label, systemImage: m.symbol).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .labelStyle(.iconOnly)
                .help("Switch between formatted Markdown, side by side, and raw text")
            }
            if #available(macOS 26.0, *) { ToolbarSpacer(.fixed) }
            ToolbarItem(id: "ocr.copy", placement: .primaryAction) {
                Button {
                    copyMarkdown()
                } label: {
                    Label(copied ? "Copied" : "Copy Markdown",
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                }
                .help("Copy the whole document as Markdown  \u{21e7}\u{2318}C")
                .disabled(session.completedPages == 0)
            }
            ToolbarItem(id: "ocr.export", placement: .primaryAction) {
                Button { exportMarkdown() } label: {
                    Label("Save Markdown\u{2026}", systemImage: "square.and.arrow.down")
                }
                .help("Save the transcription as a .md file  \u{2318}S")
                .disabled(session.completedPages == 0)
            }
            ToolbarItem(id: "ocr.open", placement: .primaryAction) {
                Button { chooseFiles() } label: {
                    Label("Open Document", systemImage: "folder")
                }
                .help("Open another document  \u{2318}O")
            }
            // Closing lives in the File menu only (Shift-Cmd-W). It is rare, it is undone by
            // reopening, and a toolbar earns its density from what people reach for often.
            if #available(macOS 26.0, *) { ToolbarSpacer(.fixed) }
            ToolbarItem(id: "ocr.rail", placement: .primaryAction) {
                Button {
                    withAnimation(.easeOut(duration: 0.2)) { session.railVisible.toggle() }
                } label: {
                    Label("Pages", systemImage: "sidebar.trailing")
                }
                .help("Show or hide the page list")
            }
        }
    }

    private var hasDocument: Bool {
        session.phase != .empty && session.phase != .needsModel && !session.pages.isEmpty
    }

    private func copyMarkdown() {
        session.copyMarkdownToPasteboard()
        // Every action needs feedback; a copy that changes nothing on screen reads as a no-op.
        withAnimation(.easeOut(duration: 0.15)) { copied = true }
        Task {
            try? await Task.sleep(for: .seconds(1.6))
            withAnimation(.easeOut(duration: 0.2)) { copied = false }
        }
    }

    private func exportMarkdown() { session.exportMarkdown() }
}

// MARK: - Page navigator

/// The pages, as a real `List` with a selection binding: arrow keys, focus ring, selection colour
/// and VoiceOver come from the platform. The previous version was a `LazyVStack` of tap gestures,
/// which had none of those and could not be driven from the keyboard at all.
private struct PageRail: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollViewReader { proxy in
            List(selection: Binding<Int?>(get: { session.visibleIndex },
                                          set: { if let id = $0 { session.select(id) } })) {
                ForEach(session.pages) { page in
                    PageThumb(page: page)
                        .id(page.id)
                        .tag(page.id)
                        .listRowInsets(EdgeInsets(top: 3, leading: 4, bottom: 3, trailing: 4))
                        .listRowSeparator(.hidden)
                }
            }
            .listStyle(.sidebar)
            .onChange(of: session.visibleIndex) { _, id in
                guard let id else { return }
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .center) }
            }
        }
        .accessibilityLabel("Pages")
    }
}

private struct PageThumb: View {
    let page: OCRSession.Page
    @Environment(OCRSession.self) private var session

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
            .frame(height: 132)
            // Unprocessed pages are dimmed rather than hidden: the rail doubles as the progress
            // display, so the shape of what is left has to stay visible.
            .opacity(page.state == .done ? 1 : (page.state == .running ? 0.85 : 0.4))
            .animation(.easeOut(duration: 0.25), value: page.state)

            Text(page.label)
                .font(.caption2)
                .lineLimit(1)
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        // Content OUT. A transcribed page drags into Notes, Mail, TextEdit or any editor as its
        // Markdown; the whole document goes to disk through Save.
        .draggable(page.state == .done ? session.displayText(at: page.id) : "")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(page.label)
        .accessibilityValue(stateDescription)
        .help("\(page.label) - \(stateDescription)")
    }

    private var stateDescription: String {
        switch page.state {
        case .done: return "\(page.tokens) tokens"
        case .running: return "transcribing"
        case .failed: return "could not be transcribed"
        case .pending: return "not transcribed yet"
        }
    }
}

// MARK: - Panes

/// Reads the page text itself rather than taking it as a parameter. Passing it down would make
/// `OCRView`'s body - and with it the toolbar and the inspector - re-evaluate on every streamed
/// token; here the 24 Hz write reaches only the view that draws it.
private struct MarkdownPane: View {
    let index: Int
    let streaming: Bool
    @Environment(OCRSession.self) private var session

    var body: some View {
        let blocks = MarkdownBlock.parse(session.displayText(at: index))
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                    block.view.transition(StreamFade.transition)
                }
                if streaming { TypingCaret() }
            }
            .modifier(StreamFade(count: blocks.count))
            .frame(maxWidth: 760, alignment: .leading)
            .padding(.horizontal, 28)
            .padding(.top, 24)
            // Room for the floating readout, so the last line of a page is never parked underneath it.
            .padding(.bottom, 72)
            .frame(maxWidth: .infinity, alignment: .center)
            .textSelection(.enabled)
        }
    }
}

/// Raw Markdown. Editable only once the page has finished decoding.
///
/// That is not a policy choice, it is a correctness one. A `TextEditor` bound to a getter that
/// reads streaming text and a setter that writes state mutates state during view update, and
/// SwiftUI responds by wedging the subtree - the whole side-by-side pane stopped refreshing while
/// the formatted view beside it kept streaming. Editing text that the next token is about to
/// overwrite was never meaningful anyway.
private struct RawPane: View {
    let index: Int
    let page: OCRSession.Page
    @Environment(OCRSession.self) private var session

    private var editable: Bool { page.state == .done }

    var body: some View {
        Group {
            if editable {
                TextEditor(text: Binding(get: { session.displayText(at: index) },
                                         set: { session.setEdit($0, at: index) }))
                    .font(.system(.body, design: .monospaced))
                    .scrollContentBackground(.hidden)
            } else {
                ScrollView {
                    Text(session.displayText(at: index))
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 5)
                        .padding(.top, 8)
                        .padding(.bottom, 72)
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
/// Under Reduce Motion it holds steady instead of blinking: a permanent animation is exactly what
/// that setting exists to stop.
private struct TypingCaret: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var on = true

    var body: some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(Color.accentColor)
            .frame(width: 7, height: 15)
            .opacity(on ? 1 : 0.15)
            .task {
                guard !reduceMotion else { return }
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
                indicator
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
                    .help("Stop transcribing  \u{238b}")
                    .accessibilityLabel("Stop transcribing")
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .glassChip(interactive: session.isBusy)
        }
        .shadow(color: .black.opacity(0.16), radius: 10, y: 3)
        .padding(.bottom, 18)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .accessibilityElement(children: .contain)
    }

    /// A determinate bar only where there is something determinate to show. macOS renders the
    /// circular style as a spinner, so a multi-page run's progress fraction was invisible; a
    /// single image has no fraction worth drawing, and gets the spinner honestly.
    @ViewBuilder private var indicator: some View {
        if !session.isBusy {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        } else if session.pages.count > 1 {
            ProgressView(value: session.progress)
                .progressViewStyle(.linear)
                .frame(width: 74)
        } else {
            ProgressView().progressViewStyle(.circular).controlSize(.small)
        }
    }

    private var detail: String {
        var parts: [String] = []
        if session.currentTokensPerSecond > 0 {
            parts.append(String(format: "%.0f tok/s", session.currentTokensPerSecond))
        }
        if session.elapsed > 0 { parts.append(session.elapsedText) }
        parts.append(session.documentName)
        return parts.joined(separator: "  \u{b7}  ")
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

/// A problem worth saying out loud that is not worth losing the document over.
private struct NoticeChip: View {
    let text: String

    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle")
            .font(.callout)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .glassChip()
            .shadow(color: .black.opacity(0.14), radius: 8, y: 2)
            .padding(.top, 14)
            .transition(.move(edge: .top).combined(with: .opacity))
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
                .keyboardShortcut(.defaultAction)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
}

/// An empty state with no way out of it is a dead end. This one carries the action it is asking
/// for, rather than naming a Settings pane and leaving the user to find it.
private struct ModelMissing: View {
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "arrow.down.circle")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.secondary)
            Text("The OCR model is not downloaded").font(.title3.weight(.medium))
            Text("About 4.5 GB, and it runs entirely on this Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
            SettingsLink {
                Text("Open Settings\u{2026}")
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
