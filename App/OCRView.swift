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
    /// Drives Quick Look, exactly as the results list does.
    @State private var previewURL: URL?

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
            PageRail(onPreview: { previewURL = session.previewURL(for: $0) })
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
            if let notice = session.notice {
                NoticeChip(text: notice, symbol: session.noticeSymbol)
            }
        }
        .animation(.easeOut(duration: 0.2), value: session.notice)
        // Escape is the system's "stop what you are doing". It reaches here because the workspace
        // is the focused content of the window.
        .onExitCommand { if session.isBusy { session.cancel() } }
        .quickLookPreview($previewURL)
        // Space previews the current page, the way it does in Finder and in the results list. A
        // focus-based key handler is not enough: the navigator List swallows the space key before
        // an ancestor sees it, which is the same reason the results list uses this monitor.
        .background(QuickLookKeyMonitor(
            onSpace: { previewCurrentPage() },
            onPreviewArrow: { vertical, forward in
                guard previewURL != nil, vertical else { return false }
                session.step(by: forward ? 1 : -1)
                previewCurrentPage(force: true)
                return true
            },
            isPreviewOpen: { previewURL != nil }))
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
        VStack(spacing: 0) {
            if session.documents.count > 1 {
                DocumentTabs()
                Divider()
            }
            content
        }
            .overlay(alignment: .bottom) {
                if session.readoutVisible { ProgressReadout() }
            }
            // The readout's own `.transition` can only play if the animation is attached where the
            // condition lives. Declared on the view itself it animated nothing.
            .animation(.easeOut(duration: 0.28), value: session.readoutVisible)
    }

    @ViewBuilder private var content: some View {
        if session.sectionIDs.isEmpty {
            CenteredHint(symbol: "text.viewfinder", title: "Preparing",
                         detail: "Rendering pages and loading the model.")
        } else {
            switch session.mode {
            case .rendered:
                RenderedDocument()
            case .raw:
                RawDocument()
            case .split:
                // A plain proportional split, not HSplitView. HSplitView propagates its children's
                // minimum widths up as its own, and inside a NavigationSplitView detail pane that
                // squeezes the app's sidebar past its own minimum - the sidebar labels start
                // clipping. Halves that simply divide what is available cannot do that.
                HStack(spacing: 0) {
                    RenderedDocument()
                        .frame(maxWidth: .infinity)
                    Divider()
                    RawDocument()
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private func chooseFiles() { session.chooseAndOpen() }

    /// Space toggles; an arrow inside an open preview replaces it with the next page.
    private func previewCurrentPage(force: Bool = false) {
        if previewURL != nil && !force { previewURL = nil; return }
        guard let id = session.visibleIndex else { return }
        previewURL = session.previewURL(for: id)
    }

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
            // Closing lives in the File menu only (Shift-Cmd-W). It is rare, it is undone by
            // reopening, and a toolbar earns its density from what people reach for often.
            ToolbarItem(id: "ocr.open", placement: .primaryAction) {
                Button { chooseFiles() } label: {
                    Label("Open Document", systemImage: "folder")
                }
                .help("Open another document  \u{2318}O")
            }
            ToolbarItem(id: "ocr.export", placement: .primaryAction) {
                Button { exportMarkdown() } label: {
                    Label("Save Markdown\u{2026}", systemImage: "square.and.arrow.down")
                }
                .help("Save the transcription as a .md file  \u{2318}S")
                .disabled(session.completedPages == 0)
            }
            ToolbarItem(id: "ocr.copy", placement: .primaryAction) {
                Button { session.copyMarkdownToPasteboard() } label: {
                    Label("Copy Markdown", systemImage: "doc.on.doc")
                }
                .help("Copy the whole document as Markdown  \u{21e7}\u{2318}C")
                .disabled(session.completedPages == 0)
            }
            ToolbarItem(id: "ocr.share", placement: .primaryAction) {
                // The system share sheet, not a menu of our own: AirDrop, Mail, Messages, Notes
                // and every service the user has enabled come from the platform and stay current
                // without this app knowing about any of them.
                ShareLink(item: transcript,
                          preview: SharePreview(session.documentName,
                                                image: Image(systemName: "doc.plaintext")))
                    .help("Share the transcription")
                    .disabled(session.completedPages == 0)
            }
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

    /// What the share sheet hands over: a Markdown FILE, so a service that wants an attachment
    /// gets one and a service that wants text still gets the text.
    private var transcript: TranscriptFile {
        TranscriptFile(name: session.suggestedFileName, markdown: session.documentMarkdown)
    }

    private var hasDocument: Bool {
        session.phase != .empty && session.phase != .needsModel && !session.pages.isEmpty
    }

    private func exportMarkdown() { session.exportMarkdown() }
}

// MARK: - Page navigator

/// The pages, as a real `List` with a selection binding: arrow keys, focus ring, selection colour
/// and VoiceOver come from the platform. The previous version was a `LazyVStack` of tap gestures,
/// which had none of those and could not be driven from the keyboard at all.
private struct PageRail: View {
    var onPreview: (Int) -> Void
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollViewReader { proxy in
            List(selection: Binding<Int?>(get: { session.visibleIndex },
                                          set: { if let id = $0 { session.select(id) } })) {
                ForEach(session.visiblePages) { page in
                    PageThumb(page: page, onPreview: { onPreview(page.id) })
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
    /// Open the page itself, the way double-clicking a Finder icon does.
    var onPreview: () -> Void
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
        // High priority, on the row's own content: a List with a selection binding consumes
        // ordinary and simultaneous tap gestures before they arrive, so neither `.onTapGesture`
        // nor `.simultaneousGesture` on the row ever fires.
        .highPriorityGesture(TapGesture(count: 2).onEnded { onPreview() })
        .contextMenu {
            Button("Quick Look") { onPreview() }
        }
        // Content OUT. A transcribed page drags into Notes, Mail, TextEdit or any editor as its
        // Markdown; the whole document goes to disk through Save.
        .draggable(page.state == .done ? session.pageText(at: page.id) : "")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(page.label)
        .accessibilityValue(stateDescription)
        .help("\(page.label) - \(stateDescription).  Space or double-click to preview")
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

/// One tab per dropped file, the way Preview opens several documents at once. A multi-page PDF is
/// a single tab whose pages run continuously inside it - the page rule separates pages, the tab
/// separates FILES, and conflating the two is what made a four-file drop read as one 40-page
/// document with no way to tell where one ended.
///
/// The ring is the same `CloudSyncPie` the folder sidebar uses for indexing progress: the app
/// already has one way of saying "this much of this thing is done", and a second one would be a
/// second thing to learn.
private struct DocumentTabs: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 6) {
                ForEach(session.documents) { doc in
                    let selected = session.visibleDocument?.id == doc.id
                    HStack(spacing: 6) {
                        CloudSyncPie(fraction: session.progress(ofDocument: doc.id))
                        Text(doc.name)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Button {
                            session.closeDocument(doc.id)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption2.weight(.semibold))
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Close \(doc.name)")
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .frame(maxWidth: 240)
                    .background(selected ? AnyShapeStyle(.selection) : AnyShapeStyle(.clear),
                                in: RoundedRectangle(cornerRadius: Design.cornerSmall))
                    .contentShape(Rectangle())
                    .onTapGesture { session.selectDocument(doc.id) }
                    .help(doc.name)
                    .accessibilityAddTraits(selected ? [.isSelected, .isButton] : .isButton)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .scrollIndicators(.never)
        .background(.background.secondary)
    }
}

/// The transcription as ONE document: every page in order, separated by a dim rule.
///
/// Pages are sections of a document, not screens to page through. A reader scrolls a scan the way
/// they scroll the PDF it came from, and a transcript that only ever shows page k of n cannot be
/// read straight through, searched with one Find, or selected across a page break - which is most
/// of what anyone wants from a transcript. The navigator scrolls this view; it does not swap it.
///
/// `LazyVStack` is load-bearing, not tidiness: a 200-page document eagerly builds tens of thousands
/// of block views, and each visible section reads `texts`, so keeping the built set to what is on
/// screen is also what keeps the 24 Hz stream write cheap.
private struct RenderedDocument: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        DocumentScroll { ids in
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(ids, id: \.self) { id in
                    if id != ids.first { PageBreak() }
                    RenderedSection(id: id, state: session.sectionState(id))
                        .id(id)
                }
            }
            .textSelection(.enabled)
        }
    }
}


/// One section's blocks. Reads its own text rather than taking it as a parameter, so a streamed
/// token invalidates this section alone - not the document, the toolbar, or the sections above it.
private struct RenderedSection: View {
    let id: Int
    let state: OCRSession.PageState
    @Environment(OCRSession.self) private var session

    var body: some View {
        let blocks = MarkdownBlock.parse(session.sectionText(id))
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                block.view.transition(StreamFade.transition)
            }
            if state == .running { TypingCaret() }
            if state == .failed { PageFailed() }
        }
        .modifier(StreamFade(count: blocks.count))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The same document as Markdown source.
///
/// Read-only while any page is still decoding, then editable as a whole. That split is a
/// correctness one, not a policy: a `TextEditor` whose getter reads streaming text and whose setter
/// writes state mutates state during view update, and SwiftUI answers by wedging the subtree - the
/// side-by-side pane stopped refreshing while the formatted view beside it kept streaming. The
/// streaming form is also per-page sections so it stays lazy; the editable form is the single
/// string that Copy and Save produce, so what is edited is exactly what leaves the app.
private struct RawDocument: View {
    @Environment(OCRSession.self) private var session

    private var editable: Bool { !session.isBusy && session.completedPages > 0 }

    var body: some View {
        Group {
            if editable {
                SourceEditor()
            } else {
                DocumentScroll(width: nil) { ids in
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(ids, id: \.self) { id in
                            if id != ids.first { PageBreak() }
                            RawSection(id: id).id(id)
                        }
                    }
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

/// The editable Markdown source, highlighted.
///
/// Tahoe's `TextEditor` takes an `AttributedString` binding, so the syntax colouring survives
/// editing instead of being a read-only decoration; on macOS 14-15 the same text is edited plain,
/// which is what those systems offer. Re-highlighting is debounced rather than run per keystroke:
/// rebuilding the attributed text under a live caret on every character is what makes a
/// hand-rolled highlighter feel wrong, and 200 ms after the typing stops is invisible.
private struct SourceEditor: View {
    @Environment(OCRSession.self) private var session
    @State private var attributed = AttributedString()
    @State private var plain = ""

    var body: some View {
        Group {
            if #available(macOS 26.0, *) {
                TextEditor(text: $attributed)
                    .onChange(of: attributed) { _, new in
                        let text = String(new.characters)
                        guard text != plain else { return }   // our own re-highlight, not a keystroke
                        plain = text
                        session.setDocumentEdit(text)
                    }
                    .task(id: plain) {
                        try? await Task.sleep(for: .milliseconds(200))
                        guard !Task.isCancelled else { return }
                        let coloured = MarkdownSource.highlighted(plain)
                        if String(coloured.characters) == plain { attributed = coloured }
                    }
            } else {
                TextEditor(text: Binding(get: { session.documentMarkdown },
                                         set: { session.setDocumentEdit($0) }))
            }
        }
        .font(.system(.body, design: .monospaced))
        .scrollContentBackground(.hidden)
        // The same insets the read-only sections use. Without the top one the first line of the
        // document sits under the translucent toolbar when scrolled to the top.
        .padding(.horizontal, 8)
        .padding(.top, 24)
        .padding(.bottom, 8)
        .onAppear {
            plain = session.documentMarkdown
            attributed = MarkdownSource.highlighted(plain)
        }
    }
}

private struct RawSection: View {
    let id: Int
    @Environment(OCRSession.self) private var session

    var body: some View {
        Text(MarkdownSource.highlighted(session.sectionText(id)))
            .font(.system(.body, design: .monospaced))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The scroller both panes share: page sections, the reading measure, room for the floating
/// readout, and the one place that follows the run.
private struct DocumentScroll<Content: View>: View {
    var width: CGFloat? = 760
    @ViewBuilder var content: ([Int]) -> Content
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                content(session.sectionIDs)
                    .frame(maxWidth: width ?? .infinity, alignment: .leading)
                    .padding(.horizontal, width == nil ? 12 : 28)
                    .padding(.top, 24)
                    // Room for the floating readout, so the end of the document is never parked
                    // underneath it.
                    .padding(.bottom, 72)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            // Follows the run, and follows the navigator, through the same value: `visibleIndex` is
            // the running page until the user picks one. It does NOT change while a page decodes,
            // so the document never scrolls out from under someone mid-read.
            .onChange(of: session.visibleIndex) { _, id in
                guard let id else { return }
                withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .top) }
            }
            // Switching Formatted/Side by Side/Markdown builds a NEW scroller, which starts at the
            // top. Without this you land on page 1 of a document whose run is on page 13, and
            // `visibleIndex` has not changed so nothing above would put you back.
            .task {
                try? await Task.sleep(for: .milliseconds(40))   // let the lazy stack lay out first
                if let id = session.visibleIndex { proxy.scrollTo(id, anchor: .top) }
            }
            .modifier(YieldFollowOnScroll())
        }
    }
}

/// Hands the document back to the reader the moment they scroll it themselves.
///
/// Only user-driven phases count - `.animating` is this view's own scroll-to and would otherwise
/// switch following off the instant it turned on. On macOS 14, where scroll phases do not exist,
/// following stays on until a thumbnail is clicked; that is the behaviour without this, not a
/// regression from it.
private struct YieldFollowOnScroll: ViewModifier {
    @Environment(OCRSession.self) private var session

    func body(content: Content) -> some View {
        if #available(macOS 15.0, *) {
            content.onScrollPhaseChange { _, phase in
                if phase == .tracking || phase == .interacting { session.stopFollowing() }
            }
        } else {
            content
        }
    }
}

/// The page boundary. A dim rule and air, which is how a printed document marks one - not a
/// heading, and not a label repeating what the navigator already shows.
private struct PageBreak: View {
    var body: some View {
        Rectangle()
            .fill(Color.secondary.opacity(0.2))
            .frame(height: 1)
            .padding(.vertical, 26)
    }
}

/// A page that could not be transcribed still occupies its place in the document. Skipping it
/// silently would make the pages either side read as continuous when they are not.
private struct PageFailed: View {
    var body: some View {
        Label("This page could not be transcribed.", systemImage: "exclamationmark.triangle")
            .font(.callout)
            .foregroundStyle(.secondary)
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

                if session.isBusy && !session.isFollowingRun {
                    // The way back to the live end of a document you have scrolled away from.
                    // Without it, following is a door that only locks.
                    Button { session.follow() } label: {
                        Image(systemName: "arrow.down.to.line")
                    }
                    .buttonStyle(.borderless)
                    .help("Jump to the page being transcribed")
                    .accessibilityLabel("Jump to the page being transcribed")
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

/// The transcription as something the share sheet can hand to another app.
struct TranscriptFile: Transferable {
    let name: String
    let markdown: String

    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: OCRSession.markdownType) { file in
            Data(file.markdown.utf8)
        }
        .suggestedFileName { $0.name }
    }
}

/// A problem worth saying out loud that is not worth losing the document over.
private struct NoticeChip: View {
    let text: String
    let symbol: String

    var body: some View {
        Label(text, systemImage: symbol)
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
