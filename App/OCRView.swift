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

    var body: some View {
        @Bindable var session = session
        Group {
            switch session.phase {
            case .empty:
                SearchWaysPrompt(title: "Drop a document to transcribe",
                                 symbol: "text.viewfinder",
                                 ways: SearchWaysPrompt.transcribeWays)
                    .accessibilityIdentifier("ocr.dropzone")
            case .needsModel:
                ModelMissing().accessibilityIdentifier("ocr.needsmodel")
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
            // In every state, including the empty one: the drop zone that used to carry its own
            // targeting is gone, so this chip is the only feedback a drag gets.
            if dropTargeted { DropOverlay() }
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
        .quickLookPreview(Binding(get: { session.previewing }, set: { session.previewing = $0 }))
        // Space previews the current page, the way it does in Finder and in the results list. A
        // focus-based key handler is not enough: the navigator List swallows the space key before
        // an ancestor sees it, which is the same reason the results list uses this monitor.
        .background(QuickLookKeyMonitor(
            onSpace: { previewCurrentPage() },
            onPreviewArrow: { vertical, forward in
                guard session.previewing != nil, vertical else { return false }
                session.step(by: forward ? 1 : -1)
                previewCurrentPage(force: true)
                return true
            },
            isPreviewOpen: { session.previewing != nil }))
        .toolbar { toolbar }
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
            .overlay(alignment: .top) {
                if !session.find.isEmpty { FindBar() }
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
            // A tab opened while another document is still decoding is waiting its turn, and
            // saying so is the difference between a queue and a stall.
            CenteredHint(symbol: "text.viewfinder",
                         title: session.isBusy ? "Queued" : "Preparing", detail: "")
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
                    RawDocument()
                        .frame(maxWidth: .infinity)
                    Divider()
                    RenderedDocument()
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private func chooseFiles() { session.chooseAndOpen() }

    /// Space toggles; an arrow inside an open preview replaces it with the next page.
    private func previewCurrentPage(force: Bool = false) {
        if session.previewing != nil && !force { session.previewing = nil; return }
        guard let id = session.visibleIndex else { return }
        session.previewing = session.previewURL(for: id)
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
                .help("Raw text, Markdown, or both")
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
struct PageRail: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(session.visiblePages) { page in
                        PageThumb(page: page,
                                  selected: session.railSelection == page.id,
                                  onPreview: { session.previewing = session.previewURL(for: page.id) })
                            .id(page.id)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 6)
            }
            // A scroller and not a `List`. A sidebar list draws the SYSTEM's row selection, which
            // greys out the moment the text pane takes focus - and here the selection means "the
            // page you are looking at", not "the focused row", so it has to stay lit. Preview's own
            // navigator is a collection view for the same reason.
            .focusable()
            .focusEffectDisabled()
            .onMoveCommand { direction in
                switch direction {
                case .up: session.step(by: -1)
                case .down: session.step(by: 1)
                default: break
                }
            }
            // The rail follows the page being decoded. Once nothing is running it belongs to the
            // reader: scrolling it back to the selection under their hands is how a navigator
            // stops being usable.
            .onChange(of: session.visibleIndex) { _, id in
                guard let id, session.isBusy else { return }
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .center) }
            }
        }
        .accessibilityLabel("Pages")
    }
}

private struct PageThumb: View {
    let page: OCRSession.Page
    let selected: Bool
    /// Open the page itself, the way double-clicking a Finder icon does.
    var onPreview: () -> Void
    @Environment(OCRSession.self) private var session

    var body: some View {
        VStack(spacing: 5) {
            sheet
            Text(page.caption.isEmpty ? page.label : page.caption)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(selected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
        }
        .frame(maxWidth: .infinity)
        .padding(4)
        // The selection encloses the page AND its number, in the accent colour, the way Preview
        // marks the page you are on.
        .background {
            if selected { RoundedRectangle(cornerRadius: 8).fill(Color.accentColor) }
        }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { onPreview() }
        .onTapGesture { session.select(page.id) }
        // The app's existing file actions, on the file this page came from - not a second set of
        // them. Reveal and Open are the same `PhotoActions` calls the results list makes, so a
        // page behaves like any other file the app knows about.
        .contextMenu {
            Button("Quick Look") { onPreview() }
            if let url = session.sourceURL(for: page.id) {
                Divider()
                Button("Open in Preview") { PhotoActions.open(url.path) }
                Button("Reveal in Finder") { PhotoActions.reveal(paths: [url.path]) }
                Button("Copy Path") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(url.path, forType: .string)
                }
            }
        }
        // Content OUT. A transcribed page drags into Notes, Mail, TextEdit or any editor as its
        // Markdown; the whole document goes to disk through Save.
        .draggable(page.state == .done ? session.pageText(at: page.id) : "")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(page.label)
        .accessibilityValue(stateDescription)
        .help("\(page.label) - \(stateDescription).  Space or double-click to preview")
    }

    /// The page itself, at its own proportions.
    ///
    /// It used to sit on a grey plate the width of the rail, which letterboxed every portrait scan
    /// and drew a second edge around one that already had a border - two rectangles reading as a
    /// frame around a frame. A hairline and a soft drop shadow instead: a sheet of paper lying on
    /// the sidebar, which is what Preview draws and what makes a white page legible on a light
    /// background.
    @ViewBuilder private var sheet: some View {
        let placeholder = Color(nsColor: .textBackgroundColor)
        Group {
            if let thumbnail = page.thumbnail {
                Image(nsImage: thumbnail).resizable().aspectRatio(contentMode: .fit)
            } else {
                placeholder.aspectRatio(1 / 1.414, contentMode: .fit)   // A4, so rows do not jump
            }
        }
        .frame(maxHeight: 132)
        .overlay {
            if page.state == .running { ProgressView().controlSize(.small) }
            if page.state == .failed {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        }
        // A shadow and no drawn line. Preview's pages read as paper because they cast one, not
        // because anything is stroked around them; a hairline on top of it is the frame the rail
        // is not supposed to have.
        .shadow(color: .black.opacity(0.16), radius: 1.5, y: 0.5)
        // Unprocessed pages are dimmed rather than hidden: the rail doubles as the progress
        // display, so the shape of what is left has to stay visible. Never the selected page,
        // whatever its state: a translucent sheet over the accent fill turns the paper blue.
        .opacity(page.state == .pending && !selected ? 0.45 : 1)
        .animation(.easeOut(duration: 0.25), value: page.state)
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
    @State private var hovered: Int?

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(session.documents.enumerated()), id: \.element.id) { index, doc in
                tab(doc, at: index)
            }
        }
        .frame(height: 28)
        .padding(3)
        // An explicit window grey under an explicit control face, rather than two shades of
        // `.background`: the semantic pair is the one that keeps the selected tab LIGHTER than the
        // track in both appearances, which is the whole shape of a macOS tab bar. Two background
        // tints left the capsule and its track the same colour in light mode, so nothing read as
        // raised.
        .background(Color(nsColor: .windowBackgroundColor))
    }

    @ViewBuilder private func tab(_ doc: OCRSession.Document, at index: Int) -> some View {
        let selected = session.visibleDocument?.id == doc.id
        // On hover only, as Preview does. A close button parked on the active tab is Safari's
        // idiom, not this one, and it competes with the title for a narrow tab.
        let showsClose = hovered == doc.id
        ZStack {
            // The selected tab is a raised light capsule on the track; the others are flat on it,
            // separated by a hairline. That is the macOS document-tab shape - a selection FILL
            // behind every tab, which is what this had, is the list-row idiom, not the tab one.
            if selected {
                RoundedRectangle(cornerRadius: 7)
                    .fill(Color(nsColor: .controlColor))
                    .overlay(RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5))
                    .shadow(color: .black.opacity(0.14), radius: 1, y: 0.5)
                    .padding(.horizontal, 1)
            } else if index > 0, session.documents[index - 1].id != session.visibleDocument?.id {
                HStack {
                    Divider().frame(height: 14)
                    Spacer()
                }
            }

            HStack(spacing: 4) {
                // Close sits on the leading edge and appears on hover, as it does in Preview and
                // Safari. Reserved space rather than inserted space: a button that appears by
                // widening the row makes the title jump under the pointer.
                Button { session.closeDocument(doc.id) } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .frame(width: 14, height: 14)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .opacity(showsClose ? 1 : 0)
                .accessibilityLabel("Close \(doc.name)")
                .accessibilityHidden(!showsClose)

                Spacer(minLength: 0)
                if let fraction = session.progress(ofDocument: doc.id) {
                    CloudSyncPie(fraction: fraction).frame(width: 11, height: 11)
                }
                Text(doc.name)
                    .font(.system(size: 12))
                    .foregroundStyle(selected ? .primary : .secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 0)
                Color.clear.frame(width: 14, height: 14)   // balances the close button
            }
            .padding(.horizontal, 4)
        }
        // Equal widths, the way a tab bar divides its track - not sized to the file name, which
        // made a long name crowd every other tab out.
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        // A container element: without this the tab is only its children (a close button and a
        // label) and nothing answers to the tab itself - for VoiceOver or for a test.
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ocr.tab.\(doc.id)")
        .onTapGesture { session.selectDocument(doc.id) }
        .onHover { hovered = $0 ? doc.id : (hovered == doc.id ? nil : hovered) }
        .help(doc.name)
        .accessibilityAddTraits(selected ? [.isSelected, .isButton] : .isButton)
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
            if state == .running {
                // While the page is decoding, one view per block: arrival is what the fade is
                // driven by, and a block that appears inside a merged string cannot animate.
                ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                    block.view(highlighting: session.find).transition(StreamFade.transition)
                }
            } else {
                // Finished: contiguous prose merged into one `Text` so a drag selects across
                // paragraphs rather than stopping at each one.
                ForEach(MarkdownBlock.runs(blocks, highlighting: session.find)) { run in
                    switch run {
                    case .text(_, let text):
                        Text(text)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    case .block(_, let block):
                        block.view(highlighting: session.find)
                    }
                }
            }
            if state == .running { TypingCaret() }
            if state == .failed { PageFailed() }
        }
        .modifier(StreamFade(count: blocks.count))
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityIdentifier("ocr.section.\(id)")
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
                            RawSection(id: id)
                                .id(id)
                        }
                    }
                }
            }
        }
        .background(.background.secondary)
    }
}

/// The editable Markdown source, highlighted.
///
/// No line-number gutter. The hand-rolled one built a `Text` per line in an eager `VStack`, and on
/// a real transcript that view tree left the window's sidebar drawing NOTHING and unscrollable -
/// the page navigator went blank the moment a run finished. It also numbered logical lines, which
/// a soft-wrapping editor does not lay out one to a row. A number column is not worth a broken
/// navigator, and the packages that provide a real one do not fit (see CLAUDE.md).
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
        editor
        // Keyed on the TAB, not on appearance. Loading in `onAppear` meant the editor kept the
        // first document it was ever shown: selecting another tab moved the navigator and the
        // title and left the text where it was.
        .task(id: session.visibleDocument?.id) {
            plain = session.documentMarkdown
            attributed = MarkdownSource.highlighted(plain)
        }
    }

    @ViewBuilder private var editor: some View {
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
    }
}


private struct RawSection: View {
    let id: Int
    @Environment(OCRSession.self) private var session

    var body: some View {
        Text(FindHighlight.mark(session.find, in: MarkdownSource.highlighted(session.sectionText(id))))
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

    /// Outside the page-id namespace: page ids are non-negative.
    private var tailID: Int { -1 }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    content(session.sectionIDs)
                    // The tail anchor and the clearance under the floating readout, in one view.
                    // It sits OUTSIDE the lazy stack deliberately: `scrollTo` cannot reach an item
                    // a `LazyVStack` has not built yet, and while following a run the item it would
                    // have to reach is precisely the one just below the fold - so the scroll stuck
                    // about a page behind the decode, one page at a time. This view always exists,
                    // and putting ITS bottom at the viewport's leaves the newest line clear of the
                    // readout.
                    Color.clear.frame(height: 72).id(tailID)
                }
                .frame(maxWidth: width ?? .infinity, alignment: .leading)
                .padding(.horizontal, width == nil ? 12 : 28)
                .padding(.top, 24)
                .frame(maxWidth: .infinity, alignment: .center)
            }
            // Follows the run, and follows the navigator, through the same value: `visibleIndex` is
            // the running page until the user picks one. It does NOT change while a page decodes,
            // so the document never scrolls out from under someone mid-read.
            // Jump to the top of a page only when the READER asked for that page. While following
            // the run, the page that becomes current is still EMPTY at that moment, and putting an
            // empty section at the top of the viewport scrolls everything else out and leaves the
            // pane blank until the first token lands. The rendered pane hid that behind scroll
            // clamping; the taller raw pane showed it as a blank half of the window at every page
            // boundary. Following is the tail's job, below.
            .onChange(of: session.visibleIndex) { _, id in
                guard let id, !session.isFollowingRun else { return }
                withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .top) }
            }
            // Follow the text as it is written, not just when the page turns. A long page grows
            // well past the fold, and a reader watching it transcribe should not have to chase it
            // down the window. Unanimated on purpose: a 24 Hz animated scroll never settles, and
            // the whole point is that the last line stays where the eye already is.
            // Stepping through find matches moves the document to the section holding the match,
            // the way Preview scrolls to what it found.
            .onChange(of: session.activeMatchSection) { _, id in
                guard let id else { return }
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .center) }
            }
            .onChange(of: session.streamTick) { _, _ in
                guard session.isFollowingRun else { return }
                proxy.scrollTo(tailID, anchor: .bottom)
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
            HStack(spacing: 10) {
                indicator
                VStack(alignment: .leading, spacing: 1) {
                    Text(headline)
                        .font(.callout.weight(.medium))
                    Text(detail)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                if session.isBusy && !session.isFollowingRun {
                    // The way back to the live end of a document you have scrolled away from.
                    // Without it, following is a door that only locks.
                    ChipButton(symbol: "arrow.down.to.line",
                               help: "Jump to the page being transcribed") { session.follow() }
                }
                if session.isPaused {
                    ChipButton(symbol: "play.fill", help: "Resume transcribing") { session.resume() }
                    ChipButton(symbol: "stop.fill",
                               help: "Stop transcribing  \u{2318}.") { session.cancel() }
                } else if session.isBusy {
                    ChipButton(symbol: "pause.fill", help: "Pause transcribing") { session.pause() }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .glassChip(interactive: session.isBusy)
        }
        .shadow(color: .black.opacity(0.16), radius: 10, y: 3)
        .accessibilityIdentifier("ocr.readout")
        .padding(.bottom, 18)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .accessibilityElement(children: .contain)
        // The file name is the one thing here a reader already knows - it is in the tab bar and in
        // the window title - so it costs width in the chip and earns it back only when a document
        // is ambiguous. Hovering asks.
        .help(session.runningDocumentName)
    }

    /// The same ring the document tabs use, so progress means one thing everywhere in the app. A
    /// single image has no fraction worth drawing and gets an honest spinner.
    @ViewBuilder private var indicator: some View {
        if !session.isBusy {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        } else if session.batchTotal > 1 {
            CloudSyncPie(fraction: session.progress)
        } else {
            ProgressView().progressViewStyle(.circular).controlSize(.small)
        }
    }

    private var headline: String {
        session.isBusy
            ? "Page \(min(session.batchCompleted + 1, session.batchTotal)) of \(session.batchTotal)"
            : "\(session.batchCompleted) of \(session.batchTotal) pages"
    }

    private var detail: String {
        // A pause takes effect at the next page boundary, so say which of the two states this is
        // rather than claiming the run has stopped while a page is still decoding.
        if session.isPaused { return session.isHolding ? "Paused" : "Finishing this page" }
        var parts: [String] = []
        if session.currentTokensPerSecond > 0 {
            parts.append(String(format: "%.0f tok/s", session.currentTokensPerSecond))
        }
        if session.elapsed > 0 { parts.append(session.elapsedText) }
        return parts.joined(separator: "  \u{b7}  ")
    }
}

/// A control on the floating chip.
///
/// `.borderless` draws nothing under the pointer, so these read as glyphs rather than buttons until
/// one is clicked. This is the system's own hover treatment - a round fill that comes up under the
/// cursor - at chip scale, and the symbol swap between pause and play uses the replace effect so
/// the control morphs rather than blinking.
private struct ChipButton: View {
    let symbol: String
    let help: String
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .semibold))
                .contentTransition(.symbolEffect(.replace))
                .frame(width: 22, height: 22)
                .background(Circle().fill(.primary.opacity(hovered ? 0.12 : 0)))
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
        .animation(.easeOut(duration: 0.12), value: hovered)
        .help(help)
        .accessibilityLabel(help)
    }
}

// MARK: - Empty and error states

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

/// Match count and step controls for find.
///
/// It cannot live in the search field's prompt, which is the obvious place: a prompt is only shown
/// while the field is EMPTY, so the count would disappear at the exact moment there is one. This is
/// Safari's answer - a small bar that says where you are among the matches and lets you walk them.
private struct FindBar: View {
    @Environment(OCRSession.self) private var session

    var body: some View {
        GlassGroup(spacing: 10) {
            HStack(spacing: 8) {
                Text(session.matchCount == 0
                     ? "No matches"
                     : "\(session.activeMatch + 1) of \(session.matchCount)")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(session.matchCount == 0 ? .secondary : .primary)

                Button { session.stepMatch(by: -1) } label: { Image(systemName: "chevron.left") }
                    .help("Previous match  \u{21e7}\u{2318}G")
                    .accessibilityLabel("Previous match")
                Button { session.stepMatch(by: 1) } label: { Image(systemName: "chevron.right") }
                    .help("Next match  \u{2318}G")
                    .accessibilityLabel("Next match")
            }
            .buttonStyle(.borderless)
            .disabled(session.matchCount == 0)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .glassChip(interactive: session.matchCount > 0)
        }
        .shadow(color: .black.opacity(0.14), radius: 8, y: 2)
        .padding(.top, 12)
        .transition(.move(edge: .top).combined(with: .opacity))
        .accessibilityIdentifier("ocr.findbar")
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
            if !detail.isEmpty {
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 380)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
