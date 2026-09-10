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
    @Environment(AppModel.self) private var model
    @State private var dropTargeted = false
    @State private var split = SplitScroll()

    /// The add-on is not here yet. Said in the empty state rather than only when a drop fails: a
    /// pane that invites a document it cannot read is a trap.
    private var needsModel: Bool { !model.ocrInstalled.contains(.balanced) }

    var body: some View {
        @Bindable var session = session
        Group {
            switch session.phase {
            case .empty:
                SearchWaysPrompt(title: "Drop a document to transcribe",
                                 symbol: "text.viewfinder",
                                 ways: SearchWaysPrompt.transcribeWays,
                                 // Something to click. The rows describe ways in, but an empty
                                 // pane whose only affordance is a drag leaves anyone without a
                                 // file already in hand with nothing to do.
                                 footer: needsModel
                                     ? AnyView(OCRDownloadAction())
                                     : AnyView(Button("Choose Files\u{2026}") { session.chooseAndOpen() }
                                         .controlSize(.large)
                                         .accessibilityIdentifier("ocr.choosefiles")))
                    .accessibilityIdentifier(needsModel ? "ocr.needsmodel" : "ocr.dropzone")
                    .task { model.refreshOCRInstalled() }
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
            // Say which of the two waits this is. "Queued" was shown for both, and for the first
            // run of a session it was simply wrong: nothing is queued behind anything, the app is
            // reading four and a half gigabytes of weights off disk.
            CenteredHint(symbol: session.phase == .loading ? "gearshape.arrow.trianglehead.2.clockwise.rotate.90" : "text.viewfinder",
                         title: waitingTitle, detail: "", spinner: session.isBusy)
        } else {
            switch session.mode {
            case .rendered:
                RenderedDocument(sync: $split)
            case .raw:
                RawDocument(sync: $split)
            case .split, .triple:
                // ONE case for both, so the two text columns keep their identity when the image
                // column is added or removed. As separate `switch` branches SwiftUI saw two
                // unrelated hierarchies and rebuilt 40 sections in each pane on every toggle -
                // measured at 2.7 s on a 40-page transcript, for a change that only concerns a
                // third column.
                //
                // A plain proportional split, not HSplitView. HSplitView propagates its children's
                // minimum widths up as its own, and inside a NavigationSplitView detail pane that
                // squeezes the app's sidebar past its own minimum - the sidebar labels start
                // clipping. Halves that simply divide what is available cannot do that.
                // Scrolling either half moves the other. The pointer decides which one leads, so
                // they cannot chase each other; they meet at page boundaries, which is the
                // granularity both panes share - the same text sets to different heights.
                HStack(spacing: 0) {
                    // The page itself beside what was read off it. The image column follows
                    // whichever text column is being scrolled, which the split's sync knows.
                    if session.mode == .triple {
                        PageImage(id: split.section ?? session.visibleIndex)
                            .frame(maxWidth: .infinity)
                        Divider()
                    }
                    RawDocument(side: 0, sync: $split)
                        .frame(maxWidth: .infinity)
                    Divider()
                    RenderedDocument(side: 1, sync: $split)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    /// Three different waits, three different labels. "Queued" was shown for all of them, and on
    /// a first run it was simply untrue twice over: nothing is queued behind anything while the
    /// weights load, and nothing is queued while this document's own first page is being read -
    /// at a wide batch that alone is twenty seconds of prefill before a single token lands.
    private var waitingTitle: String {
        if session.phase == .loading { return "Loading the model" }
        if visibleDocumentIsRunning { return "Reading the page" }
        return session.isBusy ? "Queued" : "Preparing"
    }

    private var visibleDocumentIsRunning: Bool {
        guard let doc = session.visibleDocument else { return false }
        return doc.pageIDs.contains {
            session.pages.indices.contains($0) && session.pages[$0].state == .running
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
                .help("Raw text, Markdown, both, or the page beside them")
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
                // Eager, like the transcript's sections and for the same reason: `scrollTo` cannot
                // reach a row a lazy stack has not built, which is exactly the row a run is moving
                // towards. One small view per page is tens of views, not thousands.
                VStack(spacing: 2) {
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
            // Not a `List`. A sidebar list draws its own row highlight - the system selection,
            // which greys out the moment the text pane takes focus, and a hover fill on top of it -
            // so the accent mark this rail needs ended up sitting inside a second background. Here
            // the selection means "the page you are looking at", not "the focused row", and it is
            // drawn once, by the thumbnail.
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
        .opacity(untranscribed && !selected ? 0.45 : 1)
        .animation(.easeOut(duration: 0.25), value: page.state)
    }

    private var untranscribed: Bool { page.state == .pending || page.state == .stopped }

    private var stateDescription: String {
        switch page.state {
        case .done: return "\(page.tokens) tokens"
        case .running: return "transcribing"
        case .failed: return "could not be transcribed"
        case .pending: return "not transcribed yet"
        case .stopped: return "not transcribed - click to transcribe it"
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
                Button { session.closeDocument(id: doc.id) } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .frame(width: 14, height: 14)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .opacity(showsClose ? 1 : 0)
                // An invisible button still takes the click. Reserved space is the point; a hole
                // in the tab is not.
                .allowsHitTesting(showsClose)
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
        // made a long name crowd every other tab out. FULL HEIGHT too: the row is 28pt and the
        // label inside it is about 16, so without this the clickable band was the label's and the
        // few points above and below it did nothing.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        // A container element: without this the tab is only its children (a close button and a
        // label) and nothing answers to the tab itself - for VoiceOver or for a test.
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ocr.tab.\(doc.id)")
        .onTapGesture { session.selectDocument(id: doc.id) }
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
/// The stack is EAGER. A lazy one estimates the height of every section it has not built, and on a
/// document whose sections run from a two-line note to a hundred-row table that estimate was far
/// enough out that following the tail scrolled into empty space: the pane went blank for seconds
/// and the page arrived all at once when the layout caught up. Sections are per PAGE, so this is
/// tens of views and not thousands, and each one still reads only its own text - which is what
/// keeps the 24 Hz stream write cheap.
private struct RenderedDocument: View {
    var side: Int? = nil
    @Binding var sync: SplitScroll
    @Environment(OCRSession.self) private var session

    var body: some View {
        DocumentScroll(side: side, sync: $sync) { ids in
            // LAZY, and each element is exactly ONE subview. Those two go together: a lazy stack
            // addresses its subviews by index, so a `ForEach` body that resolves to a page break
            // AND a section - two subviews for every element but the first - breaks that indexing,
            // which is what made `scrollTo` unreachable and left the pane blank the first time
            // this was tried. Wrapping the pair in a `VStack` restores the one-to-one mapping.
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(ids, id: \.self) { id in
                    VStack(alignment: .leading, spacing: 0) {
                        if id != ids.first { PageBreak() }
                        RenderedSection(id: id, state: session.sectionState(id))
                    }
                    .modifier(SectionTop(id: id))
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
    var side: Int? = nil
    @Binding var sync: SplitScroll
    @Environment(OCRSession.self) private var session

    /// Editable only when this is the WHOLE pane. Beside another column it is the sectioned,
    /// read-only form: the editor is an `NSTextView` with no sections to report a scroll position
    /// from or scroll to, so a split with the editor in it could not keep its halves together.
    /// Editing the transcript is what the single Raw Text view is for.
    private var editable: Bool { side == nil && !session.isBusy && session.completedPages > 0 }

    var body: some View {
        Group {
            if editable {
                SourceEditor(document: session.visibleDocument?.id, page: session.visibleIndex,
                             find: session.find, activeMatch: session.activeMatch)
            } else {
                DocumentScroll(width: nil, side: side, sync: $sync) { ids in
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(ids, id: \.self) { id in
                            VStack(alignment: .leading, spacing: 0) {
                                if id != ids.first { PageBreak() }
                                RawSection(id: id)
                            }
                            .modifier(SectionTop(id: id))
                            .id(id)
                        }
                    }
                }
            }
        }
        .background(.background.secondary)
    }
}

/// An `NSTextView` that hands file drops to the workspace instead of pasting their paths.
///
/// AppKit registers a text view for `NSFilenamesPboardType` and friends, so it wins the drop over
/// any SwiftUI `dropDestination` above it and inserts the path as text. Declining is not enough
/// either - the drag has to be answered here, because this view is the deepest one under the
/// pointer. So it answers, and forwards.
final class OCRSourceTextView: NSTextView {
    var onFiles: (([URL]) -> Void)?

    static func scrollable() -> NSScrollView {
        let scroll = NSScrollView()
        let big = CGFloat.greatestFiniteMagnitude
        let container = NSTextContainer(size: NSSize(width: 0, height: big))
        container.widthTracksTextView = true
        let layout = NSLayoutManager()
        layout.addTextContainer(container)
        let storage = NSTextStorage()
        storage.addLayoutManager(layout)
        let text = OCRSourceTextView(frame: .zero, textContainer: container)
        text.autoresizingMask = [NSView.AutoresizingMask.width]
        text.isVerticallyResizable = true
        text.isHorizontallyResizable = false
        text.minSize = NSSize(width: 0, height: 0)
        text.maxSize = NSSize(width: big, height: big)
        scroll.documentView = text
        scroll.hasVerticalScroller = true
        return scroll
    }

    private func droppedFiles(_ sender: NSDraggingInfo) -> [URL] {
        let objects = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self]) as? [URL]
        return (objects ?? []).filter(\.isFileURL)
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFiles(sender).isEmpty ? super.draggingEntered(sender) : .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFiles(sender).isEmpty ? super.draggingUpdated(sender) : .copy
    }

    override func prepareForDragOperation(_ sender: NSDraggingInfo) -> Bool {
        droppedFiles(sender).isEmpty ? super.prepareForDragOperation(sender) : true
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = droppedFiles(sender)
        guard !urls.isEmpty else { return super.performDragOperation(sender) }
        onFiles?(urls)
        return true
    }
}

/// The editable Markdown source, highlighted.
///
/// An `NSTextView` rather than SwiftUI's `TextEditor`. The navigator has to be able to scroll this
/// to a page, which needs a character offset and a scroll call that `TextEditor` does not expose -
/// clicking a thumbnail in raw view moved the highlight and left the text where it was. AppKit's
/// text system also lays a document of this size out lazily, where handing SwiftUI one
/// `AttributedString` per frame does not.
///
/// Read-only while any page is still decoding, then editable as a whole. That split is a
/// correctness one, not a policy: the streaming form is per-page sections so it stays lazy and can
/// be scrolled to a page; the editable form is the single string that Copy and Save produce, so
/// what is edited is exactly what leaves the app.
private struct SourceEditor: NSViewRepresentable {
    /// Passed in rather than read inside `updateNSView`: a representable is re-run when its own
    /// value changes, and only a read in the PARENT's body is tracked by Observation.
    let document: Int?
    let page: Int?
    let find: String
    let activeMatch: Int
    @Environment(OCRSession.self) private var session

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = OCRSourceTextView.scrollable()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        guard let text = scroll.documentView as? OCRSourceTextView else { return scroll }
        // A file dropped on the editor is a document to transcribe, the same as everywhere else
        // in this pane. An `NSTextView` would otherwise take it as an insertion and paste the
        // path into the transcript, which is neither what was meant nor undoable in an obvious way.
        text.onFiles = { [weak session] urls in session?.open(urls: urls) }
        text.delegate = context.coordinator
        text.allowsUndo = true
        text.drawsBackground = false
        text.isAutomaticQuoteSubstitutionEnabled = false
        text.isAutomaticDashSubstitutionEnabled = false
        text.isAutomaticTextReplacementEnabled = false
        text.isAutomaticSpellingCorrectionEnabled = false
        text.textContainerInset = NSSize(width: 8, height: 14)
        text.font = context.coordinator.face
        text.typingAttributes = [.font: context.coordinator.face,
                                 .foregroundColor: NSColor.labelColor]
        context.coordinator.textView = text
        context.coordinator.session = session
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.session = session
        context.coordinator.show(document: document, page: page, find: find, activeMatch: activeMatch)
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
        let face = NSFont.monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)

        weak var textView: OCRSourceTextView?
        var session: OCRSession?
        private var loaded: Int??
        private var scrolledTo: Int?
        private var marked: String?
        private var steppedTo: Int?
        private var recolour: Task<Void, Never>?

        func show(document: Int?, page: Int?, find: String, activeMatch: Int) {
            guard let text = textView, let session else { return }
            let source = session.documentSource()
            let reload = loaded != .some(document)
            if reload {
                loaded = .some(document)
                scrolledTo = nil
                marked = nil
                steppedTo = nil
            }
            // Find marks live in the text storage, so a change of query rebuilds it. The syntax
            // pass underneath is memoised, and this only runs when the query itself changes.
            if reload || marked != find {
                marked = find
                let selected = text.selectedRange()
                let origin = text.enclosingScrollView?.contentView.bounds.origin ?? .zero
                text.textStorage?.setAttributedString(NSAttributedString(
                    FindHighlight.mark(find, in: MarkdownSource.highlighted(source.text))))
                if reload {
                    text.setSelectedRange(NSRange(location: 0, length: 0))
                    text.scroll(.zero)
                } else {
                    text.setSelectedRange(selected)
                    text.enclosingScrollView?.contentView.scroll(to: origin)
                }
                steppedTo = nil
            }
            // Step through matches the way the find bar's chevrons say it does.
            if !find.isEmpty, steppedTo != activeMatch,
               let range = Self.match(activeMatch, of: find, in: text.string) {
                steppedTo = activeMatch
                scrolledTo = page
                text.scrollRangeToVisible(range)
                return
            }
            guard let page, page != scrolledTo, let offset = source.offsets[page] else { return }
            scrolledTo = page
            let length = (text.string as NSString).length
            guard offset < length else { return }
            // Scroll the WHOLE page into view, not its first character. `scrollRangeToVisible`
            // moves the least it can, so a one-character range that happens to be below the fold
            // lands at the bottom edge; given the page's full extent it puts the start at the top.
            // Asking the layout manager for a rectangle instead is not an option here - the text
            // view is TextKit 2, and the geometry of a range it has not laid out yet is not there
            // to be read.
            let next = source.offsets.values.filter { $0 > offset }.min() ?? length
            text.scrollRangeToVisible(NSRange(location: offset, length: max(1, next - offset - 1)))
        }

        /// The nth occurrence, counted the way `OCRSession.rebuildMatches` counts them: in order
        /// through the same string the panes render.
        private static func match(_ index: Int, of find: String, in text: String) -> NSRange? {
            let ns = text as NSString
            var from = 0
            for step in 0 ... max(index, 0) {
                let found = ns.range(of: find, options: .caseInsensitive,
                                     range: NSRange(location: from, length: ns.length - from))
                guard found.location != NSNotFound else { return nil }
                if step == index { return found }
                from = found.upperBound
            }
            return nil
        }

        func textDidChange(_ notification: Notification) {
            guard let text = textView else { return }
            session?.setDocumentEdit(text.string)
            // Re-colouring under a live caret on every keystroke is what makes a hand-rolled
            // highlighter feel wrong; 200 ms after the typing stops is invisible.
            recolour?.cancel()
            recolour = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled, let text = self?.textView else { return }
                let selected = text.selectedRange()
                let scroll = text.enclosingScrollView?.contentView.bounds.origin ?? .zero
                text.textStorage?.setAttributedString(NSAttributedString(
                    FindHighlight.mark(self?.marked ?? "", in: MarkdownSource.highlighted(text.string))))
                text.setSelectedRange(selected)
                text.enclosingScrollView?.contentView.scroll(to: scroll)
            }
        }
    }
}

/// One section of the source, highlighted. Read-only: this is the streaming form.
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
    /// Which half of the split this is, and the section the OTHER half is showing. Nil outside
    /// side-by-side, where there is nothing to keep in step and the probes below cost nothing.
    var side: Int? = nil
    @Binding var sync: SplitScroll
    @ViewBuilder var content: ([Int]) -> Content
    @Environment(OCRSession.self) private var session
    /// Set when this half is moved to follow the other one, so it does not report that move back
    /// as a reader scrolling.
    @State private var quietUntil = Date.distantPast

    /// Outside the page-id namespace: page ids are non-negative.
    private var tailID: Int { -1 }
    private static var space: String { "ocr.scroll" }

    var body: some View {
        // The named space is on a view OUTSIDE the scroller. Named ON the `ScrollView` it is the
        // CONTENT's space, so a section's position in it never changes as you scroll - the probes
        // reported once at layout and then went quiet, and the two halves never moved together.
        ZStack {
            scroller
        }
        .coordinateSpace(name: Self.space)
    }

    private var scroller: some View {
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
                    Color.clear.frame(height: 72).id(tailID).modifier(SectionTop(id: tailID))
                }
                .frame(maxWidth: width ?? .infinity, alignment: .leading)
                .padding(.horizontal, width == nil ? 12 : 28)
                .padding(.top, 24)
                .frame(maxWidth: .infinity, alignment: .center)
                // Sections report where they sit only in the split, and only while the pointer is
                // in THIS half: one probe per section per frame is not something to pay for in a
                // single-pane view that has nothing to keep in step with.
                .environment(\.sectionProbeSpace, side == nil ? nil : Self.space)
            }
            // Whichever half MOVED ON ITS OWN leads. Not hover, and not the scroll phase: the
            // pointer can sit over one half while the wheel turns in the other, and a follower that
            // reported the position it was just moved to would start the two panes chasing each
            // other. So the follower simply stays quiet for a moment after being moved, and
            // anything that moves outside that window is a reader scrolling.
            .onPreferenceChange(SectionTopKey.self) { tops in
                guard let side, quietUntil < Date(), let now = Self.position(tops) else { return }
                sync = SplitScroll(driver: side, section: now.section, fraction: now.fraction)
            }
            .onChange(of: sync) { _, now in
                guard let side, now.driver != side, let id = now.section else { return }
                quietUntil = Date().addingTimeInterval(0.2)
                // Anchored by the FRACTION of the section that has gone past, not just its top: the
                // same page sets to very different heights as source and as prose, so matching only
                // the page boundary left the follower a whole page behind by the time the driver
                // reached the end of one.
                proxy.scrollTo(id, anchor: UnitPoint(x: 0, y: now.fraction))
            }
            .onChange(of: session.streamTick) { _, _ in
                guard session.isFollowingRun else { return }
                // With a group in flight every page in it grows at once, so the tail is the LAST
                // page of the group rather than the one being read. Follow the page the navigator
                // is marking instead, which is the first of the group.
                if session.isGroupRunning, let id = session.visibleIndex {
                    proxy.scrollTo(id, anchor: .top)
                } else {
                    proxy.scrollTo(tailID, anchor: .bottom)
                }
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

    /// The section under the viewport's top edge, and how far into it the reader is.
    private static func position(_ tops: [Int: CGFloat]) -> (section: Int, fraction: Double)? {
        let above = tops.filter { $0.value <= 1 }
        guard let current = above.max(by: { $0.value < $1.value })
                ?? tops.min(by: { $0.value < $1.value }) else { return nil }
        // The next thing down is what gives this one its height. The tail spacer reports too, so
        // even the last section has one.
        let next = tops.values.filter { $0 > current.value }.min()
        let height = max((next ?? current.value + 1) - current.value, 1)
        return (current.key, min(max(Double(-current.value / height), 0), 1))
    }
}

/// Which half of a side-by-side view the pointer is in, and where in the document it is.
struct SplitScroll: Equatable {
    var driver: Int?
    var section: Int?
    var fraction: Double = 0
}

private struct SectionTopKey: PreferenceKey {
    static let defaultValue: [Int: CGFloat] = [:]
    static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) {
        value.merge(nextValue()) { first, _ in first }
    }
}

/// The coordinate space a section should report its position in, or nil for "do not report".
private struct SectionProbeSpaceKey: EnvironmentKey { static let defaultValue: String? = nil }

extension EnvironmentValues {
    var sectionProbeSpace: String? {
        get { self[SectionProbeSpaceKey.self] }
        set { self[SectionProbeSpaceKey.self] = newValue }
    }
}

/// Reports a section's distance from the top of the scroll's viewport, when asked to.
struct SectionTop: ViewModifier {
    let id: Int
    @Environment(\.sectionProbeSpace) private var space

    func body(content: Content) -> some View {
        content.background {
            if let space {
                GeometryReader { geometry in
                    Color.clear.preference(key: SectionTopKey.self,
                                           value: [id: geometry.frame(in: .named(space)).minY])
                }
            }
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

/// The scanned page, beside the text taken off it.
///
/// Rendered through the same cache Quick Look uses, so opening a page full size and reading it in
/// this column cost one render between them.
private struct PageImage: View {
    let id: Int?
    @Environment(OCRSession.self) private var session

    var body: some View {
        ZStack {
            if let id, let url = session.previewURL(for: id), let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .overlay(Rectangle().strokeBorder(Color.primary.opacity(0.12), lineWidth: 0.5))
                    .shadow(color: .black.opacity(0.16), radius: 2, y: 1)
                    .padding(16)
            } else {
                Image(systemName: "doc")
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background.secondary)
        .accessibilityLabel(id.map { "Page \($0 + 1)" } ?? "No page")
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
                    // Red, because it is the one control here that throws work away: pausing and
                    // resuming are free, stopping is not.
                    ChipButton(symbol: "stop.fill", help: "Stop transcribing  \u{2318}.",
                               tint: .red) { session.cancel() }
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
        } else if session.queueTotal > 1 {
            CloudSyncPie(fraction: session.progress)
        } else {
            ProgressView().progressViewStyle(.circular).controlSize(.small)
        }
    }

    private var headline: String {
        // Pages FINISHED over pages queued, running or not. A group decodes several pages at once,
        // so there is no single page being worked on, and naming the group's range answered a
        // question nobody asked - how far along is this - with the width of the batch.
        "\(session.queueCompleted) of \(session.queueTotal) pages"
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
    var tint: Color?
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .semibold))
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(tint ?? .primary)
                .frame(width: 22, height: 22)
                .background(Circle().fill((tint ?? .primary).opacity(hovered ? 0.12 : 0)))
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

    /// A file that is already text is not a failure, it is a file that does not need this pane.
    /// Saying "could not transcribe" over it reads as a defect in the app or the document.
    private var isNothingToDo: Bool { message.contains("already text") }

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: isNothingToDo ? "text.document" : "exclamationmark.triangle")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(isNothingToDo ? AnyShapeStyle(.secondary) : AnyShapeStyle(.orange))
            Text(isNothingToDo ? "Nothing to transcribe" : "Could not transcribe that document")
                .font(.title3.weight(.medium))
            ScrollView {
                Text(message)
                    .font(.callout.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: 520, alignment: .leading)
            }
            .frame(maxHeight: 160)
            Button(isNothingToDo ? "Choose Another" : "Start Over", action: onDismiss)
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
            Text("It runs entirely on this Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
            OCRDownloadAction().padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Get the model from where you noticed it was missing.
///
/// The same control the first-run screen offers, in the workspace: pointing at Settings sent the
/// reader to another window to answer a question they had already answered by turning OCR on.
struct OCRDownloadAction: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.isOCRDownloading {
            VStack(spacing: 8) {
                ProgressView(value: model.ocrDownloadFraction).frame(width: 300)
                HStack(spacing: 10) {
                    Text(model.ocrDownloadLabel)
                    Text(model.ocrDownloadSpeed).foregroundStyle(.secondary)
                }
                .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                Button("Cancel") { model.cancelOCRDownload() }.controlSize(.small)
            }
        } else {
            VStack(spacing: 6) {
                Button { model.downloadOCRModel(.balanced) } label: {
                    HStack {
                        Image(systemName: "arrow.down.circle")
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Download OCR model").fontWeight(.medium)
                            Text("~4.5 GB").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .font(.callout)
                    .fixedSize(horizontal: true, vertical: false)
                }
                .controlSize(.large).buttonStyle(.borderedProminent)
                if model.ocrDownloadFailed {
                    Text(model.ocrDownloadLabel).font(.caption).foregroundStyle(.red)
                }
            }
        }
    }
}

private struct CenteredHint: View {
    let symbol: String
    let title: String
    let detail: String
    var spinner: Bool = false

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
            if spinner { ProgressView().controlSize(.small).padding(.top, 2) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}


