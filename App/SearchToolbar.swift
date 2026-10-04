import SwiftUI
import AppKit
import OmniKit

/// The window toolbar outside search's OCR mode. A modifier of its own so that SwiftUI re-runs it
/// only when something it shows changes: as part of `ContentView.body` it was rebuilt on every
/// evaluation of the window, and each rebuild made AppKit re-lay out the items.
struct SearchToolbar: ViewModifier {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(OCRSession.self) private var ocr
    @Binding var columns: NavigationSplitViewVisibility
    /// What the detail pane is browsing, decided by `ContentView` (a query always wins).
    let browse: BrowseMode

    enum BrowseMode: Equatable { case none, recents, clipboardOff, photos, folder }
    private var showsBrowser: Bool { browse == .recents || browse == .photos || browse == .folder }

    func body(content: Content) -> some View {
        content.toolbar { toolbar }
    }

    // MARK: - Toolbar

    // On Tahoe, place the filter with sort/view (trailing) so the three result controls share one
    // Liquid Glass pill; on earlier macOS keep it leading so the existing toolbar layout is untouched.
    private var filterPlacement: ToolbarItemPlacement {
        if #available(macOS 26.0, *) { return .primaryAction } else { return .automatic }
    }

    /// Whether the back/forward chevrons have anything to show. Also decides whether the leading
    /// toolbar item gets a Liquid Glass background on Tahoe - an empty item must not draw one.
    private var showsHistoryControls: Bool {
        // NOT IN OCR MODE. This trail is the SEARCH history; stepping it while reading a transcript
        // changes a result set you cannot see, and leaves the chevrons looking like page navigation
        // for the document - which they are not. OCR has its own page rail for that.
        model.phase == .ready && !model.ocrMode && (model.canGoBack || model.canGoForward)
    }

    /// Only rendered inside an item that exists solely when `showsHistoryControls` is true, so
    /// there is no empty state here and no need for the 1pt clear fillers an always-present item
    /// used to carry against AppKit's "ambiguous width" warning.
    /// The leading navigation group: back/forward when there is somewhere to go, then the name of
    /// whatever is being browsed. 10pt between them - Finder's gap, measured at 11.
    @ViewBuilder private var navAndTitle: some View {
        HStack(spacing: 0) {
            // A completely empty toolbar item has zero intrinsic size and AppKit logs an
            // "ambiguous width/height" warning for it on every layout pass, so 1pt of clear keeps
            // it measurable while nothing is being browsed.
            Color.clear.frame(width: 1, height: 1)
            if showsHistoryControls {
                historyControls
                    .modifier(NavPill())
                    .padding(.trailing, 10)
            }
            browseTitle
            Color.clear.frame(width: 1, height: 1)
        }
    }

    /// The chevrons' own Liquid Glass capsule, since the toolbar item they sit in deliberately
    /// draws none (it also holds the name, which must stay outside the glass).
    private struct NavPill: ViewModifier {
        func body(content: Content) -> some View {
            if #available(macOS 26.0, *) {
                // Interactive: it holds buttons, and the glass should answer the pointer the way the
                // system's own toolbar capsules do.
                content.glassEffect(.regular.interactive(), in: .capsule)
            } else {
                content
            }
        }
    }

    /// ONE capsule holding both chevrons with a hairline between them - Finder's shape, measured
    /// at 75x28 with a divider down the middle. A `ControlGroup` cannot draw it here: with the
    /// toolbar item's shared background hidden (which it must be, so the name stays outside the
    /// glass) each of its buttons grew a capsule of ITS own and the pair rendered as two circles.
    /// So the capsule is drawn once, around a plain HStack.
    @ViewBuilder private var historyControls: some View {
        HStack(spacing: 0) {
            // The View menu owns Cmd-[ / Cmd-] (single owner, avoids a duplicate-shortcut
            // conflict); these buttons are click targets that name the same chords.
            navButton("chevron.backward", enabled: model.canGoBack,
                      help: "Back  \u{2318}[", label: "Back") { model.goBack() }
            Divider().frame(height: 15)
            navButton("chevron.forward", enabled: model.canGoForward,
                      help: "Forward  \u{2318}]", label: "Forward") { model.goForward() }
        }
        .fixedSize()
    }

    private func navButton(_ symbol: String, enabled: Bool, help: String, label: String,
                           action: @escaping () -> Void) -> some View {
        Button(action: action) {
            // Titled, for the same reason as sidebarToggleButton: a bare Image collapses the
            // enclosing toolbar item to 10x10 on Sequoia.
            Label(label, systemImage: symbol)
                // `.borderless` tints its own label and IGNORES this, which left the enabled
                // chevron at a measured 127 against the mode glyphs' 77 - so `.plain`, which does
                // not. Finder's own chevrons sample at 110 enabled and 196 disabled: secondary and
                // quaternary, not primary and tertiary. Its back arrow is not black either.
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(enabled ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
                // 36pt, which is what the SYSTEM draws for a toolbar item's capsule - measured off
                // the mode pill next door, whose fill runs y=8..43 in a toolbar strip where this
                // one ran y=12..39. Hand-drawing the capsule means hand-matching its height too.
                .frame(width: 37, height: 36)
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .help(help)
        .accessibilityLabel(label)
    }

    /// The name of whatever is being browsed, or nil when nothing is. Nil is what removes the
    /// toolbar item entirely - an always-present item holding an empty `Text` still cost a full
    /// Tahoe inter-group gap, which is the hole the name used to sit behind.
    private var browseTitleText: (name: String, help: String)? {
        // The open document, where every other mode puts the thing being looked at. Preview names
        // its document in the same slot; leaving it blank made OCR the one mode whose toolbar did
        // not say what was on screen.
        if model.ocrMode {
            return ocr.documentName.isEmpty ? nil : (ocr.documentName, ocr.documentName)
        }
        if browse == .recents { return ("Recents", "Recents") }
        if browse == .clipboardOff { return ("Clipboard", "Clipboard") }
        if browse == .photos, let source = model.browsedPhotoSource {
            return (source.title, source.title)
        }
        if browse == .folder, let folder = model.filterFolder {
            // The folder whose rows are ON SCREEN, falling back to the requested one only before
            // the first listing exists. Reading `filterFolder` directly renamed the toolbar the
            // instant a row was clicked, while the previous folder's files were still listed
            // underneath it - which reads as "I am in the new folder" when you are not.
            let shown = model.browsingFolderShown ?? folder
            return (shown.lastPathComponent, shown.path)
        }
        return nil
    }

    @ViewBuilder private var browseTitle: some View {
        if let t = browseTitleText {
            // PLAIN TEXT, like Finder. A Menu was tried and is wrong: Finder's title carries no
            // disclosure chevron and no button chrome, and the ancestors already have two ways up
            // (the back chevron and the sidebar). The path is on hover.
            //
            // SECONDARY, not primary. Sampled from a side-by-side: Finder's title is a mid grey
            // (darkest pixel 160 against a 255 ground), where this was rendering near-black.
            //
            // SF SEMIBOLD 15, not `.headline`. Measured off a real NSWindow's titlebar text field
            // rather than matched by eye: it reports .SFNS-Semibold at 15.0pt, where `.headline`
            // resolves to Bold 13 on macOS. Two points smaller and a weight heavier is exactly the
            // mismatch that reads as "nearly Finder" beside a Finder window.
            Text(t.name)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.secondary)
                .help(t.help)
        }
    }

    /// A TITLED `Label`, not a bare `Image`, and that is load-bearing rather than cosmetic. On
    /// Sequoia a toolbar item whose label carries no title gets no intrinsic size and the item
    /// collapses: dumped from a live macOS 15 toolbar, this button and its three neighbours each
    /// measured 10x10 while `search.open` and `search.share` beside them - built with
    /// `Label(title, systemImage:)` - measured 33x28 and 29x28. macOS 26 sizes either form, which
    /// is why it looked fine here. The toolbar renders the icon alone on both, so nothing changes
    /// visually; the title is what AppKit sizes from, and it is also what the overflow menu and
    /// VoiceOver read.
    private var sidebarToggleButton: some View {
        Button {
            withAnimation(.easeOut(duration: 0.2)) {
                columns = columns == .detailOnly ? .all : .detailOnly
            }
        } label: {
            Label("Hide or show the sidebar", systemImage: "sidebar.leading")
        }
        .help("Hide or show the sidebar")
        .accessibilityLabel("Hide or show the sidebar")
        .accessibilityIdentifier("sidebar.toggle")
    }

    private var ocrToggleButton: some View {
        // No `withAnimation` on the change: the two modes are different content, not a moved view,
        // and animating the swap made the whole pane slide in from the window's leading edge.
        ToolbarToggle(isOn: Binding(get: { model.ocrMode }, set: { model.ocrMode = $0 }),
                      symbol: "text.viewfinder",
                      title: model.ocrMode ? "Back to search" : "Transcribe a document")
    }

    /// Either browser is on screen. The two are one mode as far as the toolbar is concerned.

    /// Gallery first, then list - Finder's order (icon view, list view, ...), and the order this
    /// app's own shortcuts already use: Cmd-1 gallery, Cmd-2 list. The segments used to run the
    /// other way round, so the control contradicted both Finder and the View menu above it.
    ///
    /// Shared by the search cluster and the browser's, which is the point: one control, one place
    /// to change it, and no way for the two to end up offering different segments.
    private var viewPicker: some View {
        Picker("View", selection: Binding(get: { model.viewMode }, set: { model.viewMode = $0 })) {
            Image(systemName: "square.grid.2x2").accessibilityLabel("Gallery view").tag(ResultViewMode.grid)
            Image(systemName: "list.bullet").accessibilityLabel("List view").tag(ResultViewMode.list)
        }
        .pickerStyle(.segmented)
        // FIXED WIDTH, the same fix as the OCR view picker: an `NSSegmentedControl` recomputes its
        // intrinsic size through the constraint system whenever the toolbar re-lays out, which is
        // every result set. 38pt a segment is what the live toolbar measures for icon-only ones.
        .frame(width: 2 * 38)
        .fixedSize()
        .help("Switch between list and gallery")
    }

    /// Search by a file, and share what is selected - the same pair, in the same order, that OCR
    /// mode puts next to its own search field (`ocr.open`, `ocr.share`). This USED to be a glyph
    /// installed inside the search field itself, which had two problems: it was invisible as an
    /// affordance, and it had to hide whenever the field held text, so it disappeared exactly when
    /// a query was on screen. A toolbar button is always there and always the same size.
    /// Tahoe lays the trailing cluster out after its `ToolbarSpacer`. macOS 14/15 have no spacer
    /// that survives (a SwiftUI `Spacer` item is dropped, an injected `.flexibleSpace` is pruned on
    /// every state change, and a stretched item's width request pushes its neighbours into the
    /// overflow menu), so there the trailing items say `.primaryAction`, which puts them on the
    /// trailing side of the leading group. Verified on macOS 15.7 in a VM.
    private var trailingPlacement: ToolbarItemPlacement {
        if #available(macOS 26.0, *) { return .automatic }
        return .primaryAction
    }

    @ToolbarContentBuilder private var fileActions: some ToolbarContent {
        ToolbarItem(id: "search.open", placement: trailingPlacement) {
            Button { model.searchByFilePanel() } label: {
                // `folder`, the same symbol OCR's Open Document uses. The two are the same verb.
                Label("Search by File\u{2026}", systemImage: "folder")
            }
            .help("Search by File  \u{21e7}\u{2318}O")
            .accessibilityLabel("Search by File")
        }
        ToolbarItem(id: "search.share", placement: trailingPlacement) {
            // The system share sheet, not a menu of our own - same as OCR's. Disabled rather than
            // hidden when nothing is selected, so the group does not change width as you click
            // around; that is how the OCR group behaves too.
            // A button that builds the share list when clicked, not a ShareLink holding it: see
            // the File menu's Share. Holding it re-laid out the toolbar on every arrow press.
            Button { SelectionShare.present(model.selectedURLsOrdered) } label: {
                Label("Share\u{2026}", systemImage: "square.and.arrow.up")
            }
            // Both states say something true and useful: what it will do, or what is missing.
            .help(model.menuSelection.count == 0 ? "Select a file to share" : "Share the selection")
            .disabled(model.menuSelection.count == 0)
        }
    }

    /// Names the port when it is actually listening, because that is the thing a user needs next.
    /// Names the port when it is actually listening, because that is what a reader needs next. No
    /// "click to stop": it is a toggle, and its filled state already says which way it is.
    private var servingHelp: String {
        model.serving.state == .running ? "Serving on port \(model.serving.port)" : "Serve over HTTP"
    }

    /// The four leading controls, emitted at whatever placement the running system actually
    /// renders. See the note at the call site for why Sequoia cannot have `.navigation`.
    ///
    /// OCR mode sits next to the sidebar toggle because it switches what the content area IS - the
    /// same class of thing as showing or hiding the sidebar - rather than acting on results; in the
    /// trailing cluster it drifted with the search field's width. It is deliberately NOT gated on
    /// `phase == .ready`: transcription reads a dropped file and writes Markdown, touching neither
    /// the vector index nor the embedding model, so gating it on the index would strand the feature
    /// exactly when it is most useful - while a large index loads.
    ///
    /// Serving is here for the same reason: a global mode of the app, on until turned off, rather
    /// than an action on whatever is on screen. Same switch as Settings > Serving, never a second
    /// source of truth.
    @ToolbarContentBuilder
    private func leadingModeItems(_ placement: ToolbarItemPlacement) -> some ToolbarContent {
        ToolbarItem(id: "sidebar.mode", placement: placement) { sidebarToggleButton }
        ToolbarItem(id: "ocr.mode", placement: placement) {
            // On is a FILLED accent circle with a white glyph, the way Preview draws Markup while
            // it is on. The fill is drawn INSIDE the label rather than by `.borderedProminent`: a
            // prominent button takes a background of its own, which broke this item out of the
            // glass capsule it shares with the sidebar toggle.
            ocrToggleButton
                .help(model.ocrMode ? "Back to search  \u{2318}\u{2325}O" : "Transcribe a document  \u{2318}\u{2325}O")
                .accessibilityLabel(model.ocrMode ? "Back to search" : "Transcribe a document")
                .accessibilityIdentifier("ocr.toggle")
        }
        ToolbarItem(id: "serve.mode", placement: placement) {
            ToolbarToggle(isOn: Binding(get: { model.serving.enabled },
                                        set: { model.serving.enabled = $0 }),
                          symbol: "network",
                          title: "Serve over HTTP")
                .help(servingHelp)
                .accessibilityLabel("Serve over HTTP")
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        // No explicit sidebar toggle: Sequoia's system toggle (next to the traffic lights, like
        // Finder) lives and dies with the split-view TRACKING SEPARATOR item, which this app keeps
        // (transparent - see the tuner). Hiding the title does NOT collapse it; only removing the
        // separator item does. An explicit button here therefore always duplicates the system one.
        // Finder-style back/forward at the leading edge of the content toolbar. The window's title TEXT
        // is hidden (WindowTitleHider), so the chevrons own the leading edge with no "Omni" label. The
        // toolbar ITEM is unconditional and wraps the chevrons in an always-present HStack: a directly
        // conditional .navigation item reorders unpredictably, whereas the always-present HStack holds a
        // stable leading slot and the conditional chevrons inside it just appear/vanish. Progressive
        // disclosure: the chevrons show only once there's somewhere to go, so the idle state is empty
        // here. Cmd-[ / Cmd-] match Finder and Safari; each chevron disables independently at the end of
        // its trail. Grouped so on Tahoe they share one Liquid Glass pill.
        // OCR mode sits next to the sidebar toggle, at the leading edge. It switches what the
        // content area IS - the same class of thing as showing or hiding the sidebar - rather
        // than acting on results, and in the trailing cluster it drifted with the search field's
        // width and stranded itself mid-toolbar whenever the sidebar was collapsed.
        //
        // Deliberately NOT gated on `phase == .ready`: transcription reads a dropped file and
        // writes Markdown, and touches neither the vector index nor the embedding model. Gating
        // it on the index would strand the feature exactly when it is most useful - while a large
        // index loads, or when another copy of Omni holds it open.
        // THE SAME PLACEMENT ON EVERY SYSTEM, VERIFIED ON macOS 15.7 (tart VM, 2026-09-22). The
        // leading items were moved to `.primaryAction` pre-Tahoe on the theory that Sequoia drops
        // `.navigation` items, and then given titled labels on the theory that an untitled label
        // cannot be sized. Neither was the cause: the items were present and collapsed to 10x10
        // because the TOOLBAR WAS REBUILT - the root swapped whole split views between modes and
        // the detail swapped its content under a Group - and macOS 14/15 do not size SwiftUI items
        // in a rebuilt toolbar. With one stable split view and a ZStack owner (see `body` and
        // `split`) they measure 34x28 / 31x28 / 29x28 at launch and through OCR on/off cycles, at
        // the leading edge where Tahoe puts them.
        leadingModeItems(.navigation)
        if #available(macOS 26.0, *) {
            ToolbarItem(id: "nav.title", placement: .navigation) { navAndTitle }
                .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(id: "nav.title", placement: .navigation) { navAndTitle }
            // macOS 14/15: the window title is hidden (the tuner), and the title item was what
            // filled the free space - without it the search field packed against the leading
            // buttons. An empty centre item takes that job: AppKit gives a principal item flexible
            // room on both sides. Checked on 15.7 in search, results and OCR mode; the window keeps
            // its name "Omni" for the Window menu and Mission Control.
            ToolbarItem(id: "center", placement: .principal) { Color.clear.frame(width: 1, height: 1) }
        }
        // Flexible space after back/forward pushes every other control to the trailing edge (chevrons
        // own the left, everything else is right-aligned), and on Tahoe it's also the correct separator
        // between Liquid Glass toolbar groups - the leading chevron pill and the trailing control pills
        // read as distinct glass surfaces.
        if #available(macOS 26.0, *) {
            ToolbarSpacer(.flexible)
        }
        // NO SPACER PRE-TAHOE, AND NONE IS NEEDED ANY MORE. This used to say the tuner inserted an
        // AppKit `.flexibleSpace` item to do the same job; it did once, and 692c238 took that
        // machinery out with the in-field search-by-file button while leaving the claim here - so
        // the comment described code that had not existed for a while. A SwiftUI
        // `ToolbarItem { Spacer() }` is genuinely dropped on macOS 14/15 (verified against the live
        // NSToolbar's item list), which is why the AppKit item was there.
        //
        // Pre-Tahoe the trailing cluster is `.primaryAction` instead - see `trailingPlacement`.
        // Bookmark the current search. The only way into History when recording is set to "Only when
        // I bookmark", and a quick save otherwise. Appears once there's a search to keep.
        if model.phase == .ready, !model.ocrMode, model.hasActiveSearch {
            ToolbarItem(placement: .primaryAction) {
                Button { model.toggleBookmarkCurrentSearch() } label: {
                    // No explicit color in the unbookmarked state, so the toolbar can dim it like every
                    // other button when the window resigns key (e.g. while Settings is open). Yellow is
                    // applied only when bookmarked, where the lit status color is intentional.
                    // Titled, so the item sizes on Sequoia - see sidebarToggleButton.
                    if model.currentSearchIsBookmarked {
                        Label("Remove Bookmark", systemImage: "star.fill").foregroundStyle(.yellow)
                    } else {
                        Label("Bookmark Search", systemImage: "star")
                    }
                }
                // Cmd-D is owned by the File-menu "Bookmark Search" command (single owner, avoids a
                // duplicate-shortcut conflict); the tooltip names it, and accessibilityLabel is what
                // VoiceOver reads and what the toolbar-overflow menu shows for this icon-only button.
                .help(model.currentSearchIsBookmarked ? "Remove Bookmark  \u{2318}D" : "Bookmark Search  \u{2318}D")
                .accessibilityLabel(model.currentSearchIsBookmarked ? "Remove Bookmark" : "Bookmark Search")
            }
        }
        // Progressive disclosure: the filter/sort/view chrome appears only once there are results
        // to act on - hidden, not greyed out, during onboarding and the idle/empty states.
        // Exception: keep the filter menu reachable whenever a filter is active, so a filter that
        // hides every result can still be cleared (otherwise the menu vanishes with the results).
        // NOT WHILE BROWSING. Entering a folder sets `filterFolder`, which makes `filtersActive`
        // true, so this menu used to appear over a browser whose listing it cannot change - the
        // browser lists indexed children, not filtered results. Inert chrome. (The star stays:
        // bookmarking a browsed folder saves a state you can actually return to.)
        if model.phase == .ready, !model.ocrMode, !showsBrowser, browse != .clipboardOff,
           model.hasResults || model.filtersActive {
        // Filter joins sort/view in the trailing placement so on Tahoe the three result controls
        // share ONE Liquid Glass pill (search-by-file + bookmark form the other). filterPlacement
        // keeps filter leading on pre-26 so the Sequoia toolbar layout is unchanged.
        ToolbarItem(placement: filterPlacement) {
            filterMenu.disabled(!model.hasIndexedFiles)
        }
        }
        // Result presentation - sort + view.
        //
        // SORT IS SEARCH-ONLY. It orders RESULTS, which are ranked; a browser sorts by clicking a
        // column header (see FolderBrowser) and this menu does nothing there. It used to be shown
        // while browsing, where it was inert chrome.
        //
        // VIEW IS BOTH. A listing is a list of things with a name and a date whichever way you
        // arrived at it, and both browsers draw a gallery - which nothing could switch to while
        // the Photos browser was missing from this condition.
        if model.phase == .ready, !model.ocrMode, showsBrowser, !model.hasResults {
            ToolbarItem(id: "browse.view", placement: .primaryAction) { viewPicker }
        }
        if model.phase == .ready, !model.ocrMode, model.hasResults {
        ToolbarItem(placement: .primaryAction) {
            if #available(macOS 26.0, *) {
                // Tahoe: the inline sort menu + segmented view toggle render and overflow cleanly.
                ControlGroup {
                    Menu {
                        Picker("Sort By", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                            ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                        }
                    } label: { Image(systemName: "arrow.up.arrow.down") }
                    .help("Sort by \(model.sortOrder.title)")
                    .accessibilityLabel("Sort results")

                    viewPicker
                }
            } else {
                // Sequoia and earlier: a ControlGroup of a menu + segmented picker overflows into an
                // empty, icon-less toolbar dropdown. Use one compact labeled menu instead so it always
                // shows its icon and survives overflow.
                Menu {
                    Picker("View", selection: Binding(get: { model.viewMode }, set: { model.viewMode = $0 })) {
                        Label("as Gallery", systemImage: "square.grid.2x2").tag(ResultViewMode.grid)
                        Label("as List", systemImage: "list.bullet").tag(ResultViewMode.list)
                    }
                    Divider()
                    Picker("Sort By", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                        ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                    }
                } label: {
                    Label("View Options", systemImage: "slider.horizontal.3")
                }
                .help("Sort and view")
            }
        }
        }
        // LAST, so it sits immediately before the search field - the slot OCR mode puts the same
        // pair in. Not in OCR mode: that mode has its own Open Document and Share, for the
        // document rather than for the index.
        if model.phase == .ready, !model.ocrMode {
            if #available(macOS 26.0, *) { ToolbarSpacer(.fixed) }
            fileActions
        }
    }

    private var filterKinds: [FileKind] {
        // Show indexed kinds, plus any kind currently being filtered on - otherwise a filter for a
        // kind that is not (yet) in the index would be invisible and impossible to untoggle.
        let present = FileKind.indexable.filter { model.indexedKinds.contains($0.rawValue) || model.filterKinds.contains($0) }
        // Scanned is ALWAYS offered (unlike the four detection kinds): it is an extraction-time
        // sub-kind users may not know they have until they filter for it.
        return (present.isEmpty ? [.image, .video, .audio] : present) + [.scan]
    }

    private var filterMenu: some View {
        Menu {
            // Only when the box actually SPELLS a qualifier. On a plain query the two modes
            // produce the same search, so the control offered a choice with no consequence at the
            // top of every menu. Gated on the raw text, not on the parse, so it stays reachable
            // once literal mode has emptied the qualifier list.
            if model.rawQueryHasQualifiers {
                Section {
                    Toggle(isOn: Binding(get: { model.literalQuery },
                                         set: { _ in model.toggleLiteralQuery() })) {
                        // Quotation marks, not "Aa": the question is whether `type:image` is a
                        // filter or four literal characters, which is quoting, not formatting.
                        Label("As Plain Query", systemImage: "quote.opening")
                    }
                    .help("Ignore filters in the query")
                }
            }
            Section("Show") {
                ForEach(filterKinds, id: \.self) { kind in
                    Toggle(isOn: Binding(
                        get: { model.filterKinds.contains(kind) },
                        // Text carries its sub-kind along (scanned PDFs are documents too), so
                        // toggling Text never silently drops scans from the results. The Scanned
                        // PDFs toggle stays independent for narrowing within text.
                        set: { on in
                            if on {
                                model.filterKinds.insert(kind)
                                if kind == .text { model.filterKinds.insert(.scan) }
                            } else {
                                model.filterKinds.remove(kind)
                                if kind == .text { model.filterKinds.remove(.scan) }
                            }
                        }
                    )) { Label(kind.title, systemImage: kind.symbol) }
                }
            }
            Picker("Folder", selection: Binding(
                get: { model.filterFolder?.path ?? "" },
                set: { model.filterFolder = $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
            )) {
                Text("All Folders").tag("")
                ForEach(model.roots, id: \.self) { Text($0.lastPathComponent).tag($0.path) }
            }
            Picker("Extension", selection: Binding(get: { model.filterExt }, set: { model.filterExt = $0 })) {
                Text("Any Extension").tag("")
                ForEach(model.indexedExts, id: \.self) { Text(".\($0)").tag($0) }
            }
            Picker("Date", selection: Binding(get: { model.dateRange }, set: { model.dateRange = $0 })) {
                ForEach(DateRange.allCases) { Text($0.title).tag($0) }
            }
            // Two options, on the SCORE. 50% is the floor 0.60 was measured down from: 0.60
            // returned nothing at all for ordinary queries whose best hit was 0.59, while 0.50
            // keeps 94.7% of known answers. Scaled per kind on the way through, so it does not
            // delete the media (see AppModel.defaultMinScore). `score:` in the query language
            // still sets any other floor.
            Picker("Relevance", selection: Binding(get: { model.minScore }, set: { model.minScore = $0 })) {
                Text("Only Strong Matches").tag(0.5)
                Text("All").tag(0.0)
            }
            Divider()
            Button("Clear Filters") { model.clearFilters() }.disabled(!model.filtersActive)
        } label: {
            Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .help("Filter results")
        .accessibilityLabel("Filter results")
    }
}
