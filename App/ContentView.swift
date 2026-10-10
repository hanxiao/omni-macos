import SwiftUI
import AppKit
import OmniKit
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model: AppModel
    /// The two typing timers, in a reference the view holds rather than as `@State`: they are
    /// replaced on every keystroke, and a state write is a reason to re-render the window.
    @State private var timers = TypingTimers()
    @State private var fileDropTargeted = false
    /// Owned rather than left to the system, so the toggle that hides the drawer can also bring it
    /// back - see `sidebarToggleButton`.
    @State private var columns: NavigationSplitViewVisibility = .automatic
    /// The OCR workspace owns its own state so a document survives toggling back to search and
    /// returning - closing it is an explicit action, not a side effect of looking away. It is
    /// created by the App, not here, because the File menu's OCR commands have to reach it: a key
    /// equivalent declared only on a toolbar button never fires on macOS.
    @Environment(OCRSession.self) private var ocr

    // Progressive disclosure: only offer search once there is something to search. During model
    // loading, onboarding, and the no-folders state the search field stays hidden (not dimmed).
    private var showsSearch: Bool { model.phase == .ready && !model.roots.isEmpty }

    /// Whether the drawer has anything to show: the search sidebar always does, the page rail only
    /// once a document is open.
    private var ocrDrawerWanted: Bool { !model.ocrMode || ocr.hasPages }

    /// Apply a user edit of the search box: parse it into the semantic query + qualifiers, apply the
    /// filters, clear a file query if real text was typed, and schedule the (debounced) search. The
    /// box binds to the RAW typed string; `set` (user edits only) routes here.
    private func handleQueryEdit(_ typed: String) {
        if !model.suggestionsAllowed { model.suggestionsAllowed = true }   // this fires only on real keystrokes (the .searchable set:), so arm the dropdown
        // The field holds the SEMANTIC text; finished filters live beside it as chips. A qualifier
        // becomes a chip only once a space ends it - mid-word, `type:i` has to stay editable text
        // or the chip is made from half a word and cannot be corrected.
        model.setSemanticText(typed)
        // ONLY when the finished word really is a qualifier. Re-parsing on every space also
        // re-normalises the text, which swallowed the space itself and ran the next word into the
        // last one: "a very long" typed straight through arrived as "averylong".
        if typed.last?.isWhitespace ?? false,
           !SearchQueryParser.parse(typed).qualifiers.isEmpty {
            model.promoteQualifiers()
        }
        let raw = typed
        if !model.query.isEmpty, model.fileQuery != nil { model.fileQuery = nil; model.queryError = nil }
        // Instant search off: typing still parses filters and pops suggestions, but the search
        // itself waits for Return (.onSubmit) - kinder to low-end GPUs. Clearing the box always
        // runs (its empty-query path clears the stale results); auto-history records only what
        // actually searched (Return records via onSubmit).
        let cleared = raw.trimmingCharacters(in: .whitespaces).isEmpty
        if model.fileQuery == nil, model.instantSearchEnabled || cleared { scheduleSearch() }
        if model.instantSearchEnabled { scheduleHistoryRecord() }
    }

    var body: some View {
        // ONE split view, never swapped. This used to be three branches - OCR, search, neither -
        // each holding its own `split` with a different `.searchable`, so turning OCR mode on (and
        // the launch going ready) replaced the whole split view and rebuilt the window's toolbar
        // from nothing. On macOS 14 and 15 a rebuilt toolbar comes back with every SwiftUI item
        // collapsed to 10x10 - present, and invisible. The search field is configured by MODE
        // instead: OCR mode binds it to find-in-document, search mode to the query.
        split
            // The field, its chips and its suggestions, in a MODIFIER: the field's binding reads the
            // query, and read from this body every keystroke re-ran the whole window - the split
            // view, the results list and each visible row, ~60 ms a character (measured with the
            // stall detector's body counters, 2026-09-22). A modifier re-runs without its content.
            .modifier(SearchField(onEdit: handleQueryEdit, suggest: searchSuggestions))
            .onSubmit(of: .search) {
                // In OCR mode Return steps to the next match, Preview's find. In search mode it
                // finishes the word too: a qualifier typed without a trailing space still becomes
                // a chip rather than being embedded as prose.
                if model.ocrMode { ocr.stepMatch(by: 1); return }
                model.promoteQualifiers()
                model.search(); model.recordCurrentSearchToHistory(viaSubmit: true)
            }
            // Escape clears the whole query, not just the text: the chips and filters are the
            // same query, so leaving them behind is what made a "cleared" box still return a
            // filtered, empty result set.
            .onKeyPress(.escape) {
                guard !model.ocrMode, model.hasActiveSearch || !model.searchTokens.isEmpty else { return .ignored }
                model.clearSearch()
                return .handled
            }
        // An empty page rail is a column of nothing: fold it when OCR mode opens with no document
        // and unfold it the moment one arrives. On the BODY, not on the split - the split is a
        // branch of the Group above, so flipping the mode replaces it and takes any `onChange`
        // declared there with it, which is why the drawer stayed open.
        //
        // The INITIAL call is not animated. `columns` starts `.automatic`, so the launch call
        // changes it, and doing that inside `withAnimation` animated the window's first layout -
        // the centered "Search N files" prompt flew in from the top-left corner on every launch.
        .onChange(of: ocrDrawerWanted, initial: true) { old, wanted in
            let target: NavigationSplitViewVisibility = wanted ? .all : .detailOnly
            if old == wanted {
                columns = target
            } else {
                withAnimation(.easeOut(duration: 0.2)) { columns = target }
            }
        }
        .onChange(of: columns, initial: true) { _, c in
            let shown = c != .detailOnly
            if model.sidebarShown != shown { model.sidebarShown = shown }
        }
        // Spotlight-style: put the caret in the search field as soon as the app can search.
        .onChange(of: showsSearch, initial: true) { _, shows in if shows { focusSearchField() } }
        // Benchmark progress and the paper result as ONE native sheet on the main window (not a
        // stray floating panel). A single route rather than two `.sheet` modifiers on the same
        // view: stacked sheets race each other on presentation, and the paper run has to hand the
        // progress sheet over to the result sheet.
        .sheet(item: Binding(get: { model.activeSheet }, set: { model.activeSheet = $0 })) { route in
            switch route {
            case .progress: ProfilingSheet()
            case .paperResult: PaperResultSheet()
            }
        }
    }

    private func focusSearchField() { SearchFieldFocus.focus() }

    private var split: some View {
        NavigationSplitView(columnVisibility: $columns) {
            // ONE drawer. In OCR mode the sidebar becomes the page navigator rather than the app
            // growing a second column on the trailing edge: the window keeps its shape, the system
            // sidebar toggle shows and hides it like any sidebar, and there is no inspector to
            // restructure the split - which is what moved the toolbar 300pt.
            // The drawer is sized to what is in it. A page navigator needs the width of a page
            // thumbnail and no more, where a list of folders and history needs room for names, so
            // the search sidebar's 260 left a portrait scan swimming in margin.
            Group {
                if model.ocrMode { PageRail() } else { Sidebar() }
            }
            // On the COLUMN ROOT, not on the branch inside it: SwiftUI reads this from the sidebar
            // view itself, and a width declared one level down was ignored. The MAXIMUM is what
            // does the work - a split view restores the divider where it was left, over any
            // `ideal` - so the OCR maximum clamps the rail on the way in and the search sidebar's
            // minimum pushes it back out on the way out.
            .navigationSplitViewColumnWidth(
                min: model.ocrMode ? 120 : 230,
                ideal: model.ocrMode ? 168 : 260,
                max: model.ocrMode ? 190 : 320)
            // The system toggle lives in the SIDEBAR's toolbar section and goes away with it, so
            // folding the drawer left no way to unfold it but the View menu. Ours is in the window
            // toolbar and stays.
            .toolbar(removing: .sidebarToggle)
        } detail: {
            // No navigationTitle/navigationSubtitle: either one claims the leading toolbar slot
            // and pushes back/forward to its right.
            //
            // The redundant "Omni" label next to the traffic lights is dropped through SwiftUI's
            // own API, `toolbar(removing: .title)`, which removes only the title ITEM and leaves
            // the toolbar background, the split tracking separator and the traffic lights intact.
            // This is deliberately NOT the AppKit route that was tried before (setting
            // NSWindow.titleVisibility from a window observer): mutating the window/toolbar in the
            // middle of SwiftUI's commit is what took the system sidebar toggle and the toolbar's
            // sidebar/detail sectioning down with it on Sequoia - see the tuner's notes below.
            //
            // Tahoe only. On macOS 14/15 the title text is hidden by the tuner instead, and an empty
            // `.principal` item fills the space the title item used to (see `toolbar`).
            //
            // A ZSTACK, NOT A GROUP, AND THAT IS THE SEQUOIA TOOLBAR FIX. A Group is not a view: its
            // modifiers are applied to each child, so `.toolbar` landed on the CONDITIONAL content
            // inside it (search / OCR, and under that the phase switch). On macOS 14 and 15, toolbar
            // items attached to a split view's detail content are dropped for good when that content
            // is replaced (FB13106004, bdewey.com/til/2023/09/04/toolbar-bugs) - and the launch
            // replaces it once, loading -> ready - so every toolbar item was gone before the window
            // was usable. Tahoe re-registers them, which is why it only ever looked right here. A
            // ZStack is a real container whose identity never changes, so the toolbar has a stable
            // owner.
            ZStack {
                if #available(macOS 26.0, *) {
                    detailOrOCR.toolbar(removing: .title)
                } else {
                    detailOrOCR
                }
            }
            // ITS OWN MODIFIER, re-evaluated only when what it shows changes. Built inside this
            // body, every evaluation of the window - two or three per search, for `isResolving`
            // alone - rebuilt the toolbar and AppKit re-laid out its items: ~320 ms of main thread
            // per result set, measured with the toolbar removed. Same fix as `OCRToolbar`.
            .modifier(SearchToolbar(columns: $columns, browse: browseMode))
            .background(WindowTitleHider(sidebarWidth: model.ocrMode ? 168 : 260))
        }
    }

    // MARK: - Detail

    /// Search results or the OCR workspace. The toggle replaces the content area rather than
    /// opening a window: this is a different way of looking at documents on this Mac, not a
    /// different app, and a separate window would strand it from the sidebar and the index.
    @ViewBuilder private var detailOrOCR: some View {
        Group {
            if model.ocrMode {
                OCRView(dropTargeted: fileDropTargeted)
            } else {
                detail
            }
        }
        // ONE drop target for both panes, because a drop is the same gesture either way - only
        // what happens at the end differs, and DropRouter is where that is decided. Two targets
        // meant two flavor lists, and the transcription pane's was quietly narrower.
        .onDrop(of: [.image, .fileURL, .url, .text, .plainText], isTargeted: $fileDropTargeted) { providers in
            // Omni's own drag (a result lifted out of the list) is never taken back in: see FileDrag.
            guard !FileDrag.isActive else { return false }
            return DropRouter.handle(NSPasteboard(name: .drag), providers: providers, model: model, ocr: ocr)
        }
        .overlay {
            // The transcription pane draws its own chip; this border is the search pane's.
            if fileDropTargeted && !model.ocrMode && !FileDrag.isActive {
                DropRing()
            }
        }
    }

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            // No subtitle: the title names the phase and the bar shows how far along it is. The
            // sentences that used to sit here explained internals (paging, warm-up) to the user.
            // A one-time index upgrade is a different thing from loading the model, and takes tens
            // of seconds on a large index - saying "loading the model" through a database rewrite
            // is how a working upgrade reads as a hang.
            // ONE bar for the whole launch - the store now scales its load and its one-time
            // upgrade into a single fraction, so this keeps moving through both. Only the WORDS
            // change with the phase, so a database rewrite is never described as loading the model.
            CenteredStatus(symbol: model.launchSymbol,
                           title: model.launchTitle,
                           subtitle: "",
                           showSpinner: true, progress: model.loadingProgress)
        case .noModel:
            OnboardingView()
        case .failed(let msg):
            EngineFailedView(message: msg)
        case .waitingForIndex(let why):
            IndexWaitingView(message: why)
        case .indexNewer:
            IndexNewerView()
        case .ready:
            ready
        }
    }

    @ViewBuilder private var ready: some View {
        content
    }

    /// The folder embedding map is shown ONLY in the empty-result region and ONLY when nothing
    /// search-related is active: a folder is selected, the query box is empty (typed AND file), no
    /// raw results, no query error, and nothing resolving. Active queries/results always win - this
    /// flips false the instant the user types, hiding the viz purely by precedence (the selected
    /// folder is not cleared, so clearing the query brings the cached map back instantly).
    private var showsFolderViz: Bool {
        model.selectedFolderForViz != nil && !model.hasQuery && model.fileQuery == nil
            && !model.hasResults && model.queryError == nil && !model.isResolving
    }

    /// A Photos source is being browsed. Same precedence rule as the folder browser below, and
    /// checked before it - `enterPhotoSource` clears `filterFolder` and `enterFolder` clears the
    /// source, so only one can be set, but the order makes that explicit rather than incidental.
    /// Recents is on screen. Same precedence rule as the two browsers: a query wins.
    private var showsRecents: Bool {
        model.filterRecents && model.filterFolders.isEmpty && !model.hasQuery && model.fileQuery == nil
            && !model.hasResults && model.queryError == nil && !model.isResolving
    }

    /// The Clipboard row was clicked with capture off and nothing to browse. With a query too: the
    /// search is scoped to the clipboard and finds nothing, and this says why where "No matches"
    /// would not. Only while the scope is still exactly the clipboard.
    private var showsClipboardOff: Bool {
        model.showsClipboardOff && model.fileQuery == nil && !model.hasResults
            && model.queryError == nil && !model.isResolving
            && model.filterFolders.map(\.standardizedFileURL.path) == [AppModel.clipboardDirectory.standardizedFileURL.path]
    }

    private var showsPhotoBrowser: Bool {
        model.browsedPhotoSource != nil && !model.hasQuery && model.fileQuery == nil
            && !model.hasResults && model.queryError == nil && !model.isResolving
    }

    /// Same precedence as the map below: a folder is being browsed, and nothing search-related is
    /// active. Typing hides it instantly and the results are already scoped to the folder, which
    /// is the whole point - the browser and the search are two views of one `filterFolder`.
    private var showsFolderBrowser: Bool {
        // EXACTLY ONE folder. Scoping to several is a search filter, not a place to browse - the
        // empty-result region holds one listing, and showing the first of two would misrepresent
        // what the search is scoped to.
        model.filterFolders.count == 1 && !model.hasQuery && model.fileQuery == nil
            && !model.hasResults && model.queryError == nil && !model.isResolving
    }

    /// The filters are chips INSIDE the search field now, so a bar repeating them is the same
    /// duplication one row lower. It survives only to explain plain-text mode, where the field's
    /// text is embedded verbatim and the chips do not apply, and to offer the way back.
    private var showsQualifierBar: Bool { model.literalQuery }

    @ViewBuilder private var content: some View {
        // The chip / qualifier bar is a SAFE-AREA BAR, not a row stacked above the content, for the
        // same reason the browsers' column headers are: a scroll view stacked BELOW a sibling does
        // not fill its pane, and on Tahoe that is what stops the scroll edge effect from applying,
        // so results clipped at a hard line instead of blurring under the chrome.
        contentBody.modifier(TopBar {
            if let fq = model.fileQuery { FileQueryChip(fileQuery: fq) }
            else if showsQualifierBar { QualifierBar() }
        })
    }

    @ViewBuilder private var contentBody: some View {
        VStack(spacing: 0) {
            if !model.results.isEmpty {
                // `.equatable()`: see the conformance on ResultsList.
                ResultsList { belowThresholdFooter }.equatable()
            } else if showsClipboardOff {
                // The one status screen with a subtitle: it asks for a turn-on, and the title alone
                // does not say what turning it on gets you.
                CenteredStatus(symbol: CenteredStatus.moleSleep, title: "Clipboard history is off",
                               subtitle: "Saves the text and images you copy so you can search them.",
                               action: ("Turn On", { model.setClipboardEnabled(true) }),
                               prominent: true)
            } else if showsRecents {
                RecentsBrowser()
            } else if showsPhotoBrowser {
                PhotoSourceBrowser(source: model.browsedPhotoSource!)
            } else if showsFolderBrowser {
                FolderBrowser(folder: model.filterFolder!)
            } else if showsFolderViz {
                FolderEmbeddingVisualization(folderName: model.selectedFolderForViz!.lastPathComponent)
            } else {
                emptyState
            }
        }
        // Leaving the map releases the folders browsed BEFORE this one. The selected folder's
        // layout stays cached, so clearing the query still puts its map back instantly.
        .onChange(of: showsFolderViz) { _, shown in
            if !shown { model.trimProjectionCacheToCurrent() }
        }
    }

    @ViewBuilder private var emptyState: some View {
        // Indexing is invisible here - the sidebar's per-folder progress is the only cue, and
        // search works while it runs. The user just adds folders and searches.
        // THE FIRST SCREEN A NEW INSTALL SEES, now that nothing is indexed by default (see
        // AppModel.loadRoots). Keyed on hasSources, not on `roots`: a user who added only their
        // photo library has a source and can search, and telling them to add a folder over the
        // top of it would be wrong.
        if !model.hasSources {
            CenteredStatus(symbol: CenteredStatus.moleSleep, title: "Add folders to search",
                           subtitle: "", showSpinner: false,
                           action: ("Add\u{2026}", { SourcePicker.add(to: model) }),
                           prominent: true)
        } else if let err = model.queryError {
            CenteredStatus(symbol: "exclamationmark.magnifyingglass", title: "Couldn't search by that file",
                           subtitle: err, showSpinner: false)
        } else if model.indexObsolete && model.hasQuery {
            // A dim/model mismatch makes every search return nothing; explain it and offer both the
            // cheap fix (switch back to the model the index was built with) and the rebuild.
            let built = model.indexBuiltVariant
            CenteredStatus(symbol: "arrow.triangle.2.circlepath",
                           title: built != nil ? "Switch to \(built!.title) or reindex" : "Reindex to search",
                           // The FACT only. The sentence that used to follow it ("Switch back to
                           // keep your index, or reindex with the current model") described the two
                           // buttons directly beneath it, which already say so themselves.
                           subtitle: built != nil
                               ? "Built with \(built!.title). \(model.modelVariant.title) is loaded."
                               : "Built with a different model than the one loaded.",
                           showSpinner: false,
                           action: built.map { v in ("Switch to \(v.title)", { model.selectVariant(v) }) },
                           secondary: ("Reindex", { model.startIndexing() }),
                           // Two ways out, and switching back is the cheap one: it keeps the index
                           // that reindexing would spend an hour rebuilding.
                           prominent: true)
        } else if !model.hasQuery || model.isResolving {
            // Idle prompt, and the in-flight search state. They share one calm placeholder so a
            // pending search only fades a small spinner in under the same prompt - it never flashes
            // "No matches" while the debounce/search for what you just typed is still running.
            SearchWaysPrompt(
                // With several folders scoped there is no browser to show (that needs exactly one),
                // so the prompt is the only place that says what the next query will cover.
                title: model.filterFolders.count > 1
                    ? "Search \(model.filterFolders.count) folders"
                    : (model.indexedFiles > 0 ? "Search \(model.indexedFiles.formatted()) file\(model.indexedFiles == 1 ? "" : "s")" : "Search your files"),
                count: model.indexedFiles,
                showSpinner: model.isResolving)
        } else if model.hiddenByThreshold > 0 {
            // The count sits in the BUTTON, the only thing that acts on it. As a subtitle it
            // restated the title and then named the mechanism doing it.
            CenteredStatus(symbol: CenteredStatus.moleSad,
                           title: "No results above \(Int(model.minScore * 100))%",
                           subtitle: "", showSpinner: false,
                           action: (Self.weakerMatchesTitle(model.hiddenByThreshold),
                                    { model.showAllBelowThreshold() }))
        } else if model.filtersActive {
            // Filters can hide every result; the empty state is the only place left to escape
            // them. The cause goes in the TITLE - as a subtitle it was a sentence explaining the
            // button underneath it.
            CenteredStatus(symbol: CenteredStatus.moleSad, title: "No matches with these filters",
                           subtitle: "", showSpinner: false,
                           action: ("Clear Filters", { model.clearFilters() }))
        } else {
            // No subtitle. "Try a different phrase" is the only thing anyone could do here, so
            // saying it adds a line and no information.
            CenteredStatus(symbol: CenteredStatus.moleSad, title: "No matches", subtitle: "", showSpinner: false)
        }
    }

    /// One title for the one action, wherever it is offered.
    static func weakerMatchesTitle(_ n: Int) -> String {
        "Show \(n.formatted()) weaker \(n == 1 ? "match" : "matches")"
    }

    @ViewBuilder private var belowThresholdFooter: some View {
        // Collapsing is never silent: if the list is shorter than the matches behind it, the
        // difference is stated here. Not a button - the copies are reachable from their own stack,
        // and a global "un-collapse" would just restore the noise the feature removes.
        if model.collapsedCount > 0 {
            Text("\(model.collapsedCount) duplicate\(model.collapsedCount == 1 ? "" : "s") stacked into the results above")
                .font(.caption).foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity)
                .padding(.top, 8)
        }
        if model.hiddenByThreshold > 0 {
            // THE SAME BUTTON as the empty state's, same words, same style, same size - it is the
            // same action, so it should not look like two. It was a `.plain` text label, and a
            // plain button is hit only on its drawn glyphs: a click between the letters or beside
            // them did nothing, which is how it "sometimes did not respond".
            Button(Self.weakerMatchesTitle(model.hiddenByThreshold)) { model.showAllBelowThreshold() }
                .controlSize(.large)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
    }

    private var showsBrowser: Bool { showsFolderBrowser || showsPhotoBrowser || showsRecents }
    private var browseMode: SearchToolbar.BrowseMode {
        if showsRecents { return .recents }
        if showsClipboardOff { return .clipboardOff }
        if showsPhotoBrowser { return .photos }
        if showsFolderBrowser { return .folder }
        return .none
    }

    private func scheduleSearch() {
        timers.search?.cancel()
        timers.search = Task {
            try? await Task.sleep(nanoseconds: 180_000_000)
            if !Task.isCancelled { model.search() }
        }
    }

    // Auto-record (history mode .auto) only after the query has been settled for 3s, so a search has
    // to be one the user actually dwelled on - quick type-and-click-through queries aren't stored.
    // Cancelled on every keystroke, so it only fires once typing stops. (No effect in .onSubmit /
    // .manual modes, which record on Return / the bookmark button instead.)
    private func scheduleHistoryRecord() {
        timers.history?.cancel()
        timers.history = Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if !Task.isCancelled { model.recordCurrentSearchToHistory() }
        }
    }

    // MARK: - Query-language autocomplete

    /// `chip` is rendered as a capsule after the label, the way the search field draws a
    /// qualifier - an em dash between the words and the folder read as punctuation in a list that
    /// is otherwise all chips.
    struct Suggestion: Hashable {
        let label: String
        let completion: String
        let icon: String
        var chip: String? = nil
    }

    /// Typeahead for the search box: complete a partial qualifier key (`ty` -> `type:`) or a key's
    /// values (`type:` -> image/video/...). Returns full-string completions - the text before the
    /// active token is preserved, so selecting one keeps the rest of the query intact.
    private func searchSuggestions(_ raw: String) -> [Suggestion] {
        guard !model.literalQuery else { return [] }
        var out: [Suggestion] = []
        let prefix: String, tok: String
        if let sp = raw.lastIndex(of: " ") {
            prefix = String(raw[...sp]); tok = String(raw[raw.index(after: sp)...])
        } else {
            prefix = ""; tok = raw
        }
        if !tok.isEmpty {
            if let colon = tok.firstIndex(of: ":") {                   // value completion: key:partial
                let keyTyped = String(tok[..<colon])
                if let canon = SearchQueryParser.canonicalKey(keyTyped.lowercased()) {
                    let partial = String(tok[tok.index(after: colon)...]).lowercased()
                    out += valueSuggestions(canon).filter { $0.lowercased().hasPrefix(partial) }.prefix(8).map {
                        let v = $0.contains(" ") ? "\"\($0)\"" : $0
                        return Suggestion(label: "\(keyTyped):\($0)", completion: "\(prefix)\(keyTyped):\(v)", icon: "tag")
                    }
                }
            } else {                                                   // key completion: bare prefix
                let neg = tok.hasPrefix("-") ? "-" : ""
                let low = (neg.isEmpty ? tok : String(tok.dropFirst())).lowercased()
                if !low.isEmpty {
                    for k in ["type:", "tag:", "ext:", "in:", "filename:", "date:", "after:", "score:", "sort:"] where k.hasPrefix(low) {
                        out.append(Suggestion(label: neg + k, completion: "\(prefix)\(neg)\(k)", icon: "line.3.horizontal.decrease.circle"))
                    }
                }
            }
        }
        // Past queries as quick shortcuts: already query-side embedded (cached), so picking one
        // searches instantly without a trip to the sidebar.
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.count >= 1 {
            let needle = trimmed.lowercased()
            let hist = model.searchHistory
                .filter { !$0.isFile && $0.displayText.lowercased().contains(needle) && $0.displayText.lowercased() != needle }
                .sorted { a, b in a.bookmarked != b.bookmarked ? a.bookmarked : a.lastUsed > b.lastUsed }
                .prefix(5)
            // The LABEL is the compact form and the COMPLETION is still the full query. Showing
            // `displayText` here rendered a 340-character path as a ten-line wrapped paragraph,
            // five of them stacked - the suggestion list was taller than the window.
            out += hist.map { item in
                Suggestion(label: item.displayLabel,
                           completion: item.displayText,
                           icon: item.bookmarked ? "star.fill" : "clock",
                           chip: item.displayScope.map { "in:" + $0 })
            }
        }
        // ONE ROW PER COMPLETION: the list is keyed by it, and a past search can be exactly a
        // completion offered above (`type:image`). A duplicate id makes SwiftUI draw or hit-test the
        // wrong row; found by the chaos run in the app's own log.
        var seen = Set<String>()
        return Array(out.filter { seen.insert($0.completion).inserted }.prefix(10))
    }

    private func valueSuggestions(_ key: String) -> [String] {
        switch key {
        case "type": return ["image", "video", "audio", "text", "scanned"]
        case "date": return ["any", "week", "month", "year"]
        case "after": return ["week", "month", "year", "7d", "30d", "1y"]
        case "score": return ["25%", "50%", "70%"]
        case "sort": return ["relevance", "name", "date"]
        case "ext": return model.indexedExts
        case "in": return ["Recents"] + model.roots.map { ($0.path as NSString).abbreviatingWithTildeInPath }
        // filename: takes free text - there is nothing sensible to enumerate, and offering a
        // sample of 135,000 basenames would be noise rather than help.
        case "filename": return []
        default: return []
        }
    }

}

/// A thin bar under the search field showing the qualifiers Omni parsed from the box (or the
/// literal-mode state), with a one-click toggle to treat the box as plain text instead of filters.
/// The toolbar search field. See its use in `ContentView.body` for why it is a modifier.
private struct SearchField: ViewModifier {
    @Environment(AppModel.self) private var model
    @Environment(OCRSession.self) private var ocr
    let onEdit: (String) -> Void
    let suggest: (String) -> [ContentView.Suggestion]

    func body(content: Content) -> some View {
        content
            .searchable(text: Binding(get: { model.ocrMode ? ocr.find : model.query },
                                      set: { if model.ocrMode { ocr.find = $0 } else { onEdit($0) } }),
                        tokens: Binding(get: { model.ocrMode ? [] : model.searchTokens },
                                        set: { if !model.ocrMode { model.setSearchTokens($0) } }),
                        placement: .toolbar,
                        prompt: model.ocrMode ? "Find in document" : "Search by meaning") { token in
                // Text, not Label: a token chip renders its title only on macOS, so an icon here
                // is carried and then thrown away.
                Text(token.label)
            }
            .searchSuggestions { QuerySuggestions(suggest: suggest) }
    }
}

@MainActor
private final class TypingTimers {
    var search: Task<Void, Never>?
    var history: Task<Void, Never>?
}

/// The search field's suggestion list. See `.searchSuggestions` in ContentView for why it is its own
/// view.
private struct QuerySuggestions: View {
    @Environment(AppModel.self) private var model
    let suggest: (String) -> [ContentView.Suggestion]

    var body: some View {
            ForEach(!model.ocrMode && model.suggestionsAllowed ? suggest(model.query) : [], id: \.completion) { sug in
                HStack(spacing: 6) {
                    Image(systemName: sug.icon).foregroundStyle(.secondary)
                    Text(sug.label).lineLimit(1).truncationMode(.middle)
                    if let chip = sug.chip {
                        // Shaped and weighted to match the token the SEARCH FIELD
                        // draws for the same qualifier, so the field and its
                        // suggestions speak one language: a rounded rect, not a
                        // capsule, and a light wash rather than `.quaternary` - which
                        // measured far heavier than the system token (a ~3% wash on
                        // its own surface) and read as a grey block.
                        //
                        // Deliberately NOT a glass effect: this popover is already a
                        // vibrant surface, and glass inside glass is the one thing
                        // Apple's guidance rules out (see the Liquid Glass notes).
                        Text(chip)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(.primary.opacity(0.06),
                                        in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            .lineLimit(1)
                    }
                }
                .searchCompletion(sug.completion)
            }
    }
}

private struct QualifierBar: View {
    @Environment(AppModel.self) private var model: AppModel
    /// Qualifier keys another view is already showing, better. While browsing, the breadcrumb
    /// carries the folder AND lets you click any ancestor; a chip of the same path is a
    /// duplicate that cannot be clicked.
    var hiding: Set<String> = []
    var body: some View {
        HStack(spacing: 6) {
            if model.literalQuery {
                Image(systemName: "quote.opening").foregroundStyle(.secondary).frame(width: 18)
                Text("Plain query").foregroundStyle(.secondary)
                Text("- filters ignored").font(.caption).foregroundStyle(.tertiary)
            }
            Spacer(minLength: 8)
            Button { model.toggleLiteralQuery() } label: {
                Label(model.literalQuery ? "Use Filters" : "As Plain Query",
                      systemImage: model.literalQuery ? "line.3.horizontal.decrease.circle" : "quote.opening")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .help(model.literalQuery
                  ? "Use filters in the query"
                  : "Ignore filters in the query")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 6)
    }
}

/// A thin bar above the results showing the active file query (a file used as the search subject),
/// with a clear button. Reuses Thumbnail and a native .bar material.
private struct FileQueryChip: View {
    @Environment(AppModel.self) private var model: AppModel
    let fileQuery: AppModel.FileQuery
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: fileQuery.similar ? "square.on.square" : "photo.badge.magnifyingglass")
                .foregroundStyle(.secondary).frame(width: 18)
            Thumbnail(path: fileQuery.url.path, side: 18, corner: 4)
            Text(fileQuery.similar ? "Similar to" : "Searching by").foregroundStyle(.secondary)
            Text(fileQuery.url.lastPathComponent).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 8)
            Button { model.clearFileQuery() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary).help("Clear file query")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 8)
    }
}

/// The icon above a centered status: an SF Symbol, or one of the Omni mole glyphs when the name
/// starts with "Mole" (asset catalogue templates, so they take `.tertiary` like a symbol does).
/// One mole per screen, each with its own small gag - see Tools/brand/mole_family.py.
struct StatusGlyph: View {
    let symbol: String
    var body: some View {
        if symbol.hasPrefix("Mole") {
            Image(symbol).renderingMode(.template).resizable().scaledToFit()
                .frame(width: 68, height: 68).foregroundStyle(.tertiary)
                .accessibilityHidden(true)
        } else {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
        }
    }
}

struct CenteredStatus: View {
    /// The Omni mole glyphs (asset names), for `symbol`.
    static let mole = "MoleGlyph", moleSearch = "MoleSearch", moleOCR = "MoleOCR", moleSleep = "MoleSleep",
               moleSerious = "MoleSerious", moleEmpty = "MoleEmpty", moleCurious = "MoleCurious",
               moleSad = "MoleSad"
    let symbol: String
    let title: String
    let subtitle: String
    var showSpinner: Bool = false
    /// Determinate 0...1 -> a native linear progress bar replaces the indeterminate spinner.
    var progress: Double? = nil
    var action: (String, () -> Void)? = nil
    var secondary: (String, () -> Void)? = nil
    /// Blue is for the action the screen EXISTS for, or the recommended one of two. A recovery
    /// button - "Clear filters", "Show more" - is a way back, not the point of the screen, and
    /// painting every one of them prominent means none of them reads as prominent. Finder's panes
    /// are mostly plain buttons for the same reason. Defaults off so it has to be asked for.
    var prominent: Bool = false

    var body: some View {
        VStack(spacing: 12) {
            StatusGlyph(symbol: symbol)
            Text(title).font(.title)
            if !subtitle.isEmpty {
                Text(subtitle).font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 400)
            }
            if let progress {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 280)
                    .padding(.top, 4)
            } else if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
            if action != nil || secondary != nil {
                HStack(spacing: 10) {
                    if let action {
                        let b = Button(action.0, action: action.1)
                        if prominent { b.buttonStyle(.borderedProminent) } else { b }
                    }
                    if let secondary { Button(secondary.0, action: secondary.1) }
                }
                .controlSize(.large).padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// The idle search prompt. Same icon + title as CenteredStatus, but instead of one sentence it lists
/// every way to start a search as bullets - typing, dragging, pasting, picking a file, Find Similar.
/// Icons match the real controls (photo.badge.magnifyingglass = the search-by-file button,
/// square.on.square = the Find Similar / file-query chip) so the list maps onto the actual UI.
struct SearchWaysPrompt: View {
    let title: String
    /// The number inside `title` when it has one, so the headline's digits roll as the index grows
    /// instead of the whole line cutting to a new string. 0 for a title with no number in it.
    var count: Int = 0
    var showSpinner: Bool = false
    /// The empty state is ONE view whose contents cross-fade between searching and transcribing.
    /// Both modes want the same thing said the same way - an icon, what this pane is for, and the
    /// handful of ways in - so building a second layout for OCR only made the window restructure
    /// itself for no gain.
    var symbol: String = CenteredStatus.moleSearch
    var ways: [(icon: String, text: String)] = SearchWaysPrompt.searchWays
    /// Something to do about what the rows just described - the OCR pane hangs its download button
    /// here when the model is missing. `AnyView` because this view is built once per empty state
    /// and a generic parameter would spread through every call site for nothing.
    var footer: AnyView?

    static let searchWays: [(icon: String, text: String)] = [
        ("character.cursor.ibeam", "Type a phrase"),
        ("arrow.down.doc", "Drag in an image, file, or text"),
        ("doc.on.clipboard", "Paste an image or text  \u{2318}V"),
        ("folder", "Search by File  \u{21E7}\u{2318}O"),
        ("sparkle.magnifyingglass", "Right-click a result for Find Similar"),
    ]

    /// WAYS IN, not a feature list. This carried three more rows - find in the transcript, copy as
    /// Markdown, save as .md - which are commands on a document that is not open yet. The search
    /// pane's rows are all ways to START a search; these now match.
    static let transcribeWays: [(icon: String, text: String)] = [
        ("arrow.down.doc", "Drop a PDF or images"),
        // PASTE WAS ALREADY WIRED AND NOWHERE ON SCREEN. `pasteCommand` has handled Cmd-V in OCR
        // mode since the router was split - it even checks OCR before the search test, so a
        // pasted image does not leave the document being read - but this list never said so, and
        // an affordance nobody can see may as well not exist. The search pane has carried the
        // same row all along; these two are meant to match.
        ("doc.on.clipboard", "Paste an image  \u{2318}V"),
        ("folder", "Choose a document  \u{2318}O"),
    ]

    var body: some View {
        VStack(spacing: 16) {
            StatusGlyph(symbol: symbol)
            Text(title)
                .font(.title)
                .contentTransition(.numericText(value: Double(count)))
                // Rolling digits are for a count that ticks while indexing. The first count to
                // arrive replaces the words "your files", and rolling letters into digits read as
                // scrambled text - so that one change is a new view, not a transition.
                .id(count > 0)
                .animation(.snappy(duration: 0.3), value: count)
            VStack(alignment: .leading, spacing: 10) {
                ForEach(ways, id: \.icon) { w in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Image(systemName: w.icon).foregroundStyle(.tertiary).frame(width: 20)
                        Text(w.text)
                    }
                }
            }
            .font(.callout).foregroundStyle(.secondary)   // content-width block; the outer VStack centers it
            if showSpinner { ProgressView().controlSize(.small).padding(.top, 4) }
            if let footer { footer.padding(.top, 6) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct EngineFailedView: View {
    @Environment(AppModel.self) private var model: AppModel
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text("Omni can't load its model").font(.title)
            HStack {
                Button("Retry") { model.retryBootstrap() }.buttonStyle(.borderedProminent)
                Button("Choose model folder\u{2026}") { pickModel() }
            }
            .controlSize(.large)
            DisclosureGroup("Details") {
                Text(message).font(.caption.monospaced()).foregroundStyle(.secondary)
                    .textSelection(.enabled).frame(maxWidth: 460, alignment: .leading)
            }
            .frame(maxWidth: 460)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { model.setModelDir(url) }
    }
}

/// The INDEX could not be opened - which is a different problem from the model failing to load, and
/// has different remedies. The one that ships is disk: the one-time 0.5.0 upgrade rewrites the
/// chunk table and declines to start when the volume cannot hold the copy, so the message is
/// actionable ("needs 5.6 GB free, 1.2 GB available") and the only useful button is Retry once the
/// user has freed some. Reveal is there because the next question is always "where is it?".
/// The index is there but cannot be opened yet: another process holds it, its volume is not
/// mounted, or an upgrade is waiting for disk space. The app retries on its own and says what it is
/// waiting for; there is nothing for the user to decide (AppModel.waitForIndex).
struct IndexWaitingView: View {
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            ProgressView().controlSize(.large)
            Text("Waiting for the index").font(.title2)
            Text(message).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 460)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// The index was written by a newer Omni. Only an update can open it, and opening it with this
/// build would mean rewriting it in an older format.
struct IndexNewerView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "arrow.down.circle").font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text("This index needs a newer Omni").font(.title2)
            Text("It was written by a newer version. Update Omni to open it.").foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 460)
            Button("Check for Updates") { Updater.check(userInitiated: true) }
                .buttonStyle(.borderedProminent).controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// Hides the window's TITLE TEXT (not the title bar) so the toolbar's leading slot is free for the
/// back/forward chevrons and there's no redundant "Omni" label - while the title bar (and its Liquid
/// Glass toolbar material) stays intact, unlike `.hiddenTitleBar`. Re-applied on every update because
/// SwiftUI re-asserts `.visible` from the Window scene's title; the window keeps its "Omni" title for
/// the Window menu, Mission Control, and Stage Manager.
private struct WindowTitleHider: NSViewRepresentable {
    /// The drawer width this mode wants. Declaring it on the split view is not enough: the window
    /// restores the divider it was last left at and that restoration wins, so a cold launch
    /// straight into OCR opened the page rail at the search sidebar's width.
    var sidebarWidth: CGFloat

    /// Window tuner (an invisible background view holding coalesced observers). Despite the legacy
    /// name it does not touch the window title: stock Sequoia titlebar chrome - the visible "Omni"
    /// title, the system sidebar toggle, the split tracking separator - proved load-bearing, and
    /// every attempt to hide any of it broke another state (the toggle vanished, section layout
    /// collapsed, or divider drags misrendered).
    ///
    /// WHAT IT ACTUALLY DOES, and this list is now the code rather than a memory of it: sets the
    /// restored sidebar width once per wanted value, turns off the titlebar separator, pins the
    /// full-height `.unified` toolbar style, keeps the window alive when it closes, and hides the
    /// 1pt rule inside Tahoe's titlebar scroll pocket.
    ///
    /// It no longer caps the search field, installs a flexible-space constraint, or embeds the
    /// search-by-file button in the field - 692c238 removed all three when that button became a
    /// toolbar item, and this comment went on claiming them. If a pre-Tahoe layout problem is ever
    /// traced back here, that is the history: the tuner is smaller than it reads.
    ///
    /// All work runs in a coalesced main.async pass - never synchronously inside a window or
    /// toolbar notification, where mutations mid-SwiftUI-commit are unsafe.
    final class TunerView: NSView {
        var sidebarWidth: CGFloat = 0 {
            didSet { if sidebarWidth != oldValue { appliedWidth = nil; scheduleApply() } } }
        private var appliedWidth: CGFloat?
        // nonisolated(unsafe): deinit is nonisolated under strict concurrency; the view lives and
        // dies on the main thread, so the unregistration is race-free in practice.
        nonisolated(unsafe) private var observers: [NSObjectProtocol] = []

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard observers.isEmpty, window != nil else { return }
            scheduleApply()
            let nc = NotificationCenter.default
            // didUpdate fires after EVERY event (each key, each mouse move), and a pass re-sets the
            // window chrome and walks the titlebar views. Toolbar adds/removes and width changes
            // still apply at once; plain window updates at most once a second, as a backstop.
            observers.append(nc.addObserver(forName: NSWindow.didUpdateNotification, object: window, queue: nil) { [weak self] _ in
                guard let self, CFAbsoluteTimeGetCurrent() - self.lastApply > 1 else { return }
                self.scheduleApply()
            })
            for name in [NSToolbar.didRemoveItemNotification, NSToolbar.willAddItemNotification] {
                observers.append(nc.addObserver(forName: name, object: nil, queue: nil) { [weak self] _ in
                    self?.scheduleApply()
                })
            }
        }
        deinit {
            for o in observers { NotificationCenter.default.removeObserver(o) }
        }

        // nonisolated(unsafe): touched from notification closures that are main-thread in practice
        // (window updates, toolbar mutations); a stale read only coalesces one extra pass.
        nonisolated(unsafe) private var applyScheduled = false
        nonisolated(unsafe) private var lastApply: CFAbsoluteTime = 0
        private nonisolated func scheduleApply() {
            guard !applyScheduled else { return }
            applyScheduled = true
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.applyScheduled = false
                self.lastApply = CFAbsoluteTimeGetCurrent()
                if let w = self.window { self.apply(w) }
            }
        }

        private func apply(_ w: NSWindow) {
            applySidebarWidth(w)
            // macOS 14/15 draw the window title in the toolbar; Tahoe removes it with
            // `.toolbar(removing: .title)`. See the `center` item in the toolbar for the space.
            if #unavailable(macOS 26.0), w.titleVisibility != .hidden { w.titleVisibility = .hidden }
            // No rule under the toolbar. Finder's column header sits on the SAME surface as the
            // chrome with no seam between them; the automatic separator drew a hairline there and
            // made the header read as a second bar stuck underneath. The header keeps its OWN
            // divider, which separates it from the rows - that one Finder has too.
            w.titlebarSeparatorStyle = .none
            // FULL HEIGHT, not the compact one. Measured against Finder at the same window size:
            // its chrome runs 0..43 and ours ran 0..39 - a 44pt bar against a 40pt one, both
            // holding the same 36pt item capsules, so ours had 2pt of air around them where
            // Finder has 4. `.unified` is the style Finder uses; hiding the title text was enough
            // for AppKit to pick the compact one on its own.
            w.toolbarStyle = .unified
            // Closing must not DESTROY the window, or there is nothing left to bring back when the
            // Dock icon is clicked (see AppDelegate.showMainWindow). The process outlives the
            // window on purpose: serving, MCP and skills keep working with nothing on screen.
            w.isReleasedWhenClosed = false
            // CONTENT DOES NOT BLUR UNDER THIS TOOLBAR, and not for want of trying. Finder's does:
            // its scroll pocket is the SOFT one, a gradient with no rule. Ours is
            // `NSHardPocketView` (named by the live view tree), owned by the SwiftUI split view's
            // titlebar background rather than by any scroll view, so `scrollEdgeEffectStyle(.soft,
            // for: .top)` does not reach it - tried on the results scroll view itself, on the
            // browser's root and on the whole detail pane. `NSSplitViewItem.titlebarSeparatorStyle`
            // has nothing to set (a SwiftUI window has no `NSSplitViewController`), and
            // `titlebarAppearsTransparent = true` DOES let content through but takes the toolbar's
            // backdrop with it - rows then run straight across the buttons. What is fixed is the
            // rule that used to sit at the boundary; the blur needs API this window structure does
            // not expose.
            // ...and the window's setting is not what draws the rule that was actually there.
            // Measured: two hairlines, one at the titlebar boundary and one under our own header;
            // Finder has only the second. The live view tree named the first one - a 1pt
            // `_NSLayerBasedFillColorView` inside `NSHardPocketView < NSScrollPocket <
            // NSTitlebarBackgroundView < NSSplitView`, i.e. Tahoe's scroll-edge effect in its HARD
            // style. Finder's list AND icon views both use the soft one (a faint gradient, no
            // rule - sampled at 245 fading to 253 over ~25pt). SwiftUI's `scrollEdgeEffectStyle`
            // does not reach it: tried on the list, on the browser's root, and on the whole detail
            // pane, and the rule survived all three, because the pocket belongs to the split view's
            // titlebar background rather than to any scroll view in the subtree. So hide the rule.
            hidePocketRule(w.contentView?.superview)
        }

        /// Hides the 1pt rule inside Tahoe's titlebar scroll pocket. Deliberately shallow: it
        /// stops at `NSTitlebarBackgroundView`, a handful of levels below the theme frame, so it
        /// never walks the content pane's view tree. Entirely defensive - if AppKit ever renames
        /// or restructures these views nothing matches and nothing is touched.
        private func hidePocketRule(_ v: NSView?, depth: Int = 0) {
            guard let v else { return }
            if String(describing: type(of: v)).contains("TitlebarBackgroundView") {
                hideThinFills(in: v)
                return
            }
            guard depth < 8 else { return }
            for c in v.subviews { hidePocketRule(c, depth: depth + 1) }
        }

        private func hideThinFills(in v: NSView) {
            if v.bounds.height <= 1 && v.bounds.width > 100 && !v.isHidden { v.isHidden = true }
            for c in v.subviews { hideThinFills(in: c) }
        }

        /// Set once per wanted width, never on every window update: this is a correction to what
        /// the window restored, not a policy, and re-applying it would fight a divider drag.
        private func applySidebarWidth(_ w: NSWindow) {
            guard sidebarWidth > 0, appliedWidth != sidebarWidth,
                  let split = Self.firstSplitView(in: w.contentView),
                  split.subviews.count >= 2 else { return }
            appliedWidth = sidebarWidth
            if abs(split.subviews[0].frame.width - sidebarWidth) > 1 {
                split.setPosition(sidebarWidth, ofDividerAt: 0)
            }
        }

        private static func firstSplitView(in view: NSView?) -> NSSplitView? {
            guard let view else { return nil }
            if let split = view as? NSSplitView { return split }
            for sub in view.subviews {
                if let split = firstSplitView(in: sub) { return split }
            }
            return nil
        }

    }

    func makeNSView(context: Context) -> NSView {
        let v = TunerView()
        v.sidebarWidth = sidebarWidth
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? TunerView)?.sidebarWidth = sidebarWidth
    }
}


/// A bar pinned in the top safe area, with scroll content passing under it. On Tahoe that is
/// `safeAreaBar`, which is what lets the scroll edge effect apply; earlier systems just stack it.
/// An EMPTY bar must cost nothing, so the modifier checks before it inserts one.
private struct TopBar<Bar: View>: ViewModifier {
    @ViewBuilder var bar: () -> Bar

    func body(content: Content) -> some View {
        // No fill and no rule on Tahoe: the bar sits in the scroll-edge pocket, and a background
        // behind it blocks the soft edge ("remove extra backgrounds behind bar items", WWDC25 323).
        // Before Tahoe it stacks above the content, where the bar material and a divider separate it.
        if #available(macOS 26.0, *) {
            content.safeAreaBar(edge: .top, spacing: 0) { bar() }
        } else {
            VStack(spacing: 0) {
                bar().background(.bar).overlay(alignment: .bottom) { Divider() }
                content
            }
        }
    }
}
