import SwiftUI
import AppKit
import OmniKit
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var debounce: Task<Void, Never>?
    @State private var historyDebounce: Task<Void, Never>?
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
    private var ocrDrawerWanted: Bool { !model.ocrMode || !ocr.pages.isEmpty }

    /// Apply a user edit of the search box: parse it into the semantic query + qualifiers, apply the
    /// filters, clear a file query if real text was typed, and schedule the (debounced) search. The
    /// box binds to the RAW typed string; `set` (user edits only) routes here.
    private func handleQueryEdit(_ typed: String) {
        model.suggestionsAllowed = true   // this fires only on real keystrokes (the .searchable set:), so arm the dropdown
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
        Group {
            if model.ocrMode {
                // The same field, a different question. Searching the vector index while reading a
                // transcript answers something nobody asked; what a reader wants here is Preview's
                // find - matches marked where they are, with a way to step through them. No
                // suggestions: there is nothing to complete against one document.
                split
                    .searchable(text: Binding(get: { ocr.find }, set: { ocr.find = $0 }),
                                placement: .toolbar,
                                prompt: "Find in document")
                    .onSubmit(of: .search) { ocr.stepMatch(by: 1) }
            } else if showsSearch {
                split
                    // Filters are CHIPS in the field, not text the reader has to retype and step
                    // over. `searchable(text:tokens:)` is the platform's own control for this
                    // (macOS 13+), so there is no token-field component to import and no second
                    // source of truth: the chips are a projection of the canonical query string,
                    // which is still the thing history and back/forward replay.
                    .searchable(text: Binding(get: { model.query }, set: { handleQueryEdit($0) }),
                                tokens: Binding(get: { model.searchTokens },
                                                set: { model.setSearchTokens($0) }),
                                placement: .toolbar, prompt: "Search by meaning") { token in
                        // Text, not Label: a token chip renders its title only on macOS, so an
                        // icon here is carried and then thrown away.
                        Text(token.label)
                    }
                    // Typeahead: keys (ty -> type:), values (type: -> image/...), and matching past
                    // queries as instant (cached) shortcuts. Navigate with arrows + Return. Only while
                    // the user is typing - a programmatic box change (history replay, filter menu) keeps
                    // the dropdown closed (suggestionsAllowed is false unless handleQueryEdit armed it).
                    .searchSuggestions {
                        ForEach(model.suggestionsAllowed ? searchSuggestions(model.query) : [], id: \.completion) { sug in
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
                    // Return finishes the word too: a qualifier typed without a trailing space
                    // still becomes a chip rather than being embedded as prose.
                    .onSubmit(of: .search) {
                        model.promoteQualifiers()
                        model.search(); model.recordCurrentSearchToHistory(viaSubmit: true)
                    }
                    // Escape clears the whole query, not just the text: the chips and filters are
                    // the same query, so leaving them behind is what made a "cleared" box still
                    // return a filtered, empty result set.
                    .onKeyPress(.escape) {
                        guard model.hasActiveSearch || !model.searchTokens.isEmpty else { return .ignored }
                        model.clearSearch()
                        return .handled
                    }
            } else {
                split
            }
        }
        // An empty page rail is a column of nothing: fold it when OCR mode opens with no document
        // and unfold it the moment one arrives. On the BODY, not on the split - the split is a
        // branch of the Group above, so flipping the mode replaces it and takes any `onChange`
        // declared there with it, which is why the drawer stayed open.
        .onChange(of: ocrDrawerWanted, initial: true) { _, wanted in
            withAnimation(.easeOut(duration: 0.2)) { columns = wanted ? .all : .detailOnly }
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
            // Gated to macOS 26 because that is the only system this was verified on. The API
            // itself exists from macOS 15, so widening the check is a one-token change once the
            // same pass (title gone, system toggle present, divider drag clean, trailing cluster
            // right-aligned) has been run on a Sequoia machine. macOS 14/15 keep stock chrome,
            // which is the known-good state there.
            Group {
                if #available(macOS 26.0, *) {
                    detailOrOCR.toolbar(removing: .title)
                } else {
                    detailOrOCR
                }
            }
            .toolbar { toolbar }
            .background(WindowTitleHider(sidebarWidth: model.ocrMode ? 168 : 260))
        }
    }

    // MARK: - Detail

    /// Search results or the OCR workspace. The toggle replaces the content area rather than
    /// opening a window: this is a different way of looking at documents on this Mac, not a
    /// different app, and a separate window would strand it from the sidebar and the index.
    @ViewBuilder private var detailOrOCR: some View {
        if model.ocrMode {
            OCRView()
        } else {
            detail
        }
    }

    @ViewBuilder private var detail: some View {
        switch model.phase {
        case .loadingModel:
            // The subtitle sets expectations for something that happens AFTER this screen is gone:
            // going .ready does not await the warm-up (see AppModel - gating readiness on it made
            // launch look hung on an M2), so a search fired in that window queues behind it.
            //
            // It says "loads into memory", not "compiles", because that is where the time measured
            // as: omni-verify warmbench <model> <db>, 4.5M rows / 6.9 GB of bf16. Metal pipelines
            // cost 4 ms (they ship precompiled in default.metallib); the first search costs 621 ms
            // with the vector file cold and 17 ms once the page cache holds it. It is paging, so no
            // duration is promised - a Mac that cannot cache 6.9 GB pays it on every launch.
            // A one-time index upgrade is a different thing from loading the model, and takes tens
            // of seconds on a large index - saying "loading the model" through a database rewrite
            // is how a working upgrade reads as a hang.
            // ONE bar for the whole launch - the store now scales its load and its one-time
            // upgrade into a single fraction, so this keeps moving through both. Only the WORDS
            // change with the phase, so a database rewrite is never described as loading the model.
            CenteredStatus(symbol: model.launchSymbol,
                           title: model.launchTitle,
                           subtitle: model.launchSubtitle,
                           showSpinner: true, progress: model.loadingProgress)
        case .noModel:
            OnboardingView()
        case .failed(let msg):
            EngineFailedView(message: msg)
        case .failedIndex(let why):
            IndexFailedView(message: why)
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
            && model.rawResults.isEmpty && model.queryError == nil && !model.isResolving
    }

    /// A Photos source is being browsed. Same precedence rule as the folder browser below, and
    /// checked before it - `enterPhotoSource` clears `filterFolder` and `enterFolder` clears the
    /// source, so only one can be set, but the order makes that explicit rather than incidental.
    private var showsPhotoBrowser: Bool {
        model.browsedPhotoSource != nil && !model.hasQuery && model.fileQuery == nil
            && model.rawResults.isEmpty && model.queryError == nil && !model.isResolving
    }

    /// Same precedence as the map below: a folder is being browsed, and nothing search-related is
    /// active. Typing hides it instantly and the results are already scoped to the folder, which
    /// is the whole point - the browser and the search are two views of one `filterFolder`.
    private var showsFolderBrowser: Bool {
        // EXACTLY ONE folder. Scoping to several is a search filter, not a place to browse - the
        // empty-result region holds one listing, and showing the first of two would misrepresent
        // what the search is scoped to.
        model.filterFolders.count == 1 && !model.hasQuery && model.fileQuery == nil
            && model.rawResults.isEmpty && model.queryError == nil && !model.isResolving
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
                ResultsList(results: model.results) { belowThresholdFooter }
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
        // Drag an image, file, or text from anywhere (Finder, a browser, another app) - or paste one
        // (Cmd-V) - to search by it. SwiftUI's .onDrop gives us reachability over the results list and
        // the empty state alike, but a web image dragged from Chrome/Safari arrives as inline encoded
        // bytes, a file promise, or a remote URL - none of which loadObject(NSImage/NSURL) can resolve.
        // So we read the raw drag pasteboard for the full flavor set, and fall back to the providers.
        .onDrop(of: [.image, .fileURL, .url, .text, .plainText], isTargeted: $fileDropTargeted) { providers in
            handleSearchDrop(providers)
        }
        .overlay {
            if fileDropTargeted {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: 2).padding(6).allowsHitTesting(false)
            }
        }
    }

    // MARK: - Drag to search

    /// Route a drop onto the search surface. Reachability is SwiftUI's (the closure fires over the
    /// results list and the empty state alike); for the payload we read the raw drag pasteboard,
    /// which carries the full flavor set a browser provides, then fall back to the item providers.
    private func handleSearchDrop(_ providers: [NSItemProvider]) -> Bool {
        if handleDragPasteboard(NSPasteboard(name: .drag)) { return true }
        return handleDropProviders(providers)
    }

    /// Read a drag/paste pasteboard in fidelity order: a local file, inline image bytes (Chrome's
    /// public.jpeg/png, Safari/Firefox public.tiff), a file promise (Safari/Chrome), a remote image
    /// URL it downloads (Firefox / URL-only), then any other bitmap, then text. Model calls run on
    /// the main actor; the async promise/download paths hop back to it.
    private func handleDragPasteboard(_ pb: NSPasteboard) -> Bool {
        // 1) Local file (Finder, Mail attachment): the existing file-query path. Take the first
        //    SUPPORTED file, so a mixed multi-file drag still finds the one Omni can search.
        if let url = (pb.readObjects(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) as? [URL])?
            .first(where: { FileExtractor.kind(for: $0) != nil }) {
            Task { @MainActor in model.setFileQuery(url) }
            return true
        }
        // 2) Inline image bytes - synchronous, no network; preferred (Omni re-encodes to PNG anyway).
        if let (data, ext) = Self.pasteboardImageBytes(pb) {
            Task { @MainActor in model.searchByImage(data: data, suggestedExtension: ext) }
            return true
        }
        // 3) File promise (Safari, Chrome): the browser writes the file into our temp dir, async.
        //    Honor a single promise (the first) so one drag is one search - a multi-image drag must
        //    not fire N staggered searches with a nondeterministic winner. The omni-drop- prefix lets
        //    the launch sweep reclaim the dir. (FileManager.temporaryDirectory is per-user-writable.)
        if let promise = (pb.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil)
            as? [NSFilePromiseReceiver])?.first {
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("omni-drop-promise-\(UUID().uuidString)", isDirectory: true)
            try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            promise.receivePromisedFiles(atDestination: dest, options: [:],
                                         operationQueue: OperationQueue()) { url, error in
                guard error == nil, FileExtractor.kind(for: url) != nil else { return }
                Task { @MainActor in model.setFileQuery(url) }
            }
            return true
        }
        // 4) Remote image URL (Firefox / URL-only): download, then confirm it decodes as an image
        //    (a linked <img> can put the link href on public.url instead of the image src).
        if let remote = (pb.readObjects(forClasses: [NSURL.self], options: nil) as? [URL])?
            .first(where: { !$0.isFileURL && ($0.scheme == "http" || $0.scheme == "https") }) {
            downloadImage(remote, textFallback: remote.absoluteString)   // a bare link -> text search
            return true
        }
        // 5) Any other bitmap the pasteboard can vend.
        if let img = NSImage(pasteboard: pb) {
            Task { @MainActor in model.searchByImage(img) }
            return true
        }
        // 6) Text.
        if let s = pb.string(forType: .string), !s.isEmpty {
            Task { @MainActor in model.searchByText(s) }
            return true
        }
        return false
    }

    /// Fallback when the drag pasteboard is unavailable: resolve through the SwiftUI item providers.
    /// Covers the common cases (Finder file, a data-backed image, a Chrome web image whose remote URL
    /// surfaces as a provider), and only text-searches a dropped URL when it is not an image.
    private func handleDropProviders(_ providers: [NSItemProvider]) -> Bool {
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, url.isFileURL, FileExtractor.kind(for: url) != nil else { return }
                Task { @MainActor in model.setFileQuery(url) }
            }
            return true
        }
        if let p = providers.first(where: { $0.canLoadObject(ofClass: NSImage.self) }) {
            _ = p.loadObject(ofClass: NSImage.self) { obj, _ in
                guard let img = obj as? NSImage else { return }
                Task { @MainActor in model.searchByImage(img) }
            }
            return true
        }
        if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }) {
            _ = p.loadObject(ofClass: NSURL.self) { obj, _ in
                guard let url = obj as? URL, !url.isFileURL,
                      url.scheme == "http" || url.scheme == "https" else { return }
                downloadImage(url, textFallback: url.absoluteString)
            }
            return true
        }
        if let p = providers.first(where: { $0.canLoadObject(ofClass: NSString.self) }) {
            _ = p.loadObject(ofClass: NSString.self) { obj, _ in
                guard let s = obj as? String else { return }
                Task { @MainActor in model.searchByText(s) }
            }
            return true
        }
        return false
    }

    /// Download a remote URL and search by it if the bytes decode as an image; otherwise, if a text
    /// fallback was given (a bare hyperlink), run a text search on it.
    private func downloadImage(_ url: URL, textFallback: String? = nil) {
        URLSession.shared.dataTask(with: url) { data, response, _ in
            guard let data, NSImage(data: data) != nil else {
                if let textFallback { Task { @MainActor in model.searchByText(textFallback) } }
                return
            }
            let ext = response?.mimeType
                .flatMap { UTType(mimeType: $0)?.preferredFilenameExtension } ?? "png"
            Task { @MainActor in model.searchByImage(data: data, suggestedExtension: ext) }
        }.resume()
    }

    /// Encoded image bytes off a pasteboard: a named bitmap type first, then any image UTI
    /// (catches Chrome's public.jpeg / public.gif / public.webp).
    static func pasteboardImageBytes(_ pb: NSPasteboard) -> (Data, String)? {
        for t in [NSPasteboard.PasteboardType.png, .tiff] {
            if let d = pb.data(forType: t), let e = UTType(t.rawValue)?.preferredFilenameExtension {
                return (d, e)
            }
        }
        for t in pb.types ?? [] {
            guard let ut = UTType(t.rawValue), ut.conforms(to: .image),
                  let d = pb.data(forType: t) else { continue }
            return (d, ut.preferredFilenameExtension ?? "png")
        }
        return nil
    }

    @ViewBuilder private var emptyState: some View {
        // Indexing is invisible here - the sidebar's per-folder progress is the only cue, and
        // search works while it runs. The user just adds folders and searches.
        if model.roots.isEmpty {
            CenteredStatus(symbol: "folder.badge.plus", title: "Add a folder to search",
                           subtitle: "Choose the folders you want to search. Omni indexes them automatically and keeps them up to date.",
                           showSpinner: false, action: ("Add folder\u{2026}", { pickFolder() }))
        } else if let err = model.queryError {
            CenteredStatus(symbol: "exclamationmark.triangle", title: "Couldn't search by that file",
                           subtitle: err, showSpinner: false)
        } else if model.indexObsolete && model.hasQuery {
            // A dim/model mismatch makes every search return nothing; explain it and offer both the
            // cheap fix (switch back to the model the index was built with) and the rebuild.
            let built = model.indexBuiltVariant
            CenteredStatus(symbol: "arrow.triangle.2.circlepath",
                           title: built != nil ? "Switch to \(built!.title) or reindex" : "Reindex to search",
                           subtitle: built != nil
                               ? "This index was built with \(built!.title), but \(model.modelVariant.title) is loaded. Switch back to keep your index, or reindex with the current model."
                               : "This index was built with a different model than the one loaded. Reindex to search again.",
                           showSpinner: false,
                           action: built.map { v in ("Switch to \(v.title)", { model.selectVariant(v) }) },
                           secondary: ("Reindex", { model.startIndexing() }))
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
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle",
                           title: "No results above \(Int(model.minScore * 100))%",
                           subtitle: "\(model.hiddenByThreshold) weaker \(model.hiddenByThreshold == 1 ? "match is" : "matches are") hidden by the relevance threshold.",
                           showSpinner: false, action: ("Show all matches", { model.showAllBelowThreshold() }))
        } else if model.filtersActive {
            // Filters can hide every result; the empty state is the only place left to escape them.
            CenteredStatus(symbol: "line.3.horizontal.decrease.circle", title: "No matches",
                           subtitle: "Filters are hiding every result.", showSpinner: false,
                           action: ("Clear filters", { model.clearFilters() }))
        } else {
            CenteredStatus(symbol: "magnifyingglass", title: "No matches", subtitle: "Try a different phrase.", showSpinner: false)
        }
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
            Button { model.showAllBelowThreshold() } label: {
                Label("Show \(model.hiddenByThreshold) more \(model.hiddenByThreshold == 1 ? "match" : "matches")", systemImage: "chevron.down")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
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
                content.glassEffect(.regular, in: .capsule)
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
            Image(systemName: symbol)
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
        if showsPhotoBrowser, let source = model.browsedPhotoSource {
            return (source.title, source.title)
        }
        if showsFolderBrowser, let folder = model.filterFolder {
            return (folder.lastPathComponent, folder.path)
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
            Text(t.name).font(.headline).foregroundStyle(.secondary).help(t.help)
        }
    }

    private var sidebarToggleButton: some View {
        Button {
            withAnimation(.easeOut(duration: 0.2)) {
                columns = columns == .detailOnly ? .all : .detailOnly
            }
        } label: {
            Image(systemName: "sidebar.leading")
        }
        .help("Hide or show the sidebar")
        .accessibilityLabel("Hide or show the sidebar")
        .accessibilityIdentifier("sidebar.toggle")
    }

    private var ocrToggleButton: some View {
        // No `withAnimation` on the change: the two modes are different content, not a moved view,
        // and animating the swap made the whole pane slide in from the window's leading edge.
        ToolbarToggle(isOn: Binding(get: { model.ocrMode }, set: { model.ocrMode = $0 }),
                      symbol: "text.viewfinder")
    }

    /// Either browser is on screen. The two are one mode as far as the toolbar is concerned.
    private var showsBrowser: Bool { showsFolderBrowser || showsPhotoBrowser }

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
        .help("Switch between list and gallery")
    }

    /// Search by a file, and share what is selected - the same pair, in the same order, that OCR
    /// mode puts next to its own search field (`ocr.open`, `ocr.share`). This USED to be a glyph
    /// installed inside the search field itself, which had two problems: it was invisible as an
    /// affordance, and it had to hide whenever the field held text, so it disappeared exactly when
    /// a query was on screen. A toolbar button is always there and always the same size.
    @ToolbarContentBuilder private var fileActions: some ToolbarContent {
        ToolbarItem(id: "search.open") {
            Button { model.searchByFilePanel() } label: {
                // `folder`, the same symbol OCR's Open Document uses. The two are the same verb.
                Label("Search by File\u{2026}", systemImage: "folder")
            }
            .help("Search by a file  \u{21e7}\u{2318}O")
            .accessibilityLabel("Search by a file")
        }
        ToolbarItem(id: "search.share") {
            // The system share sheet, not a menu of our own - same as OCR's. Disabled rather than
            // hidden when nothing is selected, so the group does not change width as you click
            // around; that is how the OCR group behaves too.
            ShareLink(items: model.selectedURLsOrdered) {
                Label("Share\u{2026}", systemImage: "square.and.arrow.up")
            }
            // Both states say something true and useful: what it will do, or what is missing.
            .help(model.selectedURLsOrdered.isEmpty ? "Select a file to share" : "Share the selection")
            .disabled(model.selectedURLsOrdered.isEmpty)
        }
    }

    /// Names the port when it is actually listening, because that is the thing a user needs next.
    /// Names the port when it is actually listening, because that is what a reader needs next. No
    /// "click to stop": it is a toggle, and its filled state already says which way it is.
    private var servingHelp: String {
        model.serving.isRunning ? "Serving on port \(model.serving.port)" : "Serve over HTTP"
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
        ToolbarItem(id: "sidebar.mode", placement: .navigation) { sidebarToggleButton }
        ToolbarItem(id: "ocr.mode", placement: .navigation) {
            // On is a FILLED accent circle with a white glyph, the way Preview draws Markup while
            // it is on. The fill is drawn INSIDE the label rather than by `.borderedProminent`: a
            // prominent button takes a background of its own, which broke this item out of the
            // glass capsule it shares with the sidebar toggle - two separate surfaces in OCR mode
            // where every other mode had one.
            ocrToggleButton
                .help(model.ocrMode ? "Back to search  \u{2318}\u{2325}O" : "Transcribe a document  \u{2318}\u{2325}O")
                .accessibilityLabel(model.ocrMode ? "Back to search" : "Transcribe a document")
                .accessibilityIdentifier("ocr.toggle")
        }
        // A shortcut to Settings > Serving > "Serve Omni over HTTP", on the same switch - not a
        // second source of truth. It sits with the sidebar and OCR toggles because it is the same
        // KIND of control: a global mode of the app, on until you turn it off, rather than an
        // action on whatever happens to be on screen. It was briefly next to Share, which grouped
        // it with per-file actions it has nothing to do with.
        ToolbarItem(id: "serve.mode", placement: .navigation) {
            ToolbarToggle(isOn: Binding(get: { model.serving.enabled },
                                        set: { model.serving.enabled = $0 }),
                          symbol: "network")
                .help(servingHelp)
                .accessibilityLabel("Serve over HTTP")
        }
        // ONE ITEM for the chevrons AND the name, in that order. Two items cannot do it: a
        // toolbar item that only exists sometimes is APPENDED when it comes back, so descending
        // into a folder (which is what makes the chevrons appear) put them to the RIGHT of the
        // name. An always-present empty item is no good either - Tahoe charges a full inter-group
        // gap on both sides of one, which pinned the name ~46pt right of the mode pill whether or
        // not there was anything to go back to. Inside a single item the chevrons simply appear
        // and the name slides to meet them, which is the behaviour Finder has.
        //
        // The ITEM draws no glass (`.sharedBackgroundVisibility(.hidden)`); the chevrons carry
        // their own capsule. That is the whole point: Finder's name is loose text beside the
        // navigation group, not a third button inside it.
        if #available(macOS 26.0, *) {
            ToolbarItem(id: "nav.title", placement: .navigation) { navAndTitle }
                .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(id: "nav.title", placement: .navigation) { navAndTitle }
        }
        // Flexible space after back/forward pushes every other control to the trailing edge (chevrons
        // own the left, everything else is right-aligned), and on Tahoe it's also the correct separator
        // between Liquid Glass toolbar groups - the leading chevron pill and the trailing control pills
        // read as distinct glass surfaces.
        if #available(macOS 26.0, *) {
            ToolbarSpacer(.flexible)
        }
        // Pre-Tahoe gets the equivalent flexible space from WindowTitleHider's tuner, which inserts
        // AppKit's native .flexibleSpace NSToolbarItem after the chevrons: a SwiftUI
        // `ToolbarItem { Spacer() }` is silently DROPPED on macOS 14/15 (verified via the live
        // NSToolbar's item list), so without the AppKit item nothing separates the leading chevrons
        // from the trailing cluster and the stretchy search field parks every control center-left.
        // Bookmark the current search. The only way into History when recording is set to "Only when
        // I bookmark", and a quick save otherwise. Appears once there's a search to keep.
        if model.phase == .ready, !model.ocrMode, model.hasActiveSearch {
            ToolbarItem(placement: .primaryAction) {
                Button { model.toggleBookmarkCurrentSearch() } label: {
                    // No explicit color in the unbookmarked state, so the toolbar can dim it like every
                    // other button when the window resigns key (e.g. while Settings is open). Yellow is
                    // applied only when bookmarked, where the lit status color is intentional.
                    if model.currentSearchIsBookmarked {
                        Image(systemName: "star.fill").foregroundStyle(.yellow)
                    } else {
                        Image(systemName: "star")
                    }
                }
                // Cmd-D is owned by the File-menu "Bookmark search" command (single owner, avoids a
                // duplicate-shortcut conflict); the tooltip names it, and accessibilityLabel is what
                // VoiceOver reads and what the toolbar-overflow menu shows for this icon-only button.
                .help(model.currentSearchIsBookmarked ? "Remove bookmark  \u{2318}D" : "Bookmark this search  \u{2318}D")
                .accessibilityLabel(model.currentSearchIsBookmarked ? "Remove bookmark" : "Bookmark search")
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
        if model.phase == .ready, !model.ocrMode, !showsBrowser,
           !model.rawResults.isEmpty || model.filtersActive {
        // Filter joins sort/view in the trailing placement so on Tahoe the three result controls
        // share ONE Liquid Glass pill (search-by-file + bookmark form the other). filterPlacement
        // keeps filter leading on pre-26 so the Sequoia toolbar layout is unchanged.
        ToolbarItem(placement: filterPlacement) {
            filterMenu.disabled(model.indexedFiles == 0)
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
        if model.phase == .ready, !model.ocrMode, showsBrowser, model.rawResults.isEmpty {
            ToolbarItem(id: "browse.view", placement: .primaryAction) { viewPicker }
        }
        if model.phase == .ready, !model.ocrMode, !model.rawResults.isEmpty {
        ToolbarItem(placement: .primaryAction) {
            if #available(macOS 26.0, *) {
                // Tahoe: the inline sort menu + segmented view toggle render and overflow cleanly.
                ControlGroup {
                    Menu {
                        Picker("Sort by", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
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
                    Picker("Sort by", selection: Binding(get: { model.sortOrder }, set: { model.sortOrder = $0 })) {
                        ForEach(SortOrder.allCases) { Text($0.title).tag($0) }
                    }
                } label: {
                    Label("View options", systemImage: "slider.horizontal.3")
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
            Section {
                Toggle(isOn: Binding(get: { model.literalQuery },
                                     set: { _ in model.toggleLiteralQuery() })) {
                    Label("Plain text", systemImage: "textformat")
                }
                .help("Embed the box text as-is, ignoring key:value qualifiers")
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
                Text("All folders").tag("")
                ForEach(model.roots, id: \.self) { Text($0.lastPathComponent).tag($0.path) }
            }
            Picker("Extension", selection: Binding(get: { model.filterExt }, set: { model.filterExt = $0 })) {
                Text("Any extension").tag("")
                ForEach(model.indexedExts, id: \.self) { Text(".\($0)").tag($0) }
            }
            Picker("Date", selection: Binding(get: { model.dateRange }, set: { model.dateRange = $0 })) {
                ForEach(DateRange.allCases) { Text($0.title).tag($0) }
            }
            Picker("Relevance", selection: Binding(get: { model.minScore }, set: { model.minScore = $0 })) {
                Text("Any").tag(0.0); Text("25%").tag(0.25); Text("50%").tag(0.5); Text("70%").tag(0.7)
            }
            Divider()
            Button("Clear filters") { model.clearFilters() }.disabled(!model.filtersActive)
        } label: {
            Image(systemName: model.filtersActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .help("Filter results")
        .accessibilityLabel("Filter results")
    }

    private func scheduleSearch() {
        debounce?.cancel()
        debounce = Task {
            try? await Task.sleep(nanoseconds: 180_000_000)
            if !Task.isCancelled { model.search() }
        }
    }

    // Auto-record (history mode .auto) only after the query has been settled for 3s, so a search has
    // to be one the user actually dwelled on - quick type-and-click-through queries aren't stored.
    // Cancelled on every keystroke, so it only fires once typing stops. (No effect in .onSubmit /
    // .manual modes, which record on Return / the bookmark button instead.)
    private func scheduleHistoryRecord() {
        historyDebounce?.cancel()
        historyDebounce = Task {
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
        return Array(out.prefix(10))
    }

    private func valueSuggestions(_ key: String) -> [String] {
        switch key {
        case "type": return ["image", "video", "audio", "text", "scanned"]
        case "date": return ["any", "week", "month", "year"]
        case "after": return ["week", "month", "year", "7d", "30d", "1y"]
        case "score": return ["25%", "50%", "70%"]
        case "sort": return ["relevance", "name", "date"]
        case "ext": return model.indexedExts
        case "in": return model.roots.map { ($0.path as NSString).abbreviatingWithTildeInPath }
        // filename: takes free text - there is nothing sensible to enumerate, and offering a
        // sample of 135,000 basenames would be noise rather than help.
        case "filename": return []
        default: return []
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { model.addRoots(panel.urls) }
    }

}

/// A thin bar under the search field showing the qualifiers Omni parsed from the box (or the
/// literal-mode state), with a one-click toggle to treat the box as plain text instead of filters.
private struct QualifierBar: View {
    @Environment(AppModel.self) private var model: AppModel
    /// Qualifier keys another view is already showing, better. While browsing, the breadcrumb
    /// carries the folder AND lets you click any ancestor; a chip of the same path is a
    /// duplicate that cannot be clicked.
    var hiding: Set<String> = []
    var body: some View {
        HStack(spacing: 6) {
            if model.literalQuery {
                Image(systemName: "textformat").foregroundStyle(.secondary).frame(width: 18)
                Text("Plain-text search").foregroundStyle(.secondary)
                Text("- qualifiers ignored").font(.caption).foregroundStyle(.tertiary)
            }
            Spacer(minLength: 8)
            Button { model.toggleLiteralQuery() } label: {
                Label(model.literalQuery ? "Use as query" : "Plain text",
                      systemImage: model.literalQuery ? "line.3.horizontal.decrease.circle" : "textformat")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .help(model.literalQuery
                  ? "Interpret key:value as filters again"
                  : "Embed the box text as-is, ignoring key:value qualifiers")
        }
        .font(.callout)
        .padding(.horizontal, 16).padding(.vertical, 6)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
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
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }
}

struct CenteredStatus: View {
    let symbol: String
    let title: String
    let subtitle: String
    var showSpinner: Bool = false
    /// Determinate 0...1 -> a native linear progress bar replaces the indeterminate spinner.
    var progress: Double? = nil
    var action: (String, () -> Void)? = nil
    var secondary: (String, () -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
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
                    if let action { Button(action.0, action: action.1).buttonStyle(.borderedProminent) }
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
    var symbol: String = "sparkle.magnifyingglass"
    var ways: [(icon: String, text: String)] = SearchWaysPrompt.searchWays
    /// Something to do about what the rows just described - the OCR pane hangs its download button
    /// here when the model is missing. `AnyView` because this view is built once per empty state
    /// and a generic parameter would spread through every call site for nothing.
    var footer: AnyView?

    static let searchWays: [(icon: String, text: String)] = [
        ("character.cursor.ibeam", "Type a phrase, ranked by meaning"),
        ("arrow.down.doc", "Drag in an image, file, or text"),
        ("doc.on.clipboard", "Paste an image or text  \u{2318}V"),
        ("photo.badge.magnifyingglass", "Search by a file  \u{21E7}\u{2318}O"),
        ("square.on.square", "Right-click a result for Find Similar"),
    ]

    static let transcribeWays: [(icon: String, text: String)] = [
        ("arrow.down.doc", "Drop a PDF or images"),
        ("folder", "Choose a document  \u{2318}O"),
        ("magnifyingglass", "Find in the transcript, then \u{2318}G"),
        ("doc.on.doc", "Copy as Markdown  \u{21E7}\u{2318}C"),
        ("square.and.arrow.down", "Save as .md  \u{2318}S"),
    ]

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: symbol).font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title)
                .font(.title)
                .contentTransition(.numericText(value: Double(count)))
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
struct IndexFailedView: View {
    @Environment(AppModel.self) private var model: AppModel
    let message: String
    @State private var confirmReindex = false
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "internaldrive").font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text("Omni can't open its index").font(.title)
            Text(message).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 460)
            if let repair = model.repairMessage {
                Text(repair).font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 460)
            }
            HStack {
                Button("Retry") { model.retryBootstrap() }.buttonStyle(.borderedProminent)
                // Repair corrects the vector bookkeeping when the mapping is provable, and says so
                // plainly when it is not - it never guesses, because a wrong guess here returns
                // rows their neighbour's vector with no error at all.
                Button(model.repairRunning ? "Repairing\u{2026}" : "Repair") { model.repairIndex() }
                    .disabled(model.repairRunning || model.dbPath.isEmpty)
                // And the way out of the cases Repair refuses. Destructive, so it is red and asks
                // first: everything it deletes is derived from the user's files, but rebuilding it
                // is hours of embedding on a large index.
                // .tint, not just role: on macOS a destructive ROLE only colours the button inside
                // menus and dialogs - in a plain row it renders identically to its neighbours, which
                // is the one thing this button must not do.
                Button("Reindex", role: .destructive) { confirmReindex = true }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .disabled(model.repairRunning || model.dbPath.isEmpty)
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dbPath)])
                }
                .disabled(model.dbPath.isEmpty)
            }
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .confirmationDialog("Delete the index and start over?", isPresented: $confirmReindex) {
            Button("Delete and reindex", role: .destructive) { model.reindexFromScratch() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your files are not touched. Omni reads them again, which can take a while on a large library.")
        }
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

    /// Toolbar tuner (an invisible background view holding coalesced observers; despite the
    /// legacy name it no longer touches the window title - stock Sequoia titlebar chrome, i.e.
    /// the visible "Omni" title, the system sidebar toggle, and the split tracking separator,
    /// proved load-bearing: every attempt to hide any of them broke another state - the toggle
    /// vanished, section layout collapsed, or divider drags misrendered). What it does do:
    /// - pre-Tahoe: caps the search field at Tahoe's ~300pt and installs the flexible-space
    ///   width constraint on the chevrons item so the trailing cluster hugs the right edge.
    /// - all systems: keeps the search-by-file button installed INSIDE the search field's
    ///   trailing edge (magnifier left, upload right), hidden while the field has text so it
    ///   cannot be drawn over by a long query, and shrinks the stock magnifier to match.
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
            observers.append(nc.addObserver(forName: NSWindow.didUpdateNotification, object: window, queue: nil) { [weak self] _ in
                self?.scheduleApply()
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
        private nonisolated func scheduleApply() {
            guard !applyScheduled else { return }
            applyScheduled = true
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.applyScheduled = false
                if let w = self.window { self.apply(w) }
            }
        }

        private func apply(_ w: NSWindow) {
            applySidebarWidth(w)
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
        if #available(macOS 26.0, *) {
            content.safeAreaBar(edge: .top, spacing: 0) { bar() }
        } else {
            VStack(spacing: 0) { bar(); content }
        }
    }
}
