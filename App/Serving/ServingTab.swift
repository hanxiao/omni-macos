import SwiftUI
import AppKit
import OmniKit

/// Settings > Serving. Binds only to the shared ServingController surface AppModel owns:
/// read-write enabled/scope/port/bearerToken and read-only state/boundAddress. Follows the
/// same Form { Section }.formStyle(.grouped) idiom and the explicit Binding(get:set:) pattern as the other tabs - never $model, since AppModel is @Observable.
struct ServingTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var exampleKind: ExampleKind = .search
    @State private var embedSchema: EmbedSchema = .openai
    @State private var showMCPSheet = false
    @State private var showSkillSheet = false

    /// Top-level example category: the search endpoint, or an embedding endpoint.
    private enum ExampleKind: String, CaseIterable, Identifiable {
        case search = "Search", embed = "Embed", tags = "Tags", ocr = "OCR"
        var id: String { rawValue }
    }
    /// The embedding API schema styles the server speaks (all served at once).
    private enum EmbedSchema: String, CaseIterable, Identifiable {
        case openai = "OpenAI", jina = "Jina", cohere = "Cohere", gemini = "Gemini"
        var id: String { rawValue }
    }

    var body: some View {
        Form {
            serverSection
            exampleSection
            logsSection
        }
        .formStyle(.grouped)
        // NO FIXED HEIGHT. The Settings TabView sizes itself to the selected tab
        // (`fixedSize(vertical:)`), so a pinned height here did not make the window steady - it
        // made this one tab shorter than its own content and put a scroller inside it, which no
        // other tab has.
        .sheet(isPresented: $showMCPSheet) {
            AgentConfigSheet(title: "Connect agents over MCP",
                             subtitle: "For MCP clients with HTTP transport.",
                             text: mcpConfigText, saveAs: nil)
        }
        .sheet(isPresented: $showSkillSheet) {
            AgentConfigSheet(title: "SKILL.md",
                             subtitle: "~/.claude/skills/omni-local-search/SKILL.md",
                             text: skillMarkdown, saveAs: "SKILL.md")
        }
    }

    // MARK: - Server

    @ViewBuilder private var serverSection: some View {
        Section {
            Toggle("Serve Omni over HTTP", isOn: Binding(
                get: { model.serving.enabled },
                set: { model.serving.enabled = $0 }
            ))
            .toggleStyle(.switch)

            // WHO CAN REACH IT and WITH WHAT - directly under the switch that turns it on, because
            // they are the same decision. As their own "Access" section they read as a separate
            // subject, and a reader had to hold the toggle in mind while scrolling past it.
            Picker("Reachable from", selection: Binding(
                get: { model.serving.scope },
                set: { model.serving.scope = $0 }
            )) {
                Text("This Mac only").tag(ServingScope.local)
                Text("Local network").tag(ServingScope.public)
            }

            // `LabeledContent`, not a bare titled `TextField`: a width-constrained field hugs its
            // own label, so the port sat immediately after the word "Port" while every other row
            // in this window puts its value at the trailing edge. Trailing text alignment inside
            // the box keeps the digits ending on the same column as the values above and below.
            LabeledContent("Port") {
                TextField("Port", value: Binding(
                    get: { model.serving.port },
                    set: { model.serving.port = min(65535, max(1, $0)) }   // valid TCP port range
                ), format: .number.grouping(.never))
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
            }

            LabeledContent("Bearer token") {
              HStack(spacing: 6) {
                let token = Binding(
                    get: { model.serving.bearerToken },
                    set: { model.serving.bearerToken = $0 }
                )
                // Shown, not masked. A reveal toggle guards against shoulder-surfing a password;
                // this is a local API token the user has to read to paste into a client, so hiding
                // it by default only added a click before every useful thing you can do with it.
                // Default font, like every other field here - the monospaced treatment made one row
                // of a settings form look like a code sample.
                // Fills whatever the label leaves, and reads from the right like every other
                // value in this window. Both halves are load-bearing: a FIXED width clipped the
                // last characters of a real 32-char token, and without the prompt an empty token
                // left the row as two buttons with nothing between them - a borderless field in a
                // grouped form is invisible until it has text. "Not set" is also the truth; no
                // token is needed while the scope is this Mac only.
                TextField("Bearer token", text: token, prompt: Text("Not set"))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: .infinity)
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(model.serving.bearerToken, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy").disabled(model.serving.bearerToken.isEmpty)
                Button {
                    model.serving.bearerToken = ServingController.generateToken()
                } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("New token")
              }
            }

            // One row: live status on the left, the agent hand-off buttons on the right
            // (enabled only while serving - they configure clients for a running server).
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                if !model.serving.boundAddress.isEmpty {
                    Text(model.serving.boundAddress)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                Button("MCP") { showMCPSheet = true }
                    .help("Config for MCP clients")
                Button("SKILL.md") { showSkillSheet = true }
                    .help("Skill file for agents")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .disabled(!model.serving.enabled)
        } header: {
            Text("Server")
        } footer: {
            Text("Local network access requires a token.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var statusColor: Color {
        switch model.serving.state {
        case .running: return .green
        case .portInUse, .failed: return .orange
        case .stopped: return .secondary
        }
    }

    private var statusText: String {
        switch model.serving.state {
        case .running: return "Running"
        case .stopped: return "Stopped"
        case .portInUse: return "Port in use"
        case .failed(let m): return m.isEmpty ? "Failed" : m
        }
    }

    // MARK: - Example

    @ViewBuilder private var exampleSection: some View {
        Section {
            Picker("Example", selection: $exampleKind) {
                ForEach(ExampleKind.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if exampleKind == .embed {
                // Choose which schema's request shape to show; the server answers all of them.
                Picker("Schema", selection: $embedSchema) {
                    ForEach(EmbedSchema.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            HStack(alignment: .top) {
                Text(exampleCurl)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(exampleCurl, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy")
            }
        } header: {
            Text("Example")
        }
    }

    /// MCP connection snippets, generated from the live port/scope/token.
    private var mcpConfigText: String {
        let base = exampleBase
        let token = model.serving.bearerToken
        let isLAN = model.serving.scope == .public
        var out = """
        # Claude Code (one line)
        claude mcp add --transport http omni \(base)/mcp

        # .mcp.json / mcpServers config (Cursor, VS Code, claude_desktop_config.json, ...)
        {
          "mcpServers": {
            "omni": {
              "type": "http",
              "url": "\(base)/mcp"\(isLAN && !token.isEmpty ? ",\n      \"headers\": { \"Authorization\": \"Bearer \(token)\" }" : "")
            }
          }
        }
        """
        if !model.serving.isRunning {
            out += "\n\n# Note: the server is currently stopped - turn on \"Serve Omni over HTTP\" above."
        }
        return out
    }

    /// A complete SKILL.md an instruction-following agent can use to call the HTTP API.
    private var skillMarkdown: String {
        let base = exampleBase
        let token = model.serving.bearerToken
        let isLAN = model.serving.scope == .public
        let authNote = isLAN && !token.isEmpty
            ? "All requests need the header `Authorization: Bearer \(token)`."
            : "No auth needed from this Mac (loopback)."
        let authFlag = isLAN && !token.isEmpty ? " -H 'Authorization: Bearer \(token)'" : ""
        return """
        ---
        name: omni-local-search
        description: Semantic search over the user's own files - text, code, PDFs, images, audio and video - through the Omni app's local HTTP API. Use it when the user asks to find, locate or recall their own files by content: find my notes about X, that invoice from February, photos of the beach.
        ---

        # Omni - local semantic file search

        Omni indexes the user's files into one embedding space, so describe the CONTENT you want
        in natural language. Any language works and keywords are not required. Results are
        absolute file paths; read the files yourself if you need their contents.

        Base URL: \(base)
        \(authNote)

        ## Search

        ```bash
        curl -s \(base)/v1/search\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"query": "invoice from Anthropic in February", "top_k": 10}'
        ```

        Optional `filters`: `{"kinds": ["text"|"image"|"audio"|"video"|"scan"], "folder": "/abs/path",
        "folders": ["/abs/one", "/abs/two"], "since": <epoch seconds>}`.
        `"text"` includes scanned PDFs; `"scan"` is scanned PDFs only.
        Use `folders` to search two or more folders at once. Asking the user to add them as sources
        instead does not work, because an indexed parent folder already covers its children.

        Response: `{"results": [{"path", "score", "snippet", "kind", "modified", "locator",
        "chunk_count", ...}]}`.
        `score` runs 0 to 1. Compare it only within a kind: a text query scores a photo on a
        different scale than a document, so a 0.50 image and a 0.80 document are comparable matches.
        Hits below the app's relevance floor are dropped: 0.5 unless the user changed it in the
        window, scaled per kind so media is not deleted by a text-shaped floor. Pass `min_score` to
        set it per request in `filters` - `0` returns everything. `locator` is where the best
        match sits inside the file, such as `Page 3` or `Line 1240`, and is empty when the file has no meaningful position. `chunk_count` is how
        many pages or passages the file has in the index. Hits also carry `bytes` for the indexed
        file size and `mime_type`. Media hits add `width` and `height` in pixels and `duration` in
        seconds, recorded at index time, so you can prefer a 4032x3024 original over a 192px
        thumbnail without opening either.

        Copies of one file are already collapsed. A hit standing for several carries
        `duplicate_count`, the other `duplicates` paths, and a `duplicate_kind` of `exact` for
        byte-identical files or `near` for the same kind and extension within 10% in size and at
        cosine 0.98 or above. So `top_k` counts distinct files. Pass `"group_duplicates": false`
        for the flat list, or group differently yourself with `content_key`, which is
        identity-per-size.

        An image indexed with tagging on carries a few content words as its `snippet`, such as
        `cat, couch, crib`, in place of the filename. Read those as a list, or tag an untagged
        picture, through the tag calls below. Fields are omitted when unknown.

        ## File status

        ```bash
        curl -s \(base)/v1/files/status\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"paths": ["/abs/file1.pdf", "/abs/file2.png"]}'
        ```

        Response: `{"files": [{"path", "indexed", and when indexed: "exists", "kind", "chunk_count",
        "modified", "bytes", "up_to_date", "indexed_at"?}]}`. `up_to_date` compares the on-disk
        mtime and size with the indexed version; false means the file changed or was deleted after
        it was indexed, and `exists` tells you which. `indexed_at` is epoch seconds and is absent on
        files indexed by older app versions. Files only, not folders, up to 2048 paths. A path Omni
        does not hold returns just `{"path", "indexed": false}`.

        ## Image tags

        ```bash
        curl -s \(base)/v1/files/tags\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"paths": ["/abs/photo.jpg"]}'
        ```

        Response: `{"files": [{"path", "indexed", "kind", "taggable", "tags": [...]}]}`. These are
        the tags Omni generated at index time, the same words `tag:` matches in a search, so the
        call is instant and costs nothing. `taggable` is false for text and audio, which carry no
        tags. An empty `tags` on taggable media means it has not been tagged yet.

        To tag an image Omni has not indexed, or to re-tag a changed file, compute on demand:

        ```bash
        curl -s \(base)/v1/tag\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"path": "/abs/photo.jpg", "top_k": 5}'
        ```

        `path` must be inside the user's indexed folders; otherwise send the bytes as
        `{"image": "<base64 or data: URI>"}`. Multi-crop refinement is on by default so the result
        matches what the index would store; pass `"hq": false` for one forward pass per image
        instead of six. Nothing is written to the index. Max 4 images per request, or 16 with
        `hq: false`.

        ## Sources - what Omni indexes

        ```bash
        curl -s \(base)/v1/sources\(authFlag)
        ```

        Response: `{"indexing", "photos_authorized", "sources": [{"key", "kind", "name", "paused",
        "indexing", "queued", "indexed_files", "progress"?: {"done", "total"}}],
        "available_photo_albums": [{"id", "title", "count", "smart"}]}`. `kind` is `folder` or
        `photos`; `key` is the folder path or `photos://<id>`. A folder that is not a source is not
        indexed. `available_photo_albums` lists the albums not yet added, with the ids
        `{"album": ...}` takes.

        Add a folder, or the Apple Photos library whole or by album:

        ```bash
        curl -s \(base)/v1/sources/add\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"path": "/abs/folder"}'
        ```

        Send `{"album": "all"}` or an album id from the list instead of `path`; give one or the
        other, never both. Indexing starts immediately. `POST \(base)/v1/sources/pause` takes
        `{"key": "...", "paused": true|false}` and keeps what is already indexed;
        `POST \(base)/v1/sources/remove` takes `{"key": "..."}` and drops the source and its rows.
        Both keys come from the list above. Adding and especially removing change what the user
        sees in the app, so do them on request, not on your own initiative.

        ## OCR

        Transcribes a scanned PDF or an image to Markdown on this Mac: tables as HTML, formulas as
        LaTeX, headers and footers dropped. The model's instruction is fixed, so text parts in a
        request are ignored. A page takes seconds; the first call also loads the model.

        OpenAI chat shape, streamable:

        ```bash
        curl -sN \(base)/v1/chat/completions\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"model": "jina-ocr-v1", "stream": true, "messages": [{"role": "user", "content":
               [{"type": "image_url", "image_url": {"url": "file:///abs/scan.pdf"}}]}]}'
        ```

        An attachment is an `image_url` part or a `file` part with `file_data`. Its URL is a
        `file://` path inside the indexed folders, or a `data:` URI of an image or a PDF
        (`data:application/pdf;base64,...`). Remote URLs are refused. Every page of every
        attachment is transcribed, in order, joined by `\n\n---\n\n`; at most 200 pages, request
        bodies up to 48 MB. `finish_reason` is `length` when a page hit the token budget. The
        stream is standard `chat.completion.chunk` events ending in `data: [DONE]`;
        `stream_options.include_usage` adds a usage chunk.

        Page by page, Mistral OCR shape:

        ```bash
        curl -s \(base)/v1/ocr\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"document": {"type": "document_url", "document_url": "file:///abs/scan.pdf"}, "pages": "0-4"}'
        ```

        `document` is `{"type": "document_url", "document_url": ...}` or
        `{"type": "image_url", "image_url": ...}` with the same URL forms; `{"path": "/abs/scan.pdf"}`
        is shorthand. `pages` counts from 0: a list `[0, 2]` or a string `"0,2-4"`; default all.
        Response: `{"pages": [{"index", "markdown", "images": [], "dimensions": {"dpi", "height",
        "width"}}], "model", "usage_info": {"pages_processed", "doc_size_bytes"}}`. `dimensions` is
        null for a page answered by the transcript cache.

        Pages the app has transcribed before, in its OCR workspace or here, return from the cache
        at once. 503 with `Retry-After` means the app's OCR workspace is transcribing a document;
        503 without it means the OCR model is not installed (it downloads from OCR mode in the app).

        ## Health and model

        `GET \(base)/health` -> `{"status":"ok", ...}`. A refused connection means the server is
        off; ask the user to enable Settings -> Serving in the Omni app.
        `GET \(base)/v1/models` lists the embedding model, and the OCR model when it is installed.

        ## Embeddings

        L2-normalized vectors from the model behind the index, for your own similarity logic;
        searching Omni's index does not need them. Queries and documents are embedded differently:
        embed what you search WITH as a query and what you search IN as a document. A missing role
        means document.

        | Schema | Endpoint | Query | Document |
        |---|---|---|---|
        | OpenAI, Jina | `POST /v1/embeddings` | `"input_type": "query"` or `"task": "retrieval.query"` | `"input_type": "document"` or `"task": "retrieval.passage"` |
        | Cohere v1, v2 | `POST /v1/embed`, `POST /v2/embed` | `"input_type": "search_query"` | `"input_type": "search_document"` |
        | Gemini | `POST /v1beta/models/omni:embedContent`, `:batchEmbedContents` | `"taskType": "RETRIEVAL_QUERY"` | `"taskType": "RETRIEVAL_DOCUMENT"` |

        ```bash
        curl -s \(base)/v1/embeddings\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"model": "omni", "input": ["what to find"], "input_type": "query"}'
        curl -s \(base)/v1/embeddings\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"model": "omni", "input": ["text to be found", "another passage"], "input_type": "document"}'
        ```

        An unknown role is a 400. A Gemini batch takes each request's own `taskType`. Gemini
        authenticates with `x-goog-api-key` rather than a bearer header.
        """
    }

    /// Base URL for examples: the live bound address, or the configured local address when stopped.
    private var exampleBase: String {
        model.serving.boundAddress.isEmpty ? "http://127.0.0.1:\(model.serving.port)" : model.serving.boundAddress
    }

    /// A ready-to-run curl for the selected API, including the auth header when a token is set.
    private var exampleCurl: String {
        let base = exampleBase
        let token = model.serving.bearerToken
        let auth = token.isEmpty ? "" : " -H 'Authorization: Bearer \(token)'"
        let geminiAuth = token.isEmpty ? "" : " -H 'x-goog-api-key: \(token)'"
        let ct = " -H 'Content-Type: application/json'"
        if exampleKind == .search {
            return "curl \(base)/v1/search\(ct)\(auth) -d '{\"query\":\"invoices\",\"top_k\":5}'"
        }
        if exampleKind == .tags {
            return "curl \(base)/v1/files/tags\(ct)\(auth) -d '{\"path\":\"/path/to/photo.jpg\"}'"
        }
        if exampleKind == .ocr {
            return "curl -N \(base)/v1/chat/completions\(ct)\(auth) -d '{\"model\":\"jina-ocr-v1\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file:///path/to/scan.pdf\"}}]}]}'"
        }
        // QUERY AND DOCUMENT, both, in every schema. The two roles embed differently, and a single
        // example taught one of them: the OpenAI line had no role at all, which is the document
        // side, so a reader copying it for their queries got document vectors without being told.
        let pair: (query: String, document: String)
        switch embedSchema {
        case .openai:
            pair = ("curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"what to find\"],\"input_type\":\"query\"}'",
                    "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"text to be found\"],\"input_type\":\"document\"}'")
        case .jina:
            pair = ("curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"what to find\"],\"task\":\"retrieval.query\"}'",
                    "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"text to be found\"],\"task\":\"retrieval.passage\"}'")
        case .cohere:
            pair = ("curl \(base)/v2/embed\(ct)\(auth) -d '{\"model\":\"omni\",\"texts\":[\"what to find\"],\"input_type\":\"search_query\",\"embedding_types\":[\"float\"]}'",
                    "curl \(base)/v2/embed\(ct)\(auth) -d '{\"model\":\"omni\",\"texts\":[\"text to be found\"],\"input_type\":\"search_document\",\"embedding_types\":[\"float\"]}'")
        case .gemini:
            pair = ("curl \(base)/v1beta/models/omni:embedContent\(ct)\(geminiAuth) -d '{\"content\":{\"parts\":[{\"text\":\"what to find\"}]},\"taskType\":\"RETRIEVAL_QUERY\"}'",
                    "curl \(base)/v1beta/models/omni:embedContent\(ct)\(geminiAuth) -d '{\"content\":{\"parts\":[{\"text\":\"text to be found\"}]},\"taskType\":\"RETRIEVAL_DOCUMENT\"}'")
        }
        return "# query\n\(pair.query)\n\n# document\n\(pair.document)"
    }

    // MARK: - Requests

    @ViewBuilder private var logsSection: some View {
        Section("Logs") {
            LogTextView()
                .frame(height: 200)
            LabeledContent("Log file") {
                Text(ServingLogFile.url.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }
}

/// `tail -n 100` of serving.log as plain read-only text: selectable and copyable, colored by level.
/// The file is the only log; this re-reads its tail when it changes, so what is shown is what is
/// on disk. Follows the end while the reader is at the end, and leaves the scroll position alone
/// once they scroll up to read something.
private struct LogTextView: NSViewRepresentable {
    static let lineCount = 100

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.drawsBackground = false
        scroll.autohidesScrollers = true
        let text = scroll.documentView as! NSTextView
        text.isEditable = false
        text.isSelectable = true
        text.isRichText = false
        text.drawsBackground = false
        text.textContainerInset = NSSize(width: 0, height: 4)
        // No wrapping: a log line reads as one row, and its payload scrolls sideways.
        text.isHorizontallyResizable = true
        text.textContainer?.widthTracksTextView = false
        text.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                                   height: CGFloat.greatestFiniteMagnitude)
        scroll.hasHorizontalScroller = true
        context.coordinator.start(scroll)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {}

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) { coordinator.stop() }

    func makeCoordinator() -> Coordinator { Coordinator() }

    /// Polls the file's size and date twice a second, which is a stat call: it survives the file
    /// being created, rotated or deleted without re-arming a file watch for each case.
    @MainActor final class Coordinator {
        private var timer: Timer?
        private var seen: (size: Int, date: Date)?
        private weak var scroll: NSScrollView?

        func start(_ scroll: NSScrollView) {
            self.scroll = scroll
            refresh()
            let t = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.refresh() }
            }
            RunLoop.main.add(t, forMode: .common)
            timer = t
        }

        func stop() { timer?.invalidate(); timer = nil }

        private func refresh() {
            let attrs = try? FileManager.default.attributesOfItem(atPath: ServingLogFile.url.path)
            let now = (size: attrs?[.size] as? Int ?? 0, date: attrs?[.modificationDate] as? Date ?? .distantPast)
            if let seen, seen == now { return }
            let first = seen == nil
            seen = now
            guard let scroll, let text = scroll.documentView as? NSTextView else { return }
            let atEnd = first || scroll.documentVisibleRect.maxY >= text.bounds.maxY - 4
            text.textStorage?.setAttributedString(LogTextView.render(ServingLogFile.tail(LogTextView.lineCount)))
            if atEnd { text.scrollToEndOfDocument(nil) }
        }
    }

    static func render(_ lines: [String]) -> NSAttributedString {
        let font = NSFont.monospacedSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
        let out = NSMutableAttributedString()
        for (i, line) in lines.enumerated() {
            let color: NSColor
            switch ServingLogFile.level(of: Substring(line)) {
            case .error: color = .systemRed
            case .warn: color = .systemOrange
            default: color = .labelColor
            }
            let row = NSMutableAttributedString(string: i == lines.count - 1 ? line : line + "\n",
                                                attributes: [.font: font, .foregroundColor: color])
            // The timestamp recedes on every line; the level color carries the rest.
            if color == .labelColor {
                row.addAttribute(.foregroundColor, value: NSColor.secondaryLabelColor,
                                 range: NSRange(location: 0, length: min(23, (line as NSString).length)))
            }
            out.append(row)
        }
        return out
    }
}

/// Modal sheet showing a generated agent-config blob: selectable monospaced text with Copy
/// (and optionally Save as a file). Used by the MCP and SKILL.md buttons in the Serving tab.
private struct AgentConfigSheet: View {
    @Environment(\.dismiss) private var dismiss
    let title: String
    let subtitle: String
    let text: String
    /// Suggested filename to enable the Save button (nil = copy-only).
    let saveAs: String?
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)
            Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                Text(text)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.primary.opacity(0.1)))
            HStack {
                Button(copied ? "Copied" : "Copy") {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(text, forType: .string)
                    copied = true
                    Task { try? await Task.sleep(for: .seconds(1.5)); copied = false }
                }
                if let saveAs {
                    Button("Save\u{2026}") {
                        let panel = NSSavePanel()
                        panel.nameFieldStringValue = saveAs
                        if panel.runModal() == .OK, let url = panel.url {
                            try? text.write(to: url, atomically: true, encoding: .utf8)
                        }
                    }
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 640, height: 460)
        .onExitCommand { dismiss() }   // Esc closes, matching the native sheet expectation
    }
}
