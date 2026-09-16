import SwiftUI
import AppKit
import OmniKit

/// Settings > Serving. Binds only to the shared ServingController surface AppModel owns:
/// read-write enabled/scope/port/bearerToken, read-only state/boundAddress/counters/log, and the
/// clearLog() action. Follows the same Form { Section }.formStyle(.grouped) idiom and the explicit
/// Binding(get:set:) pattern as the other tabs - never $model, since AppModel is @Observable.
struct ServingTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var exampleKind: ExampleKind = .search
    @State private var embedSchema: EmbedSchema = .openai
    @State private var showMCPSheet = false
    @State private var showSkillSheet = false

    /// Top-level example category: the search endpoint, or an embedding endpoint.
    private enum ExampleKind: String, CaseIterable, Identifiable {
        case search = "Search", embed = "Embed", tags = "Tags"
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
            requestsSection
        }
        .formStyle(.grouped)
        // NO FIXED HEIGHT. The Settings TabView sizes itself to the selected tab
        // (`fixedSize(vertical:)`), so a pinned height here did not make the window steady - it
        // made this one tab shorter than its own content and put a scroller inside it, which no
        // other tab has.
        .sheet(isPresented: $showMCPSheet) {
            AgentConfigSheet(title: "Connect agents over MCP",
                             subtitle: "For any MCP client with HTTP transport. Eight tools: search, search_inline, file_status, tag_image, list_sources, add_source, pause_source, remove_source.",
                             text: mcpConfigText, saveAs: nil)
        }
        .sheet(isPresented: $showSkillSheet) {
            AgentConfigSheet(title: "SKILL.md for instruction-following agents",
                             subtitle: "Save where your agent reads skills (e.g. ~/.claude/skills/omni-search/SKILL.md), or paste into its instructions.",
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
                .help("Replace the token with a new one")
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
                    .help("Connection config for MCP clients such as Claude Code, Cursor or VS Code")
                Button("SKILL.md") { showSkillSheet = true }
                    .help("A ready skill file for instruction-following agents")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .disabled(!model.serving.enabled)
        } header: {
            Text("Server")
        } footer: {
            Text("A local HTTP API for search, tags, embeddings, and the folders Omni indexes. Local network needs a token; changes restart the server.")
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
        } footer: {
            Text("Run this in a terminal with the server on.")
                .font(.caption).foregroundStyle(.secondary)
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
        "chunk_count", ...}]}` over HTTP. Over MCP the same facts arrive as one text line per hit,
        `N. /path  (kind, score%, locator, N passages, yyyy-MM-dd)`, followed by the snippet.
        `score` runs 0 to 1. Compare it only within a kind: a text query scores a photo on a
        different scale than a document, so a 0.50 image and a 0.80 document are comparable matches.
        Pass `"min_score"` in `filters` to drop weak hits; the default keeps everything.
        A search that matches nothing well returns NO hits and says so instead: semantic search
        always has a top result, so Omni compares the best score against the most confusable files
        in the index and withholds the page when it is not clearly better. Retry with
        `"include_weak": true` to see the nearest files anyway. `locator` is where the best match sits inside the file, such as `Page 3` or
        `Line 1240`, and is empty when the file has no meaningful position. `chunk_count` is how
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

        Response: `{"sources": [{"key", "title", "path", "kind", "indexed", "paused", "indexing"}]}`.
        `kind` is `folder` or `photos`. Call this before concluding a file is not on the Mac: a
        folder that is not a source is simply not indexed.

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

        ## Health and model

        `GET \(base)/health` -> `{"status":"ok", ...}`. A refused connection means the server is
        off; ask the user to enable Settings -> Serving in the Omni app.
        `GET \(base)/v1/models` lists the loaded model.

        ## Embeddings

        Four request schemas, all returning L2-normalized vectors over the same model. Use them to
        build your own similarity logic; searching Omni's index does not need them.

        - `POST \(base)/v1/embeddings` - OpenAI and Jina bodies, `{"model":"omni","input":[...]}`,
          with an optional Jina `task` such as `retrieval.query`.
        - `POST \(base)/v1/embed` and `POST \(base)/v2/embed` - Cohere v1 and v2 bodies.
        - `POST \(base)/v1beta/models/omni:embedContent` and `:batchEmbedContents` - Gemini bodies,
          authenticated with `x-goog-api-key` rather than a bearer header.

        ## MCP

        The server also speaks MCP over streamable HTTP at `\(base)/mcp`. Point any MCP client at
        that URL. Eight tools, each one the call of the same name above:
        `search`, `search_inline`, `file_status`, `tag_image`, `list_sources`, `add_source`,
        `pause_source`, `remove_source`.

        Two differ from their HTTP form. `search` takes `include_images`, which attaches an inline
        JPEG thumbnail to image and scanned-PDF hits so they render in the client, and returns one
        text line per hit rather than JSON rows: `N. /path  (kind, score%, locator, N passages,
        yyyy-MM-dd)` followed by the snippet. Open a result by its path. `search_inline` ranks the
        best passages within an explicit set of files or folders, taking `query` and `paths` plus
        `top_k` and `max_snippet`; only the query is embedded, so use it to pinpoint where a topic
        is discussed across documents you already know.
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
        switch embedSchema {
        case .openai:
            return "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"your text\"]}'"
        case .jina:
            return "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":\"your text\",\"task\":\"retrieval.query\"}'"
        case .cohere:
            return "curl \(base)/v2/embed\(ct)\(auth) -d '{\"model\":\"omni\",\"texts\":[\"your text\"],\"input_type\":\"search_document\",\"embedding_types\":[\"float\"]}'"
        case .gemini:
            return "curl \(base)/v1beta/models/omni:embedContent\(ct)\(geminiAuth) -d '{\"content\":{\"parts\":[{\"text\":\"your text\"}]}}'"
        }
    }

    // MARK: - Requests

    @ViewBuilder private var requestsSection: some View {
        Section {
            if model.serving.log.isEmpty {
                Text("No requests yet")
                    .foregroundStyle(.secondary)
            } else {
                List(model.serving.log) { entry in
                    LogRow(entry: entry)
                        .listRowSeparator(.hidden)
                }
                .listStyle(.plain)
                .frame(height: 200)
            }
        } header: {
            HStack {
                Text("Requests")
                Spacer()
                Text("\(model.serving.requestCount) served \u{00B7} \(model.serving.errorCount) failed")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Button("Clear") { model.serving.clearLog() }
                    .buttonStyle(.link)
                    .disabled(model.serving.log.isEmpty)
            }
        } footer: {
            Text("Recent requests, newest first.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// One row in the live request log: time, method, path, status, and latency. Kept fixed-height and
/// monospaced so the columns line up and the List inside the grouped Form scrolls cleanly.
private struct LogRow: View {
    let entry: LogEntry

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    private var statusColor: Color {
        switch entry.status {
        case 200 ..< 300: return .green
        case 400 ..< 500: return .orange
        default: return .red
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(Self.timeFormatter.string(from: entry.time))
                .foregroundStyle(.secondary)
            Text(entry.method)
                .frame(width: 44, alignment: .leading)
            Text(entry.path)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("\(entry.status)")
                .foregroundStyle(statusColor)
            Text(String(format: "%.0f ms", entry.ms))
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .trailing)
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.vertical, 1)
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
