import SwiftUI
import AppKit
import OmniKit

/// Settings > Serving. Binds only to the shared ServingController surface AppModel owns:
/// read-write enabled/scope/port/bearerToken, read-only state/boundAddress/counters/log, and the
/// clearLog() action. Follows the same Form { Section }.formStyle(.grouped) idiom and the explicit
/// Binding(get:set:) pattern as the other tabs - never $model, since AppModel is @Observable.
struct ServingTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var revealToken = false
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
            accessSection
            exampleSection
            requestsSection
        }
        .formStyle(.grouped)
        .frame(height: 520)   // matches the Content tab so switching tall tabs doesn't jump
        .sheet(isPresented: $showMCPSheet) {
            AgentConfigSheet(title: "Connect agents over MCP",
                             subtitle: "Works with every MCP client that speaks the HTTP transport (Claude Code, Cursor, VS Code, ...). Tools: search, search_inline, file_status, tag_image.",
                             text: mcpConfigText, saveAs: nil)
        }
        .sheet(isPresented: $showSkillSheet) {
            AgentConfigSheet(title: "SKILL.md for instruction-following agents",
                             subtitle: "Drop this file where your agent reads skills (e.g. ~/.claude/skills/omni-search/SKILL.md) or paste it into its instructions.",
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
                    .help("Connection config for MCP clients (Claude Code, Cursor, VS Code)")
                Button("SKILL.md") { showSkillSheet = true }
                    .help("A ready skill file for instruction-following agents")
            }
            .buttonStyle(.bordered).controlSize(.small)
            .disabled(!model.serving.enabled)
        } header: {
            Text("Server")
        } footer: {
            Text("Serves OpenAI, Jina, Cohere, and Gemini embedding endpoints plus search, per-file index status, and image tags, backed by the local model. MCP hands the server to protocol clients; SKILL.md to instruction-following agents.")
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

    // MARK: - Access

    @ViewBuilder private var accessSection: some View {
        Section {
            Picker("Reachable from", selection: Binding(
                get: { model.serving.scope },
                set: { model.serving.scope = $0 }
            )) {
                Text("This Mac only").tag(ServingScope.local)
                Text("Local network").tag(ServingScope.public)
            }

            TextField("Port", value: Binding(
                get: { model.serving.port },
                set: { model.serving.port = min(65535, max(1, $0)) }   // valid TCP port range
            ), format: .number.grouping(.never))
            .frame(width: 90)

            HStack(spacing: 6) {
                let token = Binding(
                    get: { model.serving.bearerToken },
                    set: { model.serving.bearerToken = $0 }
                )
                Group {
                    if revealToken {
                        TextField("Bearer token", text: token).font(.callout.monospaced())
                    } else {
                        SecureField("Bearer token", text: token)
                    }
                }
                Button { revealToken.toggle() } label: {
                    Image(systemName: revealToken ? "eye.slash" : "eye")
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help(revealToken ? "Hide" : "Show")
                .disabled(model.serving.bearerToken.isEmpty)
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(model.serving.bearerToken, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy").disabled(model.serving.bearerToken.isEmpty)
                Button("Generate new") {
                    model.serving.bearerToken = ServingController.generateToken()
                    revealToken = true   // show it so it can be copied
                }
                .buttonStyle(.link)
            }
        } header: {
            Text("Access")
        } footer: {
            Text("Local network always requires a token (one is generated if empty). Token, port, or scope changes restart the server.")
                .font(.caption).foregroundStyle(.secondary)
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
            Text("Start the server, then run the selected example in a terminal.")
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
        description: Search the user's local files by MEANING (semantic search over text, code, PDFs, images, audio, and video) via the Omni app's local HTTP API. Use when the user asks to find, locate, or recall their own files by content ("find my notes about X", "that invoice from February", "photos of the beach").
        ---

        # Omni - local semantic file search

        Omni indexes the user's files into one embedding space, so describe the CONTENT you want
        in natural language (any language); keywords are not required. Results are absolute file
        paths - read the files yourself if you need their contents.

        Base URL: \(base)
        \(authNote)

        ## Search (the main call)

        ```bash
        curl -s \(base)/v1/search\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"query": "invoice from Anthropic in February", "top_k": 10}'
        ```

        Optional `filters`: `{"kinds": ["text"|"image"|"audio"|"video"|"scan"], "folder": "/abs/path", "since": <epoch seconds>}`.
        `"text"` includes scanned PDFs; `"scan"` is scanned PDFs only.
        Response: `{"results": [{"path", "score" (0..1), "snippet", "kind", "modified", "locator", "chunk_count", ...}]}`.
        `locator` is where the best match sits inside the file ("Page 3", "Line 1240"; "" if n/a);
        `chunk_count` is how many chunks (pages/passages) the file has in the index. Hits also
        carry `bytes` (indexed file size) and `mime_type`, and media hits add `width`/`height`
        (px) and `duration` (seconds) recorded at index time - so you can prefer, say, the
        4032x3024 original over a 192px thumbnail without opening either. Hits that are
        byte-identical copies share a `content_key`: collapse them before choosing. Image hits
        indexed with tagging on carry a few content words as their `snippet` ("cat, couch,
        crib") instead of the filename - see Image tags below to read those as a list, or to
        tag a picture that is not indexed. Fields are omitted when unknown. Scores above ~0.45
        are usually relevant; below ~0.3 usually noise.

        ## File status (is this file indexed, and is the index fresh?)

        ```bash
        curl -s \(base)/v1/files/status\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"paths": ["/abs/file1.pdf", "/abs/file2.png"]}'
        ```

        Response: `{"files": [{"path", "indexed", and when indexed: "exists", "kind", "chunk_count",
        "modified", "bytes", "up_to_date", "indexed_at"?}]}`. `up_to_date` compares the on-disk
        (mtime, size) with the indexed version - false means the file on disk changed (or was
        deleted, see `exists`) after it was indexed. `indexed_at` (epoch seconds) is when the
        indexer last wrote the file; absent on files indexed by older app versions. Files only
        (not folders); up to 2048 paths. Non-indexed paths return just `{"path", "indexed": false}`.

        ## Image tags (what is in this picture?)

        ```bash
        curl -s \(base)/v1/files/tags\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"paths": ["/abs/photo.jpg"]}'
        ```

        Response: `{"files": [{"path", "indexed", "kind", "taggable", "tags": [...]}]}`. These are
        the tags Omni generated at index time - the same words `tag:` matches in a search - so the
        call is instant and costs nothing. `taggable` is false for text and audio, which carry no
        tags. An empty `tags` on taggable media means it has not been tagged yet.

        To tag an image Omni has NOT indexed, or to re-tag a changed file, compute on demand:

        ```bash
        curl -s \(base)/v1/tag\(authFlag) -H 'Content-Type: application/json' \\
          -d '{"path": "/abs/photo.jpg", "top_k": 5}'
        ```

        `path` must be inside the user's indexed folders; otherwise send the bytes as
        `{"image": "<base64 or data: URI>"}`. Multi-crop refinement is on by default so the result
        matches what the index would store - pass `"hq": false` for one forward per image instead
        of six. Nothing is written to the index. Max 4 images per request (16 with `hq: false`).

        ## Health check

        `curl -s \(base)/health` -> `{"status":"ok", ...}`. If the connection is refused, the
        server is off - ask the user to enable Settings -> Serving in the Omni app.

        ## Embeddings (optional)

        `POST \(base)/v1/embeddings` accepts OpenAI/Jina-style bodies (`{"model":"omni","input":[...]}`)
        and returns L2-normalized vectors - useful for building your own similarity logic.

        ## MCP

        The server also speaks MCP (streamable HTTP) at `\(base)/mcp` - point any MCP client at
        that URL. Three tools:

        - `search` - the same semantic search as above. Args: `query` (required), `top_k` (default
          10, max 50), `kinds`, `folder`, `max_snippet` (snippet chars per result, default 200), and
          `include_images` (when true, image and scanned-PDF hits carry an inline JPEG thumbnail so
          they render in the client). Each result also returns a `resource_link` - a `file://` URI
          the client can open or preview - and media hits carry resolution/duration/size.
        - `search_inline` - rank the best passages WITHIN a specific set of files or folders. Args:
          `query` and `paths` (absolute file or folder paths), plus `top_k` and `max_snippet`. Reuses
          the index (only the query is embedded), so it is fast - use it to pinpoint where something
          is discussed across documents you already know.
        - `file_status` - per-file index coverage. Args: `paths` (absolute file paths, up to 2048).
          Reports indexed, kind, chunk count, the indexed version's (modified, bytes), exists, and
          `up_to_date` - whether the on-disk file still matches the index. Same data as
          `/v1/files/status` above.
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
