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
    @State private var cliState = CommandLineTool.state
    @State private var cliError = ""
    @State private var logHasLines = false
    /// The token as typed. Committed on Return or when the field loses focus: every keystroke used
    /// to restart a running server, and on the LAN scope clearing the field to paste a new token
    /// put a random one straight back.
    @State private var tokenDraft: String?
    @FocusState private var tokenFocused: Bool

    private func commitToken() {
        guard let d = tokenDraft else { return }
        tokenDraft = nil
        if d != model.serving.bearerToken { model.serving.bearerToken = d }
    }

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
            commandLineSection
            serverSection
            exampleSection
            logsSection
        }
        .formStyle(.grouped)
        .task {
            // One stat a second while the tab is open, for `logsSection`; the text view polls its own.
            while !Task.isCancelled {
                let size = (try? FileManager.default.attributesOfItem(atPath: ServingLogFile.url.path)[.size] as? Int) ?? 0
                if (size > 0) != logHasLines { logHasLines = size > 0 }
                try? await Task.sleep(for: .seconds(1))
            }
        }
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
                    get: { tokenDraft ?? model.serving.bearerToken },
                    set: { tokenDraft = $0 }
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
                    .focused($tokenFocused)
                    .onSubmit { commitToken() }
                    .onChange(of: tokenFocused) { _, focused in if !focused { commitToken() } }
                    .onDisappear { commitToken() }
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: .infinity)
                Button {
                    OmniPasteboard.copy(model.serving.bearerToken, concealed: true)
                } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy").disabled(model.serving.bearerToken.isEmpty)
                Button {
                    tokenDraft = nil
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

    // MARK: - Command line

    /// The `omni` command and its skill file. Not tied to the server above: the command reaches
    /// the app over a private socket that is up whenever Omni runs.
    private var commandLineSection: some View {
        Section {
            HStack(spacing: 8) {
                Text(cliState == .installed ? "omni" : CommandLineTool.bundledPath)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Button(cliState == .installed ? "Installed" : "Install\u{2026}") {
                    cliError = CommandLineTool.install() ?? ""
                    cliState = CommandLineTool.state
                }
                .disabled(cliState == .installed)
                .help("Link omni into \((CommandLineTool.linkPath as NSString).deletingLastPathComponent)")
                Button("SKILL.md") { showSkillSheet = true }
                    .help("Skill file for agents")
            }
            .buttonStyle(.bordered).controlSize(.small)
        } header: {
            Text("Command Line")
        } footer: {
            Group {
                if !cliError.isEmpty {
                    Text(cliError).foregroundStyle(.red)
                } else if case .stale(let other) = cliState {
                    Text("\(CommandLineTool.linkPath) points to \(other). Install again to use this copy of Omni.")
                } else {
                    Text("Agents run `omni search \"...\"` against this app, with or without the server on.")
                }
            }
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var statusColor: Color {
        switch model.serving.state {
        case .running: return .green
        case .portInUse, .failed: return .orange
        case .stopped, .starting: return .secondary
        }
    }

    private var statusText: String {
        switch model.serving.state {
        case .running: return "Running"
        case .starting: return "Starting"
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
                    OmniPasteboard.copy(exampleCurl)
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

    private var skillMarkdown: String { CommandLineTool.skillMarkdown }

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

    /// Absent until the log has a line: an empty text box and a path to an empty file say nothing.
    @ViewBuilder private var logsSection: some View {
        if logHasLines {
            Section("Logs") {
                LogTextView()
                    .frame(height: 200)
                // The OCR cache's Location row, exactly: path on the label line, buttons below.
                VStack(spacing: 6) {
                    HStack(spacing: 8) {
                        Text("Log file")
                        Spacer()
                        Text((ServingLogFile.url.path as NSString).abbreviatingWithTildeInPath)
                            .foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                            .help(ServingLogFile.url.path)
                    }
                    HStack(spacing: 8) {
                        Spacer()
                        Button("Copy Path") {
                            OmniPasteboard.copy(ServingLogFile.url.path)
                        }
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([ServingLogFile.url])
                        }
                    }
                    .controlSize(.small)
                }
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
                    OmniPasteboard.copy(text)
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
