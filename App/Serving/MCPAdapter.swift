import Foundation
import OmniKit
import UniformTypeIdentifiers
import ImageIO
import AppKit

/// MCP (Model Context Protocol) endpoint: JSON-RPC 2.0 over streamable HTTP at POST /mcp.
/// Stateless by design - every request is self-contained, no session ids, and responses are
/// single JSON bodies (the spec permits a plain `application/json` reply for servers that do
/// not stream). That makes the endpoint work with every MCP client that speaks the HTTP
/// transport - `claude mcp add --transport http`, Cursor, VS Code, etc. - with zero setup
/// beyond the URL.
///
/// Tools: `search` (whole-index), `search_inline` (rank passages within given paths), `read_file`
/// (the text Omni read from an indexed file, by page, line or offset), `file_status` (index
/// coverage/freshness per file), `tag_image`, `ocr` (transcribe a scan to Markdown), and the four
/// source tools when the app has wired them. Agents that want raw
/// vectors use the OpenAI-style /v1/embeddings endpoint instead; MCP is for the things agents
/// actually do with Omni - reach the user's files by meaning and trust what the index covers.
enum MCPAdapter {
    /// The revisions this server answers, newest first. Its replies are valid under each: tools
    /// with annotations and titles, no sessions, no batching, nothing a later revision removed.
    /// A client asking for one of these gets it back; any other request gets the newest, which the
    /// spec says the client then accepts or disconnects on. It used to echo whatever was asked -
    /// "2099-01-01" and "bogus" included - so a client believed it had negotiated anything.
    static let supportedVersions = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
    /// MODERN revisions: no handshake, no session - every request names its revision and the
    /// client's capabilities in `_meta`, and the server answers each on its own (2026-07-28). A
    /// request that carries that metadata is served this way; one that opens with `initialize` is a
    /// legacy client and is served as before. Both on one endpoint, as the spec allows a dual-era
    /// server to.
    static let modernVersions = ["2026-07-28"]
    static let versionMetaKey = "io.modelcontextprotocol/protocolVersion"
    static let serverInfoMetaKey = "io.modelcontextprotocol/serverInfo"

    /// The passage width an agent gets by default. 500 was A/B'd against it with real agents: no
    /// fewer calls, no more answers, more input tokens on 81 of 125 queries (index.md, "Passages").
    static let defaultSnippet = 200

    /// Longest edge of an inlined thumbnail (matches jina-ai/MCP's image handling).
    private static let inlineImageMaxEdge = 800
    /// Hard cap on inlined thumbnails per response, so an all-images search can't flood the
    /// client's context with base64. Image hits past this are still returned as links/text.
    private static let maxInlineImages = 10

    /// yyyy-MM-dd in the local calendar. The only date format that is unambiguous to a reader who
    /// does not know the server's locale, and the only precision a relevance question ever needs.
    static func shortDate(_ epoch: Double) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: Date(timeIntervalSince1970: epoch))
    }

    /// What `initialize` tells a client about the whole server. Also the body of SKILL.md
    /// (AgentSkill), so it is written for an agent reading it cold.
    static let instructions = "Search the user's local files by meaning. Files of every kind - text, code, PDFs, images, audio, video - share one embedding space, so describe the CONTENT you want in natural language (any language). `search` finds files across the whole index; `search_inline` ranks the best passages WITHIN a specific set of files or folders you already know; `read_file` reads an indexed file's text by page, line or offset, including formats you cannot open; `file_status` reports whether given files are indexed and whether the index is still fresh for them. `ocr` transcribes a scanned PDF or an image to Markdown, page by page. Results are file paths with scores, the passage that matched and media metadata; a search ends with how many more there are (`offset` pages) and where the top matches are.\n\nOmni only finds what it has indexed. If a search comes back empty for something the user says is on their Mac, check `list_sources` before concluding the file is not there - the folder may simply not be a source yet. Linked Macs (other Macs running Omni that this one is linked with) are searched too, and their results are listed under each Mac's name: get_linked_file copies one here to read. `add_source` adds a folder (or the Apple Photos library, whole or by album) and indexing starts immediately; `pause_source` stops work on one without losing what it already indexed; `remove_source` drops it and its rows. These change what the user sees in the app, so treat add and especially remove as actions to take on request rather than on your own initiative."

    /// Every tool, as `tools/list` returns them - and as the `omni` command line and SKILL.md
    /// present them. The source tools are only advertised once the app has wired the control:
    /// a tool a client can see but not call is worse than one that is absent.
    static func tools(withSources: Bool) -> [[String: Any]] {
        var tools = [searchToolDescriptor(), searchInlineToolDescriptor(), readFileToolDescriptor(),
                     fileStatusToolDescriptor(), tagImageToolDescriptor(), ocrToolDescriptor()]
        if withSources {
            tools += [listSourcesDescriptor(), addSourceDescriptor(),
                      pauseSourceDescriptor(), removeSourceDescriptor(), linkedFileDescriptor()]
        }
        return tools.map { tool in
            var t = tool
            // Closed schemas: an argument a tool does not take is refused (tools/call), and saying
            // so here lets a client catch the typo before it calls.
            if var schema = t["inputSchema"] as? [String: Any] {
                schema["additionalProperties"] = false
                t["inputSchema"] = schema
            }
            guard let name = tool["name"] as? String, let ex = examples[name] else { return t }
            t["_meta"] = [examplesMetaKey: ex]
            return t
        }
    }

    /// Example arguments per tool, in the tool's `_meta`. Arguments rather than command lines, so
    /// the one place they live stays true to the schema: `omni <tool> --help` renders them as
    /// commands, and an MCP client may show them as they are.
    static let examplesMetaKey = CLIProtocol.examplesKey
    nonisolated(unsafe) private static let examples: [String: [[String: Any]]] = [   // immutable
        "search": [
            ["query": "invoice from anthropic in february"],
            ["query": "photos of a red sports car", "kinds": ["image"], "top_k": 5],
            ["query": "meeting notes about the launch", "modified_after": "2026-09-01", "ext": "md"],
            ["query": "tax return 2025", "macs": []],
        ],
        "search_inline": [["query": "termination clause", "paths": ["/Users/me/Documents/contract.pdf"]]],
        "read_file": [["path": "/Users/me/Documents/contract.pdf", "page": 7], ["path": "/Users/me/Projects/app/server.log", "line": 1200],
                      ["path": "/Users/me/Documents/proposal.docx", "offset": 4000]],
        "file_status": [["paths": ["/Users/me/Documents/report.pdf", "/Users/me/Desktop/notes.md"]]],
        "tag_image": [["paths": ["/Users/me/Pictures/IMG_0001.jpg"]]],
        "ocr": [["path": "/Users/me/Downloads/scan.pdf"], ["path": "/Users/me/Downloads/scan.pdf", "pages": "11-20"]],
        "list_sources": [[:]],
        "get_linked_file": [["mac": "Studio", "path": "/Users/me/Documents/tax-2025.pdf"]],
        "add_source": [["path": "/Users/me/Projects"], ["album": "all"]],
        "pause_source": [["key": "/Users/me/Downloads"], ["key": "/Users/me/Downloads", "paused": false]],
        "remove_source": [["key": "/Users/me/Old Projects"]],
    ]

    /// How deep the pool a page is cut from goes, at least: deep enough to say where the matches sit.
    static let summaryPool = 30

    /// WHAT ELSE THERE IS, in one line under the results: how many more ranked results a next page
    /// holds, and where the top matches sit - the folders under each source and the kinds. A list of
    /// ten says nothing about the eleventh; measured on agents over a large corpus, exact counts beat
    /// "there are more" (IndexAct, arXiv 2610.07960: 74.0 vs 62.0), and the folders are the corpus's
    /// own structure, which the agent can pass back as `folder` to narrow the next search.
    static func poolSummary(_ pool: [ResultGroup], offset: Int, shown: Int, pageSize: Int, roots: [String]) -> String? {
        guard pool.count > shown else { return nil }
        let end = offset + shown
        var parts: [String] = []
        if end < pool.count { parts.append("More: offset=\(end) (\(pool.count - end) more).") }
        // A source's own top-level folders, named from the source's folder name ("Documents/Work"):
        // specific enough to scope by - the result paths above give the full spelling - and short
        // enough that the line costs a few tokens, not a few hundred.
        func label(_ path: String) -> String {
            if PhotoLibrary.isPhotoPath(path) { return "Photos" }
            let dir = (path as NSString).deletingLastPathComponent
            guard let root = roots.filter({ dir == $0 || dir.hasPrefix($0 + "/") }).max(by: { $0.count < $1.count }) else {
                return (dir as NSString).lastPathComponent
            }
            let name = (root as NSString).lastPathComponent
            return dir.dropFirst(root.count).split(separator: "/").first.map { name + "/" + $0 } ?? name
        }
        var byFolder: [String: Int] = [:], byKind: [String: Int] = [:]
        for g in pool {
            byFolder[label(g.representative.path), default: 0] += 1
            byKind[g.representative.fileKind, default: 0] += 1
        }
        let ranked = byFolder.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        var folders = ranked.prefix(4).map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        if ranked.count > 4 { folders += ", +\(ranked.count - 4)" }
        let kinds = byKind.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        parts.append("Top \(pool.count): \(folders); \(kinds).")
        return parts.joined(separator: " ")
    }

    /// A passage on one line, the way the stored snippet is: line breaks and tabs become one space.
    static func oneLine(_ s: String) -> String {
        s.split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
    }

    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend, appVersion: String,
                       sources: SourcesControl? = nil, surface: ServedSurface = .mcp) async -> HTTPResponse {
        // GET is the SSE stream in the full spec; this server has no server-initiated
        // messages, and the spec allows refusing it.
        guard req.method == "POST" else {
            return HTTPResponse.json(jsonRPCError(id: NSNull(), code: -32600,
                                                  message: "use POST; this server does not offer an SSE stream"),
                                     status: 405)
        }
        guard let obj = try? JSONSerialization.jsonObject(with: req.body) else {
            return HTTPResponse.json(jsonRPCError(id: NSNull(), code: -32700, message: "parse error"), status: 400)
        }
        // JSON-RPC batching was removed in 2025-06-18; answer arrays explicitly.
        guard let msg = obj as? [String: Any] else {
            return HTTPResponse.json(jsonRPCError(id: NSNull(), code: -32600,
                                                  message: "batch requests are not supported"), status: 400)
        }

        let method = msg["method"] as? String ?? ""
        let id = msg["id"] ?? NSNull()
        let params = msg["params"] as? [String: Any] ?? [:]

        if let version = (params["_meta"] as? [String: Any])?[versionMetaKey] as? String {
            return await handleModern(req, method: method, id: id, params: params, isNotification: msg["id"] == nil,
                                      version: version, backend, appVersion: appVersion, sources: sources, surface: surface)
        }
        // After initialize the client names its revision on every request; one this server does
        // not speak is a 400, as the spec says. (A modern request was checked above, against its
        // `_meta`, so an unknown modern revision gets the error that lists the supported ones.)
        if let v = req.headers["mcp-protocol-version"], !supportedVersions.contains(v) {
            return HTTPResponse.json(jsonRPCError(id: NSNull(), code: -32600,
                message: "unsupported MCP-Protocol-Version \(v); supported: \(supportedVersions.joined(separator: ", "))"),
                status: 400)
        }

        // Notifications (no id) get a 202 with no body.
        if msg["id"] == nil {
            return HTTPResponse(status: 202, headers: ["Content-Type": "application/json"], body: Data())
        }

        switch method {
        case "initialize":
            let requested = params["protocolVersion"] as? String
            return result(id: id, [
                "protocolVersion": requested.flatMap { supportedVersions.contains($0) ? $0 : nil } ?? supportedVersions[0],
                "capabilities": ["tools": [:] as [String: Any]],
                "serverInfo": ["name": "omni", "title": "Omni - local semantic file search",
                               "version": appVersion],
                "instructions": instructions
            ])

        case "ping":
            return result(id: id, [:])

        case "tools/list":
            // Sorted keys: the same bytes every launch, so a client's prompt cache holds across them.
            return HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": ["tools": tools(withSources: sources != nil)]], sortedKeys: true)

        case "tools/call":
            return await callTool(id: id, params: params, backend, sources: sources, surface: surface)

        default:
            return HTTPResponse.json(jsonRPCError(id: id, code: -32601, message: "method not found: \(method)"))
        }
    }

    // MARK: - The search tool

    private static func searchToolDescriptor() -> [String: Any] {
        [
            "name": "search",
            "title": "Search local files by meaning",
            "description": "Semantic search over the files Omni has indexed on this Mac (text, code, PDFs, images, audio, video - one embedding space). Describe the content in any language; keywords are not required. Returns paths with scores and the passage that matched; copies of a file count once.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": [
                        "type": "string",
                        "description": "What to find, described by meaning, e.g. 'invoice from Anthropic in February' or 'photos of a red sports car'."
                    ],
                    "top_k": [
                        "type": "integer",
                        "description": "Number of results (default 10, max 50).",
                        "minimum": 1, "maximum": 50
                    ],
                    "max_snippet": [
                        "type": "integer",
                        "description": "Max characters of text shown per result (default \(defaultSnippet), max 2000): for a text result, the part of its matching passage closest to the query, or the whole passage when it fits. 0 omits it.",
                        "minimum": 0, "maximum": 2000
                    ],
                    "include_images": [
                        "type": "boolean",
                        "description": "Inline a small JPEG of each image and scan result (at most \(maxInlineImages)). Off by default."
                    ],
                    "kinds": [
                        "type": "array",
                        "items": ["type": "string", "enum": ["text", "image", "audio", "video", "scan"]],
                        "description": "Restrict to these file kinds. 'text' includes scanned PDFs; 'scan' is scanned PDFs only."
                    ],
                    "folder": [
                        "type": "string",
                        "description": "Restrict to files under this absolute folder path."
                    ],
                    "folders": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Restrict to files under any of these absolute folder paths."
                    ],
                    "modified_after": [
                        "type": "string",
                        "description": "Only files modified at or after this ISO 8601 date or date-time ('2026-10-01')."
                    ],
                    "modified_before": [
                        "type": "string",
                        "description": "Only files modified before this date or date-time."
                    ],
                    "ext": [
                        "type": "string",
                        "description": "Only files with this extension, e.g. 'pdf' or '.md' (case-insensitive)."
                    ],
                    "exclude_folders": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Leave out files under these folders."
                    ],
                    "exclude_ext": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Leave out these extensions."
                    ],
                    "tags": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Only media with any of these tags (words tag_image reports)."
                    ],
                    "exclude_tags": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Leave out media with these tags."
                    ],
                    "filename": [
                        "type": "string",
                        "description": "Only files whose name matches these words."
                    ],
                    "sort": [
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "description": "date: newest first among the top matches."
                    ],
                    "offset": [
                        "type": "integer",
                        "description": "Skip this many results: the next page.",
                        "minimum": 0, "maximum": 140
                    ],
                    "min_score": [
                        "type": "number",
                        "description": "Relevance floor, 0 to 1, scaled per kind (default: none; weaker matches are marked instead). 0.5 keeps strong matches only.",
                        "minimum": 0, "maximum": 1
                    ],
                    "macs": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Linked Macs to search too, by name (default: all; [] for this Mac only). Their paths are on that Mac: get_linked_file copies one here."
                    ],
                    "group_duplicates": [
                        "type": "boolean", "default": true,
                        "description": "Collapse copies of a file into one result (default true); false lists every copy."
                    ]
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false]
        ]
    }

    private static func callSearch(id: Any, args: [String: Any], backend: any ServingBackend,
                                   sources: SourcesControl?, surface: ServedSurface, cli: Bool) async -> HTTPResponse {
        guard let query = args["query"] as? String, !query.isEmpty else {
            // Tool-level (not protocol-level) failure: isError true with a readable message.
            return result(id: id, [
                "content": [["type": "text", "text": "search failed: 'query' is required"]],
                "isError": true
            ])
        }
        var topK = (args["top_k"] as? Int) ?? 10
        topK = max(1, min(topK, 50))
        var maxSnippet = (args["max_snippet"] as? Int) ?? Self.defaultSnippet
        maxSnippet = max(0, min(maxSnippet, 2000))
        let includeImages = (args["include_images"] as? Bool) ?? false
        let (parsed, err) = SearchArgs.filter(args)
        if let err { return toolError(id: id, "search failed: \(err)") }
        let filter = parsed
        let offset = max(0, min((args["offset"] as? Int) ?? 0, 140))
        let byDate: Bool
        switch (args["sort"] as? String)?.lowercased() ?? "relevance" {
        case "relevance": byDate = false
        case "date": byDate = true
        default: return toolError(id: id, "search failed: 'sort' must be 'relevance' or 'date'")
        }

        // INDEX STATE, FETCHED CONCURRENTLY. An agent cannot tell an empty result set caused by
        // "not on this Mac" from one caused by "not indexed yet", and the second reading is the
        // one that makes it stop trusting the tool. The snapshot is a main-actor hop, so it is
        // started BEFORE the blocking scan and awaited after: the hop overlaps the search rather
        // than adding to it.
        let tIn = omniPerfEnabled ? Date() : nil
        async let snapshot: SourcesSnapshot? = { if let sources { return await sources.recentSnapshot() }; return nil }()

        // Same duplicate collapsing as the HTTP endpoint and the app's own list. Copies cost an
        // agent more than they cost a human - every one is context spent re-reading a file it has
        // already seen - so this defaults ON, with "group_duplicates": false for the flat list.
        let group = (args["group_duplicates"] as? Bool) ?? true
        let k = topK, width = maxSnippet
        // THE POOL: the top matches a page is cut from, at least 30 deep, so the answer can say what
        // else there is (where those matches sit, how many more) and `offset` can page through the
        // same ranking. A page is a slice of it - a second call with offset=10 runs the same search
        // and returns the next ten, and no ranking is held between calls.
        let poolSize = min(max(k + offset, Self.summaryPool), 190)
        let (hits, pool, groups, passages, tQueued, tRan) = await ServedWork.run {
            () -> ([SearchHit], [ResultGroup], [ResultGroup], [String?], Date, Date) in
            let tQueued = Date()
            let (hits, vector) = backend.searchReturningVector(query, topK: group ? min(poolSize * 3, 300) : poolSize, filter: filter, surface: surface)
            var pool = backend.groupedResults(hits, enabled: group, limit: poolSize)
            if byDate { pool.sort { $0.representative.modified > $1.representative.modified } }
            let groups = offset < pool.count ? Array(pool[offset ..< min(pool.count, offset + k)]) : []
            // The text a result shows is its chunk's own, recovered from the file: the whole chunk
            // when it fits `max_snippet`, else the window of it that matches the query. Media and
            // anything that cannot be recovered keep the stored snippet.
            let passages = SearchAdapter.textPassages(groups.map(\.representative), query: query, vector: vector, width: width, backend: backend)
            return (hits, pool, groups, passages, tQueued, Date())
        }
        let reps = groups.map(\.representative)
        let dupesByPath = Dictionary(uniqueKeysWithValues: groups.map { ($0.representative.path, $0) })

        // One text block per result, and nothing else.
        //
        // This used to interleave a `resource_link` carrying the same path as a file:// URI. MCP
        // defines a resource_link as something the client then READS BACK through `resources/read`,
        // and this server implements no resources at all - `resources/list` answers "method not
        // found" and the handshake declares `{"tools": {}}`. So the links named a door that was
        // never built, restating a path the text line above already gives in full. Measured, they
        // were 2,079 bytes of a 9,656-byte ten-result response, about 519 tokens of a model's
        // context per search. An agent that wants the file opens the path; a local search server
        // is talking to a client that already has the filesystem.
        //
        // Implementing resources/read instead would be the other way out, and it would mean serving
        // bytes the client can already open for itself.
        let snap = await snapshot
        if let tIn {
            omniPerfLog(String(format: "served-search queued=%.0fms work=%.0fms snapshot=%.0fms",
                               tQueued.timeIntervalSince(tIn) * 1000, tRan.timeIntervalSince(tQueued) * 1000,
                               -tRan.timeIntervalSinceNow * 1000))
        }
        let note = indexStateLine(snap)

        var content: [[String: Any]] = []
        if let note { content.append(diagnostic(note)) }
        if hits.isEmpty {
            // Say WHICH kind of nothing this is, so the agent's next move is right.
            var text = "No results for \"\(query)\"."
            if let snap {
                if snap.sources.isEmpty {
                    text += " Omni has no sources yet, so nothing is searchable"
                        + " - add_source indexes a folder or the Photos library."
                }
            }
            content.append(diagnostic(text))
        } else {
            var inlined = 0          // thumbnails emitted so far (capped)
            var cappedSkips = 0      // media hits skipped specifically because the cap was already reached
            // The app folds these under "Show N weaker matches"; an agent is told the same thing.
            // Judged over the whole pool, so a later page is folded by the same measure as the first.
            let poolWeak = VectorStore.weakFold(pool.map(\.representative))
            let weaker = reps.indices.map { offset + $0 < poolWeak.count ? poolWeak[offset + $0] : false }
            for (i, h) in reps.enumerated() {
                let score = Int((max(0, min(1, h.score)) * 100).rounded())
                let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                // Resolution/duration/size suffix on media hits only (text hits stay clean):
                // lets the reading agent weigh quality without a follow-up call.
                let isTextKind = h.fileKind == FileKind.text.rawValue
                var meta = isTextKind ? "" : mediaLabel(width: h.width, height: h.height,
                                                        duration: h.duration, bytes: h.size)
                // State what the row stands for, in the text block itself.
                if let g = dupesByPath[h.path], g.isStack {
                    meta += g.reason == .exact
                        ? ", \(g.count) identical copies"
                        : ", \(g.count) near-identical files"
                }
                // Carried here because the text block is now the ONLY representation. Both were
                // reachable solely through structuredContent before, and both change what an agent
                // does next: chunk_count says whether search_inline has more of this file to give,
                // and the date is what a "most recent" question turns on. A few characters each.
                if h.chunkCount > 1 { meta += ", \(h.chunkCount) passages" }
                meta += ", \(Self.shortDate(h.modified))"
                if weaker[i] { meta += ", weaker match" }
                var text = "\(offset + i + 1). \(h.path)  (\(h.fileKind), \(score)%\(loc)\(meta))"
                if maxSnippet > 0 {
                    let snippet = oneLine(passages[i] ?? h.snippet)
                    if !snippet.isEmpty { text += "\n   \(String(snippet.prefix(maxSnippet)))" }
                }
                content.append(["type": "text", "text": text])
                let isMedia = h.fileKind == FileKind.image.rawValue || h.fileKind == FileKind.scan.rawValue
                if includeImages && isMedia {
                    if inlined >= maxInlineImages {
                        cappedSkips += 1                         // genuinely skipped by the cap
                    } else if let b64 = thumbnailBase64(for: h) {
                        content.append(["type": "image", "data": b64, "mimeType": "image/jpeg"])
                        inlined += 1
                    }
                    // else: a decode failure (missing/corrupt file) - the text block still carries the
                    // path; don't count it against the cap message, which is about hitting the limit.
                }
            }
            if cappedSkips > 0 {
                content.append(diagnostic("(\(cappedSkips) more image result(s) not inlined; cap is \(maxInlineImages). Open them by path.)"))
            }
            let roots = snap?.sources.filter { $0.kind == "folder" }.map(\.key) ?? []
            if reps.isEmpty {
                content.append(diagnostic("No results past \(pool.count) for \"\(query)\": offset \(offset) is beyond the end."))
            } else if let line = poolSummary(pool, offset: offset, shown: reps.count, pageSize: topK, roots: roots) {
                content.append(["type": "text", "text": line])
            }
        }

        // Linked Macs, after this Mac's results: each under its name, its paths on that Mac.
        var linked: [[String: Any]] = []
        // Every linked Mac unless the caller names some, or none with [].
        var named = args["macs"] as? [String]
        if named?.count == 1, named?.first?.caseInsensitiveCompare("none") == .orderedSame { named = [] }
        if named?.isEmpty != true, let search = sources?.linkedSearch {
            let floor = filter.minScore ?? EngineServingBackend.minScore
            for a in await search(query, filter, topK, named ?? ["all"], named != nil) {
                let hits = Array(a.hits.filter { floor <= 0 || Double($0.score) / $0.lift >= VectorStore.relevanceFloor($0, base: floor) }.prefix(topK))
                if let note = a.note {
                    content.append(diagnostic("On \(a.mac): \(note.hasSuffix(".") ? note : note + ".")"))
                } else if hits.isEmpty {
                    content.append(diagnostic("On \(a.mac): no results for \"\(query)\"."))
                } else {
                    // THE WAY TO THE FILE GOES WITH THE PATHS. An agent given paths on another Mac and
                    // nothing else concluded it could only read the snippets: the tool's name was in
                    // the server instructions, which many clients never show the model.
                    let how = !a.canOpen ? "\(a.mac) does not allow opening its files, so only these snippets"
                        : cli ? "paths on \(a.mac); copy one here with: omni get_linked_file <path> --mac \"\(a.mac)\""
                        : "paths on \(a.mac); copy one here to read with get_linked_file, mac \"\(a.mac)\""
                    content.append(["type": "text", "text": "On \(a.mac) (\(how)):"])
                    for (i, h) in hits.enumerated() {
                        let score = Int((max(0, min(1, h.score)) * 100).rounded())
                        let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                        var text = "\(i + 1). \(h.path)  (\(h.fileKind), \(score)%\(loc), \(Self.shortDate(h.modified)))"
                        if maxSnippet > 0 {
                            let snippet = h.snippet.replacingOccurrences(of: "\n", with: " ")
                            if !snippet.isEmpty { text += "\n   \(String(snippet.prefix(maxSnippet)))" }
                        }
                        content.append(["type": "text", "text": text])
                    }
                }
                linked.append(["mac": a.mac, "note": a.note ?? NSNull(), "canOpen": a.canOpen, "results": hits.map { h -> [String: Any] in
                    ["path": h.path, "kind": h.fileKind, "score": Double(max(0, min(1, h.score))), "snippet": h.snippet,
                     "locator": h.locator, "modified": h.modified, "size": h.size]
                }])
            }
        }

        return answer(id: id, content, cli: cli) {
            var body: [String: Any] = ["query": query, "results": SearchAdapter.rows(groups, backend: backend, passages: passages), "grouped": group,
                                       "offset": offset, "ranked": pool.count]
            if !linked.isEmpty { body["linked"] = linked }
            return body
        }
    }

    // MARK: - The search_inline tool

    private static func searchInlineToolDescriptor() -> [String: Any] {
        [
            "name": "search_inline",
            "title": "Search within specific files",
            "description": "Semantic search restricted to the files (or folders) you name: ranks the best-matching passages (chunks) INSIDE those paths for the query. Reuses the existing index - only the query is embedded, nothing is re-read from disk - so it is fast. Use it to pinpoint where in a known set of documents something is discussed. Each path may be an exact indexed file or a folder (all indexed files under it are included); paths that are not indexed are skipped.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": [
                        "type": "string",
                        "description": "What to find inside the given files, described by meaning."
                    ],
                    "paths": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Absolute file paths to search within. A folder path includes every indexed file under it. Files not yet indexed are silently skipped."
                    ],
                    "top_k": [
                        "type": "integer",
                        "description": "Number of passages to return across all the given paths (default 10, max 100).",
                        "minimum": 1, "maximum": 100
                    ],
                    "max_snippet": [
                        "type": "integer",
                        "description": "Max characters of each passage (default 400, max 4000): the passage's own text, or the part of it closest to the query when it is longer. 0 omits passages.",
                        "minimum": 0, "maximum": 4000
                    ],
                    "exact": [
                        "type": "boolean",
                        "description": "Find the query as an exact phrase (any case, any spacing) in the files' text instead of by meaning: for names, numbers, codes and quotes. Scans are not searched."
                    ]
                ] as [String: Any],
                "required": ["query", "paths"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false]
        ]
    }

    private static func callSearchInline(id: Any, args: [String: Any], backend: any ServingBackend,
                                         surface: ServedSurface, cli: Bool) -> HTTPResponse {
        guard let query = args["query"] as? String, !query.isEmpty else {
            return result(id: id, [
                "content": [["type": "text", "text": "search_inline failed: 'query' is required"]],
                "isError": true
            ])
        }
        let paths = (args["paths"] as? [String])?.filter { !$0.isEmpty } ?? []
        guard !paths.isEmpty else {
            return result(id: id, [
                "content": [["type": "text", "text": "search_inline failed: 'paths' must be a non-empty array of absolute file or folder paths"]],
                "isError": true
            ])
        }
        var topK = (args["top_k"] as? Int) ?? 10
        topK = max(1, min(topK, 100))
        var maxSnippet = (args["max_snippet"] as? Int) ?? 400
        maxSnippet = max(0, min(maxSnippet, 4000))
        if args["exact"] as? Bool == true {
            return callPhrase(id: id, phrase: query, paths: paths, keep: topK, backend: backend, cli: cli)
        }

        var hits = backend.searchInline(query, paths: paths, topK: topK, surface: surface)
        // Each passage is the chunk's own text now, not its stored 220-character opening: the
        // whole chunk when it fits `max_snippet`, else the window of it that matches the query.
        if maxSnippet > 0 {
            let texts = backend.passages(for: hits.map { ($0.path, $0.chunkIndex, $0.locator) }, query: query, vector: nil, width: maxSnippet)
            for i in hits.indices { if let t = texts[i] { hits[i].snippet = t } }
        }

        var content: [[String: Any]] = []
        if hits.isEmpty {
            content.append(diagnostic("No indexed passages in the given paths matched \"\(query)\". (search_inline only reads files Omni has already indexed; unindexed paths are skipped.)"))
        } else {
            for (i, h) in hits.enumerated() {
                let score = Int((max(0, min(1, h.score)) * 100).rounded())
                let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                var text = "\(i + 1). \(h.path)  (\(h.kind), chunk \(h.chunkIndex), \(score)%\(loc))"
                if maxSnippet > 0 {
                    let snippet = oneLine(h.snippet)
                    if !snippet.isEmpty { text += "\n   \(String(snippet.prefix(maxSnippet)))" }
                }
                content.append(["type": "text", "text": text])
            }
        }
        return answer(id: id, content, cli: cli) {
            ["query": query, "results": hits.map { h -> [String: Any] in
                ["path": h.path, "kind": h.kind, "chunk_index": h.chunkIndex,
                 "score": Double(max(0, min(1, h.score))), "snippet": h.snippet, "locator": h.locator]
            }]
        }
    }

    /// search_inline with `exact`: the phrase found in the files' text, where it occurs and how often.
    /// GREP OVER THE FILES AN AGENT IS WORKING WITH, for what meaning does not find - a name, an amount,
    /// a code, a quote - in formats it cannot grep itself (Word, PDF, Pages). Agents quote phrases
    /// constantly, and taking cross-document matching away from a research agent cut its accuracy from
    /// 82 to 40 (arXiv 2606.14885); the index reads no content words, only names and identifiers, so
    /// the text is read here, from the files the scope names.
    private static func callPhrase(id: Any, phrase: String, paths: [String], keep: Int,
                                   backend: any ServingBackend, cli: Bool) -> HTTPResponse {
        let found = backend.findPhrase(phrase, paths: paths, keep: keep)
        var content: [[String: Any]] = []
        for (i, p) in found.places.enumerated() {
            content.append(["type": "text", "text": "\(i + 1). \(p.path)  (\(p.locator))\n   \(p.context)"])
        }
        var summary = found.total == 0
            ? "\"\(phrase)\" does not occur in the text of \(found.filesSearched) indexed file\(found.filesSearched == 1 ? "" : "s") in those paths."
            : "\(found.total) match\(found.total == 1 ? "" : "es") in \(found.filesWithMatches) of \(found.filesSearched) file\(found.filesSearched == 1 ? "" : "s")"
                + (found.total > found.places.count ? "; the first \(found.places.count) shown (top_k for more)." : ".")
        if found.filesInScope > found.filesSearched + found.scans + found.skipped {
            summary += " Only the first \(EngineServingBackend.maxPhraseFiles) of \(found.filesInScope) files were searched: name narrower paths."
        }
        if found.scans > 0 { summary += " \(found.scans) scanned PDF\(found.scans == 1 ? " was" : "s were") not searched: its text is OCR (ocr reads it)." }
        if !found.partial.isEmpty { summary += " Searched only partly (very long): \(found.partial.prefix(3).joined(separator: ", "))." }
        if found.filesInScope == 0 { summary = "No indexed files in those paths." }
        content.append(diagnostic(summary))
        return answer(id: id, content, cli: cli) {
            ["query": phrase, "exact": true, "total": found.total, "files_searched": found.filesSearched,
             "files_with_matches": found.filesWithMatches,
             "matches": found.places.map { ["path": $0.path, "locator": $0.locator, "context": $0.context] }]
        }
    }

    // MARK: - The read_file tool

    private static func readFileToolDescriptor() -> [String: Any] {
        [
            "name": "read_file",
            "title": "Read the text of an indexed file",
            "description": "DOCUMENTS ONLY: the text Omni read from an indexed text, code, PDF, Word, Pages, RTF or Keynote file, a stretch at a time (PDF pages, lines, or characters). For reading past a search result in formats you may not open yourself. Audio, video and images have no text to read; scans: use ocr.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "path": ["type": "string", "description": "Absolute path of a file Omni has indexed."],
                    "page": ["type": "integer", "minimum": 1, "description": "PDF: first page (default 1)."],
                    "line": ["type": "integer", "minimum": 1, "description": "Text and code: first line (default 1)."],
                    "offset": ["type": "integer", "minimum": 0, "description": "First character, for documents (default 0)."],
                    "max_chars": ["type": "integer", "minimum": 100, "maximum": 20000, "description": "Default 4000."]
                ] as [String: Any],
                "required": ["path"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false]
        ]
    }

    /// Runs on the served work queue. Only files the index holds, outside the clipboard history:
    /// this is not a general file reader for whatever path an agent names.
    private static func callReadFile(id: Any, args: [String: Any], backend: any ServingBackend, cli: Bool) -> HTTPResponse {
        guard let raw = args["path"] as? String, !raw.trimmingCharacters(in: .whitespaces).isEmpty else {
            return toolError(id: id, "read_file failed: 'path' is required")
        }
        let path = normalizeStorePath((raw.trimmingCharacters(in: .whitespaces) as NSString).expandingTildeInPath)
        let clip = AppModel.clipboardDirectory.path
        guard path.hasPrefix("/"), path != clip, !path.hasPrefix(clip + "/"),
              backend.fileStatus(paths: [path])[path] != nil else {
            return toolError(id: id, "read_file failed: \(path) is not in Omni's index. read_file reads only indexed files; file_status says which are.")
        }
        let maxChars = max(100, min((args["max_chars"] as? Int) ?? 4000, 20000))
        let outcome = TextSlice.read(path: path, page: args["page"] as? Int, line: args["line"] as? Int,
                                     offset: args["offset"] as? Int, maxChars: maxChars)
        switch outcome {
        case .scanned(let n):
            return toolError(id: id, "read_file: \(path) is a scanned PDF (\(n) page\(n == 1 ? "" : "s")) with no text layer. Use ocr with this path and the pages you want, e.g. pages \"1-3\".")
        case .media(let kind):
            return toolError(id: id, "read_file: \(path) is \(kind == "image" ? "an image" : kind == "audio" ? "audio" : "a video"); Omni searches it by meaning but keeps no text of it to read.\(kind == "image" ? " tag_image describes it." : "")")
        case .unreadable(let why):
            return toolError(id: id, "read_file failed: \(why) (\(path)).")
        case .text(let s):
            let plural = s.unit == "character" ? "characters" : s.unit + "s"
            var head = s.start == s.end || s.unit == "character" && s.end - s.start <= 1
                ? "\(path): \(s.unit) \(s.start) of \(s.total)"
                : "\(path): \(plural) \(s.start)-\(s.end) of \(s.total)"
            if s.unit == "character" { head = "\(path): characters \(s.start)-\(s.end) of \(s.total)" }
            if let next = s.next { head += "; next: \(s.unit == "character" ? "offset" : s.unit)=\(next)" } else { head += "; end of file" }
            return answer(id: id, [["type": "text", "text": head + "\n\n" + s.text]], cli: cli) {
                ["path": path, "unit": s.unit, "start": s.start, "end": s.end, "total": s.total,
                 "next": s.next.map { $0 as Any } ?? NSNull(), "text": s.text]
            }
        }
    }

    // MARK: - The file_status tool

    private static func fileStatusToolDescriptor() -> [String: Any] {
        [
            "name": "file_status",
            "title": "Check whether files are indexed",
            "description": "For each given absolute file path, report whether Omni's index currently covers it: indexed or not, the file kind, the number of indexed chunks (pages/passages), the modified time and byte size of the indexed version, when it was last indexed (indexed_at, when known), and up_to_date - whether the file on disk still matches the indexed version. Use it to check coverage before search_inline, or to tell whether a file's latest edits are searchable yet. Read-only and fast: nothing is embedded or read from the file contents.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "paths": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Absolute file paths to check (files, not folders; up to 2048)."
                    ] as [String: Any]
                ] as [String: Any],
                "required": ["paths"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false]
        ]
    }

    private static func callFileStatus(id: Any, args: [String: Any], backend: any ServingBackend, cli: Bool) -> HTTPResponse {
        let paths = (args["paths"] as? [String])?.filter { !$0.isEmpty } ?? []
        guard !paths.isEmpty else {
            return result(id: id, [
                "content": [["type": "text", "text": "file_status failed: 'paths' must be a non-empty array of absolute file paths"]],
                "isError": true
            ])
        }
        // Same cap as POST /v1/files/status: an unbounded batch would sweep the store's
        // queue and stat the disk once per path.
        guard paths.count <= servingMaxInputs else {
            return result(id: id, [
                "content": [["type": "text", "text": "file_status failed: 'paths' exceeds \(servingMaxInputs) items"]],
                "isError": true
            ])
        }
        let rows = FileStatusAdapter.rows(for: paths, backend: backend)

        let content: [[String: Any]] = rows.enumerated().map { i, row in
            let p = row["path"] as? String ?? ""
            let line: String
            if row["indexed"] as? Bool == true {
                let kind = row["kind"] as? String ?? "?"
                let chunks = row["chunk_count"] as? Int ?? 0
                let fresh: String
                if row["up_to_date"] as? Bool == true { fresh = "up to date" }
                else if row["exists"] as? Bool == true { fresh = "STALE: file changed on disk since indexing" }
                else { fresh = "file MISSING on disk" }
                line = "\(i + 1). \(p)  indexed (\(kind), \(chunks) chunk\(chunks == 1 ? "" : "s"), \(fresh))"
            } else {
                // WHY NOT, since the next step differs: a bare "NOT indexed" for a folder or a URL
                // read as "go and index it", and sent agents to add_source.
                let asked = i < paths.count ? paths[i] : p
                var isDir: ObjCBool = false
                let why: String
                if !asked.hasPrefix("/") && !asked.hasPrefix("~") { why = "not an absolute file path" }
                else if !FileManager.default.fileExists(atPath: p, isDirectory: &isDir) { why = "no such file" }
                else if isDir.boolValue { why = "a folder, not a file: search inside it with search's 'folder', or list_sources for what is indexed" }
                else { why = "NOT indexed (outside the indexed folders, a kind Omni skips, or not reached yet)" }
                line = "\(i + 1). \(p)  \(why)"
            }
            return ["type": "text", "text": line]
        }
        return answer(id: id, content, cli: cli) { ["files": rows] }
    }

    // MARK: - The tag_image tool

    private static func tagImageToolDescriptor() -> [String: Any] {
        [
            "name": "tag_image",
            "title": "Describe an image with open-vocabulary tags",
            "description": "Return short descriptive tags for images on this Mac (a photo, a screenshot, a scanned page, a video frame). Give absolute file paths inside the user's indexed folders. By default this reports the tags Omni already generated at index time, which are the same words the `tag:` search qualifier matches, so it is instant and costs nothing. Set recompute=true to run the vision model now instead - use that only for a file Omni has not indexed yet, or when the file changed since it was indexed. Tags are single lowercase words and may include other languages.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "paths": [
                        "type": "array",
                        "items": ["type": "string"],
                        "description": "Absolute image paths inside the user's indexed folders."
                    ] as [String: Any],
                    "recompute": [
                        "type": "boolean",
                        "description": "Run the vision model now rather than reading stored tags. Slower (a GPU forward per image, plus 5 refinement crops) and capped at \(mcpTagRecomputeMax) images. Off by default."
                    ] as [String: Any],
                    "top_k": [
                        "type": "integer",
                        "description": "How many tags per image when recomputing (1-25, default 5). Ignored when reading stored tags."
                    ] as [String: Any]
                ] as [String: Any],
                "required": ["paths"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false]
        ]
    }

    /// Recompute is far more expensive than a stored read (6 GPU forwards per image in HQ mode),
    /// and an agent cannot see that cost, so it gets a much tighter cap than servingMaxInputs.
    private static let mcpTagRecomputeMax = 4

    private static func callTagImage(id: Any, args: [String: Any], backend: any ServingBackend, cli: Bool) -> HTTPResponse {
        let paths = (args["paths"] as? [String])?.filter { !$0.isEmpty } ?? []
        guard !paths.isEmpty else {
            return result(id: id, [
                "content": [["type": "text", "text": "tag_image failed: 'paths' must be a non-empty array of absolute image paths"]],
                "isError": true
            ])
        }
        let recompute = (args["recompute"] as? Bool) ?? false
        var topK = (args["top_k"] as? Int) ?? OmniTagger.topK
        topK = max(1, min(topK, 25))

        let rows: [[String: Any]]
        if recompute {
            guard paths.count <= mcpTagRecomputeMax else {
                return result(id: id, [
                    "content": [["type": "text", "text": "tag_image failed: recompute is limited to \(mcpTagRecomputeMax) images per call; drop recompute to read stored tags for more"]],
                    "isError": true
                ])
            }
            guard let computed = TagAdapter.computeRows(paths: paths, topK: topK, backend: backend) else {
                return result(id: id, [
                    "content": [["type": "text", "text": "tag_image failed: tagging is unavailable (it is switched off, or the label cache is still building)"]],
                    "isError": true
                ])
            }
            rows = computed
        } else {
            rows = FileTagsAdapter.rows(for: paths, backend: backend)
        }

        let content: [[String: Any]] = rows.enumerated().map { i, row in
            let p = row["path"] as? String ?? ""
            let tags = row["tags"] as? [String] ?? []
            let line: String
            if let err = row["error"] as? String {
                line = "\(i + 1). \(p)  \(err)"
            } else if !tags.isEmpty {
                line = "\(i + 1). \(p)  \(tags.joined(separator: ", "))"
            } else if row["indexed"] as? Bool == false {
                line = "\(i + 1). \(p)  NOT indexed (pass recompute=true to tag it now)"
            } else if row["taggable"] as? Bool == false {
                line = "\(i + 1). \(p)  not an image (only images, scanned pages and video carry tags)"
            } else {
                line = "\(i + 1). \(p)  no tags yet"
            }
            return ["type": "text", "text": line]
        }
        return answer(id: id, content, cli: cli) { ["files": rows] }
    }

    // MARK: - The ocr tool

    private static func ocrToolDescriptor() -> [String: Any] {
        [
            "name": "ocr",
            "title": "Transcribe a scanned document or image to Markdown",
            "description": "Read the text of a scanned PDF, a photographed page or a screenshot on this Mac with Omni's on-device OCR model, as Markdown: tables as HTML, formulas as LaTeX, headers and footers dropped. Use it when a file's text is not otherwise readable - a PDF with no text layer, or an image of a document. Give an absolute path inside the user's indexed folders. At most \(OCRServing.maxPagesMCP) pages per call, the first \(OCRServing.maxPagesMCP) by default; the reply says which pages remain, so call again with `pages` for the next ones. Pages Omni has transcribed before come back instantly. A page takes a few seconds on the GPU, and the first call loads the model.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "path": [
                        "type": "string",
                        "description": "Absolute path of a PDF or image inside the user's indexed folders."
                    ],
                    "pages": [
                        "type": "string",
                        "description": "Pages to transcribe, counted from 1 as search results name them: \"3\", \"1-5\" or \"2,7-9\". Default: the first \(OCRServing.maxPagesMCP)."
                    ]
                ] as [String: Any],
                "required": ["path"]
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false]
        ]
    }

    private static func callOCR(id: Any, args: [String: Any], cli: Bool) async -> HTTPResponse {
        guard let path = args["path"] as? String, !path.isEmpty else {
            return toolError(id: id, "ocr failed: 'path' (an absolute file path) is required")
        }
        let document: OCRServing.Document
        switch OCRServing.open(path: path, inlineHint: false) {
        case .success(let d): document = d
        case .failure(let f): return toolError(id: id, "ocr failed: \(f.message)")
        }
        let count = document.pageCount
        var selected: [Int]
        switch OCRServing.parsePages(args["pages"], count: count, oneBased: true) {
        case .success(let p): selected = p ?? Array(0 ..< min(count, OCRServing.maxPagesMCP))
        case .failure(let f): return toolError(id: id, "ocr failed: \(f.message)")
        }
        guard selected.count <= OCRServing.maxPagesMCP else {
            return toolError(id: id, "ocr failed: \(selected.count) pages asked for; at most \(OCRServing.maxPagesMCP) per call")
        }
        let job: OCRServing.Job
        switch await OCRServing.prepare([(document, selected)]) {
        case .success(let j): job = j
        case .failure(let f): return toolError(id: id, "ocr failed: \(f.message)")
        }
        let pages: [OCRServing.Page]
        switch await job.run() {
        case .success(let p): pages = p
        case .failure(let f): return toolError(id: id, "ocr failed: \(f.message)")
        }

        // The path `open` resolved, not the argument: a file:// URL run through the path cleanup
        // came out as "/file:/...".
        let name = document.url?.path ?? path
        var head = count == 1 ? "\(name) (1 page)" : "\(name) (\(count) pages)"
        if count > 1 {
            head += ". This reply: pages \(Self.ranges(selected.map { $0 + 1 }))."
            let after = (selected.max() ?? -1) + 1
            if after < count {
                head += " Next: pages \"\(after + 1)-\(min(after + OCRServing.maxPagesMCP, count))\"."
            }
        }
        var content: [[String: Any]] = [diagnostic(head)]
        for p in pages {
            let body: String
            if p.failed {
                body = "(this page could not be transcribed)"
            } else if p.markdown.isEmpty {
                body = "(no text on this page)"
            } else {
                body = p.markdown
            }
            let label = count == 1 ? "" : "## Page \(p.index + 1)\n\n"
            content.append(["type": "text", "text": label + body])
        }
        let next = (selected.max() ?? -1) + 1
        return answer(id: id, content, cli: cli) {
            var body: [String: Any] = [
                "path": name, "page_count": count,
                "pages": pages.map { p -> [String: Any] in
                    var row: [String: Any] = ["index": p.index, "markdown": p.markdown]
                    if p.failed { row["failed"] = true }
                    return row
                },
            ]
            if next < count { body["next_pages"] = "\(next + 1)-\(min(next + OCRServing.maxPagesMCP, count))" }
            return body
        }
    }

    /// "1-3, 7" for [1, 2, 3, 7].
    private static func ranges(_ pages: [Int]) -> String {
        var out: [String] = []
        var start: Int?
        var prev = 0
        for p in pages.sorted() {
            if start != nil, p == prev + 1 { prev = p; continue }
            if let s = start { out.append(s == prev ? "\(s)" : "\(s)-\(prev)") }
            start = p; prev = p
        }
        if let s = start { out.append(s == prev ? "\(s)" : "\(s)-\(prev)") }
        return out.joined(separator: ", ")
    }

    // mimeType(forPath:) is shared with the HTTP adapters - see SchemaAdapters.swift.

    /// Compact human-readable media suffix for a hit's text line:
    /// resolution for images ("4032x3024"), duration for audio/video ("12:34", "1:23:45"), and
    /// the indexed byte size - the quality clues an agent needs to pick between similar hits.
    private static func mediaLabel(width: Int, height: Int, duration: Double, bytes: Int) -> String {
        var parts: [String] = []
        if width > 0 && height > 0 { parts.append("\(width)x\(height)") }
        if duration > 0 {
            let s = Int(duration.rounded())
            parts.append(s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60)
                                   : String(format: "%d:%02d", s / 60, s % 60))
        }
        if bytes > 0 { parts.append(bytesLabel(bytes)) }
        return parts.isEmpty ? "" : ", " + parts.joined(separator: ", ")
    }

    /// "8.2 MB"-style label without ByteCountFormatter (not documented thread-safe; this runs
    /// on the connection's detached task).
    private static func bytesLabel(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var v = Double(bytes), i = 0
        while v >= 1000, i < units.count - 1 { v /= 1000; i += 1 }
        return i == 0 ? "\(bytes) B" : String(format: v < 10 ? "%.1f %@" : "%.0f %@", v, units[i])
    }

    // MARK: - Inline thumbnails

    /// A base64 JPEG thumbnail for a media hit, longest edge `inlineImageMaxEdge`. Image hits
    /// are downsampled from the file; scanned-PDF hits render the matching page. nil if the
    /// file is missing or undecodable. Runs off the main actor (this adapter is called from
    /// the connection's detached task).
    private static func thumbnailBase64(for hit: SearchHit) -> String? {
        if hit.kind == FileKind.scan.rawValue {
            return pdfPageThumbnail(path: hit.path, pageIndex: pageIndex(from: hit.locator))
        }
        return imageThumbnail(path: hit.path)
    }

    private static func imageThumbnail(path: String) -> String? {
        // A Photos asset has no file to open; PhotoKit renders it at the same edge cap. Network
        // access stays off - a search response must not trigger an iCloud download.
        if let ref = PhotoLibrary.Ref(path) {
            guard let cg = PhotoLibrary.image(ref, maxDimension: inlineImageMaxEdge, allowNetwork: false) else { return nil }
            return jpegBase64(cg)
        }
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,   // honor EXIF orientation
            kCGImageSourceThumbnailMaxPixelSize: inlineImageMaxEdge
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
        return jpegBase64(cg)
    }

    /// Render the matching PDF page with Core Graphics (CGPDFDocument + a CGBitmapContext) rather than
    /// PDFKit/NSImage - CG is thread-safe off the main actor (this runs on the serving queue), and
    /// CGPDFDocument is mmap-backed so it doesn't eagerly parse the whole file.
    private static func pdfPageThumbnail(path: String, pageIndex: Int) -> String? {
        guard let doc = CGPDFDocument(URL(fileURLWithPath: path) as CFURL), doc.numberOfPages > 0,
              let page = doc.page(at: max(1, min(pageIndex + 1, doc.numberOfPages))) else { return nil }  // CGPDF pages are 1-based
        let box = page.getBoxRect(.mediaBox)
        guard box.width > 0, box.height > 0 else { return nil }
        let scale = CGFloat(inlineImageMaxEdge) / max(box.width, box.height)
        let w = max(1, Int((box.width * scale).rounded())), h = max(1, Int((box.height * scale).rounded()))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))   // PDF pages are transparent; paint a white sheet
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.scaleBy(x: scale, y: scale)
        ctx.translateBy(x: -box.minX, y: -box.minY)
        ctx.drawPDFPage(page)
        guard let cg = ctx.makeImage() else { return nil }
        return jpegBase64(cg)
    }

    private static func jpegBase64(_ cg: CGImage, quality: CGFloat = 0.7) -> String? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return (data as Data).base64EncodedString()
    }

    /// 0-based page index parsed from a locator like "Page 3" (-> 2); 0 if absent.
    private static func pageIndex(from locator: String) -> Int {
        guard let n = locator.split(whereSeparator: { !$0.isNumber }).first.flatMap({ Int($0) }), n > 0
        else { return 0 }
        return n - 1
    }

    // MARK: - JSON-RPC envelopes

    /// `tools/call`, for either era: the arguments checked against the tool's schema, then the tool.
    static func callTool(id: Any, params: [String: Any], _ backend: any ServingBackend,
                         sources: SourcesControl?, surface: ServedSurface) async -> HTTPResponse {
        guard let name = params["name"] as? String else {
            return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "missing tool name"))
        }
        let args = params["arguments"] as? [String: Any] ?? [:]
        // AN ARGUMENT THE TOOL DOES NOT TAKE IS AN ERROR, naming the ones it does. `topK` for
        // `top_k` - the slip every agent makes - used to be ignored without a word, and the call
        // ran with the default instead of what was asked.
        if let schema = tools(withSources: sources != nil).first(where: { $0["name"] as? String == name })?["inputSchema"] as? [String: Any],
           let props = schema["properties"] as? [String: Any] {
            let unknown = args.keys.filter { props[$0] == nil }.sorted()
            if !unknown.isEmpty {
                let known = props.keys.sorted().joined(separator: ", ")
                return toolError(id: id, "\(name) failed: unknown argument\(unknown.count == 1 ? "" : "s") \(unknown.map { "'\($0)'" }.joined(separator: ", ")); \(name) takes: \(known)")
            }
        }
        // The `omni` command line marks its calls (CLI.md): it gets structuredContent and the
        // diagnostic blocks marked for stderr. Every other client's answer is unchanged.
        let cli = (params["_meta"] as? [String: Any])?[cliMetaKey] as? Bool == true
        let surface: ServedSurface = cli ? .cli : surface
        switch name {
        case "search":        return await callSearch(id: id, args: args, backend: backend, sources: sources, surface: surface, cli: cli)
        case "search_inline", "file_status", "tag_image", "read_file":
            // Blocking tools, so the whole call runs on the served work queue (see ServedWork).
            let call = ReadOnlyRequest(id: id, args: args)
            return await ServedWork.run {
                switch name {
                case "search_inline": return callSearchInline(id: call.id, args: call.args, backend: backend, surface: surface, cli: cli)
                case "file_status":   return callFileStatus(id: call.id, args: call.args, backend: backend, cli: cli)
                case "read_file":     return callReadFile(id: call.id, args: call.args, backend: backend, cli: cli)
                default:              return callTagImage(id: call.id, args: call.args, backend: backend, cli: cli)
                }
            }
        case "ocr":           return await callOCR(id: id, args: args, cli: cli)
        case "get_linked_file":
            guard let fetch = sources?.linkedFetch else { return toolError(id: id, "get_linked_file failed: Omni is still loading") }
            return await callLinkedFile(id: id, args: args, fetch: fetch, cli: cli)
        case "list_sources", "add_source", "pause_source", "remove_source":
            guard let sources else {
                return toolError(id: id, "\(name) failed: indexing control is unavailable (the index is still loading)")
            }
            return await callSources(id: id, tool: name, args: args, sources: sources, cli: cli)
        default:
            return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "unknown tool: \(name)"))
        }
    }

    // MARK: - The modern revision (2026-07-28)

    /// Who answers, in each modern result's `_meta`.
    private static func serverInfo(_ appVersion: String) -> [String: Any] {
        ["name": "omni", "title": "Omni - local semantic file search", "version": appVersion]
    }

    /// A request carrying its revision in `_meta`: the revision checked, the HTTP headers that mirror
    /// the body checked against it (Streamable HTTP: a mismatch is refused, so a proxy routing on a
    /// header cannot be made to run something else), then the method.
    static func handleModern(_ req: HTTPRequest, method: String, id: Any, params: [String: Any], isNotification: Bool,
                             version: String, _ backend: any ServingBackend, appVersion: String,
                             sources: SourcesControl?, surface: ServedSurface) async -> HTTPResponse {
        guard modernVersions.contains(version) else {
            return HTTPResponse.json(["jsonrpc": "2.0", "id": id, "error": [
                "code": -32022, "message": "unsupported protocol version \(version)",
                "data": ["supported": modernVersions + supportedVersions, "requested": version]]], status: 400)
        }
        func mismatch(_ why: String) -> HTTPResponse {
            HTTPResponse.json(["jsonrpc": "2.0", "id": id, "error": ["code": -32020, "message": why]], status: 400)
        }
        guard req.headers["mcp-protocol-version"] == version else {
            return mismatch("the MCP-Protocol-Version header must be \(version), as the request's _meta says")
        }
        guard req.headers["mcp-method"] == method else {
            return mismatch("the Mcp-Method header must be \(method), as the request's method says")
        }
        if ["tools/call", "prompts/get", "resources/read"].contains(method) {
            let named = (params["name"] as? String) ?? (params["uri"] as? String)
            guard let header = req.headers["mcp-name"], let decoded = Self.headerValue(header), decoded == named else {
                return mismatch("the Mcp-Name header must name the same \(method == "resources/read" ? "resource" : "tool") as the request")
            }
        }
        if isNotification { return HTTPResponse(status: 202, headers: ["Content-Type": "application/json"], body: Data()) }
        let info = serverInfo(appVersion)
        // Every result says it is complete and who answered; the list results say how long they keep.
        func complete(_ value: [String: Any]) -> [String: Any] {
            var v = value
            v["resultType"] = v["resultType"] ?? "complete"
            var meta = v["_meta"] as? [String: Any] ?? [:]
            meta[serverInfoMetaKey] = info
            v["_meta"] = meta
            return v
        }
        // The tool list changes only when the app wires its source tools, once, while it loads.
        let ttl = sources == nil ? 5_000 : 300_000
        switch method {
        case "server/discover":
            return HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": complete([
                "supportedVersions": modernVersions + supportedVersions,
                "capabilities": ["tools": [:] as [String: Any]],
                "instructions": instructions, "ttlMs": ttl, "cacheScope": "private"])], sortedKeys: true)
        case "tools/list":
            return HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": complete([
                "tools": tools(withSources: sources != nil), "ttlMs": ttl, "cacheScope": "private"])], sortedKeys: true)
        case "tools/call":
            let r = await callTool(id: id, params: params, backend, sources: sources, surface: surface)
            guard var obj = (try? JSONSerialization.jsonObject(with: r.body)) as? [String: Any],
                  let value = obj["result"] as? [String: Any] else { return r }
            obj["result"] = complete(value)
            return HTTPResponse.json(obj, status: r.status)
        default:
            // Not in this revision (`ping` and `initialize` belong to the legacy handshake) or not offered.
            return HTTPResponse.json(jsonRPCError(id: id, code: -32601, message: "method not found: \(method)"), status: 404)
        }
    }

    /// A header value as sent, or decoded from the spec's `=?base64?...?=` form (for a value that is
    /// not plain header-safe ASCII); nil when that form does not decode.
    static func headerValue(_ raw: String) -> String? {
        guard raw.hasPrefix("=?base64?"), raw.hasSuffix("?="), raw.count >= 11 else { return raw }
        let b64 = String(raw.dropFirst(9).dropLast(2))
        return Data(base64Encoded: b64).flatMap { String(data: $0, encoding: .utf8) }
    }

    private static func result(id: Any, _ value: [String: Any]) -> HTTPResponse {
        HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": value])
    }

    private static func jsonRPCError(id: Any, code: Int, message: String) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]]
    }

    // MARK: - Managing what Omni indexes
    //
    // These four mirror the sidebar one-for-one (add / pause / resume / remove) and go through the
    // same AppModel methods it does - see SourcesAdapter. They are the only MCP tools that CHANGE
    // anything, which is why each description says plainly what the user will see happen.

    private static func linkedFileDescriptor() -> [String: Any] {
        [
            "name": "get_linked_file",
            "title": "Get a file from a linked Mac",
            "description": "Copy a file from a linked Mac - a path search listed under that Mac's name - to this Mac, and return the local path to read. The copy is read-only; editing it changes nothing on the other Mac.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "mac": ["type": "string", "description": "The linked Mac's name, as search listed it."],
                    "path": ["type": "string", "description": "The file's path on that Mac, as search listed it."],
                ] as [String: Any],
                "required": ["path", "mac"],   // path first: the command line takes it bare
            ] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false],
        ]
    }

    private static func callLinkedFile(id: Any, args: [String: Any],
                                       fetch: @Sendable (String, String) async -> (path: String?, error: String?),
                                       cli: Bool) async -> HTTPResponse {
        guard let mac = args["mac"] as? String, !mac.isEmpty, let path = args["path"] as? String, !path.isEmpty else {
            return toolError(id: id, "get_linked_file failed: 'mac' and 'path' are required")
        }
        let r = await fetch(mac, path)
        guard let local = r.path else { return toolError(id: id, "get_linked_file failed: \(r.error ?? "unknown error")") }
        return answer(id: id, [["type": "text", "text": local]], cli: cli) { ["mac": mac, "path": path, "local": local] }
    }

    private static func listSourcesDescriptor() -> [String: Any] {
        [
            "name": "list_sources",
            "title": "List what Omni indexes",
            "description": "The folders and Apple Photos sources Omni currently indexes, with how many files each holds, whether it is paused, and live progress while it is being indexed. Also lists the Photos albums that could be added but have not been. Call this before concluding that a file is not on the Mac: search only ever covers what is listed here.",
            "inputSchema": ["type": "object", "properties": [:] as [String: Any]] as [String: Any],
            "annotations": ["readOnlyHint": true, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false],
        ]
    }

    private static func addSourceDescriptor() -> [String: Any] {
        [
            "name": "add_source",
            "title": "Index a folder or photo album",
            "description": "Add a folder, or the user's Apple Photos library, to what Omni indexes. Indexing starts right away and the source appears in the app's sidebar. Give EITHER 'path' (an absolute folder path) OR 'album' ('all' for the whole Photos library, or an album id from list_sources) - not both. A folder already covered by an indexed parent is reported as such rather than added twice. The disk root, system folders and the home folder itself are refused: add the folders inside them.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "path": ["type": "string", "description": "Absolute path of a folder to index, e.g. /Users/me/Projects. '~' is expanded."],
                    "album": ["type": "string", "description": "'all' for the whole Apple Photos library, or an album id from list_sources. Requires that the user has granted Omni access to Photos in the app."],
                ] as [String: Any],
            ] as [String: Any],
            "annotations": ["readOnlyHint": false, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false],
        ]
    }

    private static func pauseSourceDescriptor() -> [String: Any] {
        [
            "name": "pause_source",
            "title": "Pause or resume a source",
            "description": "Stop indexing one folder or Photos source without losing what it has already indexed - its files stay searchable, and new or changed files are ignored until it is resumed. Pass paused=false to resume. Use 'key' exactly as list_sources reports it.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "key": ["type": "string", "description": "The source's key from list_sources (a folder path, or 'photos://<id>')."],
                    "paused": ["type": "boolean", "default": true, "description": "true to pause (the default), false to resume."],
                ] as [String: Any],
                "required": ["key"],
            ] as [String: Any],
            "annotations": ["readOnlyHint": false, "destructiveHint": false,
                            "idempotentHint": true, "openWorldHint": false],
        ]
    }

    private static func removeSourceDescriptor() -> [String: Any] {
        [
            "name": "remove_source",
            "title": "Stop indexing a source and drop its results",
            "description": "Remove a folder or Photos source from Omni. Its files stop appearing in search and their index entries are deleted. THE FILES THEMSELVES ARE NOT TOUCHED - nothing is deleted from disk or from the Photos library - but re-adding the source means indexing it again from scratch, which can take a long time. Ask the user before calling this.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "key": ["type": "string", "description": "The source's key from list_sources (a folder path, or 'photos://<id>')."],
                ] as [String: Any],
                "required": ["key"],
            ] as [String: Any],
            "annotations": ["readOnlyHint": false, "destructiveHint": true,
                            "idempotentHint": true, "openWorldHint": false],
        ]
    }

    private static func callSources(id: Any, tool: String, args: [String: Any],
                                    sources: SourcesControl, cli: Bool) async -> HTTPResponse {
        func done(_ m: SourceMutation, _ verb: String) async -> HTTPResponse {
            if let err = m.error { return toolError(id: id, "\(tool) failed: \(err)") }
            let snap = await sources.snapshot()
            return sourcesResult(id: id, snap, headline: m.note ?? "\(verb) \(m.key ?? "")", cli: cli)
        }

        switch tool {
        case "list_sources":
            return sourcesResult(id: id, await sources.snapshot(), headline: nil, cli: cli)

        case "add_source":
            let path = (args["path"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            let album = (args["album"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            switch (path?.isEmpty == false ? path : nil, album?.isEmpty == false ? album : nil) {
            case (let p?, nil): return await done(await sources.addFolder(p), "now indexing")
            case (nil, let a?): return await done(await sources.addAlbum(a), "now indexing")
            case (nil, nil):
                return toolError(id: id, "add_source failed: give 'path' (a folder) or 'album' ('all', or an album id from list_sources)")
            default:
                return toolError(id: id, "add_source failed: give 'path' or 'album', not both")
            }

        case "pause_source":
            guard let key = (args["key"] as? String), !key.isEmpty else {
                return toolError(id: id, "pause_source failed: 'key' is required (from list_sources)")
            }
            let paused = (args["paused"] as? Bool) ?? true
            return await done(await sources.setPaused(key, paused), paused ? "paused" : "resumed")

        case "remove_source":
            guard let key = (args["key"] as? String), !key.isEmpty else {
                return toolError(id: id, "remove_source failed: 'key' is required (from list_sources)")
            }
            return await done(await sources.remove(key), "removed")

        default:
            return toolError(id: id, "unknown tool: \(tool)")
        }
    }

    /// One rendering for all four: the caller's next question after any of them is "so what is
    /// indexed now", and answering it in the same turn saves a round trip.
    private static func sourcesResult(id: Any, _ snap: SourcesSnapshot, headline: String?, cli: Bool) -> HTTPResponse {
        var lines: [String] = []
        if let headline { lines.append(headline) }
        if snap.sources.isEmpty {
            lines.append("Omni is not indexing anything yet.")
        } else {
            for s in snap.sources {
                var state: [String] = ["\(s.indexedFiles.formatted()) indexed"]
                if s.paused { state.append("PAUSED") }
                else if s.queued { state.append("waiting to be indexed") }
                else if s.total > 0, s.done < s.total { state.append("indexing \(s.done.formatted())/\(s.total.formatted())") }
                lines.append("- [\(s.kind)] \(s.name)  \(s.key)  (\(state.joined(separator: ", ")))")
            }
        }
        if !snap.albums.isEmpty {
            lines.append("Photo albums that can be added: " +
                         snap.albums.map { "\($0.title) (id \($0.id), \($0.count))" }.joined(separator: "; "))
        } else if !snap.photosAuthorized {
            lines.append("Omni has no access to the Apple Photos library; the user grants it in the app.")
        }

        return answer(id: id, [["type": "text", "text": lines.joined(separator: "\n")]], cli: cli) {
            var body = SourcesAdapter.snapshotBody(snap)
            if let headline { body["done"] = headline }
            return body
        }
    }

    // MARK: - Freshness

    /// One line when the index has nothing to search, or nil. Prepended to every search response,
    /// so it has to earn its tokens. There is deliberately no "still indexing" line: the index is
    /// updated continuously (the watcher, the launch pass, tag refinement), so "indexing" is true
    /// most of the time and said nothing about whether this answer was complete - agents read it
    /// as a reason to distrust good results. Per-file freshness is on each hit and in file_status.
    private static func indexStateLine(_ snap: SourcesSnapshot?) -> String? {
        guard let snap, snap.sources.isEmpty else { return nil }
        return "index: empty - no folders or photo sources are indexed yet (see list_sources / add_source)."
    }

    // MARK: - Answers for the command line

    static let cliMetaKey = CLIProtocol.cliKey
    static let stderrMetaKey = CLIProtocol.stderrKey
    private static let diagnosticMark = "omni.diagnostic"

    /// A text block that is ABOUT the answer - nothing found, an empty index, a cap, a page range -
    /// rather than part of it. An MCP client reads it like any other block; the command line sends
    /// it to stderr, so stdout stays data a pipe can use.
    private static func diagnostic(_ text: String) -> [String: Any] {
        ["type": "text", "text": text, diagnosticMark: true]
    }

    /// A tool's successful answer. For the command line only: `structured` becomes
    /// structuredContent (the same shapes REST answers with) and diagnostic blocks are marked
    /// for stderr. Any other client gets exactly the content blocks, as before.
    private static func answer(id: Any, _ content: [[String: Any]], cli: Bool,
                               structured: () -> [String: Any]) -> HTTPResponse {
        let blocks = content.map { block -> [String: Any] in
            var b = block
            if b.removeValue(forKey: diagnosticMark) != nil, cli { b["_meta"] = [stderrMetaKey: true] }
            return b
        }
        var value: [String: Any] = ["content": blocks, "isError": false]
        if cli { value["structuredContent"] = structured() }
        return result(id: id, value)
    }

    private static func toolError(id: Any, _ message: String) -> HTTPResponse {
        result(id: id, ["content": [["type": "text", "text": message]], "isError": true])
    }

}
