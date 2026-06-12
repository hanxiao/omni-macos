import Foundation
import OmniKit

/// MCP (Model Context Protocol) endpoint: JSON-RPC 2.0 over streamable HTTP at POST /mcp.
/// Stateless by design - every request is self-contained, no session ids, and responses are
/// single JSON bodies (the spec permits a plain `application/json` reply for servers that do
/// not stream). That makes the endpoint work with every MCP client that speaks the HTTP
/// transport - `claude mcp add --transport http`, Cursor, VS Code, etc. - with zero setup
/// beyond the URL.
///
/// One tool is exposed: `search`. Agents that want raw vectors use the OpenAI-style
/// /v1/embeddings endpoint instead; MCP is for the thing agents actually do with Omni -
/// reach the user's files by meaning.
enum MCPAdapter {
    /// The newest protocol revision this server knows. If the client asks for a different
    /// one we echo the client's (every revision we care about shares this method surface).
    private static let protocolVersion = "2025-06-18"

    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend, appVersion: String) -> HTTPResponse {
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

        // Notifications (no id) get a 202 with no body.
        if msg["id"] == nil {
            return HTTPResponse(status: 202, headers: ["Content-Type": "application/json"], body: Data())
        }

        switch method {
        case "initialize":
            let requested = params["protocolVersion"] as? String
            return result(id: id, [
                "protocolVersion": requested ?? protocolVersion,
                "capabilities": ["tools": [:] as [String: Any]],
                "serverInfo": ["name": "omni", "title": "Omni - local semantic file search",
                               "version": appVersion],
                "instructions": "Search the user's local files by meaning. Files of every kind - text, code, PDFs, images, audio, video - share one embedding space, so describe the CONTENT you want in natural language (any language). Results are file paths with scores and snippets; read the files yourself if you need their contents."
            ])

        case "ping":
            return result(id: id, [:])

        case "tools/list":
            return result(id: id, ["tools": [searchToolDescriptor()]])

        case "tools/call":
            guard let name = params["name"] as? String else {
                return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "missing tool name"))
            }
            guard name == "search" else {
                return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "unknown tool: \(name)"))
            }
            let args = params["arguments"] as? [String: Any] ?? [:]
            return callSearch(id: id, args: args, backend: backend)

        default:
            return HTTPResponse.json(jsonRPCError(id: id, code: -32601, message: "method not found: \(method)"))
        }
    }

    // MARK: - The search tool

    private static func searchToolDescriptor() -> [String: Any] {
        [
            "name": "search",
            "title": "Search local files by meaning",
            "description": "Semantic search over the files Omni has indexed on this Mac (text, code, PDFs, images, audio, video - all in one embedding space). Describe the content in natural language; keywords are not required and any language works. Returns file paths with relevance scores and text snippets.",
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
                    "kinds": [
                        "type": "array",
                        "items": ["type": "string", "enum": ["text", "image", "audio", "video", "scan"]],
                        "description": "Restrict to these file kinds. 'text' includes scanned PDFs; 'scan' is scanned PDFs only."
                    ],
                    "folder": [
                        "type": "string",
                        "description": "Restrict to files under this absolute folder path."
                    ]
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any]
        ]
    }

    private static func callSearch(id: Any, args: [String: Any], backend: any ServingBackend) -> HTTPResponse {
        guard let query = args["query"] as? String, !query.isEmpty else {
            // Tool-level (not protocol-level) failure: isError true with a readable message.
            return result(id: id, [
                "content": [["type": "text", "text": "search failed: 'query' is required"]],
                "isError": true
            ])
        }
        var topK = (args["top_k"] as? Int) ?? 10
        topK = max(1, min(topK, 50))
        var filter = SearchFilter()
        if let kinds = args["kinds"] as? [String], !kinds.isEmpty {
            var set = Set(kinds)
            // Same superset rule as the app: text includes scanned PDFs ('scan').
            if set.contains(FileKind.text.rawValue) { set.insert(FileKind.scan.rawValue) }
            filter.kinds = set
        }
        if let folder = args["folder"] as? String, !folder.isEmpty { filter.folderPrefix = folder }

        let hits = backend.search(query, topK: topK, filter: filter)

        // Human/LLM-readable text block plus machine-readable structuredContent.
        let lines: [String] = hits.isEmpty
            ? ["No results for \"\(query)\"."]
            : hits.enumerated().map { i, h in
                let score = Int((max(0, min(1, h.score)) * 100).rounded())
                let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                let snippet = h.snippet.replacingOccurrences(of: "\n", with: " ")
                let snip = h.kind == "text" && !snippet.isEmpty ? "\n   \(String(snippet.prefix(160)))" : ""
                return "\(i + 1). \(h.path)  (\(h.kind), \(score)%\(loc))\(snip)"
            }
        let structured: [[String: Any]] = hits.map { h in
            ["path": h.path,
             "score": Double(max(0, min(1, h.score))),
             "kind": h.kind,
             "snippet": h.snippet,
             // Where the best-matching chunk sits inside the file ("Page 3", "Line 1240"); "" if n/a.
             "locator": h.locator,
             "modified": h.modified]
        }
        return result(id: id, [
            "content": [["type": "text", "text": lines.joined(separator: "\n")]],
            "structuredContent": ["results": structured],
            "isError": false
        ])
    }

    // MARK: - JSON-RPC envelopes

    private static func result(id: Any, _ value: [String: Any]) -> HTTPResponse {
        HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": value])
    }

    private static func jsonRPCError(id: Any, code: Int, message: String) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]]
    }
}
