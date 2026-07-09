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
/// Three tools are exposed: `search` (whole-index), `search_inline` (rank passages within
/// given paths), and `file_status` (index coverage/freshness per file). Agents that want raw
/// vectors use the OpenAI-style /v1/embeddings endpoint instead; MCP is for the things agents
/// actually do with Omni - reach the user's files by meaning and trust what the index covers.
enum MCPAdapter {
    /// The newest protocol revision this server knows. If the client asks for a different
    /// one we echo the client's (every revision we care about shares this method surface).
    private static let protocolVersion = "2025-06-18"

    /// Longest edge of an inlined thumbnail (matches jina-ai/MCP's image handling).
    private static let inlineImageMaxEdge = 800
    /// Hard cap on inlined thumbnails per response, so an all-images search can't flood the
    /// client's context with base64. Image hits past this are still returned as links/text.
    private static let maxInlineImages = 10

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
                "instructions": "Search the user's local files by meaning. Files of every kind - text, code, PDFs, images, audio, video - share one embedding space, so describe the CONTENT you want in natural language (any language). `search` finds files across the whole index; `search_inline` ranks the best passages WITHIN a specific set of files or folders you already know; `file_status` reports whether given files are indexed and whether the index is still fresh for them. Results are file paths with scores, snippets, and media metadata (resolution, duration, size); read the files yourself if you need their full contents."
            ])

        case "ping":
            return result(id: id, [:])

        case "tools/list":
            return result(id: id, ["tools": [searchToolDescriptor(), searchInlineToolDescriptor(),
                                             fileStatusToolDescriptor()]])

        case "tools/call":
            guard let name = params["name"] as? String else {
                return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "missing tool name"))
            }
            let args = params["arguments"] as? [String: Any] ?? [:]
            switch name {
            case "search":        return callSearch(id: id, args: args, backend: backend)
            case "search_inline": return callSearchInline(id: id, args: args, backend: backend)
            case "file_status":   return callFileStatus(id: id, args: args, backend: backend)
            default:
                return HTTPResponse.json(jsonRPCError(id: id, code: -32602, message: "unknown tool: \(name)"))
            }

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
                    "max_snippet": [
                        "type": "integer",
                        "description": "Max characters of the matching text snippet shown per result (default 200, max 2000). 0 omits snippets. Does not affect structuredContent, which always carries the full snippet.",
                        "minimum": 0, "maximum": 2000
                    ],
                    "include_images": [
                        "type": "boolean",
                        "description": "When true, embed a downscaled JPEG thumbnail (max 800px) for each image and scanned-PDF result so they render inline in the client. Off by default to keep responses small; at most the top \(maxInlineImages) media hits are inlined."
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
        var maxSnippet = (args["max_snippet"] as? Int) ?? 200
        maxSnippet = max(0, min(maxSnippet, 2000))
        let includeImages = (args["include_images"] as? Bool) ?? false
        var filter = SearchFilter()
        if let kinds = args["kinds"] as? [String], !kinds.isEmpty {
            var set = Set(kinds)
            // Same superset rule as the app: text includes scanned PDFs ('scan').
            if set.contains(FileKind.text.rawValue) { set.insert(FileKind.scan.rawValue) }
            filter.kinds = set
        }
        if let folder = args["folder"] as? String, !folder.isEmpty { filter.folderPrefix = folder }

        let hits = backend.search(query, topK: topK, filter: filter)
        // Duplicate grouping for structuredContent - same key = byte-identical file (see
        // SearchAdapter; lockstep rule keeps stale sidecar rows from mislabeling).
        let contentKeys = backend.contentKeys(paths: hits.map { $0.path })

        // The response interleaves, per result, a human/LLM-readable text block and a
        // `resource_link` (a file:// URI the client can open or render directly). Capable
        // clients show a list of openable files; dumb ones still read every text block in
        // order. `structuredContent` carries the full machine-readable rows regardless.
        var content: [[String: Any]] = []
        if hits.isEmpty {
            content.append(["type": "text", "text": "No results for \"\(query)\"."])
        } else {
            var inlined = 0          // thumbnails emitted so far (capped)
            var cappedSkips = 0      // media hits skipped specifically because the cap was already reached
            var firstWithKey: [String: Int] = [:]   // content_key -> 1-based rank of its first hit
            for (i, h) in hits.enumerated() {
                let score = Int((max(0, min(1, h.score)) * 100).rounded())
                let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                // Resolution/duration/size suffix on media hits only (text hits stay clean):
                // lets the reading agent weigh quality without a follow-up call.
                let isTextKind = h.kind == FileKind.text.rawValue
                var meta = isTextKind ? "" : mediaLabel(width: h.width, height: h.height,
                                                        duration: h.duration, bytes: h.size)
                // Flag byte-identical copies of an earlier hit right in the text line, so a
                // text-only agent doesn't weigh the same file twice.
                if let ck = contentKeys[h.path], ck.modified == h.modified {
                    if let first = firstWithKey[ck.key] {
                        meta += ", identical copy of result \(first)"
                    } else {
                        firstWithKey[ck.key] = i + 1
                    }
                }
                var text = "\(i + 1). \(h.path)  (\(h.kind), \(score)%\(loc)\(meta))"
                if maxSnippet > 0 {
                    let snippet = h.snippet.replacingOccurrences(of: "\n", with: " ")
                    if !snippet.isEmpty { text += "\n   \(String(snippet.prefix(maxSnippet)))" }
                }
                content.append(["type": "text", "text": text])
                content.append(resourceLink(for: h, description: "\(h.kind), \(score)%\(loc)\(meta)"))
                let isMedia = h.kind == FileKind.image.rawValue || h.kind == FileKind.scan.rawValue
                if includeImages && isMedia {
                    if inlined >= maxInlineImages {
                        cappedSkips += 1                         // genuinely skipped by the cap
                    } else if let b64 = thumbnailBase64(for: h) {
                        content.append(["type": "image", "data": b64, "mimeType": "image/jpeg"])
                        inlined += 1
                    }
                    // else: a decode failure (missing/corrupt file) - the resource_link still stands; don't
                    // count it against the cap message, which is specifically about hitting the limit.
                }
            }
            if cappedSkips > 0 {
                content.append(["type": "text",
                                 "text": "(\(cappedSkips) more image result(s) not inlined; cap is \(maxInlineImages). Open them via their file:// links.)"])
            }
        }

        let structured: [[String: Any]] = hits.map { h in
            var row: [String: Any] =
                ["path": h.path,
                 "uri": URL(fileURLWithPath: h.path).absoluteString,
                 "score": Double(max(0, min(1, h.score))),
                 "kind": h.kind,
                 "snippet": h.snippet,
                 // Where the best-matching chunk sits inside the file ("Page 3", "Line 1240"); "" if n/a.
                 "locator": h.locator,
                 // Total indexed chunks (pages/passages) in this file; > 1 means more passages exist.
                 "chunk_count": h.chunkCount,
                 "modified": h.modified]
            if let mime = mimeType(forPath: h.path) { row["mime_type"] = mime }
            if h.width > 0 { row["width"] = h.width }
            if h.height > 0 { row["height"] = h.height }
            if h.duration > 0 { row["duration"] = h.duration }
            if h.size > 0 { row["bytes"] = h.size }
            if let ck = contentKeys[h.path], ck.modified == h.modified { row["content_key"] = ck.key }
            return row
        }
        return result(id: id, [
            "content": content,
            "structuredContent": ["results": structured],
            "isError": false
        ])
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
                        "description": "Max characters of each passage in the text block (default 400, max 4000). 0 omits snippets; structuredContent always carries the full snippet.",
                        "minimum": 0, "maximum": 4000
                    ]
                ] as [String: Any],
                "required": ["query", "paths"]
            ] as [String: Any]
        ]
    }

    private static func callSearchInline(id: Any, args: [String: Any], backend: any ServingBackend) -> HTTPResponse {
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

        let hits = backend.searchInline(query, paths: paths, topK: topK)

        var content: [[String: Any]] = []
        if hits.isEmpty {
            content.append(["type": "text",
                             "text": "No indexed passages in the given paths matched \"\(query)\". (search_inline only reads files Omni has already indexed; unindexed paths are skipped.)"])
        } else {
            for (i, h) in hits.enumerated() {
                let score = Int((max(0, min(1, h.score)) * 100).rounded())
                let loc = h.locator.isEmpty ? "" : ", \(h.locator)"
                var text = "\(i + 1). \(h.path)  (\(h.kind), chunk \(h.chunkIndex), \(score)%\(loc))"
                if maxSnippet > 0 {
                    let snippet = h.snippet.replacingOccurrences(of: "\n", with: " ")
                    if !snippet.isEmpty { text += "\n   \(String(snippet.prefix(maxSnippet)))" }
                }
                content.append(["type": "text", "text": text])
            }
        }
        let structured: [[String: Any]] = hits.map { h in
            ["path": h.path,
             "uri": URL(fileURLWithPath: h.path).absoluteString,
             "score": Double(max(0, min(1, h.score))),
             "kind": h.kind,
             "chunk_index": h.chunkIndex,
             "snippet": h.snippet,
             "locator": h.locator]
        }
        return result(id: id, [
            "content": content,
            "structuredContent": ["results": structured],
            "isError": false
        ])
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
            ] as [String: Any]
        ]
    }

    private static func callFileStatus(id: Any, args: [String: Any], backend: any ServingBackend) -> HTTPResponse {
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
                line = "\(i + 1). \(p)  NOT indexed"
            }
            return ["type": "text", "text": line]
        }
        return result(id: id, [
            "content": content,
            "structuredContent": ["files": rows],
            "isError": false
        ])
    }

    /// An MCP `resource_link` content block pointing at a local file. Clients that render
    /// resources can open or preview the image/audio/video/document directly; the rest
    /// ignore it and fall back to the adjacent text block.
    private static func resourceLink(for hit: SearchHit, description: String) -> [String: Any] {
        var link: [String: Any] = [
            "type": "resource_link",
            "uri": URL(fileURLWithPath: hit.path).absoluteString,
            "name": (hit.path as NSString).lastPathComponent,
            "description": description
        ]
        if let mime = mimeType(forPath: hit.path) { link["mimeType"] = mime }
        return link
    }

    // mimeType(forPath:) is shared with the HTTP adapters - see SchemaAdapters.swift.

    /// Compact human-readable media suffix for a hit's text line / resource_link description:
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

    private static func result(id: Any, _ value: [String: Any]) -> HTTPResponse {
        HTTPResponse.json(["jsonrpc": "2.0", "id": id, "result": value])
    }

    private static func jsonRPCError(id: Any, code: Int, message: String) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]]
    }
}
