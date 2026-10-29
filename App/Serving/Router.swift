import Foundation

/// Maps a parsed request to a provider adapter. Pure value type, Sendable, so it can be
/// captured by the @Sendable handler closure passed to HTTPServer.
struct Router: Sendable {
    let backend: any ServingBackend
    /// Returns true if the request is authorized to proceed. Built by the controller as a
    /// snapshot of the current scope/token so it carries no main-actor state.
    let auth: @Sendable (HTTPRequest) -> Bool
    /// Reported by the MCP initialize handshake as serverInfo.version.
    var appVersion: String = ""
    /// What Omni indexes, and how to change it. Optional because it is wired from AppModel after
    /// the store loads; a request that arrives first gets a 503 rather than a wrong answer.
    var sources: SourcesControl? = nil
    /// What an MCP tool call through this router counts as: `.cli` on the command line's socket,
    /// where the only caller is `omni`.
    var mcpSurface: ServedSurface = .mcp
    /// Refuse requests that name another host or come from another site's page. On for the TCP
    /// server in local scope; off for the command line's socket, which no browser can reach.
    var loopbackOnly = false

    func handle(_ req: HTTPRequest) async -> HTTPResponse {
        let route = req.routePath

        // DNS REBINDING, before anything else - health included. A local-only server is still
        // reachable from a web page: the page's own domain is re-pointed at 127.0.0.1, the browser
        // then sends it requests it considers same-origin, and reads the answers. Accepting the
        // connection from loopback proves nothing about who wrote the request. What does is the
        // Host header (the browser sends the page's domain, not 127.0.0.1) and the Origin header
        // (the page's origin). Measured before this: Host: evil.test with Origin: http://evil.test
        // got a 200 and the full source list. Requests from agents and curl send no Origin and a
        // loopback Host, so they are untouched. The MCP transport spec requires the Origin check.
        if loopbackOnly, let refusal = Self.refuseForeign(req) { return refusal }

        // Liveness probes are always open, before any auth check.
        if req.method == "GET", route == "/health" || route == "/v1/models" {
            return HealthAdapter.handle(route, backend)
        }

        // Auth gate. The 401 envelope is shaped per provider by path prefix.
        guard auth(req) else {
            return unauthorized(for: route)
        }

        // MCP: JSON-RPC over streamable HTTP. Auth-gated like everything else (loopback scope
        // never requires a token; LAN scope always does, sent as a normal Authorization header).
        if route == "/mcp" {
            return await MCPAdapter.handle(req, backend, appVersion: appVersion, sources: sources, surface: mcpSurface)
        }

        // What Omni indexes (folders + Photos sources), and the four things the sidebar can do to it.
        if SourcesAdapter.owns(route) {
            return await SourcesAdapter.handle(req, route: route, sources)
        }

        switch (req.method, route) {
        case ("POST", "/v1/embeddings"):
            return await ServedWork.run { OpenAIJinaAdapter.handle(req, backend) }
        case ("POST", "/v1/embed"):
            return await ServedWork.run { CohereAdapter.handle(req, backend, v2: false) }
        case ("POST", "/v2/embed"):
            return await ServedWork.run { CohereAdapter.handle(req, backend, v2: true) }
        case ("POST", "/v1/search"):
            return await ServedWork.run { SearchAdapter.handle(req, backend) }
        case ("POST", "/v1/files/status"):
            return await ServedWork.run { FileStatusAdapter.handle(req, backend) }
        case ("POST", "/v1/files/tags"):
            return await ServedWork.run { FileTagsAdapter.handle(req, backend) }
        case ("POST", "/v1/tag"):
            return await ServedWork.run { TagAdapter.handle(req, backend) }
        case ("POST", "/v1/chat/completions"):
            return await ChatOCRAdapter.handle(req)
        case ("POST", "/v1/ocr"):
            return await DocumentOCRAdapter.handle(req)
        default:
            break
        }

        // Gemini: POST /v1beta/models/{model}:embedContent | :batchEmbedContents
        if req.method == "POST", route.hasPrefix("/v1beta/models/") {
            if let colon = route.lastIndex(of: ":") {
                let modelPart = String(route[route.index(route.startIndex, offsetBy: "/v1beta/models/".count)..<colon])
                let action = String(route[route.index(after: colon)...])
                switch action {
                case "embedContent":
                    return await ServedWork.run { GeminiAdapter.handle(req, model: modelPart, backend, batch: false) }
                case "batchEmbedContents":
                    return await ServedWork.run { GeminiAdapter.handle(req, model: modelPart, backend, batch: true) }
                default:
                    break
                }
            }
        }

        if Self.postRoutes.contains(route) || route.hasPrefix("/v1beta/models/") {
            var r = HTTPResponse.json(["error": ["message": "\(route) takes POST, not \(req.method)", "type": "invalid_request_error"]],
                                      status: 405)
            r.headers["Allow"] = "POST"
            return r
        }
        return notFound(for: route)
    }

    /// The routes above that answer POST only: another method on one is a 405, not "no route".
    static let postRoutes: Set<String> = ["/v1/embeddings", "/v1/embed", "/v2/embed", "/v1/search", "/v1/files/status",
                                          "/v1/files/tags", "/v1/tag", "/v1/chat/completions", "/v1/ocr"]

    // MARK: - Foreign requests

    static func refuseForeign(_ req: HTTPRequest) -> HTTPResponse? {
        if let host = req.headers["host"], !isLoopbackHost(host) {
            return HTTPResponse.json(["error": ["message": "Host \(host) is not this Mac; Omni answers only requests addressed to localhost",
                                                "type": "forbidden"]], status: 403)
        }
        if let origin = req.headers["origin"], !isLoopbackOrigin(origin) {
            return HTTPResponse.json(["error": ["message": "Requests from web pages at \(origin) are refused",
                                                "type": "forbidden"]], status: 403)
        }
        return nil
    }

    /// "localhost", "127.0.0.1", "[::1]", each with or without a port.
    static func isLoopbackHost(_ raw: String) -> Bool {
        var h = raw.trimmingCharacters(in: .whitespaces).lowercased()
        if h.hasPrefix("[") {                       // [::1] or [::1]:51299
            guard let close = h.firstIndex(of: "]") else { return false }
            h = String(h[h.index(after: h.startIndex)..<close])
        } else if let colon = h.lastIndex(of: ":"), h.firstIndex(of: ":") == colon {
            h = String(h[..<colon])                 // localhost:51299; a bare IPv6 has several colons
        }
        return h == "localhost" || h == "127.0.0.1" || h == "::1"
    }

    /// An http(s) origin on this Mac. "null" (a sandboxed frame, a file:// page) is not.
    static func isLoopbackOrigin(_ raw: String) -> Bool {
        guard let u = URL(string: raw.trimmingCharacters(in: .whitespaces)), let scheme = u.scheme?.lowercased(),
              scheme == "http" || scheme == "https", let host = u.host?.lowercased() else { return false }
        return host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]"
    }

    // MARK: - Error envelopes

    private func unauthorized(for route: String) -> HTTPResponse {
        if route.hasPrefix("/v1beta/") {
            return HTTPResponse.json([
                "error": ["code": 401, "message": "Unauthorized", "status": "UNAUTHENTICATED"]
            ], status: 401)
        }
        // Exact routes: "/v1/embeddings" also starts with "/v1/embed", and is OpenAI-shaped.
        if route == "/v1/embed" || route == "/v2/embed" {
            return HTTPResponse.json([
                "message": "invalid api token"
            ], status: 401)
        }
        // OpenAI / Jina / search shape.
        return HTTPResponse.json([
            "error": ["message": "Unauthorized", "type": "invalid_request_error", "code": "invalid_api_key"]
        ], status: 401)
    }

    private func notFound(for route: String) -> HTTPResponse {
        HTTPResponse.json([
            "error": ["message": "no route for \(route)", "type": "invalid_request_error"]
        ], status: 404)
    }
}
