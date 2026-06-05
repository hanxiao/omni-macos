import Foundation

/// Maps a parsed request to a provider adapter. Pure value type, Sendable, so it can be
/// captured by the @Sendable handler closure passed to HTTPServer.
struct Router: Sendable {
    let backend: any ServingBackend
    /// Returns true if the request is authorized to proceed. Built by the controller as a
    /// snapshot of the current scope/token so it carries no main-actor state.
    let auth: @Sendable (HTTPRequest) -> Bool

    func handle(_ req: HTTPRequest) async -> HTTPResponse {
        let route = req.routePath

        // Liveness probes are always open, before any auth check.
        if req.method == "GET", route == "/health" || route == "/v1/models" {
            return HealthAdapter.handle(route, backend)
        }

        // Auth gate. The 401 envelope is shaped per provider by path prefix.
        guard auth(req) else {
            return unauthorized(for: route)
        }

        switch (req.method, route) {
        case ("POST", "/v1/embeddings"):
            return OpenAIJinaAdapter.handle(req, backend)
        case ("POST", "/v1/embed"):
            return CohereAdapter.handle(req, backend, v2: false)
        case ("POST", "/v2/embed"):
            return CohereAdapter.handle(req, backend, v2: true)
        case ("POST", "/v1/search"):
            return SearchAdapter.handle(req, backend)
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
                    return GeminiAdapter.handle(req, model: modelPart, backend, batch: false)
                case "batchEmbedContents":
                    return GeminiAdapter.handle(req, model: modelPart, backend, batch: true)
                default:
                    break
                }
            }
        }

        return notFound(for: route)
    }

    // MARK: - Error envelopes

    private func unauthorized(for route: String) -> HTTPResponse {
        if route.hasPrefix("/v1beta/") {
            return HTTPResponse.json([
                "error": ["code": 401, "message": "Unauthorized", "status": "UNAUTHENTICATED"]
            ], status: 401)
        }
        if route.hasPrefix("/v1/embed") || route.hasPrefix("/v2/embed") {
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
