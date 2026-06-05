import Foundation
import OmniKit

// All adapters are stateless enums. They parse JSON via JSONSerialization, call the
// backend, and emit a provider-shaped JSON response. The engine emits fixed 1024-d
// L2-normalized float vectors; adapters never truncate, requantize, or fabricate vectors.
// Only the usage/billed-units token counts are an acknowledged whitespace heuristic.

// MARK: - Shared helpers

private enum JSONBody {
    /// Parse the request body into a JSON object dictionary, or nil if invalid.
    static func object(_ req: HTTPRequest) -> [String: Any]? {
        guard !req.body.isEmpty,
              let obj = try? JSONSerialization.jsonObject(with: req.body),
              let dict = obj as? [String: Any] else { return nil }
        return dict
    }
}

/// Whitespace token estimate for usage fields only. Never touches vectors.
private func tokenEstimate(_ texts: [String]) -> Int {
    texts.reduce(0) { acc, t in
        acc + max(1, t.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).count)
    }
}

/// Little-endian Float32 bytes, base64-encoded (OpenAI/Jina base64 encoding_format).
private func base64Encode(_ vec: [Float]) -> String {
    var data = Data(capacity: vec.count * 4)
    for f in vec {
        var le = f.bitPattern.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
    return data.base64EncodedString()
}

private func badRequest(_ message: String, type: String = "invalid_request_error") -> HTTPResponse {
    HTTPResponse.json(["error": ["message": message, "type": type]], status: 400)
}

// MARK: - OpenAI + Jina (one OpenAI-shaped emitter)

enum OpenAIJinaAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }

        let model = (body["model"] as? String) ?? backend.modelName
        let texts = flattenInput(body["input"])
        if texts.isEmpty { return badRequest("'input' is required") }

        // task suffix ".query" or == "query" -> query path. Field is "task" (Jina).
        let task = (body["task"] as? String)?.lowercased() ?? ""
        let asQuery = task == "query" || task.hasSuffix(".query")

        // base64 if either OpenAI's encoding_format or Jina's embedding_type asks for it.
        let wantsBase64 = matchesBase64(body["encoding_format"]) || matchesBase64(body["embedding_type"])

        let vectors = backend.embedBatch(texts, query: asQuery)

        var data: [[String: Any]] = []
        data.reserveCapacity(vectors.count)
        for (i, vec) in vectors.enumerated() {
            let embedding: Any = wantsBase64 ? base64Encode(vec) : vec
            data.append([
                "object": "embedding",
                "index": i,
                "embedding": embedding
            ])
        }

        let tokens = tokenEstimate(texts)
        let payload: [String: Any] = [
            "object": "list",
            "data": data,
            "model": model,
            "usage": ["prompt_tokens": tokens, "total_tokens": tokens]
        ]
        return HTTPResponse.json(payload)
    }

    /// Accepts String, [String], [{"text": ...}], or a mix; flattens to [String].
    private static func flattenInput(_ raw: Any?) -> [String] {
        if let s = raw as? String { return [s] }
        if let arr = raw as? [Any] {
            var out: [String] = []
            for item in arr {
                if let s = item as? String {
                    out.append(s)
                } else if let obj = item as? [String: Any], let t = obj["text"] as? String {
                    out.append(t)
                }
            }
            return out
        }
        return []
    }

    private static func matchesBase64(_ raw: Any?) -> Bool {
        if let s = raw as? String { return s.lowercased() == "base64" }
        if let arr = raw as? [String] { return arr.contains { $0.lowercased() == "base64" } }
        return false
    }
}

// MARK: - Cohere (v1 + v2)

enum CohereAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend, v2: Bool) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return cohereError("invalid JSON body") }

        let texts = parseTexts(body)
        if texts.isEmpty { return cohereError("'texts' is required") }

        let inputType = (body["input_type"] as? String)?.lowercased()
        if v2, inputType == nil {
            return cohereError("input_type is required")
        }
        let asQuery = inputType == "search_query"

        let requestedTypes = (body["embedding_types"] as? [String])?.map { $0.lowercased() } ?? []
        // We only produce float vectors. Quantized types are never fabricated.
        if let bad = requestedTypes.first(where: { $0 != "float" }) {
            return cohereError("unsupported embedding_type: \(bad)")
        }

        let vectors = backend.embedBatch(texts, query: asQuery)
        let floatRows: [[Float]] = vectors

        // v2 always emits the object form; v1 emits bare list unless embedding_types given.
        let embeddings: Any
        if v2 || !requestedTypes.isEmpty {
            embeddings = ["float": floatRows]
        } else {
            embeddings = floatRows
        }

        let tokens = tokenEstimate(texts)
        let payload: [String: Any] = [
            "id": UUID().uuidString,
            "embeddings": embeddings,
            "texts": texts,
            "meta": [
                "api_version": ["version": v2 ? "2" : "1"],
                "billed_units": ["input_tokens": tokens]
            ]
        ]
        return HTTPResponse.json(payload)
    }

    /// texts from "texts", or inputs[].text (v4 multimodal item form, text parts only).
    private static func parseTexts(_ body: [String: Any]) -> [String] {
        if let texts = body["texts"] as? [String] { return texts }
        if let inputs = body["inputs"] as? [Any] {
            var out: [String] = []
            for item in inputs {
                if let obj = item as? [String: Any], let t = obj["text"] as? String {
                    out.append(t)
                }
            }
            return out
        }
        return []
    }

    private static func cohereError(_ message: String) -> HTTPResponse {
        HTTPResponse.json(["message": message], status: 400)
    }
}

// MARK: - Gemini (embedContent + batchEmbedContents)

enum GeminiAdapter {
    private static let queryTaskTypes: Set<String> = [
        "RETRIEVAL_QUERY", "QUESTION_ANSWERING", "CODE_RETRIEVAL_QUERY"
    ]

    static func handle(_ req: HTTPRequest, model: String, _ backend: any ServingBackend, batch: Bool) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return geminiError("invalid JSON body") }

        if batch {
            guard let requests = body["requests"] as? [Any] else {
                return geminiError("'requests' is required")
            }
            var texts: [String] = []
            var anyQuery = false
            for item in requests {
                guard let obj = item as? [String: Any] else { continue }
                texts.append(partsText(obj["content"]))
                if let tt = obj["taskType"] as? String, queryTaskTypes.contains(tt) { anyQuery = true }
                if let dimErr = checkDimension(obj["outputDimensionality"], backend) { return dimErr }
            }
            if texts.isEmpty { return geminiError("no content to embed") }

            let vectors = backend.embedBatch(texts, query: anyQuery)
            let embeddings = vectors.map { ["values": $0] }
            return HTTPResponse.json(["embeddings": embeddings])
        } else {
            if let dimErr = checkDimension(body["outputDimensionality"], backend) { return dimErr }
            let text = partsText(body["content"])
            let tt = body["taskType"] as? String
            let asQuery = tt.map { queryTaskTypes.contains($0) } ?? false
            let vectors = backend.embedBatch([text], query: asQuery)
            let values = vectors.first ?? []
            return HTTPResponse.json(["embedding": ["values": values]])
        }
    }

    /// Join all parts[].text in a content object.
    private static func partsText(_ raw: Any?) -> String {
        guard let content = raw as? [String: Any], let parts = content["parts"] as? [Any] else { return "" }
        var pieces: [String] = []
        for p in parts {
            if let obj = p as? [String: Any], let t = obj["text"] as? String { pieces.append(t) }
        }
        return pieces.joined(separator: " ")
    }

    /// outputDimensionality is accepted only if it equals the engine dimension; the engine
    /// emits fixed-width vectors and never truncates.
    private static func checkDimension(_ raw: Any?, _ backend: any ServingBackend) -> HTTPResponse? {
        guard let n = raw as? Int else { return nil }
        if n != backend.dim {
            return geminiError("outputDimensionality must equal \(backend.dim)")
        }
        return nil
    }

    private static func geminiError(_ message: String) -> HTTPResponse {
        HTTPResponse.json([
            "error": ["code": 400, "message": message, "status": "INVALID_ARGUMENT"]
        ], status: 400)
    }
}

// MARK: - Custom search

enum SearchAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }
        guard let query = body["query"] as? String, !query.isEmpty else {
            return badRequest("'query' is required")
        }

        var topK = (body["top_k"] as? Int) ?? 20
        topK = max(1, min(topK, 200))

        var filter = SearchFilter()
        if let filters = body["filters"] as? [String: Any] {
            if let kinds = filters["kinds"] as? [String] { filter.kinds = Set(kinds) }
            if let folder = filters["folder"] as? String, !folder.isEmpty { filter.folderPrefix = folder }
            if let ext = filters["ext"] as? String, !ext.isEmpty { filter.ext = ext }
            if let since = filters["since"] as? Double { filter.since = since }
            else if let sinceInt = filters["since"] as? Int { filter.since = Double(sinceInt) }
        }

        let hits = backend.search(query, topK: topK, filter: filter)
        let results: [[String: Any]] = hits.map { hit in
            [
                "path": hit.path,
                "score": Double(max(0, min(1, hit.score))),
                "snippet": hit.snippet,
                "kind": hit.kind,
                "modified": hit.modified
            ]
        }
        return HTTPResponse.json(["query": query, "results": results])
    }
}

// MARK: - Health / models

enum HealthAdapter {
    static func handle(_ routePath: String, _ backend: any ServingBackend) -> HTTPResponse {
        if routePath == "/health" {
            return HTTPResponse.json([
                "status": "ok",
                "model": backend.modelName,
                "dim": backend.dim,
                "running": true
            ])
        }
        // /v1/models
        return HTTPResponse.json([
            "object": "list",
            "data": [["id": backend.modelName, "object": "model"]]
        ])
    }
}
