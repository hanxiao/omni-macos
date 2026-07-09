import Foundation
import OmniKit
import UniformTypeIdentifiers

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

/// IANA MIME type inferred from the path extension (e.g. "image/jpeg"); nil if unknown.
/// Shared by the search/status adapters and MCP so every surface labels files identically.
func mimeType(forPath path: String) -> String? {
    let ext = (path as NSString).pathExtension
    guard !ext.isEmpty else { return nil }
    return UTType(filenameExtension: ext)?.preferredMIMEType
}

// MARK: - OpenAI + Jina (one OpenAI-shaped emitter)

/// One request must not occupy the engine for minutes: cap the batch like hosted APIs do
/// (Jina caps at 2048 inputs). Callers get an explicit 400, not a silent truncation.
let servingMaxInputs = 2048

enum OpenAIJinaAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }

        let model = (body["model"] as? String) ?? backend.modelName
        guard let texts = flattenInput(body["input"]) else {
            // Silently dropping unrecognized items (token-id arrays, numbers, nulls) would return
            // fewer `data` entries than inputs with shifted indexes - corrupt for the caller.
            return badRequest("unsupported 'input' element: expected a string or {\"text\": ...}")
        }
        if texts.isEmpty { return badRequest("'input' is required") }
        if texts.count > servingMaxInputs { return badRequest("'input' exceeds \(servingMaxInputs) items") }

        // The model serves exactly one dimension; a client asking for another (OpenAI `dimensions`)
        // must hear "no" here, not discover a mismatch downstream in its vector store.
        if let want = body["dimensions"] as? Int, want != backend.dim {
            return badRequest("'dimensions' must equal \(backend.dim) for \(backend.modelName)")
        }

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
    /// Returns nil if the array contains an unrecognized element (the caller responds 400) so the
    /// response `data` always has one entry per input, indexes aligned.
    private static func flattenInput(_ raw: Any?) -> [String]? {
        if let s = raw as? String { return [s] }
        if let arr = raw as? [Any] {
            var out: [String] = []
            for item in arr {
                if let s = item as? String {
                    out.append(s)
                } else if let obj = item as? [String: Any], let t = obj["text"] as? String {
                    out.append(t)
                } else {
                    return nil
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
        if texts.count > servingMaxInputs { return badRequest("'texts' exceeds \(servingMaxInputs) items") }
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
            if requests.count > servingMaxInputs { return geminiError("'requests' exceeds \(servingMaxInputs) items") }
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
            if let kinds = filters["kinds"] as? [String] {
                var set = Set(kinds)
                // Same superset rule as the app: text documents include scanned PDFs ('scan'),
                // so API clients asking for text don't silently lose them.
                if set.contains(FileKind.text.rawValue) { set.insert(FileKind.scan.rawValue) }
                filter.kinds = set
            }
            if let folder = filters["folder"] as? String, !folder.isEmpty { filter.folderPrefix = folder }
            if let ext = filters["ext"] as? String, !ext.isEmpty { filter.ext = ext }
            if let since = filters["since"] as? Double { filter.since = since }
            else if let sinceInt = filters["since"] as? Int { filter.since = Double(sinceInt) }
        }

        let hits = backend.search(query, topK: topK, filter: filter)
        // Duplicate grouping: hits sharing a content_key are byte-identical files (the dedup
        // sidecar), so an agent picking between matches knows which ones are literally copies.
        // Lockstep rule as duplicateChunks: only trust a key whose modified matches the hit's.
        let contentKeys = backend.contentKeys(paths: hits.map { $0.path })
        let results: [[String: Any]] = hits.map { hit in
            var row: [String: Any] = [
                "path": hit.path,
                "score": Double(max(0, min(1, hit.score))),
                "snippet": hit.snippet,
                "kind": hit.kind,
                "modified": hit.modified,
                // Best-matching chunk's position inside the file ("Page 3", "Line 1240"); "" if n/a.
                "locator": hit.locator,
                // Total indexed chunks (pages/passages) in this file.
                "chunk_count": hit.chunkCount
            ]
            // Media metadata recorded at index time, so downstream agents can weigh image
            // resolution or clip length without opening the file. All of it is already resident
            // in the hit (or fetched with the snippet for the winners only) - zero extra cost.
            // 0 = unknown/not applicable (e.g. rows indexed before these columns existed): omitted.
            if hit.width > 0 { row["width"] = hit.width }
            if hit.height > 0 { row["height"] = hit.height }
            if hit.duration > 0 { row["duration"] = hit.duration }
            if hit.size > 0 { row["bytes"] = hit.size }
            if let mime = mimeType(forPath: hit.path) { row["mime_type"] = mime }
            if let ck = contentKeys[hit.path], ck.modified == hit.modified { row["content_key"] = ck.key }
            return row
        }
        return HTTPResponse.json(["query": query, "results": results])
    }
}

// MARK: - File index status

/// POST /v1/files/status: for each given absolute file path, whether Omni's index currently
/// covers it and how fresh that coverage is. `up_to_date` uses the indexer's own change-detection
/// signature - on-disk (mtime, size) equal to the stored (modified, size) - read with the exact
/// resourceValues expressions the crawler uses, so the answer always agrees with what a reconcile
/// pass would do. Read-only, index-backed metadata lookups: nothing is embedded, the GPU is never
/// touched, and indexing/search speed is unaffected.
enum FileStatusAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }
        // "paths": [...] is the canonical form; a single "path" string is accepted for convenience.
        // A present-but-mistyped "paths" gets its own message, not the generic "is required".
        var paths: [String]
        if let raw = body["paths"] {
            guard let arr = raw as? [String] else {
                return badRequest("'paths' must be an array of strings")
            }
            paths = arr
        } else if let one = body["path"] as? String {
            paths = [one]
        } else {
            paths = []
        }
        paths = paths.filter { !$0.isEmpty }
        if paths.isEmpty { return badRequest("'paths' (array of absolute file paths) is required") }
        if paths.count > servingMaxInputs { return badRequest("'paths' exceeds \(servingMaxInputs) items") }

        return HTTPResponse.json(["files": rows(for: paths, backend: backend)])
    }

    /// One status row per input path, input order preserved. Shared with the MCP file_status tool.
    static func rows(for paths: [String], backend: any ServingBackend) -> [[String: Any]] {
        let normalized = paths.map(normalize)
        let status = backend.fileStatus(paths: normalized)

        return normalized.map { p in
            guard let st = status[p] else {
                // Not indexed: report only that, with NO disk probe. Stat-ing arbitrary paths
                // would let any (LAN-token) caller test the existence of files far outside the
                // indexed roots; this endpoint reports index coverage, nothing more.
                return ["path": p, "indexed": false]
            }
            // Freshness: same attribute source as the crawler (FileCrawler.swift) -
            // contentModificationDate + fileSize via resourceValues - so double-for-double
            // equality against the stored signature mirrors the indexer's own change
            // detection ("prev.modified == mtime, prev.size == size").
            let vals = try? URL(fileURLWithPath: p).resourceValues(
                forKeys: [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey])
            let exists = vals?.isRegularFile ?? false
            let diskMtime = vals?.contentModificationDate?.timeIntervalSince1970 ?? 0
            let diskSize = vals?.fileSize ?? 0
            var row: [String: Any] = [
                "path": p,
                "indexed": true,
                // False when the indexed file has since been deleted (or replaced by a non-file).
                "exists": exists,
                "kind": st.kind,
                "chunk_count": st.chunkCount,
                // mtime + byte size of the file VERSION the index holds.
                "modified": st.modified,
                "bytes": st.size,
                "up_to_date": exists && diskMtime == st.modified && diskSize == st.size
            ]
            // When the indexer last wrote these rows; absent if the rows predate the
            // indexed_at column (they get a real stamp on their next reindex).
            if st.indexedAt > 0 { row["indexed_at"] = st.indexedAt }
            return row
        }
    }

    /// Lexical-only cleanup to the store's path identity (the crawler's verbatim absolute paths):
    /// tilde expansion, duplicate/trailing slashes removed, "." and ".." collapsed - WITHOUT
    /// consulting the filesystem. NSString.standardizingPath is deliberately NOT used: it rewrites
    /// "/private/tmp/..." to "/tmp/..." (a symlink-derived transform), while the store keys
    /// /private-form paths verbatim (AppModel.canonicalizeRoots converts /tmp-style roots INTO
    /// /private form), so a stored path fed back in would stop matching. The index never resolves
    /// symlinks, so neither does this.
    private static func normalize(_ raw: String) -> String {
        let expanded = (raw as NSString).expandingTildeInPath
        var parts: [String] = []
        for comp in expanded.split(separator: "/") {
            if comp == "." { continue }
            if comp == ".." { if !parts.isEmpty { parts.removeLast() }; continue }
            parts.append(String(comp))
        }
        return "/" + parts.joined(separator: "/")
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
