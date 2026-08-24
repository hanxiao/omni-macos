import Foundation
import ImageIO
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

/// Lexical-only cleanup to the store's path identity (the crawler's verbatim absolute paths):
/// tilde expansion, duplicate/trailing slashes removed, "." and ".." collapsed - WITHOUT
/// consulting the filesystem. NSString.standardizingPath is deliberately NOT used: it rewrites
/// "/private/tmp/..." to "/tmp/..." (a symlink-derived transform), while the store keys
/// /private-form paths verbatim (AppModel.canonicalizeRoots converts /tmp-style roots INTO
/// /private form), so a stored path fed back in would stop matching. The index never resolves
/// symlinks, so neither does this.
func normalizeStorePath(_ raw: String) -> String {
    let expanded = (raw as NSString).expandingTildeInPath
    var parts: [String] = []
    for comp in expanded.split(separator: "/") {
        if comp == "." { continue }
        if comp == ".." { if !parts.isEmpty { parts.removeLast() }; continue }
        parts.append(String(comp))
    }
    return "/" + parts.joined(separator: "/")
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

        // Duplicate collapsing, on by default: copies of one file waste an agent's top-k and its
        // context exactly as they waste a human's screen. Pass "group_duplicates": false for the
        // flat, pre-grouping list. When on, the search over-fetches so collapsing cannot leave the
        // caller with fewer distinct files than it asked for.
        let group = (body["group_duplicates"] as? Bool) ?? true
        let fetch = group ? min(topK * 3, 300) : topK
        let hits = backend.search(query, topK: fetch, filter: filter)
        let groups = backend.groupedResults(hits, enabled: group, limit: topK)
        // Lockstep rule as duplicateChunks: only trust a key whose modified matches the hit's.
        let contentKeys = backend.contentKeys(paths: groups.map { $0.representative.path })
        let results: [[String: Any]] = groups.map { g in
            let hit = g.representative
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
            // What this row stands for. Present only on a stack, so a flat result set is byte-for-byte
            // what it was before grouping existed. "exact" = byte-identical (same content key AND
            // size); "near" = same kind and extension, sizes within 10%, cosine >= 0.98.
            if g.isStack {
                row["duplicate_count"] = g.count
                row["duplicates"] = g.members.dropFirst().map(\.path)
                row["duplicate_kind"] = g.reason == .exact ? "exact" : "near"
            }
            return row
        }
        return HTTPResponse.json(["query": query, "results": results,
                                  "grouped": group])
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
            //
            // A Photos asset has no file to stat, and the same two numbers come from PhotoKit
            // instead - the indexer compares exactly these (asset modification date, pixel count),
            // so "up to date" means here what it means there. Reporting exists:false for one, as a
            // failed stat would, is simply wrong: the photo is right where the index says it is.
            let exists: Bool, diskMtime: Double, diskSize: Int
            if let ref = PhotoLibrary.Ref(p) {
                let info = PhotoLibrary.assetSignature(ref)
                exists = info != nil
                diskMtime = info?.modified ?? 0
                diskSize = info?.size ?? 0
            } else {
                let vals = try? URL(fileURLWithPath: p).resourceValues(
                    forKeys: [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey])
                exists = vals?.isRegularFile ?? false
                diskMtime = vals?.contentModificationDate?.timeIntervalSince1970 ?? 0
                diskSize = vals?.fileSize ?? 0
            }
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

    /// Local alias; the definition is `normalizeStorePath`, at file scope so every path-taking
    /// adapter (status, tags) resolves to the same store identity.
    static func normalize(_ raw: String) -> String { normalizeStorePath(raw) }
}

// MARK: - Tags

/// Longest edge an incoming image is decoded to before patchifying. Matches
/// IndexSettings.maxImageDimension, so a tag computed here is the tag the indexer would store.
private let tagMaxImageDimension = 1568

/// A path argument is honored only when it sits inside one of the user's indexed roots.
///
/// Without this an authenticated LAN caller could tag ANY image on the Mac, which is a far larger
/// disclosure than the existence probe /v1/files/status already refuses to be (see the "NO disk
/// probe" note in FileStatusAdapter.rows). Inside a root the content is already reachable through
/// /v1/search, so tagging adds no new exposure. Read per request rather than snapshotted at attach
/// time: ServingController.attach runs once at engine load (AppModel.swift), and the user can add
/// or remove roots at any point afterwards.
private func pathIsInIndexedRoot(_ path: String) -> Bool {
    let roots = (UserDefaults.standard.array(forKey: "omni.roots") as? [String]) ?? []
    for r in roots {
        let root = normalizeStorePath(r)
        if path == root || path.hasPrefix(root.hasSuffix("/") ? root : root + "/") { return true }
    }
    return false
}

/// Decode one request image to a CGImage, downscaled the way the indexer does.
/// Accepts a filesystem path (root-gated by the caller) or inline bytes as base64 / a data: URI.
private func decodeRequestImage(path: String?, base64: String?) -> CGImage? {
    var source: CGImageSource?
    if let path {
        source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)
    } else if var b64 = base64 {
        // "data:image/png;base64,AAAA..." - keep only the payload.
        if b64.hasPrefix("data:"), let comma = b64.firstIndex(of: ",") { b64 = String(b64[b64.index(after: comma)...]) }
        guard let data = Data(base64Encoded: b64, options: [.ignoreUnknownCharacters]) else { return nil }
        source = CGImageSourceCreateWithData(data as CFData, nil)
    }
    guard let src = source else { return nil }
    // Thumbnail decode rather than full decode + resize: one pass, honors EXIF orientation, and
    // never materializes a 100MP buffer for a photo we are about to shrink anyway.
    let opts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: tagMaxImageDimension
    ]
    return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
}

/// POST /v1/files/tags: the tags Omni ALREADY generated for the given indexed paths - the same
/// strings the `tag:` search qualifier matches on. Read-only index metadata: nothing is embedded,
/// the GPU is never touched, and indexing/search speed is unaffected. Use /v1/tag to tag an image
/// that is not indexed (or to re-tag one without writing to the index).
enum FileTagsAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }
        var paths: [String]
        if let raw = body["paths"] {
            guard let arr = raw as? [String] else { return badRequest("'paths' must be an array of strings") }
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

    /// One row per input path, input order preserved. Shared with the MCP tag tool.
    static func rows(for paths: [String], backend: any ServingBackend) -> [[String: Any]] {
        let normalized = paths.map(normalizeStorePath)
        let tags = backend.storedTags(paths: normalized)
        let status = backend.fileStatus(paths: normalized)

        return normalized.map { p in
            // Absent from storedTags means "no indexed media rows for this path". Distinguish the
            // two reasons using the status lookup we already have, so a caller can tell a text
            // file (never tagged, by design) from a path Omni has never seen. No disk probe here
            // either, for the same reason FileStatusAdapter refuses one.
            guard let t = tags[p] else {
                guard let st = status[p] else { return ["path": p, "indexed": false, "tags": []] }
                return ["path": p, "indexed": true, "kind": st.kind, "taggable": false, "tags": []]
            }
            return [
                "path": p,
                "indexed": true,
                "kind": status[p]?.kind ?? "",
                "taggable": true,
                // Empty means indexed media that carries no tags yet: tagging was off, the label
                // cache was still building, or the forward returned nothing usable.
                "tags": t
            ]
        }
    }
}

/// POST /v1/tag: compute open-vocabulary tags for an image, on demand.
///
/// Input is either {"path": "/abs/file.jpg"} (must sit inside an indexed root) or
/// {"image": "<base64 | data: URI>"}; "paths"/"images" arrays are accepted for batches. Runs the
/// vision tower + tagger on the GPU at INDEXING priority, so an interactive search still preempts
/// it. The result is ephemeral: nothing is written to the index, and - unlike the indexer's own
/// tagging - the image is never folded into the corpus prior (see OmniTagger.finalize).
enum TagAdapter {
    static func handle(_ req: HTTPRequest, _ backend: any ServingBackend) -> HTTPResponse {
        guard let body = JSONBody.object(req) else { return badRequest("invalid JSON body") }

        var paths: [String] = []
        if let raw = body["paths"] {
            guard let arr = raw as? [String] else { return badRequest("'paths' must be an array of strings") }
            paths = arr.filter { !$0.isEmpty }
        } else if let one = body["path"] as? String, !one.isEmpty {
            paths = [one]
        }
        var images: [String] = []
        if let raw = body["images"] {
            guard let arr = raw as? [String] else { return badRequest("'images' must be an array of base64 strings") }
            images = arr.filter { !$0.isEmpty }
        } else if let one = body["image"] as? String, !one.isEmpty {
            images = [one]
        }
        if paths.isEmpty && images.isEmpty {
            return badRequest("'path' (inside an indexed folder) or 'image' (base64) is required")
        }
        let hqRequested = (body["hq"] as? Bool) ?? true
        let maxImages = hqRequested ? tagMaxImagesHQ : tagMaxImages
        if paths.count + images.count > maxImages {
            return badRequest("at most \(maxImages) images per request\(hqRequested ? " in hq mode (each image is 6 forwards; pass \"hq\": false for \(tagMaxImages))" : "")")
        }

        var topK = (body["top_k"] as? Int) ?? OmniTagger.topK
        topK = max(1, min(topK, 25))

        // HQ (CWR multi-crop) refinement is ON by default so this endpoint returns the SAME tags
        // the indexer would store for the image (Indexer.swift cuts the same 5 crops on its retag
        // pass). Base quality is a different answer, not just a faster one - a small object that
        // only becomes legible inside a crop is missed without it. "hq": false opts out: 1 forward
        // per image instead of 6, at the cost of disagreeing with the stored tags.
        let hq = (body["hq"] as? Bool) ?? true

        // Availability is a server state, not a bad request: vision may be unloaded, tagging may be
        // switched off, or the label cache may still be building on first launch. 503 tells a
        // client to retry rather than to fix its request.
        guard backend.canTag else {
            return HTTPResponse.json([
                "error": ["message": "image tagging is not available (tagging disabled, or the label cache is still building)",
                          "type": "service_unavailable"]
            ], status: 503)
        }

        // Decode everything BEFORE taking the GPU: preprocessRaw saturates every core via
        // concurrentPerform, and holding the engine gate across it would stall search for no reason.
        var raws: [OmniVisionPreprocess.RawPatches] = []
        var crops: [[OmniVisionPreprocess.RawPatches]] = []
        var labels: [String] = []
        // The same 5-crop CWR geometry the indexer cuts, so HQ tags here match HQ tags there.
        func cutCrops(_ cg: CGImage) -> [OmniVisionPreprocess.RawPatches] {
            guard hq else { return [] }
            return OmniTagger.cwrCropRects(width: cg.width, height: cg.height)
                .compactMap { cg.cropping(to: $0) }
                .map { OmniVisionPreprocess.preprocessRaw($0) }
        }
        for p in paths {
            let np = normalizeStorePath(p)
            guard pathIsInIndexedRoot(np) else {
                return badRequest("'\(np)' is outside the indexed folders; pass the image inline as 'image' instead")
            }
            guard let cg = decodeRequestImage(path: np, base64: nil) else {
                return badRequest("could not decode an image at '\(np)'")
            }
            raws.append(OmniVisionPreprocess.preprocessRaw(cg))
            crops.append(cutCrops(cg))
            labels.append(np)
        }
        for (i, b64) in images.enumerated() {
            guard let cg = decodeRequestImage(path: nil, base64: b64) else {
                return badRequest("'image[\(i)]' is not decodable base64 image data")
            }
            raws.append(OmniVisionPreprocess.preprocessRaw(cg))
            crops.append(cutCrops(cg))
            labels.append("")
        }

        guard let tagged = backend.tagImages(raws, crops: crops, topK: topK) else {
            return HTTPResponse.json([
                "error": ["message": "tagging failed (the vision model is unavailable)", "type": "service_unavailable"]
            ], status: 503)
        }

        let results: [[String: Any]] = tagged.enumerated().map { i, tags in
            var row: [String: Any] = ["tags": tags]
            if i < labels.count, !labels[i].isEmpty { row["path"] = labels[i] }
            return row
        }
        return HTTPResponse.json(["hq": hq, "results": results])
    }

    /// Recompute tags for indexed-root paths, shaped like FileTagsAdapter.rows so the MCP tag tool
    /// can render stored and recomputed answers with one code path. Always HQ, matching the tags
    /// the indexer would store. nil when tagging is unavailable (the caller reports that once);
    /// a per-path problem lands in that row's "error" instead of failing the whole call.
    static func computeRows(paths: [String], topK: Int, backend: any ServingBackend) -> [[String: Any]]? {
        guard backend.canTag else { return nil }
        var raws: [OmniVisionPreprocess.RawPatches] = []
        var crops: [[OmniVisionPreprocess.RawPatches]] = []
        var ok: [String] = []
        var rows: [String: [String: Any]] = [:]
        for raw in paths {
            let p = normalizeStorePath(raw)
            guard pathIsInIndexedRoot(p) else {
                rows[p] = ["path": p, "tags": [], "error": "outside the indexed folders"]
                continue
            }
            guard let cg = decodeRequestImage(path: p, base64: nil) else {
                rows[p] = ["path": p, "tags": [], "error": "not a decodable image"]
                continue
            }
            raws.append(OmniVisionPreprocess.preprocessRaw(cg))
            crops.append(OmniTagger.cwrCropRects(width: cg.width, height: cg.height)
                .compactMap { cg.cropping(to: $0) }
                .map { OmniVisionPreprocess.preprocessRaw($0) })
            ok.append(p)
        }
        if !raws.isEmpty {
            guard let tagged = backend.tagImages(raws, crops: crops, topK: topK) else { return nil }
            for (i, p) in ok.enumerated() where i < tagged.count {
                rows[p] = ["path": p, "tags": tagged[i], "taggable": true, "recomputed": true]
            }
        }
        return paths.map { rows[normalizeStorePath($0)] ?? ["path": normalizeStorePath($0), "tags": []] }
    }
}

/// Far below servingMaxInputs: each image is a full vision-tower forward (tens of ms even on this
/// hardware) and the whole batch is decoded into memory up front, so the text-batch cap would let
/// one request hold the GPU for minutes. The 8 MB request-body cap (HTTPMessage.maxBody) is the
/// other practical limit on inline images: base64 inflates by 4/3, so ~5.9 MB of source bytes.
private let tagMaxImages = 16
/// HQ scores 5 crops on top of the image, one gate hold each, and holds all six raw pixel buffers
/// at once - a real memory spike on a low-RAM Mac. Cap the default (HQ) mode far lower.
private let tagMaxImagesHQ = 4

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
