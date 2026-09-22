import Foundation
import OmniKit

// The HTTP faces of OCRServing. Two shapes, because OCR has two kinds of caller:
//
// - POST /v1/chat/completions, OpenAI's chat shape. Every OpenAI SDK, and every tool built on one,
//   already speaks it, and it is how OCR vision models are commonly served (vLLM and friends). It
//   streams. The reply is one message, so a multi-page document comes back as one Markdown text
//   with the workspace's page rule between pages.
// - POST /v1/ocr, Mistral's OCR shape: a document in, a list of pages out, with page selection.
//   The shape for a caller that wants page N of a 300-page scan, or wants to know where pages end.
//
// jina-ocr-v1 is trained with one fixed instruction and does not act on others (see
// OCRSession.Settings.prompt), so the text of a chat message is not a prompt here: it is ignored,
// and the attachments are what get transcribed.

private func ocrError(_ f: OCRServing.Failure) -> HTTPResponse {
    let type: String
    switch f {
    case .badInput: type = "invalid_request_error"
    case .notInstalled, .busy: type = "service_unavailable"
    case .failed: type = "server_error"
    }
    var r = HTTPResponse.json(["error": ["message": f.message, "type": type]], status: f.status)
    if case .busy = f { r.headers["Retry-After"] = "30" }
    return r
}

private func jsonBody(_ req: HTTPRequest) -> [String: Any]? {
    guard !req.body.isEmpty, let obj = try? JSONSerialization.jsonObject(with: req.body) else { return nil }
    return obj as? [String: Any]
}

/// A document named by URL, the way both shapes name one: a `data:` URI (inline bytes), a
/// `file://` URL or absolute path (inside the indexed folders), or bare base64.
func ocrDocument(url raw: String) -> Result<OCRServing.Document, OCRServing.Failure> {
    let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if s.hasPrefix("data:") { return OCRServing.open(inline: s) }
    if s.hasPrefix("file://") || s.hasPrefix("/") || s.hasPrefix("~") { return OCRServing.open(path: s) }
    if s.hasPrefix("http://") || s.hasPrefix("https://") {
        return .failure(.badInput("remote URLs are not fetched - Omni runs offline; send the bytes as a data: URI"))
    }
    return OCRServing.open(inline: s)
}

/// Sum of a job's page counters, for `usage`.
private func tokenTotals(_ pages: [OCRServing.Page]) -> (prompt: Int, completion: Int) {
    pages.reduce((0, 0)) { ($0.0 + $1.promptTokens, $0.1 + $1.tokens) }
}

// MARK: - OpenAI chat completions

enum ChatOCRAdapter {
    static func handle(_ req: HTTPRequest) async -> HTTPResponse {
        guard let body = jsonBody(req) else { return ocrError(.badInput("invalid JSON body")) }
        guard let messages = body["messages"] as? [[String: Any]], !messages.isEmpty else {
            return ocrError(.badInput("'messages' is required"))
        }
        if let n = body["n"] as? Int, n != 1 { return ocrError(.badInput("'n' must be 1")) }
        let stream = (body["stream"] as? Bool) ?? false
        let includeUsage = ((body["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false

        guard let user = messages.last(where: { ($0["role"] as? String) == "user" }) else {
            return ocrError(.badInput("no user message"))
        }
        var documents: [OCRServing.Document] = []
        for part in (user["content"] as? [[String: Any]]) ?? [] {
            let type = part["type"] as? String ?? ""
            let url: String?
            switch type {
            case "text", "input_text":
                continue   // not a prompt for this model - see the note at the top of the file
            case "image_url":
                url = (part["image_url"] as? [String: Any])?["url"] as? String ?? part["image_url"] as? String
            case "input_image":
                url = part["image_url"] as? String
            case "file":
                let file = part["file"] as? [String: Any] ?? [:]
                if file["file_id"] != nil {
                    return ocrError(.badInput("file_id is not supported; send file_data as a data: URI or base64"))
                }
                url = file["file_data"] as? String
            default:
                return ocrError(.badInput("unsupported content part type '\(type)'; use image_url or file"))
            }
            guard let url, !url.isEmpty else { return ocrError(.badInput("a '\(type)' part carries no data")) }
            switch ocrDocument(url: url) {
            case .success(let d): documents.append(d)
            case .failure(let f): return ocrError(f)
            }
        }
        guard !documents.isEmpty else {
            return ocrError(.badInput("the last user message carries no image or document: attach one as an image_url part (a data: URI, or a file:// path inside the indexed folders) or a file part with file_data"))
        }
        let total = documents.reduce(0) { $0 + $1.pageCount }
        guard total <= OCRServing.maxPagesHTTP else {
            return ocrError(.badInput("\(total) pages in one message; the limit is \(OCRServing.maxPagesHTTP). Use /v1/ocr with 'pages' to take a long document in parts"))
        }

        let job: OCRServing.Job
        switch await OCRServing.prepare(documents.map { ($0, Array(0 ..< $0.pageCount)) }) {
        case .success(let j): job = j
        case .failure(let f): return ocrError(f)
        }
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let created = Int(Date().timeIntervalSince1970)
        let model = job.modelID

        if !stream {
            let pages: [OCRServing.Page]
            switch await job.run() {
            case .success(let p): pages = p
            case .failure(let f): return ocrError(f)
            }
            let (prompt, completion) = tokenTotals(pages)
            return HTTPResponse.json([
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [[
                    "index": 0,
                    "message": ["role": "assistant",
                                "content": pages.map(\.markdown).joined(separator: OCRServing.pageSeparator)],
                    "finish_reason": finishReason(pages)
                ]],
                "usage": ["prompt_tokens": prompt, "completion_tokens": completion,
                          "total_tokens": prompt + completion]
            ])
        }

        return HTTPResponse.eventStream { send in
            func event(_ obj: [String: Any]) async -> Bool {
                guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return true }
                var line = Data("data: ".utf8); line.append(data); line.append(contentsOf: Array("\n\n".utf8))
                return await send(line)
            }
            func chunk(_ delta: [String: Any], finish: String? = nil) -> [String: Any] {
                ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [["index": 0, "delta": delta, "finish_reason": finish.map { $0 as Any } ?? NSNull()]]]
            }

            let gone = OCRFlag()
            let (pieces, sink) = AsyncStream<String>.makeStream()
            let ordered = OCROrderedText(count: job.pages.count) { sink.yield($0) }
            if !(await event(chunk(["role": "assistant", "content": ""]))) { gone.set() }
            let runner = Task {
                let r = await job.run(onText: { ordered.update($0, $1) },
                                      onDone: { ordered.finish($0, $1.markdown) },
                                      shouldContinue: { !gone.isSet })
                sink.finish()
                return r
            }
            // Drained to the end even after the client leaves, so the decode sees `gone` and stops
            // and the job gives its slot back.
            for await piece in pieces where !gone.isSet {
                if !(await event(chunk(["content": piece]))) { gone.set() }
            }
            let result = await runner.value
            guard !gone.isSet else { return }
            switch result {
            case .success(let pages):
                _ = await event(chunk([:], finish: finishReason(pages)))
                if includeUsage {
                    let (prompt, completion) = tokenTotals(pages)
                    _ = await event(["id": id, "object": "chat.completion.chunk", "created": created,
                                     "model": model, "choices": [] as [Any],
                                     "usage": ["prompt_tokens": prompt, "completion_tokens": completion,
                                               "total_tokens": prompt + completion]])
                }
            case .failure(let f):
                _ = await event(["error": ["message": f.message, "type": "server_error"]])
            }
            _ = await send(Data("data: [DONE]\n\n".utf8))
        }
    }

    /// "length" when any page hit the token budget, which is the one way a page comes back short.
    private static func finishReason(_ pages: [OCRServing.Page]) -> String {
        pages.contains { $0.stop == OCRModel.StopReason.cap.rawValue } ? "length" : "stop"
    }
}

// MARK: - Mistral OCR

enum DocumentOCRAdapter {
    static func handle(_ req: HTTPRequest) async -> HTTPResponse {
        guard let body = jsonBody(req) else { return ocrError(.badInput("invalid JSON body")) }

        // Mistral's `document` chunk, or Omni's `path` shorthand for a file in the indexed folders.
        var url: String?
        if let doc = body["document"] as? [String: Any] {
            switch doc["type"] as? String ?? (doc["document_url"] != nil ? "document_url" : "image_url") {
            case "document_url":
                url = doc["document_url"] as? String
            case "image_url":
                url = (doc["image_url"] as? [String: Any])?["url"] as? String ?? doc["image_url"] as? String
            case "file":
                return ocrError(.badInput("file_id is not supported; send a document_url with a data: URI"))
            case let other:
                return ocrError(.badInput("unsupported document type '\(other)'; use document_url or image_url"))
            }
        } else if let s = body["document"] as? String {
            url = s
        } else if let p = body["path"] as? String {
            url = p
        }
        guard let url, !url.isEmpty else {
            return ocrError(.badInput("'document' is required: {\"type\": \"document_url\", \"document_url\": \"data:application/pdf;base64,...\"}, or 'path' for a file inside the indexed folders"))
        }
        let document: OCRServing.Document
        switch ocrDocument(url: url) {
        case .success(let d): document = d
        case .failure(let f): return ocrError(f)
        }
        let selected: [Int]
        switch OCRServing.parsePages(body["pages"], count: document.pageCount, oneBased: false) {
        case .success(let p): selected = p ?? Array(0 ..< document.pageCount)
        case .failure(let f): return ocrError(f)
        }
        guard selected.count <= OCRServing.maxPagesHTTP else {
            return ocrError(.badInput("\(selected.count) pages requested; the limit is \(OCRServing.maxPagesHTTP) per request. Pass 'pages', such as \"0-\(OCRServing.maxPagesHTTP - 1)\", and continue from there"))
        }

        let job: OCRServing.Job
        switch await OCRServing.prepare([(document, selected)]) {
        case .success(let j): job = j
        case .failure(let f): return ocrError(f)
        }
        let pages: [OCRServing.Page]
        switch await job.run() {
        case .success(let p): pages = p
        case .failure(let f): return ocrError(f)
        }
        let rows: [[String: Any]] = pages.map { p in
            [
                "index": p.index,
                "markdown": p.markdown,
                "images": [] as [Any],
                // A cached page was not rendered for this request, so there is nothing to measure.
                "dimensions": p.width > 0 ? ["dpi": p.dpi, "height": p.height, "width": p.width] : NSNull()
            ]
        }
        return HTTPResponse.json([
            "pages": rows,
            "model": job.modelID,
            "usage_info": ["pages_processed": pages.count, "doc_size_bytes": document.bytes]
        ])
    }
}

/// A one-way latch readable from any thread.
final class OCRFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}
