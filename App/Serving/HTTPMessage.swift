import Foundation

// Pure value types + a hand-rolled HTTP/1.1 parser. No I/O here, so this file is
// unit-testable in isolation. HTTPServer feeds raw bytes in; gets requests out.

enum HTTPError: Error {
    case badRequest
    case payloadTooLarge
}

struct HTTPRequest {
    /// Uppercased HTTP method (GET, POST, ...).
    let method: String
    /// Raw request target including any query string, e.g. "/v1/search?key=abc".
    let path: String
    /// `path` with the query string stripped, e.g. "/v1/search".
    let routePath: String
    /// Parsed query parameters (percent-decoded best-effort).
    let query: [String: String]
    /// Header keys are lowercased; values are trimmed.
    let headers: [String: String]
    /// Request body bytes (length == contentLength once fully buffered).
    var body: Data

    var contentLength: Int { Int(headers["content-length"] ?? "") ?? 0 }

    /// HTTP/1.1 defaults to keep-alive unless Connection: close is sent.
    var wantsKeepAlive: Bool { (headers["connection"]?.lowercased() ?? "") != "close" }

    /// Bearer token from Authorization header, case-insensitive scheme match.
    var bearer: String? {
        guard let raw = headers["authorization"] else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        let lower = trimmed.lowercased()
        let prefix = "bearer "
        guard lower.hasPrefix(prefix) else { return nil }
        let token = String(trimmed.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        return token.isEmpty ? nil : token
    }

    /// Google-style API key header (Gemini).
    var googApiKey: String? { headers["x-goog-api-key"] }
}

enum HTTPParse {
    /// Try to parse one full request from the head of `buf`.
    /// Returns (request, bytesConsumed) once the head and the Content-Length body are
    /// fully buffered; returns nil to signal "read more bytes"; throws HTTPError.badRequest
    /// on a malformed head, non-UTF8 head, or a chunked transfer encoding (unsupported).
    static func tryParse(_ buf: Data) throws -> (HTTPRequest, Int)? {
        // Find the CRLFCRLF that ends the head.
        guard let headerEnd = rangeOfHeaderTerminator(in: buf) else { return nil }
        let headData = buf.subdata(in: buf.startIndex..<headerEnd.lowerBound)
        guard let headString = String(data: headData, encoding: .utf8) else {
            throw HTTPError.badRequest
        }

        // Split head into lines on CRLF (tolerate a bare LF too).
        let lines = headString.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard let requestLine = lines.first, !requestLine.isEmpty else { throw HTTPError.badRequest }

        // Request line: METHOD SP TARGET SP VERSION  (we accept >= 2 parts).
        let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard parts.count >= 2 else { throw HTTPError.badRequest }
        let method = parts[0].uppercased()
        let target = parts[1]

        // Headers: everything after the request line, first ":" splits name/value.
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            if line.isEmpty { continue }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            if name.isEmpty { continue }
            headers[name] = value
        }

        // Reject chunked bodies: we only support Content-Length framing.
        if let te = headers["transfer-encoding"], !te.isEmpty {
            throw HTTPError.badRequest
        }

        let contentLength = Int(headers["content-length"] ?? "") ?? 0
        if contentLength < 0 { throw HTTPError.badRequest }

        let bodyStart = headerEnd.upperBound
        let available = buf.distance(from: bodyStart, to: buf.endIndex)
        if available < contentLength { return nil } // need more bytes

        let bodyEnd = buf.index(bodyStart, offsetBy: contentLength)
        let body = buf.subdata(in: bodyStart..<bodyEnd)
        let consumed = buf.distance(from: buf.startIndex, to: bodyEnd)

        let (routePath, query) = splitTarget(target)
        let req = HTTPRequest(
            method: method,
            path: target,
            routePath: routePath,
            query: query,
            headers: headers,
            body: body
        )
        return (req, consumed)
    }

    /// Locate the "\r\n\r\n" head terminator. Returns the Range covering those 4 bytes.
    private static func rangeOfHeaderTerminator(in buf: Data) -> Range<Data.Index>? {
        let terminator = Data([0x0d, 0x0a, 0x0d, 0x0a]) // \r\n\r\n
        return buf.range(of: terminator)
    }

    private static func splitTarget(_ target: String) -> (String, [String: String]) {
        guard let q = target.firstIndex(of: "?") else { return (target, [:]) }
        let routePath = String(target[target.startIndex..<q])
        let queryString = String(target[target.index(after: q)...])
        var query: [String: String] = [:]
        for pair in queryString.split(separator: "&", omittingEmptySubsequences: true) {
            let kv = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = decodeURLComponent(String(kv[0]))
            let value = kv.count > 1 ? decodeURLComponent(String(kv[1])) : ""
            if !key.isEmpty { query[key] = value }
        }
        return (routePath, query)
    }

    private static func decodeURLComponent(_ s: String) -> String {
        s.replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? s
    }
}

struct HTTPResponse {
    var status: Int
    var headers: [String: String]
    var body: Data

    init(status: Int, headers: [String: String] = [:], body: Data = Data()) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    /// Build a JSON response from a JSONSerialization-compatible object.
    static func json(_ obj: Any, status: Int = 200) -> HTTPResponse {
        let data: Data
        if let d = try? JSONSerialization.data(withJSONObject: obj, options: []) {
            data = d
        } else {
            data = Data("{\"error\":\"serialization failure\"}".utf8)
        }
        return HTTPResponse(
            status: status,
            headers: ["Content-Type": "application/json"],
            body: data
        )
    }

    /// Serialize to wire bytes: status line + framing headers + body.
    func serialize(keepAlive: Bool) -> Data {
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        var h = headers
        if h["Content-Type"] == nil { h["Content-Type"] = "application/json" }
        h["Content-Length"] = String(body.count)
        h["Connection"] = keepAlive ? "keep-alive" : "close"
        h["Date"] = Self.httpDate()
        for (k, v) in h {
            head += "\(k): \(v)\r\n"
        }
        head += "\r\n"
        var out = Data(head.utf8)
        out.append(body)
        return out
    }

    static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 413: return "Payload Too Large"
        case 500: return "Internal Server Error"
        default: return "OK"
        }
    }

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "GMT")
        f.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        return f
    }()

    private static func httpDate() -> String {
        dateFormatter.string(from: Date())
    }
}
