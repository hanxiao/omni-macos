import Foundation

/// One serviced request, as the HTTP server reports it. Sendable so it can cross from the network
/// queue to the main actor.
struct LogEntry: Identifiable, Sendable {
    let id = UUID()
    let time: Date
    let method: String
    let path: String
    let status: Int
    let ms: Double
    let client: String
    /// The request target with its query string, any `key=` token masked.
    var target = ""
    var userAgent = ""
    /// Request and response bodies as one line each: see `summarize`.
    var request = ""
    var response = ""

    /// Characters of a body kept in the log. Enough for a whole search or MCP call and the start
    /// of its results; an OCR upload or a long result list is cut, with its full size noted.
    static let bodyLimit = 4096

    /// A body as a single log line: whitespace runs collapsed, base64 runs (images, audio, PDFs
    /// sent inline) replaced by their size, cut at `bodyLimit` with the full size noted. A body
    /// that is not text is logged as its type and size only.
    static func summarize(_ body: Data, type: String?, total: Int? = nil) -> String {
        let size = total ?? body.count
        guard size > 0 else { return "" }
        let isText = type.map { $0.contains("json") || $0.contains("text") || $0.contains("x-www-form") } ?? true
        guard isText else { return "<\(type ?? "binary") \(bytes(size))>" }
        // Decoding repairs a character the cut split in half, rather than failing the whole body.
        var text = String(decoding: body.prefix(bodyLimit * 4), as: UTF8.self)
        text = text.replacingOccurrences(of: #"[A-Za-z0-9+/=_-]{256,}"#, with: "<base64>", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        let cut = text.count > bodyLimit || size > body.count
        if text.count > bodyLimit { text = String(text.prefix(bodyLimit)) }
        return cut ? "\(text) ... (\(bytes(size)))" : text
    }

    static func redact(_ target: String) -> String {
        target.replacingOccurrences(of: #"([?&]key=)[^&]*"#, with: "$1***", options: .regularExpression)
    }

    private static func bytes(_ n: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(n), countStyle: .file)
    }
}

/// Where the server binds. Persisted as its rawValue string in UserDefaults.
enum ServingScope: String, Sendable {
    case local
    case `public`
}

/// Serving's log: one plain line per event, appended to ~/Library/Logs/Omni/serving.log so it
/// outlives the app and can be grepped, tailed or attached to a report. The Serving tab shows the
/// tail of the same lines, so what the user copies from the tab matches the file byte for byte.
///
/// Line format: `2026-09-23 19:03:07.123  INFO   POST /mcp  200  156 ms  127.0.0.1  ...`, then the
/// user agent, `> request body` and `< response body`.
/// The level is the second column and fixed width, which is all the tab's coloring reads.
final class ServingLogFile: @unchecked Sendable {
    enum Level: String { case info = "INFO", warn = "WARN", error = "ERROR" }

    static let url: URL = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Logs/Omni/serving.log")

    /// Rotated at open past this size: one previous file (`serving.log.1`) is kept.
    private static let rotateBytes = 32 << 20
    private let queue = DispatchQueue(label: "omni.serving.log", qos: .utility)
    private var handle: FileHandle?

    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    static func line(_ level: Level, _ message: String, at time: Date = Date()) -> String {
        let pad = String(repeating: " ", count: 6 - level.rawValue.count)
        return "\(stamp.string(from: time))  \(level.rawValue)\(pad) \(message)"
    }

    /// `POST /v1/search  200  37 ms  ::1  curl/8.7.1  > {"query":"invoice"}  < {"results":[...]}`
    static func line(for e: LogEntry) -> String {
        let level: Level = e.status >= 500 ? .error : e.status >= 400 ? .warn : .info
        var parts = ["\(e.method) \(e.target.isEmpty ? e.path : e.target)", "\(e.status)",
                     String(format: "%.0f ms", e.ms), e.client]
        if !e.userAgent.isEmpty { parts.append(e.userAgent) }
        if !e.request.isEmpty { parts.append("> " + e.request) }
        if !e.response.isEmpty { parts.append("< " + e.response) }
        return line(level, parts.joined(separator: "  "), at: e.time)
    }

    /// The level column of a line written by `line(_:_:)`, or nil for anything else.
    static func level(of line: Substring) -> Level? {
        let parts = line.split(separator: " ", maxSplits: 3, omittingEmptySubsequences: true)
        guard parts.count >= 3 else { return nil }
        return Level(rawValue: String(parts[2]))
    }

    /// The last `count` lines on disk, oldest first: what the Serving tab shows.
    static func tail(_ count: Int) -> [String] {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? fh.close() }
        let size = (try? fh.seekToEnd()) ?? 0
        let window: UInt64 = 1 << 20   // 100 lines of up to ~8 KB each
        try? fh.seek(toOffset: size > window ? size - window : 0)
        guard let data = try? fh.readToEnd(), let text = String(data: data, encoding: .utf8) else { return [] }
        var lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        if size > window, !lines.isEmpty { lines.removeFirst() }   // the window cut it mid-line
        return Array(lines.suffix(count))
    }

    func append(_ line: String) {
        queue.async { [self] in
            if handle == nil { open() }
            try? handle?.write(contentsOf: Data((line + "\n").utf8))
        }
    }

    private func open() {
        let fm = FileManager.default
        try? fm.createDirectory(at: Self.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? fm.attributesOfItem(atPath: Self.url.path))?[.size] as? Int, size > Self.rotateBytes {
            let old = Self.url.appendingPathExtension("1")
            try? fm.removeItem(at: old)
            try? fm.moveItem(at: Self.url, to: old)
        }
        if !fm.fileExists(atPath: Self.url.path) { fm.createFile(atPath: Self.url.path, contents: nil) }
        handle = try? FileHandle(forWritingTo: Self.url)
        _ = try? handle?.seekToEnd()
    }
}
