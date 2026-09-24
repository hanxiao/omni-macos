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
/// Line format: `2026-09-23 19:03:07.123  INFO   POST /mcp  200  156 ms  127.0.0.1`.
/// The level is the second column and fixed width, which is all the tab's coloring reads.
final class ServingLogFile: @unchecked Sendable {
    enum Level: String { case info = "INFO", warn = "WARN", error = "ERROR" }

    static let url: URL = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Logs/Omni/serving.log")

    /// Rotated at open past this size: one previous file (`serving.log.1`) is kept.
    private static let rotateBytes = 8 << 20
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

    static func line(for e: LogEntry) -> String {
        let level: Level = e.status >= 500 ? .error : e.status >= 400 ? .warn : .info
        return line(level, "\(e.method) \(e.path)  \(e.status)  \(String(format: "%.0f ms", e.ms))  \(e.client)",
                    at: e.time)
    }

    /// The level column of a line written by `line(_:_:)`, or nil for anything else.
    static func level(of line: Substring) -> Level? {
        let parts = line.split(separator: " ", maxSplits: 3, omittingEmptySubsequences: true)
        guard parts.count >= 3 else { return nil }
        return Level(rawValue: String(parts[2]))
    }

    /// The last `count` lines already on disk, oldest first. Read once at launch, so the tab
    /// shows the previous session's tail instead of an empty box.
    static func tail(_ count: Int) -> [String] {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? fh.close() }
        let size = (try? fh.seekToEnd()) ?? 0
        let window: UInt64 = 64 << 10
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
