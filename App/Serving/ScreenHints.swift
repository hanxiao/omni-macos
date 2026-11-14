import Foundation
import OmniKit

/// What a served result says about a screenshot Omni took (docs/screenshots.md): when it was
/// taken, what was in front, and the stretch of work around it. An agent that found a moment by
/// what was on screen can then ask for the files changed in that stretch - `search` with a time
/// range and no query - which is the reason a shot's answer carries a time at all.
///
/// Everything here reads the screen history's own tables (ScreenHistory, its lock) and nothing of
/// the app's state, so it runs on the served work queue with the search it annotates.
///
/// A linked Mac answers with the same thing about its own screenshots (LinkWire.SearchReply), so a
/// moment is plain values that travel: the asking Mac holds no history of another's screen.
enum ScreenHints {
    struct Moment: Codable, Sendable {
        /// The shot's time: its file is dated at it.
        var taken: Double
        var app: String
        var title: String
        /// The stretch of work around it. nil when the history around the shot is gone (cleared, or
        /// pruned ahead of the index).
        var span: Span?
        /// What a served answer names the shot by (`ScreenHistory.link(of:)`). nil from a Mac older
        /// than 0.20.4, whose answers name a shot by its path.
        var link: String?
    }

    struct Span: Codable, Sendable {
        struct App: Codable, Sendable { var name: String; var windows: [String] }
        var start: Double
        var end: Double
        /// Most seen first.
        var apps: [App]

        init(_ s: ScreenHistory.Session) {
            start = s.start; end = s.end
            apps = s.apps.map { App(name: $0.name, windows: $0.titles) }
        }
    }

    static func isShot(_ path: String) -> Bool {
        path.hasPrefix(AppModel.screenshotsDirectory.path + "/")
    }

    /// The moment of each hit that is one of Omni's screenshots, by path. Empty, and the history
    /// never opened, when none is.
    static func moments(_ hits: [SearchHit]) -> [String: Moment] {
        let shots = hits.filter { isShot($0.path) }
        guard !shots.isEmpty else { return [:] }
        var out: [String: Moment] = [:]
        for (h, session) in zip(shots, AppModel.screenHistory.sessions(around: shots.map(\.modified))) {
            // The file is dated at its shot; the shot within a second of it is the same one.
            let shot = session.map(\.nearest).flatMap { abs($0.time - h.modified) < 1 ? $0 : nil }
            out[h.path] = Moment(taken: h.modified, app: shot?.app ?? "", title: shot?.title ?? "", span: session.map(Span.init),
                                 link: AppModel.screenHistory.link(forPath: h.path))
        }
        return out
    }

    /// The moment of one screenshot, for a read of it. nil for any other path.
    static func moment(ofShot path: String) -> Moment? {
        guard isShot(path), let shot = AppModel.screenHistory.shot(atPath: path) else { return nil }
        let session = AppModel.screenHistory.sessions(around: [shot.time]).first ?? nil
        return Moment(taken: shot.time, app: shot.app, title: shot.title, span: session.map(Span.init),
                      link: AppModel.screenHistory.link(of: shot))
    }

    /// Said once under results that hold a screenshot, not on every row: its omni:// link names a
    /// frame of the screen history's video, and an agent that opens it with its own reader finds
    /// nothing. On the command line read_file saves the picture as a JPEG and prints where.
    static func notAFile(cli: Bool, mac: String? = nil) -> String {
        if let mac {
            return cli ? "A screenshot on \(mac) is no file: omni get_linked_file <omni:// link> --mac \"\(mac)\" copies it here."
                       : "A screenshot on \(mac) is no file: get_linked_file with its omni:// link and mac \"\(mac)\" copies it here."
        }
        return cli ? "A screenshot is no file: omni read_file <omni:// link> saves it as a JPEG and prints the path."
                   : "A screenshot is no file: read_file with its omni:// link returns it as a JPEG."
    }

    /// ISO 8601 to the second with the local offset ("2026-11-10T13:16:27+01:00"): what
    /// `modified_after` and `modified_before` read back (SearchArgs.parseTime).
    static func stamp(_ epoch: Double) -> String {
        let f = ISO8601DateFormatter()
        f.timeZone = .current
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: Date(timeIntervalSince1970: epoch))
    }

    /// The range that takes in a stretch: its first minute to the end of its last. `until` is
    /// exclusive, as the filter reads it.
    static func range(_ s: Span) -> (since: Double, until: Double) {
        ((s.start / 60).rounded(.down) * 60, ((s.end / 60).rounded(.down) + 1) * 60)
    }

    /// A stretch's apps, most seen first, each with the windows it showed: "Xcode (Sidebar.swift;
    /// AppModel.swift), Safari (Docs)". Cut to what a line can carry.
    static func apps(_ s: Span) -> String {
        var parts = s.apps.prefix(4).map { app -> String in
            let titles = app.windows.prefix(3).map { $0.count > 60 ? String($0.prefix(60)) + "..." : $0 }
            var text = app.name
            if !titles.isEmpty { text += " (" + titles.joined(separator: "; ") + (app.windows.count > 3 ? "; +\(app.windows.count - 3)" : "") + ")" }
            return text
        }
        if s.apps.count > 4 { parts.append("+\(s.apps.count - 4)") }
        return parts.joined(separator: ", ")
    }

    /// One line per stretch of work the shown screenshots fall in, under the results: its span,
    /// what was in front, and the call that lists the files changed in it. One line a stretch and
    /// not one a result: ten shots of one afternoon would say the same thing ten times. `mac` for a
    /// linked Mac's results: the stretch was on that Mac, and so are the files changed in it.
    static func sessionLines(_ hits: [SearchHit], _ moments: [String: Moment], offset: Int, cli: Bool, mac: String? = nil) -> [String] {
        var order: [Double] = [], numbers: [Double: [Int]] = [:], sessions: [Double: Span] = [:]
        for (i, h) in hits.enumerated() {
            guard let s = moments[h.path]?.span else { continue }
            if sessions[s.start] == nil { sessions[s.start] = s; order.append(s.start) }
            numbers[s.start, default: []].append(offset + i + 1)
        }
        return order.compactMap { key in
            guard let s = sessions[key], let which = numbers[key] else { return nil }
            return "Screenshot\(which.count == 1 ? "" : "s") \(which.map(String.init).joined(separator: ", "))\(mac.map { " on \($0)" } ?? ""): "
                + sessionText(s, cli: cli, mac: mac)
        }
    }

    /// What one stretch of work says: its span, what was in front, the call that lists the files
    /// changed in it.
    static func sessionText(_ s: Span, cli: Bool, mac: String? = nil) -> String {
        let (since, until) = range(s)
        let who = apps(s)
        let call = cli ? "omni search --modified-after \(stamp(since)) --modified-before \(stamp(until))" + (mac.map { " --macs \"\($0)\"" } ?? "")
                       : "search with modified_after \"\(stamp(since))\", modified_before \"\(stamp(until))\"" + (mac.map { ", macs [\"\($0)\"]" } ?? "") + " and no query"
        return "taken while \(mac ?? "the Mac") was in use from \(stamp(s.start)) to \(stamp(s.end))"
            + (who.isEmpty ? "" : ", in \(who)")
            + ". Files changed then: \(call)."
    }

    /// The same, as a result row's `screenshot` object (REST and `omni search --json`).
    static func row(_ m: Moment) -> [String: Any] {
        var row: [String: Any] = ["taken": stamp(m.taken)]
        if !m.app.isEmpty { row["app"] = m.app }
        if !m.title.isEmpty { row["title"] = m.title }
        if let s = m.span {
            let (since, until) = range(s)
            row["session"] = [
                "start": stamp(s.start), "end": stamp(s.end),
                "apps": s.apps.map { ["name": $0.name, "windows": $0.windows] as [String: Any] },
                // What to pass back to list the files changed in it.
                "modified_after": stamp(since), "modified_before": stamp(until),
            ] as [String: Any]
        }
        return row
    }
}
