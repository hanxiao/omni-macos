import Foundation
import OmniKit

/// The search filters, read in ONE place for every surface: the MCP `search` tool (and through
/// it the `omni` command line) and REST /v1/search's `filters` object. They were two parsers
/// that had drifted - REST took an unknown kind silently and matched nothing, passed `ext` through
/// uncleaned, and knew only epoch seconds - and every new filter had to be added twice.
///
/// Both spellings of a date are accepted on both surfaces: `since`/`until` (epoch seconds, REST's
/// documented form) and `modified_after`/`modified_before` (ISO 8601, MCP's). Results are still
/// rendered per surface (JSON for REST, text for agents); only the reading is shared.
enum SearchArgs {
    /// The filter `a` asks for, or why it cannot be applied (a message naming the fix).
    static func filter(_ a: [String: Any]) -> (SearchFilter, String?) {
        var filter = SearchFilter()
        // Absent means "use the server's default floor", which is why this is Optional rather than
        // defaulted here: a caller that says nothing gets the same cut the window applies, and
        // min_score 0 is the explicit way to ask for everything.
        if let ms = number(a["min_score"]) { filter.minScore = Swift.max(0, Swift.min(1, ms)) }
        if let raw = a["kinds"] {
            guard let kinds = raw as? [String] else { return (filter, "'kinds' must be a list of kinds") }
            let (set, err) = normalizedKinds(kinds)
            if let err { return (filter, err) }
            if let set { filter.kinds = set }
        }
        // `folder` (one) and `folders` (several) - issue #18. An agent scoping to two project
        // folders under one indexed root could not say so: the parent subsumes its children, so
        // adding them as roots does not help either. Both spellings are accepted and merged.
        var scoped: [String] = []
        let named = [a["folder"] as? String].compactMap { $0 } + (a["folders"] as? [String] ?? [])
        for folder in named {
            let (value, err) = normalizedFolder(folder)
            if let err { return (filter, err) }
            if let value, !scoped.contains(value) { scoped.append(value) }
        }
        if !scoped.isEmpty { filter.folderPrefixes = scoped }
        // Date range and extension: the window's `date:`/`after:` and `ext:` qualifiers - "what was
        // I working on last week" needs a range, not a query.
        for (keys, apply) in [(["modified_after", "since"], { (t: Double) in filter.since = t }),
                              (["modified_before", "until"], { (t: Double) in filter.until = t })]
                              as [([String], (Double) -> Void)] {
            for key in keys {
                guard let raw = a[key] else { continue }
                if let t = number(raw) { apply(t); continue }
                guard let text = raw as? String, let t = parseTime(text) else {
                    return (filter, "'\(key)' must be an ISO 8601 date ('2026-10-01'), a date-time ('2026-10-01T09:30:00Z') or epoch seconds")
                }
                apply(t)
            }
        }
        if let since = filter.since, let until = filter.until, since >= until {
            return (filter, "the start of the date range must be earlier than its end")
        }
        if let raw = a["ext"] {
            guard let ext = raw as? String else { return (filter, "'ext' must be a string like 'pdf'") }
            let clean = ext.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: ".")).lowercased()
            if !clean.isEmpty { filter.ext = clean }
        }
        return (filter, nil)
    }

    /// A JSON number as Double, whether it arrived as an integer or not. A Bool is not a number,
    /// though Foundation would bridge it to one.
    private static func number(_ v: Any?) -> Double? {
        guard let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        return n.doubleValue
    }

    /// An ISO 8601 date (local midnight) or date-time (with or without fractional seconds), as
    /// epoch seconds; nil when it is neither.
    static func parseTime(_ s: String) -> Double? {
        let text = s.trimmingCharacters(in: .whitespaces)
        let full = ISO8601DateFormatter()
        if let d = full.date(from: text) { return d.timeIntervalSince1970 }
        full.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = full.date(from: text) { return d.timeIntervalSince1970 }
        let day = DateFormatter()
        day.calendar = Calendar(identifier: .gregorian)
        day.locale = Locale(identifier: "en_US_POSIX")
        day.timeZone = .current
        day.dateFormat = "yyyy-MM-dd"
        return day.date(from: text)?.timeIntervalSince1970
    }

    // MARK: - Filter input validation
    //
    // `folderPrefix` matching is `path == f || path.hasPrefix(f + "/")` (SearchFilter.acceptsPath),
    // so a trailing slash builds the prefix "…//" and a "~" or relative path matches nothing at
    // all. Both used to come back as an empty result set, which an agent reasonably reads as
    // "not on this Mac". Normalize what we can and reject what we cannot, with the corrected
    // value in the message so the next call is right.

    /// nil when the folder is usable (or absent); a ready-to-return error response otherwise.
    static func normalizedFolder(_ raw: String) -> (value: String?, error: String?) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return (nil, nil) }
        let expanded = (trimmed as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else {
            return (nil, "'folder' must be an absolute path (got \"\(trimmed)\"). "
                       + "Use a path like /Users/you/Documents, or omit it to search everywhere.")
        }
        // A folder that is not there at all scoped the search to nothing and came back as an empty
        // result set, which reads as "no such file on this Mac" - the one conclusion a scope typo
        // must never produce. Say which path missed instead. Existence is the decisive half and
        // costs one stat; a real folder that simply is not a source still returns empty, and the
        // index-state lines already in the response explain that case.
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir) else {
            return (nil, "no such folder: \"\(expanded)\". Check the path, or omit 'folder' to "
                       + "search everywhere. list_sources reports the folders Omni indexes.")
        }
        guard isDir.boolValue else {
            return (nil, "'folder' must be a folder, not a file (got \"\(expanded)\"). "
                       + "To search inside specific files, use search_inline.")
        }
        return (normalizeStorePath(expanded), nil)
    }

    /// Validates against FileKind, returning the expanded set (text implies scan) or an error.
    static func normalizedKinds(_ raw: [String]) -> (value: Set<String>?, error: String?) {
        let cleaned = raw.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                         .filter { !$0.isEmpty }
        guard !cleaned.isEmpty else { return (nil, nil) }
        let valid = Set(FileKind.allCases.map(\.rawValue))
        let unknown = cleaned.filter { !valid.contains($0) }
        guard unknown.isEmpty else {
            let known = FileKind.allCases.map(\.rawValue).sorted().joined(separator: ", ")
            return (nil, "unknown kind\(unknown.count == 1 ? "" : "s") "
                       + unknown.map { "\"\($0)\"" }.joined(separator: ", ")
                       + ". Valid kinds are: \(known).")
        }
        var set = Set(cleaned)
        // Same superset rule as the app: text includes scanned PDFs ('scan').
        if set.contains(FileKind.text.rawValue) { set.insert(FileKind.scan.rawValue) }
        return (set, nil)
    }
}
