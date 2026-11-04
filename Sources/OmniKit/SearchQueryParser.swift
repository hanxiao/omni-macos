import Foundation

/// The result of parsing a search-box string: the semantic (embedding) query plus the structured
/// `key:value` qualifiers found in it.
public struct ParsedQuery: Equatable, Sendable {
    public struct Qualifier: Equatable, Sendable {
        public let key: String     // canonical key: type, ext, in, date, after, score, sort
        public let value: String   // raw value, unquoted, case preserved
        public let negated: Bool
        public init(key: String, value: String, negated: Bool) {
            self.key = key; self.value = value; self.negated = negated
        }
    }
    public let semanticText: String
    public let qualifiers: [Qualifier]
}

/// Splits a search-box string into a semantic remainder (the embedding query) and structured
/// qualifiers. Pure and dependency-free so it is unit-testable and shared across layers.
///
/// The grammar follows the lexical-qualifier convention of GitHub / Gmail / Spotlight: `key:value`,
/// quoted values, `type:` multi-value (`type:image,video`) and negation (`-type:audio`). A run is a
/// qualifier ONLY if (a) its key is whitelisted and (b) a value follows the colon with no whitespace
/// on either side. So `notes about type: theory`, `12:30`, `ratio 3:1`, and `http://x` all stay
/// semantic. Anything that is not a qualifier becomes free text; the semantic remainder is the
/// free-text runs joined in original order. The two channels are disjoint by construction.
public enum SearchQueryParser {

    /// Canonical key for a lowercased word, or nil if it is not a recognized qualifier key.
    public static func canonicalKey(_ word: String) -> String? {
        switch word {
        case "type", "kind": return "type"
        case "tag", "tags": return "tag"
        case "ext", "extension": return "ext"
        case "in", "folder", "path": return "in"
        case "filename", "name", "file": return "filename"
        case "date": return "date"
        case "after", "since": return "after"
        case "score", "relevance", "min": return "score"
        case "sort": return "sort"
        default: return nil
        }
    }

    /// Whether `typed` holds a qualifier that is FINISHED: one that is not the word the edit from
    /// `previous` left the caret in. That word is still being typed: mid-word, `type:i` has to stay
    /// editable text or the chip is made from half a word - at the end of the box or inside it
    /// ("beach type:i photos", a filter added between two words). Every other qualifier is done:
    /// text pasted or typed faster than the field reports each keystroke arrives as
    /// "type:image photos" without ever ending in the space, and was searched as prose with no
    /// filter (review 2026-11-01).
    public static func hasFinishedQualifier(_ typed: String, previous: String = "") -> Bool {
        // No edit, nothing finished by it. The field reports some keystrokes twice, and on the
        // second the caret read as the end of the box: `type:i` typed between two words was a chip.
        guard typed != previous else { return false }
        let new = Array(typed), old = Array(previous)
        // The caret: where the edit ends, after the common prefix and before the common suffix.
        var head = 0
        while head < new.count, head < old.count, new[head] == old[head] { head += 1 }
        var tail = 0
        while tail < new.count - head, tail < old.count - head,
              new[new.count - 1 - tail] == old[old.count - 1 - tail] { tail += 1 }
        let caret = new.count - tail
        // The word the caret is in or at the end of. Right after whitespace it is in none.
        var from = caret, to = caret
        if caret > 0, !new[caret - 1].isWhitespace {
            while from > 0, !new[from - 1].isWhitespace { from -= 1 }
            while to < new.count, !new[to].isWhitespace { to += 1 }
        }
        let done = String(new[..<from]) + " " + String(new[to...])
        // Inside an open quote a space does not end anything: `in:"my ` is half a folder name.
        guard done.filter({ $0 == "\"" }).count % 2 == 0 else { return false }
        return !parse(done).qualifiers.isEmpty
    }

    /// The folders an `in:` value names when it is not a path. The sidebar shows each indexed folder
    /// by its name, so the name is what gets typed: `in:corpus2`, `in:Documents/Taxes`. A value
    /// whose first component is a root's name (as shown, or its last path component, either case)
    /// is that root, the rest a path inside it. Read as a path from `/`, `in:corpus2` matched
    /// nothing and said nothing (review 2026-11-02). A path (`/`, `~`) or a name no root has: none.
    public static func folderPaths(named value: String, roots: [(path: String, name: String)]) -> [String] {
        let v = value.trimmingCharacters(in: .whitespaces)
        guard !v.hasPrefix("/"), !v.hasPrefix("~") else { return [] }
        let parts = v.split(separator: "/", maxSplits: 1)
        guard let head = parts.first else { return [] }
        let rest = parts.count > 1 ? String(parts[1]) : ""
        var out: [String] = []
        for r in roots {
            let leaf = (r.path as NSString).lastPathComponent
            guard head.caseInsensitiveCompare(r.name) == .orderedSame
                    || head.caseInsensitiveCompare(leaf) == .orderedSame else { continue }
            let p = rest.isEmpty ? r.path : (r.path as NSString).appendingPathComponent(rest)
            if !out.contains(p) { out.append(p) }
        }
        return out
    }

    public static func parse(_ raw: String) -> ParsedQuery {
        let s = Array(raw)
        let n = s.count
        var i = 0
        var quals: [ParsedQuery.Qualifier] = []
        var spans: [String] = []

        while i < n {
            while i < n && s[i].isWhitespace { i += 1 }
            if i >= n { break }
            let tokenStart = i
            var negated = false
            if s[i] == "-" { negated = true; i += 1 }
            let keyStart = i
            while i < n && s[i].isLetter { i += 1 }
            let word = String(s[keyStart..<i]).lowercased()

            // Qualifier candidate: `word:value`, key whitelisted, value present, no space around `:`.
            if i < n, s[i] == ":", let canon = canonicalKey(word), i + 1 < n, !s[i + 1].isWhitespace {
                i += 1   // consume ':'
                var value = ""
                if s[i] == "\"" {                       // quoted value may contain spaces
                    i += 1
                    while i < n && s[i] != "\"" {
                        if s[i] == "\\" && i + 1 < n { i += 1 }   // \" / \\ escape
                        value.append(s[i]); i += 1
                    }
                    if i < n && s[i] == "\"" { i += 1 }
                } else {                                // bare value runs to the next whitespace
                    while i < n && !s[i].isWhitespace { value.append(s[i]); i += 1 }
                }
                if !value.isEmpty {
                    quals.append(.init(key: canon, value: value, negated: negated))
                    continue
                }
            }

            // Not a qualifier: take the whole run (including any leading '-' and inner ':') as one
            // free-text span, preserving the user's characters.
            i = tokenStart
            var span = ""
            while i < n && !s[i].isWhitespace { span.append(s[i]); i += 1 }
            spans.append(span)
        }
        return ParsedQuery(semanticText: spans.joined(separator: " "), qualifiers: quals)
    }
}
