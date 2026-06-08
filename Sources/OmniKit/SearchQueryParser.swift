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
        case "ext", "extension": return "ext"
        case "in", "folder", "path": return "in"
        case "date": return "date"
        case "after", "since": return "after"
        case "score", "relevance", "min": return "score"
        case "sort": return "sort"
        default: return nil
        }
    }

    /// NSRanges (UTF-16, for AppKit text storage) of each `key:value` qualifier token in `raw`, for
    /// inline tinting. Cosmetic-only and intentionally regex-based; it mirrors `parse`'s notion of a
    /// qualifier (whitelisted key, value right after the colon, token-start), so any drift only
    /// mis-tints, never mis-filters.
    // Compiled once: regex compilation dwarfs matching, and this runs on the main thread per keystroke
    // for inline tinting. NSRegularExpression is thread-safe for matching.
    private static let qualifierRegex: NSRegularExpression? = {
        // (start-of-string or whitespace) then an optional '-', a whitelisted key, ':', and a value
        // that is either a quoted string or a run of non-space characters.
        let pattern = #"(?:^|\s)(-?(?:type|kind|ext|extension|in|folder|path|date|after|since|score|relevance|min|sort):(?:"[^"]*"|\S+))"#
        return try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()

    public static func qualifierNSRanges(_ raw: String) -> [NSRange] {
        guard let re = qualifierRegex else { return [] }
        let ns = raw as NSString
        return re.matches(in: raw, range: NSRange(location: 0, length: ns.length)).map { $0.range(at: 1) }
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
