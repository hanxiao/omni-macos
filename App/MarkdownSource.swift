import AppKit
import SwiftUI

/// Syntax highlighting for the raw Markdown pane.
///
/// There is no system Markdown source highlighter on macOS - `AttributedString(markdown:)` parses
/// Markdown in order to RENDER it, and hands back text with the syntax removed, which is the
/// opposite of what a source view needs. So the colouring is done here, over the source, with the
/// platform's own text machinery: an `AttributedString` that `Text` and (on Tahoe) `TextEditor`
/// both take directly. No third-party highlighter, no web view.
///
/// Colours come from `NSColor`'s system palette rather than fixed hex values, so they follow the
/// active appearance and Increase Contrast the way every other control does. The scheme is
/// deliberately restrained: markup punctuation recedes, structure (headings, tags, code) is what
/// gets colour. A source view is for reading the text, not the syntax.
enum MarkdownSource {

    static func highlighted(_ text: String) -> AttributedString {
        Cache.shared.highlighted(text)
    }

    // MARK: - The scheme

    private static let punctuation = Color(nsColor: .tertiaryLabelColor)
    private static let heading = Color(nsColor: .systemBlue)
    private static let tag = Color(nsColor: .systemPurple)
    private static let attribute = Color(nsColor: .systemTeal)
    private static let code = Color(nsColor: .systemGreen)
    private static let link = Color(nsColor: .systemBrown)
    private static let rule = Color(nsColor: .secondaryLabelColor)
    private static let math = Color(nsColor: .systemIndigo)

    /// Ordered because later rules paint over earlier ones: a `<td>` inside a fenced block should
    /// read as code, not as a tag.
    private static let rules: [(pattern: String, style: Style)] = [
        // Fenced code first so nothing inside it gets re-coloured by the rules below.
        ("(?s)^```.*?(?:^```|\\z)",              .init(color: code)),
        ("^(#{1,6})\\s+(.*)$",                    .init(color: heading, bold: true)),
        ("^\\s*(?:[-*+]|\\d+\\.)\\s",             .init(color: punctuation, bold: true)),
        ("^\\s*>\\s?",                            .init(color: punctuation)),
        ("^\\s*(?:-{3,}|\\*{3,}|_{3,})\\s*$",     .init(color: rule)),
        // HTML - this model emits <table> for anything tabular, so tags are most of a real page.
        ("</?[A-Za-z][^>]*>",                     .init(color: tag)),
        ("\\s([A-Za-z-]+)=\"[^\"]*\"",            .init(color: attribute)),
        ("`[^`\\n]+`",                            .init(color: code)),
        // Maths, so the source view shows where a formula is even though it is not set here.
        ("\\$\\$?[^$\\n]+\\$\\$?",                 .init(color: math)),
        ("\\*\\*[^*\\n]+\\*\\*",                  .init(bold: true)),
        ("(?<![*\\w])\\*[^*\\n]+\\*(?![*\\w])",   .init(italic: true)),
        ("\\[[^\\]\\n]*\\]\\([^)\\n]*\\)",        .init(color: link, underline: true)),
        ("^\\|.*\\|$",                            .init(color: punctuation)),
    ]

    private struct Style {
        var color: Color?
        var bold = false
        var italic = false
        var underline = false
    }

    private static let expressions: [(NSRegularExpression, Style)] = rules.compactMap {
        guard let re = try? NSRegularExpression(pattern: $0.pattern,
                                                options: [.anchorsMatchLines]) else { return nil }
        return (re, $0.style)
    }

    fileprivate static func build(_ text: String) -> AttributedString {
        var out = AttributedString(text)
        let full = NSRange(text.startIndex ..< text.endIndex, in: text)
        for (expression, style) in expressions {
            for match in expression.matches(in: text, range: full) {
                guard let range = Range(match.range, in: text),
                      let lower = AttributedString.Index(range.lowerBound, within: out),
                      let upper = AttributedString.Index(range.upperBound, within: out)
                else { continue }
                if let color = style.color { out[lower ..< upper].foregroundColor = color }
                if style.bold || style.italic {
                    var traits: NSFontDescriptor.SymbolicTraits = []
                    if style.bold { traits.insert(.bold) }
                    if style.italic { traits.insert(.italic) }
                    out[lower ..< upper].appKit.font = monospaced(traits)
                }
                if style.underline { out[lower ..< upper].underlineStyle = .single }
            }
        }
        return out
    }

    private static func monospaced(_ traits: NSFontDescriptor.SymbolicTraits) -> NSFont {
        let size = NSFont.systemFontSize
        let base = NSFont.monospacedSystemFont(ofSize: size, weight: traits.contains(.bold) ? .bold : .regular)
        guard traits.contains(.italic) else { return base }
        let descriptor = base.fontDescriptor.withSymbolicTraits(traits)
        return NSFont(descriptor: descriptor, size: size) ?? base
    }

    /// Highlighting a streaming section runs on every frame, and only the tail has changed. The
    /// cache turns that back into one pass per new frame instead of one per section per frame.
    private final class Cache: @unchecked Sendable {
        static let shared = Cache()
        private let lock = NSLock()
        private var entries: [String: AttributedString] = [:]

        func highlighted(_ text: String) -> AttributedString {
            lock.withLock {
                if let hit = entries[text] { return hit }
                let built = MarkdownSource.build(text)
                if entries.count > 256 { entries.removeAll(keepingCapacity: true) }
                entries[text] = built
                return built
            }
        }
    }
}
