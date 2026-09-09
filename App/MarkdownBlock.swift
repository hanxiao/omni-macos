import Foundation
import SwiftUI

/// A minimal block-level Markdown renderer for transcription output.
///
/// SwiftUI's `Text(AttributedString(markdown:))` only carries INLINE styling - it parses block
/// intents but renders them flat, so headings, lists and rules all come out as body text. Getting
/// document structure means splitting into blocks here and letting `AttributedString` handle only
/// what it is good at, which is the inline span inside each one.
///
/// HTML tables get a real grid rather than a code block. That is not polish: this model is
/// instructed to emit `<table>` for anything tabular, so ledgers, invoices and spreadsheets - the
/// documents most worth transcribing - are almost entirely tables. Showing them as raw tags would
/// make the formatted view useless for exactly the content it matters most for.
enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bullets([String])
    case ordered([String])
    case code(String)
    case rule
    case table(rows: [[String]], hasHeader: Bool)

    // Main actor because it is only ever built from a SwiftUI body.
    /// `highlighting` marks find matches. It is applied to the RENDERED text rather than to the
    /// Markdown source: the source has markers the reader never sees, so a range mapped from it
    /// would land in the wrong place, and what a reader is looking for is what is on screen.
    @MainActor @ViewBuilder func view(highlighting find: String = "") -> some View {
        switch self {
        case .heading(let level, let text):
            Text(inline(text, find))
                .font(Self.headingFont(level))
                .padding(.top, level <= 2 ? 8 : 4)

        case .paragraph(let text):
            Text(inline(text, find))
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)

        case .bullets(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•").foregroundStyle(.secondary)
                        Text(inline(item, find)).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

        case .ordered(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\(index + 1).").foregroundStyle(.secondary).monospacedDigit()
                        Text(inline(item, find)).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

        case .code(let body):
            Text(FindHighlight.mark(find, in: AttributedString(body)))
                .font(.system(.callout, design: .monospaced))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.tertiary, in: RoundedRectangle(cornerRadius: Design.cornerSmall))

        case .rule:
            Divider().padding(.vertical, 6)


        case .table(let rows, let hasHeader):
            TableBlock(rows: rows, hasHeader: hasHeader, find: find)
        }
    }

    private static func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title.weight(.semibold)
        case 2: return .title2.weight(.semibold)
        case 3: return .title3.weight(.semibold)
        default: return .headline
        }
    }

    /// The block as one styled string, or nil for blocks that need a view of their own.
    ///
    /// This exists so a drag can select across paragraphs. SwiftUI's text selection does not span
    /// sibling `Text` views, so a page drawn as one view per block is a page where every paragraph
    /// break ends the selection; merged into a single `Text`, a whole run of prose selects and
    /// copies in one go. Tables and code blocks stay separate - they are objects in the flow, and
    /// a selection that stops at one reads as deliberate rather than broken.
    @MainActor func attributed(highlighting find: String) -> AttributedString? {
        switch self {
        case .heading(let level, let text):
            var out = inline(text, find)
            out.font = Self.headingFont(level)
            return out

        case .paragraph(let text):
            var out = inline(text, find)
            out.font = .body
            return out

        case .bullets(let items):
            return Self.list(items.map { ("\u{2022}  ", $0) }, find, self)

        case .ordered(let items):
            return Self.list(items.enumerated().map { ("\($0.offset + 1).  ", $0.element) },
                             find, self)

        case .code, .rule, .table:
            return nil
        }
    }

    @MainActor private static func list(_ items: [(String, String)], _ find: String,
                                        _ owner: MarkdownBlock) -> AttributedString {
        var out = AttributedString()
        for (index, item) in items.enumerated() {
            if index > 0 { out += AttributedString("\n") }
            var marker = AttributedString(item.0)
            marker.font = .body
            marker.foregroundColor = .secondary
            out += marker
            var body = owner.inline(item.1, find)
            body.font = .body
            out += body
        }
        return out
    }

    /// Inline Markdown only. `.inlineOnlyPreservingWhitespace` is deliberate: the full parser
    /// would swallow leading `#` and `-` markers that this splitter has already claimed.
    private func inline(_ text: String, _ find: String) -> AttributedString {
        FindHighlight.mark(find, in: InlineCache.attributed(text))
    }
}

/// Memoises inline parsing across renders.
///
/// A streaming page redraws 24 times a second and every redraw re-runs `AttributedString(markdown:)`
/// for every block on the page - on a long page that is hundreds of full Markdown parses per second
/// for text that has not changed since the last token. Only the final block is ever new, so a
/// straight cache turns that back into one parse per frame.
///
/// Bounded and cleared wholesale rather than evicted one entry at a time: the keys are page text
/// that goes out of scope when the document does, so precise eviction would cost more than it saves.
private final class InlineCache: @unchecked Sendable {
    private static let shared = InlineCache()
    private let lock = NSLock()
    private var entries: [String: AttributedString] = [:]

    static func attributed(_ text: String) -> AttributedString { shared.lookup(text) }

    private func lookup(_ text: String) -> AttributedString {
        lock.withLock {
            if let hit = entries[text] { return hit }
            let parsed = (try? AttributedString(
                markdown: text,
                options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
                ?? AttributedString(text)
            if entries.count > 4_000 { entries.removeAll(keepingCapacity: true) }
            entries[text] = parsed
            return parsed
        }
    }
}

// MARK: - Runs

/// One drawable unit of a finished section: a merged run of flowing text, or a block that draws
/// itself. See `MarkdownBlock.attributed(highlighting:)` for why the merge exists.
enum MarkdownRun: Identifiable {
    case text(index: Int, AttributedString)
    case block(index: Int, MarkdownBlock)

    var id: Int {
        switch self {
        case .text(let index, _), .block(let index, _): return index
        }
    }
}

extension MarkdownBlock {
    @MainActor static func runs(_ blocks: [MarkdownBlock], highlighting find: String) -> [MarkdownRun] {
        var out: [MarkdownRun] = []
        var pending: AttributedString?

        func flush() {
            guard let text = pending else { return }
            out.append(.text(index: out.count, text))
            pending = nil
        }

        for block in blocks {
            if let piece = block.attributed(highlighting: find) {
                // A blank line between blocks, which is the spacing the separate views had.
                pending = pending.map { $0 + AttributedString("\n\n") + piece } ?? piece
            } else {
                flush()
                out.append(.block(index: out.count, block))
            }
        }
        flush()
        return out
    }
}

// MARK: - Parsing

extension MarkdownBlock {
    static func parse(_ markdown: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = markdown.components(separatedBy: .newlines)
        var index = 0
        var paragraph: [String] = []

        func flushParagraph() {
            let joined = paragraph.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            if !joined.isEmpty { blocks.append(.paragraph(joined)) }
            paragraph = []
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                flushParagraph(); index += 1; continue
            }

            // An HTML table can be streamed in mid-write, so an unterminated one still renders
            // what has arrived rather than dumping tags into the formatted view.
            if trimmed.lowercased().hasPrefix("<table") {
                flushParagraph()
                var body = ""
                while index < lines.count {
                    body += lines[index] + "\n"
                    if lines[index].lowercased().contains("</table>") { index += 1; break }
                    index += 1
                }
                if let table = parseHTMLTable(body) { blocks.append(table) }
                continue
            }

            if trimmed.hasPrefix("```") {
                flushParagraph()
                index += 1
                var body: [String] = []
                while index < lines.count, !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    body.append(lines[index]); index += 1
                }
                if index < lines.count { index += 1 }
                blocks.append(.code(body.joined(separator: "\n")))
                continue
            }

            if trimmed.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }), trimmed.count >= 3 {
                flushParagraph(); blocks.append(.rule); index += 1; continue
            }

            if trimmed.hasPrefix("#") {
                flushParagraph()
                let level = trimmed.prefix(while: { $0 == "#" }).count
                let text = trimmed.drop(while: { $0 == "#" }).trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(level: min(level, 4), text: text))
                index += 1
                continue
            }

            if isBullet(trimmed) {
                flushParagraph()
                var items: [String] = []
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    guard isBullet(candidate) else { break }
                    items.append(String(candidate.dropFirst(2)).trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                blocks.append(.bullets(items))
                continue
            }

            if isOrdered(trimmed) {
                flushParagraph()
                var items: [String] = []
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    guard isOrdered(candidate), let dot = candidate.firstIndex(of: ".") else { break }
                    items.append(String(candidate[candidate.index(after: dot)...])
                        .trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                blocks.append(.ordered(items))
                continue
            }

            // Pipe tables, the other tabular form this model emits.
            if trimmed.hasPrefix("|"), trimmed.hasSuffix("|") {
                flushParagraph()
                var rows: [[String]] = []
                var sawDivider = false
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    guard candidate.hasPrefix("|") else { break }
                    let cells = candidate.split(separator: "|", omittingEmptySubsequences: false)
                        .dropFirst().dropLast()
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                    if cells.allSatisfy({ $0.allSatisfy { c in c == "-" || c == ":" } && !$0.isEmpty }) {
                        sawDivider = true
                    } else {
                        rows.append(Array(cells))
                    }
                    index += 1
                }
                if !rows.isEmpty { blocks.append(.table(rows: rows, hasHeader: sawDivider)) }
                continue
            }

            paragraph.append(trimmed)
            index += 1
        }
        flushParagraph()
        return blocks
    }

    private static func isBullet(_ line: String) -> Bool {
        (line.hasPrefix("- ") || line.hasPrefix("* ")) && line.count > 2
    }

    private static func isOrdered(_ line: String) -> Bool {
        guard let dot = line.firstIndex(of: "."), dot > line.startIndex else { return false }
        return line[line.startIndex ..< dot].allSatisfy(\.isNumber)
    }

    /// Pull `<tr>`/`<th>`/`<td>` out of an HTML table.
    ///
    /// Deliberately tolerant, and regex rather than a parser: this is model output arriving a
    /// token at a time, so a missing `</td>` or a half-written final row has to degrade into a
    /// readable row instead of losing the table.
    private static func parseHTMLTable(_ html: String) -> MarkdownBlock? {
        let rowPattern = "<tr[^>]*>(.*?)(?:</tr>|$)"
        let cellPattern = "<(t[dh])[^>]*>(.*?)(?:</t[dh]>|$)"
        guard let rowRegex = try? NSRegularExpression(pattern: rowPattern,
                                                      options: [.dotMatchesLineSeparators, .caseInsensitive]),
              let cellRegex = try? NSRegularExpression(pattern: cellPattern,
                                                       options: [.dotMatchesLineSeparators, .caseInsensitive])
        else { return nil }

        var rows: [[String]] = []
        var sawHeaderCell = false
        let full = NSRange(html.startIndex ..< html.endIndex, in: html)

        for rowMatch in rowRegex.matches(in: html, range: full) {
            guard let rowRange = Range(rowMatch.range(at: 1), in: html) else { continue }
            let rowBody = String(html[rowRange])
            var cells: [String] = []
            let rowFull = NSRange(rowBody.startIndex ..< rowBody.endIndex, in: rowBody)
            for cellMatch in cellRegex.matches(in: rowBody, range: rowFull) {
                if let tagRange = Range(cellMatch.range(at: 1), in: rowBody),
                   rowBody[tagRange].lowercased() == "th" {
                    sawHeaderCell = true
                }
                if let bodyRange = Range(cellMatch.range(at: 2), in: rowBody) {
                    cells.append(strip(String(rowBody[bodyRange])))
                }
            }
            if !cells.isEmpty { rows.append(cells) }
        }
        guard !rows.isEmpty else { return nil }
        return .table(rows: rows, hasHeader: sawHeaderCell && rows.count > 1)
    }

    private static func strip(_ html: String) -> String {
        html.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Marks find matches in text that is already laid out.
///
/// The system's own find tint, so a match here looks like a match anywhere else on the Mac, and it
/// follows the appearance rather than being a fixed yellow that turns muddy in dark mode.
enum FindHighlight {
    static func mark(_ find: String, in text: AttributedString) -> AttributedString {
        guard !find.isEmpty else { return text }
        var out = text
        let plain = String(out.characters)
        var from = plain.startIndex
        while let r = plain.range(of: find, options: .caseInsensitive, range: from ..< plain.endIndex) {
            if let lower = AttributedString.Index(r.lowerBound, within: out),
               let upper = AttributedString.Index(r.upperBound, within: out) {
                // Both scopes: `Text` reads SwiftUI's, and the editable source pane is an
                // `NSTextView` whose storage comes from the AppKit one.
                out[lower ..< upper].backgroundColor = Color(nsColor: .findHighlightColor)
                out[lower ..< upper].foregroundColor = Color(nsColor: .black)
                out[lower ..< upper].appKit.backgroundColor = .findHighlightColor
                out[lower ..< upper].appKit.foregroundColor = .black
            }
            from = r.upperBound
            if r.upperBound == plain.endIndex { break }
        }
        return out
    }
}

// MARK: - Streaming arrival

/// How newly decoded content arrives: it fades up rather than snapping in.
///
/// Driven purely by INSERTION - a block, a table row or a bullet that was not there on the last
/// frame. Nothing already on screen moves, dims or re-animates, which is what keeps a page
/// readable while it is still being written; an effect applied to the whole growing text would
/// make the reader chase it. Off entirely under Reduce Motion.
struct StreamFade: ViewModifier {
    let count: Int
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    static let transition: AnyTransition = .opacity.combined(with: .offset(y: 4))

    func body(content: Content) -> some View {
        content.animation(reduceMotion ? nil : .easeOut(duration: 0.28), value: count)
    }
}

// MARK: - Table

private struct TableBlock: View {
    let rows: [[String]]
    let hasHeader: Bool
    var find: String = ""

    private var columns: Int { rows.map(\.count).max() ?? 0 }

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 0, verticalSpacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    GridRow {
                        ForEach(0 ..< columns, id: \.self) { column in
                            let value = column < row.count ? row[column] : ""
                            Text(FindHighlight.mark(find, in: AttributedString(value)))
                                // Numerics right-align, which is what makes a ledger readable and
                                // what a plain grid gets wrong by default.
                                .font(isNumeric(value) ? .system(.callout, design: .monospaced) : .callout)
                                .fontWeight(index == 0 && hasHeader ? .semibold : .regular)
                                .frame(maxWidth: .infinity,
                                       alignment: isNumeric(value) ? .trailing : .leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                        }
                    }
                    .background(rowBackground(index))
                    .transition(StreamFade.transition)
                }
            }
            .modifier(StreamFade(count: rows.count))
            .clipShape(RoundedRectangle(cornerRadius: Design.cornerSmall))
            .overlay {
                RoundedRectangle(cornerRadius: Design.cornerSmall)
                    .strokeBorder(Color.secondary.opacity(0.25))
            }
        }
    }

    private func rowBackground(_ index: Int) -> Color {
        if index == 0 && hasHeader { return Color.secondary.opacity(0.14) }
        return index.isMultiple(of: 2) ? .clear : Color.secondary.opacity(0.06)
    }

    private func isNumeric(_ value: String) -> Bool {
        guard !value.isEmpty else { return false }
        let stripped = value.filter { !",.()%$€£ -+".contains($0) }
        return !stripped.isEmpty && stripped.allSatisfy(\.isNumber)
    }
}
