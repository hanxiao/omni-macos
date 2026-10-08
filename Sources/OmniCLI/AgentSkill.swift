import Foundation

/// SKILL.md, rendered from what the MCP server says about itself - its `initialize` instructions
/// and its `tools/list` - so the skill cannot drift from the tools. Compiled into both the `omni`
/// command (`omni skill`) and the app (Settings > Serving > SKILL.md).
///
/// PROGRESSIVE, NOT A MANUAL. It names the commands and shows a few examples; every option stays
/// behind `omni <tool> --help`, so the agent pays context only for the tool it is about to use.
/// The first version listed every flag of every tool - 125 lines loaded on every task.
enum AgentSkill {
    static let name = "omni-local-search"

    /// `command` is how the agent runs the CLI: an absolute path, so the skill works without the
    /// command being on PATH.
    static func render(instructions: String, tools: [[String: Any]], command: String) -> String {
        let width = tools.compactMap { ($0["name"] as? String)?.count }.max() ?? 0
        let list = tools.map { t -> String in
            let name = t["name"] as? String ?? ""
            return "  " + name.padding(toLength: width + 2, withPad: " ", startingAt: 0) + (t["title"] as? String ?? "")
        }.joined(separator: "\n")
        let shown = tools.flatMap { t in Examples.commands(for: t, command: command).prefix(t["name"] as? String == "search" ? 3 : 1) }
        return """
        ---
        name: \(name)
        description: Semantic search over the user's own files - text, code, PDFs, images, audio and video - through the Omni app on this Mac. Use it when the user asks to find, locate or recall their own files by content: find my notes about X, that invoice from February, photos of the beach.
        ---

        # Omni - local semantic file search

        \(instructions)

        ## Using it

        One command: `\(command)`. It talks to the running Omni app, and starts the app when it is not running, so the model is never loaded twice.

        ```
        \(list)
        ```

        ```
        \(shown.joined(separator: "\n"))
        ```

        - `\(command) <command> --help` shows its usage, examples and options; `--help-all` gives every option in full.
        - stdout is the answer, one result per line; stderr is everything about it - an empty result, a page range, a mistake and its fix. Add `--json` for structured output, the same fields as Omni's HTTP API.
        - Exit codes: 0 done, 1 the command needs fixing (stderr says how), 2 Omni is unavailable, 130 interrupted.
        """
    }
}

/// The `_meta` keys the command line and the MCP server agree on. One definition, compiled into
/// both (this file is in the app target too).
enum CLIProtocol {
    /// On a tools/call: the caller is `omni`. The answer adds structuredContent and marks blocks.
    static let cliKey = "io.hanxiao.omni/cli"
    /// On a content block: print it to stderr, not stdout.
    static let stderrKey = "io.hanxiao.omni/stderr"
    /// On a tool in tools/list: example arguments.
    static let examplesKey = "io.hanxiao.omni/examples"
}

/// A tool's examples (`_meta` in tools/list, MCPAdapter.examples), as command lines.
enum Examples {
    static let metaKey = CLIProtocol.examplesKey

    static func commands(for tool: [String: Any], command: String) -> [String] {
        guard let name = tool["name"] as? String,
              let list = (tool["_meta"] as? [String: Any])?[metaKey] as? [[String: Any]] else { return [] }
        let schema = ToolSchema(tool)
        return list.map { args in
            var words = [command, name]
            if let p = schema.params.first(where: \.positional), let v = args[p.name] {
                words += (v as? [Any] ?? [v]).map { shellQuote("\($0)") }
            }
            for p in schema.params where !p.positional {
                guard let v = args[p.name] else { continue }
                if p.type == "boolean" {
                    words.append((v as? Bool ?? true) ? "--\(p.flag)" : "--no-\(p.flag)")
                } else {
                    for item in (v as? [Any] ?? [v]) { words += ["--\(p.flag)", shellQuote("\(item)")] }
                }
            }
            return words.joined(separator: " ")
        }
    }

    static func shellQuote(_ s: String) -> String {
        let safe = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:@%+=,-")
        if !s.isEmpty, s.unicodeScalars.allSatisfy(safe.contains) { return s }
        return "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

/// A tool's input schema, read as command-line parameters.
struct ToolSchema {
    struct Param {
        let name: String            // as the schema names it: top_k
        let type: String            // string, integer, number, boolean, array, object
        let itemType: String
        let description: String
        let required: Bool
        let positional: Bool        // the first required parameter takes the bare words
        let enumValues: [String]
        var flag: String { name.replacingOccurrences(of: "_", with: "-") }
        var metavar: String {
            if !enumValues.isEmpty { return "<" + enumValues.joined(separator: "|") + ">" }
            switch type {
            case "boolean": return ""
            case "array": return "<\(itemType)>..."
            default: return "<\(type)>"
            }
        }
        /// The first sentence: the short help line.
        var summary: String { ToolSchema.firstSentence(description) }
        /// What a flag reads as in help. A yes/no that is on by default shows its off switch too,
        /// the one that changes anything: --[no-]group-duplicates.
        var display: String {
            if positional { return "<\(name)>" }
            if type == "boolean" { return defaultsOn ? "--[no-]\(flag)" : "--\(flag)" }
            return "--\(flag) \(metavar)"
        }
        let defaultsOn: Bool
    }

    let params: [Param]

    init(_ tool: [String: Any]) {
        let schema = tool["inputSchema"] as? [String: Any] ?? [:]
        let props = schema["properties"] as? [String: Any] ?? [:]
        let required = schema["required"] as? [String] ?? []
        let first = required.first { (props[$0] as? [String: Any])?["type"] as? String != "object" }
        // Required first in the schema's order, then the rest by name: JSON objects carry no order.
        let names = required.filter { props[$0] != nil } + props.keys.filter { !required.contains($0) }.sorted()
        params = names.map { n in
            let p = props[n] as? [String: Any] ?? [:]
            let items = p["items"] as? [String: Any] ?? [:]
            let enums = (p["enum"] as? [String]) ?? (items["enum"] as? [String]) ?? []
            return Param(name: n, type: p["type"] as? String ?? "string", itemType: items["type"] as? String ?? "string",
                         description: p["description"] as? String ?? "", required: required.contains(n),
                         positional: n == first, enumValues: enums, defaultsOn: p["default"] as? Bool == true)
        }
    }

    /// Up to the first full stop that ends a sentence - not the one in "e.g." or "i.e.".
    static func firstSentence(_ text: String) -> String {
        var from = text.startIndex
        while let r = text.range(of: ". ", range: from ..< text.endIndex) {
            let before = text[text.startIndex ..< r.lowerBound]
            if !(before.hasSuffix("e.g") || before.hasSuffix("i.e") || before.hasSuffix("etc") || before.hasSuffix("vs")) {
                return String(before) + "."
            }
            from = r.upperBound
        }
        return text
    }

    func usage(_ tool: String) -> String {
        var u = tool
        if let p = params.first(where: \.positional) { u += p.type == "array" ? " <\(p.name)>..." : " <\(p.name)>" }
        if params.contains(where: { !$0.positional }) { u += " [options]" }
        return u
    }
}
