import Foundation

/// SKILL.md, rendered from what the MCP server says about itself - its `initialize` instructions
/// and its `tools/list` - so the skill cannot drift from the tools. Compiled into both the `omni`
/// command (`omni skill`) and the app (Settings > Serving > SKILL.md).
enum AgentSkill {
    static let name = "omni-local-search"

    /// `command` is how the agent runs the CLI: an absolute path, so the skill works without the
    /// command being on PATH.
    static func render(instructions: String, tools: [[String: Any]], command: String) -> String {
        var out = """
        ---
        name: \(name)
        description: Semantic search over the user's own files - text, code, PDFs, images, audio and video - through the Omni app on this Mac. Use it when the user asks to find, locate or recall their own files by content: find my notes about X, that invoice from February, photos of the beach.
        ---

        # Omni - local semantic file search

        \(instructions)

        ## Running it

        The command is `\(command)`. Every tool below is a subcommand of it. It talks to the running Omni app, and starts the app when it is not running, so the model is never loaded twice.

        A tool's parameters are its flags: `top_k` is `--top-k`, a list takes its flag once per item, and a yes/no flag is on by itself and off as `--no-<flag>`. Output is plain text written for you; add `--json` for the raw result. `<tool> --help` shows a tool's options.

        """
        for tool in tools {
            out += "\n" + section(tool, command: command)
        }
        return out
    }

    static func section(_ tool: [String: Any], command: String) -> String {
        let name = tool["name"] as? String ?? "?"
        var s = "## \(name)\n\n"
        if let d = tool["description"] as? String { s += d + "\n\n" }
        let schema = ToolSchema(tool)
        s += "```\n\(command) \(schema.usage(name))\n```\n\n"
        for p in schema.params where !p.positional {
            s += "- `--\(p.flag)" + (p.metavar.isEmpty ? "" : " \(p.metavar)") + "`" + (p.required ? " (required)" : "")
            if !p.description.isEmpty { s += ": " + p.description }
            s += "\n"
        }
        if let p = schema.params.first(where: \.positional), !p.description.isEmpty {
            s += "- `<\(p.name)>`: " + p.description + "\n"
        }
        return s
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
                         positional: n == first, enumValues: enums)
        }
    }

    func usage(_ tool: String) -> String {
        var u = tool
        if let p = params.first(where: \.positional) { u += p.type == "array" ? " <\(p.name)>..." : " <\(p.name)>" }
        if params.contains(where: { !$0.positional }) { u += " [options]" }
        return u
    }
}
