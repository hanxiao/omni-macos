import Foundation

/// The `omni` command shipped inside the app (Contents/Helpers/omni, docs/CLI.md), and putting it
/// on PATH the way other Mac apps do: a symlink in /usr/local/bin, asking for an administrator's
/// password only when that folder is not writable. The link points into the bundle, so updates
/// replace the command with the app; a moved app shows the link as stale, to be installed again.
enum CommandLineTool {
    nonisolated static var bundledPath: String {
        Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/omni").path
    }

    nonisolated static var linkPath: String {
        let args = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        let dir = (args["omni.cliLinkDir"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "/usr/local/bin"
        return (dir as NSString).appendingPathComponent("omni")
    }

    /// SKILL.md for the command, rendered from the MCP tools themselves (AgentSkill) - the same
    /// text `omni skill` prints - so it cannot drift from them the way the hand-written one did.
    nonisolated static var skillMarkdown: String {
        AgentSkill.render(instructions: MCPAdapter.instructions, tools: MCPAdapter.tools(withSources: true),
                          command: bundledPath)
    }

    enum State: Equatable { case notInstalled, installed, stale(String) }

    nonisolated static var state: State {
        guard let dest = try? FileManager.default.destinationOfSymbolicLink(atPath: linkPath) else {
            return FileManager.default.fileExists(atPath: linkPath) ? .stale(linkPath) : .notInstalled
        }
        return dest == bundledPath ? .installed : .stale(dest)
    }

    /// Link it, with a password prompt when /usr/local/bin needs one. nil on success, else why not.
    @MainActor static func install() -> String? {
        let fm = FileManager.default
        let dir = (linkPath as NSString).deletingLastPathComponent
        guard fm.isExecutableFile(atPath: bundledPath) else { return "The command is missing from this copy of Omni." }
        if (fm.fileExists(atPath: dir) ? fm.isWritableFile(atPath: dir) : fm.isWritableFile(atPath: (dir as NSString).deletingLastPathComponent)) {
            do {
                try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
                if (try? fm.destinationOfSymbolicLink(atPath: linkPath)) != nil || fm.fileExists(atPath: linkPath) {
                    try fm.removeItem(atPath: linkPath)
                }
                try fm.createSymbolicLink(atPath: linkPath, withDestinationPath: bundledPath)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
        // `do shell script ... with administrator privileges` runs in this process - no Apple
        // Events to another app, so no automation permission - and shows the system password dialog.
        func sh(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        let command = "mkdir -p \(sh(dir)) && ln -sfn \(sh(bundledPath)) \(sh(linkPath))"
        let escaped = command.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        var error: NSDictionary?
        NSAppleScript(source: "do shell script \"\(escaped)\" with administrator privileges")?.executeAndReturnError(&error)
        guard let error else { return nil }
        if (error[NSAppleScript.errorNumber] as? Int) == -128 { return "" }   // the user cancelled: say nothing
        return error[NSAppleScript.errorMessage] as? String ?? "Could not install the command."
    }
}
