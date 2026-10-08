import Foundation
import AppKit

// `omni` - the command line for the running Omni app. A generic MCP client: it asks the app for
// its tools (`tools/list`) and turns each one into a subcommand whose flags are the tool's
// parameters, so a tool or a parameter added to the app is here without a line changed. It talks
// over the app's owner-only Unix socket (always on while the app runs, Serving toggle or not), or
// over HTTP to another Mac with --url. See docs/CLI.md.
//
// stdout is the answer and nothing else; stderr is everything about it. Exit codes: 0 done,
// 1 the command needs fixing, 2 Omni unavailable or failed, 130 interrupted (the default SIGINT).

let bundleID = "io.hanxiao.omni"

enum Exit {
    static let ok: Int32 = 0
    static let user: Int32 = 1          // fix the command: stderr says how
    static let unavailable: Int32 = 2   // Omni not running, not answering, or failed
}

struct Failure: Error { let message: String; var code: Int32 = Exit.user }

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

// MARK: - Options

var args = Array(CommandLine.arguments.dropFirst())
var jsonOutput = false
var remoteURL = ProcessInfo.processInfo.environment["OMNI_URL"]
var token = ProcessInfo.processInfo.environment["OMNI_TOKEN"]
var socketOverride = ProcessInfo.processInfo.environment["OMNI_SOCKET"]
var mayLaunch = true

/// Global options may sit anywhere; they are taken out before the tool's own flags are read.
@MainActor func takeGlobals() {
    var rest: [String] = []
    var i = 0
    func value(_ flag: String) -> String {
        i += 1
        guard i < args.count else { fail("omni: \(flag) needs a value", code: Exit.user) }
        return args[i]
    }
    while i < args.count {
        let a = args[i]
        switch a {
        case "--json": jsonOutput = true
        case "--no-launch": mayLaunch = false
        case "--url": remoteURL = value(a)
        case "--token": token = value(a)
        case "--socket": socketOverride = value(a)
        case "--": rest += args[i...]; i = args.count; continue
        default:
            if a.hasPrefix("--url=") { remoteURL = String(a.dropFirst(6)) }
            else if a.hasPrefix("--token=") { token = String(a.dropFirst(8)) }
            else if a.hasPrefix("--socket=") { socketOverride = String(a.dropFirst(9)) }
            else { rest.append(a) }
        }
        i += 1
    }
    args = rest
}

// MARK: - Transport

var defaultSocketPath: String {
    let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    return support.appendingPathComponent("Omni/omni.sock").path
}

struct NotRunning: Error {}

/// One HTTP/1.1 exchange over a Unix socket: request out, `Connection: close`, read to EOF.
func unixExchange(_ path: String, body: Data) throws -> (status: Int, body: Data) {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw Failure(message: "omni: socket: \(String(cString: strerror(errno)))", code: Exit.unavailable) }
    defer { close(fd) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(path.utf8)
    guard bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else {
        throw Failure(message: "omni: --socket path is too long (over 103 bytes): \(path)")
    }
    withUnsafeMutableBytes(of: &addr.sun_path) { raw in raw.copyBytes(from: bytes); raw[bytes.count] = 0 }
    let ok = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0 }
    }
    guard ok else { throw NotRunning() }
    var request = Data("POST /mcp HTTP/1.1\r\nHost: omni\r\nContent-Type: application/json\r\nAccept: application/json\r\nUser-Agent: omni-cli\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
    request.append(body)
    try request.withUnsafeBytes { raw in
        var sent = 0
        while sent < raw.count {
            let n = write(fd, raw.baseAddress! + sent, raw.count - sent)
            if n <= 0 { throw Failure(message: "omni: Omni closed the connection (\(String(cString: strerror(errno)))).", code: Exit.unavailable) }
            sent += n
        }
    }
    var response = Data()
    var buf = [UInt8](repeating: 0, count: 65536)
    while true {
        let n = read(fd, &buf, buf.count)
        if n <= 0 { break }
        response.append(contentsOf: buf[0 ..< n])
        // A keep-alive server would not close: stop once Content-Length is satisfied.
        if let (status, head, rest) = splitHead(response), let len = contentLength(head), rest.count >= len {
            return (status, rest.prefix(len))
        }
    }
    guard let (status, head, rest) = splitHead(response) else {
        throw Failure(message: "omni: Omni sent no response.", code: Exit.unavailable)
    }
    if head.lowercased().contains("transfer-encoding: chunked") { return (status, dechunk(rest)) }
    return (status, rest)
}

func splitHead(_ d: Data) -> (Int, String, Data)? {
    guard let r = d.range(of: Data("\r\n\r\n".utf8)) else { return nil }
    let head = String(decoding: d[..<r.lowerBound], as: UTF8.self)
    let status = Int(head.split(separator: " ").dropFirst().first ?? "") ?? 0
    return (status, head, Data(d[r.upperBound...]))
}

func contentLength(_ head: String) -> Int? {
    for line in head.split(separator: "\r\n") where line.lowercased().hasPrefix("content-length:") {
        return Int(line.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces))
    }
    return nil
}

func dechunk(_ d: Data) -> Data {
    var out = Data(), i = d.startIndex
    while let r = d[i...].range(of: Data("\r\n".utf8)) {
        guard let n = Int(String(decoding: d[i ..< r.lowerBound], as: UTF8.self), radix: 16), n > 0 else { break }
        let start = r.upperBound, end = d.index(start, offsetBy: n, limitedBy: d.endIndex) ?? d.endIndex
        out.append(d[start ..< end])
        i = d.index(end, offsetBy: 2, limitedBy: d.endIndex) ?? d.endIndex
    }
    return out
}

@MainActor func httpExchange(_ base: String, body: Data) throws -> (status: Int, body: Data) {
    let trimmed = base.hasSuffix("/") ? String(base.dropLast()) : base
    guard let url = URL(string: trimmed.hasSuffix("/mcp") ? trimmed : trimmed + "/mcp"), url.scheme != nil else {
        throw Failure(message: "omni: --url must be an address like http://192.168.1.20:51234 (got \(base))")
    }
    var req = URLRequest(url: url, timeoutInterval: 600)
    req.httpMethod = "POST"
    req.httpBody = body
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.setValue("application/json", forHTTPHeaderField: "Accept")
    if let token, !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<(Int, Data), Error> = .failure(Failure(message: "omni: no response", code: Exit.unavailable))
    URLSession.shared.dataTask(with: req) { data, resp, err in
        if let err {
            result = .failure(Failure(message: "omni: cannot reach Omni at \(url.absoluteString) (\(err.localizedDescription)). Check that Serving is on there and reachable from this network.", code: Exit.unavailable))
        } else {
            result = .success(((resp as? HTTPURLResponse)?.statusCode ?? 0, data ?? Data()))
        }
        done.signal()
    }.resume()
    done.wait()
    return try result.get()
}

/// Whether an Omni is running at all. Its socket can be missing while it is (still loading its
/// model, or a version from before the socket), and then it must NOT be `open`ed: open sends a
/// running app a reopen event, and Omni answers that by showing its window.
var appIsRunning: Bool { !NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).isEmpty }

/// The app this command ships in (…/Omni.app/Contents/Helpers/omni), else Omni by bundle id.
@MainActor func launchApp() {
    let me = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let bundle = me.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    p.arguments = bundle.pathExtension == "app" ? ["-g", "-j", bundle.path] : ["-g", "-j", "-b", bundleID]
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    try? p.run()
    p.waitUntilExit()
}

var requestID = 0

@MainActor func rpc(_ method: String, _ params: [String: Any] = [:]) throws -> [String: Any] {
    requestID += 1
    let body = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": requestID, "method": method, "params": params])
    let (status, data): (Int, Data)
    if let remoteURL {
        (status, data) = try httpExchange(remoteURL, body: body)
    } else {
        let path = socketOverride ?? defaultSocketPath
        do {
            (status, data) = try unixExchange(path, body: body)
        } catch is NotRunning {
            // Not running, or still loading its model. Start it if allowed, then wait for the socket.
            if socketOverride != nil {
                throw Failure(message: "omni: nothing is answering at \(path) (from --socket or OMNI_SOCKET). Check the path, or drop it to use Omni's own socket.",
                              code: Exit.unavailable)
            }
            guard mayLaunch else {
                throw Failure(message: "omni: Omni is not running. Open Omni, or drop --no-launch to let omni start it.", code: Exit.unavailable)
            }
            let running = appIsRunning
            if !running {
                FileHandle.standardError.write(Data("Starting Omni...\n".utf8))
                launchApp()
            }
            let deadline = Date().addingTimeInterval(running ? 60 : 180)
            var answer: (Int, Data)?
            while answer == nil, Date() < deadline {
                Thread.sleep(forTimeInterval: 0.5)
                answer = try? unixExchange(path, body: body)
            }
            guard let answer else {
                throw Failure(message: running
                    ? "omni: Omni is running but not answering on \(path). It is still loading (try again in a minute) or a version without the command line (update Omni)."
                    : "omni: Omni did not start within 3 minutes. Open it once by hand to see what it needs.", code: Exit.unavailable)
            }
            (status, data) = answer
        }
    }
    if status == 401 {
        throw Failure(message: token == nil ? "omni: Omni at \(remoteURL ?? "") needs a token: pass --token <token> (Settings > Serving on that Mac)."
                                            : "omni: Omni refused the token. Copy it again from Settings > Serving on that Mac.")
    }
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw Failure(message: "omni: Omni answered HTTP \(status) without JSON: \(String(decoding: data.prefix(300), as: UTF8.self))", code: Exit.unavailable)
    }
    if let err = obj["error"] as? [String: Any] {
        throw Failure(message: "omni: " + (err["message"] as? String ?? "Omni returned an error."), code: Exit.unavailable)
    }
    return obj["result"] as? [String: Any] ?? [:]
}

func printJSON(_ obj: Any) {
    let d = (try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])) ?? Data()
    print(String(decoding: d, as: UTF8.self))
}

/// The command as an agent should type it: this binary's own resolved path, so it works off PATH.
var commandPath: String { URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().path }

// MARK: - Arguments to a tool call

/// The closest of `candidates` to `word`, when it is close enough to be a typo.
func suggestion(_ word: String, _ candidates: [String]) -> String? {
    func distance(_ a: [Character], _ b: [Character]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var row = Array(0 ... b.count)
        for i in 1 ... a.count {
            var prev = row[0]
            row[0] = i
            for j in 1 ... b.count {
                let cur = row[j]
                row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] == b[j - 1] ? 0 : 1))
                prev = cur
            }
        }
        return row[b.count]
    }
    let w = Array(word.lowercased())
    let best = candidates.map { ($0, distance(w, Array($0.lowercased()))) }.min { $0.1 < $1.1 }
    guard let best, best.1 <= max(2, w.count / 3) else { return nil }
    return best.0
}

func arguments(for tool: [String: Any], _ words: [String]) throws -> [String: Any] {
    let name = tool["name"] as? String ?? ""
    let schema = ToolSchema(tool)
    var out: [String: Any] = [:]
    var bare: [String] = []
    var i = 0
    while i < words.count {
        let w = words[i]
        guard w.hasPrefix("--"), w.count > 2 else { bare.append(w); i += 1; continue }
        var key = String(w.dropFirst(2)), inline: String?
        if let eq = key.firstIndex(of: "=") { inline = String(key[key.index(after: eq)...]); key = String(key[..<eq]) }
        var negated = false
        var pname = key.replacingOccurrences(of: "-", with: "_")
        if schema.params.first(where: { $0.name == pname }) == nil, pname.hasPrefix("no_") {
            pname = String(pname.dropFirst(3)); negated = true
        }
        guard let p = schema.params.first(where: { $0.name == pname && !$0.positional }) else {
            let flags = schema.params.filter { !$0.positional }.map { "--" + $0.flag }
            let hint = suggestion("--" + key, flags).map { "Did you mean \($0)? " } ?? ""
            throw Failure(message: "omni \(name): unknown option --\(key). \(hint)Options: \(flags.joined(separator: ", ")). See: omni \(name) --help")
        }
        func next() throws -> String {
            if let inline { return inline }
            i += 1
            guard i < words.count else {
                throw Failure(message: "omni \(name): --\(p.flag) needs a value \(p.metavar). See: omni \(name) --help")
            }
            return words[i]
        }
        func checkEnum(_ v: String) throws {
            guard !p.enumValues.isEmpty, !p.enumValues.contains(v) else { return }
            let hint = suggestion(v, p.enumValues).map { "Did you mean \($0)? " } ?? ""
            throw Failure(message: "omni \(name): --\(p.flag) takes one of \(p.enumValues.joined(separator: ", ")), not '\(v)'. \(hint)"
                          + (p.type == "array" ? "Repeat the flag for several." : ""))
        }
        switch p.type {
        case "boolean":
            if negated { out[p.name] = false }
            else if let inline { out[p.name] = !["false", "0", "no"].contains(inline.lowercased()) }
            else { out[p.name] = true }
        case "integer":
            let v = try next()
            guard let n = Int(v) else { throw Failure(message: "omni \(name): --\(p.flag) takes a whole number, not '\(v)'. Example: --\(p.flag) 5") }
            out[p.name] = n
        case "number":
            let v = try next()
            guard let n = Double(v) else { throw Failure(message: "omni \(name): --\(p.flag) takes a number, not '\(v)'. Example: --\(p.flag) 0.4") }
            out[p.name] = n
        case "array":
            let v = try next()
            try checkEnum(v)
            var list = out[p.name] as? [Any] ?? []
            list.append(p.itemType == "integer" ? (Int(v) as Any? ?? v) : (p.itemType == "number" ? (Double(v) as Any? ?? v) : v))
            out[p.name] = list
        case "object":
            let v = try next()
            guard let o = try? JSONSerialization.jsonObject(with: Data(v.utf8)) else {
                throw Failure(message: "omni \(name): --\(p.flag) takes a JSON object, e.g. --\(p.flag) '{\"key\": \"value\"}'")
            }
            out[p.name] = o
        default:
            let v = try next()
            try checkEnum(v)
            out[p.name] = v
        }
        i += 1
    }
    if !bare.isEmpty {
        guard let p = schema.params.first(where: \.positional) else {
            throw Failure(message: "omni \(name) takes no bare arguments (got '\(bare[0])'). Usage: omni \(schema.usage(name))")
        }
        switch p.type {
        case "array": out[p.name] = (out[p.name] as? [Any] ?? []) + bare
        case "integer":
            guard bare.count == 1, let n = Int(bare[0]) else {
                throw Failure(message: "omni \(name): <\(p.name)> is a whole number. Usage: omni \(schema.usage(name))")
            }
            out[p.name] = n
        default: out[p.name] = bare.joined(separator: " ")   // `omni search red sports car` needs no quotes
        }
    }
    for p in schema.params where p.required && out[p.name] == nil {
        let what = p.positional ? "<\(p.name)>" : "--\(p.flag)"
        let example = Examples.commands(for: tool, command: "omni").first.map { " Example: \($0)" } ?? ""
        throw Failure(message: "omni \(name): missing \(what). Usage: omni \(schema.usage(name)).\(example)")
    }
    return out
}

/// A tool's own error, in this command's words: the server names parameters as MCP does
/// ('modified_after'); here they are flags (--modified-after) or the bare argument (<query>).
func inCLITerms(_ message: String, tool: [String: Any]) -> String {
    var m = message
    for p in ToolSchema(tool).params {
        let cli = p.positional ? "<\(p.name)>" : "--\(p.flag)"
        m = m.replacingOccurrences(of: "'\(p.name)'", with: cli)
        if p.name.contains("_") { m = m.replacingOccurrences(of: p.name, with: cli) }
    }
    let name = tool["name"] as? String ?? ""
    if m.hasPrefix("\(name) failed: ") { m = "omni \(name): " + m.dropFirst("\(name) failed: ".count) }
    return m
}

// MARK: - Help, in three layers

/// Layer 0: what there is.
@MainActor func printTopHelp() {
    print("""
    usage: omni <command> [arguments] [options]

    Search and manage the files Omni has indexed on this Mac, through the running app (started
    if needed). The commands are the app's MCP tools.

    """)
    if let tools = try? rpc("tools/list")["tools"] as? [[String: Any]] {
        let width = tools.compactMap { ($0["name"] as? String)?.count }.max() ?? 0
        print("Commands:")
        for t in tools {
            print("  " + (t["name"] as? String ?? "").padding(toLength: width + 2, withPad: " ", startingAt: 0) + (t["title"] as? String ?? ""))
        }
        let search = tools.first { $0["name"] as? String == "search" } ?? [:]
        let examples = Examples.commands(for: search, command: "omni").prefix(2)
        if !examples.isEmpty { print("\nExamples:\n" + examples.map { "  " + $0 }.joined(separator: "\n")) }
        print("")
    } else {
        print("(Omni is not reachable, so its commands cannot be listed.)\n")
    }
    print("""
    omni <command> --help      usage, examples and options
    omni <command> --help-all  every option in full
    omni skill                 SKILL.md for agents

    Global: --json (structured output), --url <http://host:port> and --token <t> for another
    Mac's Serving, --no-launch. stdout is the answer; stderr is everything about it.
    Exit codes: 0 done, 1 fix the command (stderr says how), 2 Omni unavailable, 130 interrupted.
    """)
}

/// Layer 1 (`--help`): usage, what it does, examples, and one line per option.
/// Layer 2 (`--help-all`): the same with every description in full.
func toolHelp(_ tool: [String: Any], full: Bool) -> String {
    let name = tool["name"] as? String ?? ""
    let schema = ToolSchema(tool)
    var s = "usage: omni \(schema.usage(name))\n\n"
    if let d = tool["description"] as? String {
        s += (full ? d : ToolSchema.firstSentence(d)) + "\n"
    }
    let examples = Examples.commands(for: tool, command: "omni")
    if !examples.isEmpty { s += "\nExamples:\n" + examples.map { "  " + $0 }.joined(separator: "\n") + "\n" }
    let rows = schema.params.map { p in (p.display, full ? p.description : p.summary) }
    if !rows.isEmpty {
        s += "\nOptions:\n"
        if full {
            for (k, v) in rows { s += "  \(k)\n      \(v)\n" }
        } else {
            let width = min(30, rows.map(\.0.count).max() ?? 0)
            for (k, v) in rows {
                let key = k.count > width ? k + "\n  " + String(repeating: " ", count: width)
                                          : k.padding(toLength: width, withPad: " ", startingAt: 0)
                s += "  \(key)  \(v)\n"
            }
        }
    }
    if !full { s += "\nEvery option in full: omni \(name) --help-all\n" }
    return s
}

// MARK: - Main

takeGlobals()

do {
    let first = args.first ?? "help"
    switch first {
    case "help", "-h", "--help":
        if args.count > 1 {
            let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
            guard let tool = tools.first(where: { ($0["name"] as? String) == args[1] }) else {
                throw Failure(message: "omni: no command '\(args[1])'. Commands: \(tools.compactMap { $0["name"] as? String }.joined(separator: ", "))")
            }
            print(toolHelp(tool, full: false), terminator: "")
        } else {
            printTopHelp()
        }
    case "--version", "version":
        let info = try rpc("initialize", ["protocolVersion": "2025-06-18", "capabilities": [:] as [String: Any],
                                          "clientInfo": ["name": "omni-cli", "version": "1"]])
        print((info["serverInfo"] as? [String: Any])?["version"] as? String ?? "?")
    case "tools":
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        if jsonOutput { printJSON(tools) } else {
            for t in tools { print("\(t["name"] as? String ?? "")\t\(t["title"] as? String ?? "")") }
        }
    case "skill":
        let info = try rpc("initialize", ["protocolVersion": "2025-06-18", "capabilities": [:] as [String: Any],
                                          "clientInfo": ["name": "omni-cli", "version": "1"]])
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        print(AgentSkill.render(instructions: info["instructions"] as? String ?? "", tools: tools, command: commandPath))
    default:
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        let names = tools.compactMap { $0["name"] as? String }
        guard let tool = tools.first(where: { ($0["name"] as? String) == first }) else {
            let hint = suggestion(first, names + ["tools", "skill", "help"]).map { "Did you mean '\($0)'? " } ?? ""
            throw Failure(message: "omni: no command '\(first)'. \(hint)Commands: \(names.joined(separator: ", ")). See: omni --help")
        }
        let words = Array(args.dropFirst())
        if words.contains("--help-all") { print(toolHelp(tool, full: true), terminator: ""); exit(Exit.ok) }
        if words.contains("--help") || words.contains("-h") { print(toolHelp(tool, full: false), terminator: ""); exit(Exit.ok) }
        let result = try rpc("tools/call", ["name": first, "arguments": try arguments(for: tool, words),
                                            "_meta": [CLIProtocol.cliKey: true]])
        let content = result["content"] as? [[String: Any]] ?? []
        // A tool's error is about the command: stderr, in this command's terms, exit 1.
        if result["isError"] as? Bool ?? false {
            fail(inCLITerms(content.compactMap { $0["text"] as? String }.joined(separator: "\n"), tool: tool), code: Exit.user)
        }
        // Blocks the server marked as ABOUT the answer go to stderr; the rest is the answer.
        var out: [String] = []
        var images = 0
        for c in content {
            let toStderr = (c["_meta"] as? [String: Any])?[CLIProtocol.stderrKey] as? Bool == true
            if let t = c["text"] as? String {
                if toStderr { FileHandle.standardError.write(Data((t + "\n").utf8)) } else { out.append(t) }
            } else if c["type"] as? String == "image" {
                images += 1
            }
        }
        if jsonOutput {
            printJSON(result["structuredContent"] ?? ["content": content])
        } else {
            if images > 0 { FileHandle.standardError.write(Data("(\(images) inline image(s) not printed: the command line prints text; open the paths instead)\n".utf8)) }
            if !out.isEmpty { print(out.joined(separator: "\n")) }
        }
        exit(Exit.ok)
    }
} catch let f as Failure {
    fail(f.message, code: f.code)
} catch {
    fail("omni: \(error)", code: Exit.unavailable)
}
