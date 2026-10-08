import Foundation
import AppKit

// `omni` - the command line for the running Omni app. A generic MCP client: it asks the app for
// its tools (`tools/list`) and turns each one into a subcommand whose flags are the tool's
// parameters, so a tool or a parameter added to the app is here without a line changed. It talks
// over the app's owner-only Unix socket (always on while the app runs, Serving toggle or not), or
// over HTTP to another Mac with --url. See docs/CLI.md.

let bundleID = "io.hanxiao.omni"

struct Failure: Error { let message: String; var code: Int32 = 1 }

func fail(_ message: String, code: Int32 = 1) -> Never {
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
        guard i < args.count else { fail("\(flag) needs a value") }
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

/// One HTTP/1.1 exchange over a Unix socket: request out, `Connection: close`, read to EOF.
func unixExchange(_ path: String, body: Data) throws -> (status: Int, body: Data) {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw Failure(message: "socket: \(String(cString: strerror(errno)))") }
    defer { close(fd) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(path.utf8)
    guard bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else { throw Failure(message: "socket path too long: \(path)") }
    withUnsafeMutableBytes(of: &addr.sun_path) { raw in raw.copyBytes(from: bytes); raw[bytes.count] = 0 }
    let ok = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0 }
    }
    guard ok else { throw Failure(message: "not running", code: 69) }
    var request = Data("POST /mcp HTTP/1.1\r\nHost: omni\r\nContent-Type: application/json\r\nAccept: application/json\r\nUser-Agent: omni-cli\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
    request.append(body)
    try request.withUnsafeBytes { raw in
        var sent = 0
        while sent < raw.count {
            let n = write(fd, raw.baseAddress! + sent, raw.count - sent)
            if n <= 0 { throw Failure(message: "write: \(String(cString: strerror(errno)))") }
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
    guard let (status, head, rest) = splitHead(response) else { throw Failure(message: "no response from Omni") }
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
    guard let url = URL(string: trimmed.hasSuffix("/mcp") ? trimmed : trimmed + "/mcp") else { throw Failure(message: "bad --url: \(base)") }
    var req = URLRequest(url: url, timeoutInterval: 600)
    req.httpMethod = "POST"
    req.httpBody = body
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.setValue("application/json", forHTTPHeaderField: "Accept")
    if let token, !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<(Int, Data), Error> = .failure(Failure(message: "no response"))
    URLSession.shared.dataTask(with: req) { data, resp, err in
        if let err { result = .failure(Failure(message: "\(url.absoluteString): \(err.localizedDescription)", code: 69)) }
        else { result = .success(((resp as? HTTPURLResponse)?.statusCode ?? 0, data ?? Data())) }
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
        } catch let f as Failure where f.code == 69 {
            // Not running, or still loading its model. Start it if allowed, then wait for the socket.
            guard mayLaunch, socketOverride == nil else {
                throw Failure(message: "Omni is not running (no socket at \(path)).", code: 69)
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
                    ? "Omni is running but not answering on \(path). It may be a version without the command line socket: update Omni."
                    : "Omni did not come up within 3 minutes.", code: 69)
            }
            (status, data) = answer
        }
    }
    if status == 401 { throw Failure(message: "Omni refused the request: wrong or missing --token.", code: 77) }
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw Failure(message: "Omni answered HTTP \(status) with no JSON: \(String(decoding: data.prefix(300), as: UTF8.self))")
    }
    if let err = obj["error"] as? [String: Any] {
        throw Failure(message: err["message"] as? String ?? "error", code: 2)
    }
    return obj["result"] as? [String: Any] ?? [:]
}

func printJSON(_ obj: Any) {
    let d = (try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])) ?? Data()
    print(String(decoding: d, as: UTF8.self))
}

/// The command as the agent should type it: this binary's own resolved path.
var commandPath: String { URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().path }

// MARK: - Arguments to a tool call

func arguments(for tool: [String: Any], _ words: [String]) throws -> [String: Any] {
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
        var name = key.replacingOccurrences(of: "-", with: "_")
        if schema.params.first(where: { $0.name == name }) == nil, name.hasPrefix("no_") {
            name = String(name.dropFirst(3)); negated = true
        }
        guard let p = schema.params.first(where: { $0.name == name }) else {
            let flags = schema.params.filter { !$0.positional }.map { "--" + $0.flag }.joined(separator: ", ")
            throw Failure(message: "unknown option --\(key). Options: \(flags)", code: 64)
        }
        func next() throws -> String {
            if let inline { return inline }
            i += 1
            guard i < words.count else { throw Failure(message: "--\(p.flag) needs a value", code: 64) }
            return words[i]
        }
        switch p.type {
        case "boolean":
            if negated { out[p.name] = false }
            else if let inline { out[p.name] = !["false", "0", "no"].contains(inline.lowercased()) }
            else { out[p.name] = true }
        case "integer":
            let v = try next()
            guard let n = Int(v) else { throw Failure(message: "--\(p.flag) takes a whole number, not '\(v)'", code: 64) }
            out[p.name] = n
        case "number":
            let v = try next()
            guard let n = Double(v) else { throw Failure(message: "--\(p.flag) takes a number, not '\(v)'", code: 64) }
            out[p.name] = n
        case "array":
            let v = try next()
            var list = out[p.name] as? [Any] ?? []
            list.append(p.itemType == "integer" ? (Int(v) as Any? ?? v) : (p.itemType == "number" ? (Double(v) as Any? ?? v) : v))
            out[p.name] = list
        case "object":
            let v = try next()
            guard let o = try? JSONSerialization.jsonObject(with: Data(v.utf8)) else {
                throw Failure(message: "--\(p.flag) takes a JSON object", code: 64)
            }
            out[p.name] = o
        default:
            out[p.name] = try next()
        }
        i += 1
    }
    if !bare.isEmpty {
        guard let p = schema.params.first(where: \.positional) else {
            throw Failure(message: "unexpected argument '\(bare[0])'", code: 64)
        }
        switch p.type {
        case "array": out[p.name] = (out[p.name] as? [Any] ?? []) + bare
        case "integer":
            guard bare.count == 1, let n = Int(bare[0]) else { throw Failure(message: "<\(p.name)> is a whole number", code: 64) }
            out[p.name] = n
        default: out[p.name] = bare.joined(separator: " ")   // `omni search red sports car` needs no quotes
        }
    }
    for p in schema.params where p.required && out[p.name] == nil {
        throw Failure(message: "missing <\(p.name)>. Usage: omni \(schema.usage(tool["name"] as? String ?? ""))", code: 64)
    }
    return out
}

func toolHelp(_ tool: [String: Any]) -> String {
    let name = tool["name"] as? String ?? ""
    let schema = ToolSchema(tool)
    var s = "usage: omni \(schema.usage(name))\n\n"
    if let d = tool["description"] as? String { s += d + "\n" }
    let rows = schema.params.map { p -> (String, String) in
        (p.positional ? "<\(p.name)>" : "--\(p.flag)" + (p.metavar.isEmpty ? "" : " " + p.metavar), p.description)
    }
    if !rows.isEmpty {
        s += "\n"
        for (k, v) in rows { s += "  \(k)\n      \(v)\n" }
    }
    s += "\nGlobal: --json (raw result), --url <http://host:port> --token <t> (another Mac), --no-launch\n"
    return s
}

let usage = """
usage: omni <tool> [arguments] [options]
       omni tools | omni skill | omni <tool> --help

Search and manage the files Omni has indexed on this Mac, through the running app (started if
needed). The tools are the app's MCP tools; their options are listed by `omni <tool> --help`.

Global options: --json, --url <http://host:port>, --token <token>, --socket <path>, --no-launch
"""

// MARK: - Main

takeGlobals()

do {
    let first = args.first ?? "help"
    switch first {
    case "help", "-h", "--help":
        print(usage)
        if let tools = try? rpc("tools/list")["tools"] as? [[String: Any]] {
            print("\nTools:")
            for t in tools { print("  \((t["name"] as? String ?? "").padding(toLength: 16, withPad: " ", startingAt: 0))\(t["title"] as? String ?? "")") }
        }
    case "--version", "version":
        let info = try rpc("initialize", ["protocolVersion": "2025-06-18", "capabilities": [:] as [String: Any],
                                          "clientInfo": ["name": "omni-cli", "version": "1"]])
        print((info["serverInfo"] as? [String: Any])?["version"] as? String ?? "?")
    case "tools":
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        if jsonOutput { printJSON(tools) } else {
            for t in tools { print("\((t["name"] as? String ?? "").padding(toLength: 16, withPad: " ", startingAt: 0))\(t["title"] as? String ?? "")") }
        }
    case "skill":
        let info = try rpc("initialize", ["protocolVersion": "2025-06-18", "capabilities": [:] as [String: Any],
                                          "clientInfo": ["name": "omni-cli", "version": "1"]])
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        print(AgentSkill.render(instructions: info["instructions"] as? String ?? "", tools: tools, command: commandPath))
    default:
        let tools = try rpc("tools/list")["tools"] as? [[String: Any]] ?? []
        guard let tool = tools.first(where: { ($0["name"] as? String) == first }) else {
            let names = tools.compactMap { $0["name"] as? String }.joined(separator: ", ")
            throw Failure(message: "unknown tool '\(first)'. Tools: \(names)", code: 64)
        }
        let words = Array(args.dropFirst())
        if words.contains("--help") || words.contains("-h") { print(toolHelp(tool), terminator: ""); exit(0) }
        let result = try rpc("tools/call", ["name": first, "arguments": try arguments(for: tool, words)])
        let isError = result["isError"] as? Bool ?? false
        if jsonOutput {
            printJSON(result["structuredContent"] ?? result)
        } else {
            let content = result["content"] as? [[String: Any]] ?? []
            var texts: [String] = []
            var images = 0
            for c in content {
                if let t = c["text"] as? String { texts.append(t) } else if c["type"] as? String == "image" { images += 1 }
            }
            let text = texts.joined(separator: "\n\n") + (images > 0 ? "\n\n(\(images) image(s) omitted; use --json)" : "")
            if isError { FileHandle.standardError.write(Data((text + "\n").utf8)) } else { print(text) }
        }
        exit(isError ? 1 : 0)
    }
} catch let f as Failure {
    fail(f.message, code: f.code)
} catch {
    fail("\(error)")
}
