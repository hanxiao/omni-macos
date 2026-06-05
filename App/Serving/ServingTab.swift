import SwiftUI
import AppKit
import OmniKit

/// Settings > Serving. Binds only to the shared ServingController surface AppModel owns:
/// read-write enabled/scope/port/bearerToken, read-only state/boundAddress/counters/log, and the
/// clearLog() action. Follows the same Form { Section }.formStyle(.grouped) idiom and the explicit
/// Binding(get:set:) pattern as the other tabs - never $model, since AppModel is @Observable.
struct ServingTab: View {
    @Environment(AppModel.self) private var model: AppModel
    @State private var revealToken = false
    @State private var exampleKind: ExampleKind = .search
    @State private var embedSchema: EmbedSchema = .openai

    /// Top-level example category: the search endpoint, or an embedding endpoint.
    private enum ExampleKind: String, CaseIterable, Identifiable {
        case search = "Search", embed = "Embed"
        var id: String { rawValue }
    }
    /// The embedding API schema styles the server speaks (all served at once).
    private enum EmbedSchema: String, CaseIterable, Identifiable {
        case openai = "OpenAI", jina = "Jina", cohere = "Cohere", gemini = "Gemini"
        var id: String { rawValue }
    }

    var body: some View {
        Form {
            serverSection
            accessSection
            exampleSection
            requestsSection
        }
        .formStyle(.grouped)
        .frame(height: 520)   // matches the Content tab so switching tall tabs doesn't jump
    }

    // MARK: - Server

    @ViewBuilder private var serverSection: some View {
        Section {
            Toggle("Serve Omni over HTTP", isOn: Binding(
                get: { model.serving.enabled },
                set: { model.serving.enabled = $0 }
            ))
            .toggleStyle(.switch)

            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                Spacer()
                if !model.serving.boundAddress.isEmpty {
                    Text(model.serving.boundAddress)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        } header: {
            Text("Server")
        } footer: {
            Text("Serves OpenAI, Jina, Cohere, and Gemini embedding endpoints plus search, backed by the local model.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var statusColor: Color {
        switch model.serving.state {
        case .running: return .green
        case .portInUse, .failed: return .orange
        case .stopped: return .secondary
        }
    }

    private var statusText: String {
        switch model.serving.state {
        case .running: return "Running"
        case .stopped: return "Stopped"
        case .portInUse: return "Port in use"
        case .failed(let m): return m.isEmpty ? "Failed" : m
        }
    }

    // MARK: - Access

    @ViewBuilder private var accessSection: some View {
        Section {
            Picker("Reachable from", selection: Binding(
                get: { model.serving.scope },
                set: { model.serving.scope = $0 }
            )) {
                Text("This Mac only").tag(ServingScope.local)
                Text("Local network").tag(ServingScope.public)
            }

            TextField("Port", value: Binding(
                get: { model.serving.port },
                set: { model.serving.port = min(65535, max(1, $0)) }   // valid TCP port range
            ), format: .number.grouping(.never))
            .frame(width: 90)

            HStack(spacing: 6) {
                let token = Binding(
                    get: { model.serving.bearerToken },
                    set: { model.serving.bearerToken = $0 }
                )
                Group {
                    if revealToken {
                        TextField("Bearer token", text: token).font(.callout.monospaced())
                    } else {
                        SecureField("Bearer token", text: token)
                    }
                }
                Button { revealToken.toggle() } label: {
                    Image(systemName: revealToken ? "eye.slash" : "eye")
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help(revealToken ? "Hide" : "Show")
                .disabled(model.serving.bearerToken.isEmpty)
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(model.serving.bearerToken, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy").disabled(model.serving.bearerToken.isEmpty)
                Button("Generate New") {
                    model.serving.bearerToken = Self.newToken()
                    revealToken = true   // show it so it can be copied
                }
                .buttonStyle(.link)
            }
        } header: {
            Text("Access")
        } footer: {
            Text("Local network reaches other devices, so set a token to require it. Port or scope changes restart the server.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private static func newToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 24)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    // MARK: - Example

    @ViewBuilder private var exampleSection: some View {
        Section {
            Picker("Example", selection: $exampleKind) {
                ForEach(ExampleKind.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if exampleKind == .embed {
                // Choose which schema's request shape to show; the server answers all of them.
                Picker("Schema", selection: $embedSchema) {
                    ForEach(EmbedSchema.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            HStack(alignment: .top) {
                Text(exampleCurl)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(exampleCurl, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Copy")
            }
        } header: {
            Text("Example")
        } footer: {
            Text("Start the server, then run the selected example in a terminal.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Base URL for examples: the live bound address, or the configured local address when stopped.
    private var exampleBase: String {
        model.serving.boundAddress.isEmpty ? "http://127.0.0.1:\(model.serving.port)" : model.serving.boundAddress
    }

    /// A ready-to-run curl for the selected API, including the auth header when a token is set.
    private var exampleCurl: String {
        let base = exampleBase
        let token = model.serving.bearerToken
        let auth = token.isEmpty ? "" : " -H 'Authorization: Bearer \(token)'"
        let geminiAuth = token.isEmpty ? "" : " -H 'x-goog-api-key: \(token)'"
        let ct = " -H 'Content-Type: application/json'"
        if exampleKind == .search {
            return "curl \(base)/v1/search\(ct)\(auth) -d '{\"query\":\"invoices\",\"top_k\":5}'"
        }
        switch embedSchema {
        case .openai:
            return "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":[\"your text\"]}'"
        case .jina:
            return "curl \(base)/v1/embeddings\(ct)\(auth) -d '{\"model\":\"omni\",\"input\":\"your text\",\"task\":\"retrieval.query\"}'"
        case .cohere:
            return "curl \(base)/v2/embed\(ct)\(auth) -d '{\"model\":\"omni\",\"texts\":[\"your text\"],\"input_type\":\"search_document\",\"embedding_types\":[\"float\"]}'"
        case .gemini:
            return "curl \(base)/v1beta/models/omni:embedContent\(ct)\(geminiAuth) -d '{\"content\":{\"parts\":[{\"text\":\"your text\"}]}}'"
        }
    }

    // MARK: - Requests

    @ViewBuilder private var requestsSection: some View {
        Section {
            if model.serving.log.isEmpty {
                Text("No requests yet")
                    .foregroundStyle(.secondary)
            } else {
                List(model.serving.log) { entry in
                    LogRow(entry: entry)
                        .listRowSeparator(.hidden)
                }
                .listStyle(.plain)
                .frame(height: 200)
            }
        } header: {
            HStack {
                Text("Requests")
                Spacer()
                Text("\(model.serving.requestCount) served \u{00B7} \(model.serving.errorCount) failed")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Button("Clear") { model.serving.clearLog() }
                    .buttonStyle(.link)
                    .disabled(model.serving.log.isEmpty)
            }
        } footer: {
            Text("Recent requests, newest first.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// One row in the live request log: time, method, path, status, and latency. Kept fixed-height and
/// monospaced so the columns line up and the List inside the grouped Form scrolls cleanly.
private struct LogRow: View {
    let entry: LogEntry

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    private var statusColor: Color {
        switch entry.status {
        case 200 ..< 300: return .green
        case 400 ..< 500: return .orange
        default: return .red
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(Self.timeFormatter.string(from: entry.time))
                .foregroundStyle(.secondary)
            Text(entry.method)
                .frame(width: 44, alignment: .leading)
            Text(entry.path)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("\(entry.status)")
                .foregroundStyle(statusColor)
            Text(String(format: "%.0f ms", entry.ms))
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .trailing)
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.vertical, 1)
    }
}
