import AppIntents
import AppKit
import Foundation
import OmniKit

// SEARCH BY MEANING FROM SHORTCUTS, SIRI AND SPOTLIGHT.
//
// Agents reach Omni over MCP; the rest of macOS reaches apps through App Intents. One background
// action returns the files a query finds - each with the passage that matched - through the very
// search an agent gets (`ServingController.searchForIntents`: the served search, its duplicate
// collapsing and its passages), so a Shortcut can hand them to "Use Model", to Mail, to Finder.
// A file entity whose query answers by meaning lets Siri resolve "the lease agreement" from words;
// two foreground actions open the window on a search or on Find Similar for a file - the second is
// what a Finder Quick Action is built from. Nothing here is a second search.

/// A file Omni found, as Shortcuts and Siri see it.
struct OmniFileEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "File"
    static let defaultQuery = OmniFileQuery()

    /// The absolute path: what Omni's index is keyed by.
    var id: String
    @Property(title: "Name") var name: String
    @Property(title: "Passage") var passage: String
    @Property(title: "Kind") var kind: String
    /// Where in the file it matched: "Page 7", "Line 240", a time in a recording.
    @Property(title: "Location") var location: String
    @Property(title: "File") var url: URL

    init(path: String, passage: String = "", kind: String = "", location: String = "") {
        id = path
        name = (path as NSString).lastPathComponent
        self.passage = passage
        self.kind = kind
        self.location = location
        url = URL(fileURLWithPath: path)
    }

    var displayRepresentation: DisplayRepresentation {
        let where_ = location.isEmpty ? "" : " (\(location))"
        return DisplayRepresentation(title: "\(name)\(where_)",
                                     subtitle: passage.isEmpty ? "\((id as NSString).deletingLastPathComponent)" : "\(passage)")
    }
}

/// Resolves files by MEANING: Siri and the Shortcuts picker pass the words, the served search
/// answers. By identifier, a path is a file.
struct OmniFileQuery: EntityStringQuery {
    func entities(matching string: String) async throws -> [OmniFileEntity] {
        try await OmniIntentSearch.run(string, limit: 10)
    }

    func entities(for identifiers: [String]) async throws -> [OmniFileEntity] {
        identifiers.filter { FileManager.default.fileExists(atPath: $0) }.map { OmniFileEntity(path: $0) }
    }

    func suggestedEntities() async throws -> [OmniFileEntity] { [] }
}

enum OmniIntentSearch {
    struct NotReady: Error, CustomLocalizedStringResourceConvertible {
        var localizedStringResource: LocalizedStringResource { "Omni is still loading its index. Try again in a moment." }
    }

    @MainActor static func run(_ query: String, limit: Int) async throws -> [OmniFileEntity] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        guard let model = AppModel.shared, model.phase == .ready else { throw NotReady() }
        let found = await model.serving.searchForIntents(q, limit: max(1, min(limit, 50)))
        return found.map { r in
            OmniFileEntity(path: r.hit.path, passage: AppModel.oneLine(r.passage ?? r.hit.snippet),
                           kind: r.hit.fileKind, location: r.hit.locator)
        }
    }
}

/// The background action: files for a query, with the passage each matched. Does not open Omni.
struct SearchFilesIntent: AppIntent {
    static let title: LocalizedStringResource = "Search Files by Meaning"
    static let description = IntentDescription(
        "Finds files on this Mac by what they contain, in any language, and returns them with the passage that matched.",
        categoryName: "Search")
    static let openAppWhenRun = false

    @Parameter(title: "Query", requestValueDialog: "What are you looking for?")
    var query: String

    @Parameter(title: "Limit", default: 10, inclusiveRange: (1, 50))
    var limit: Int

    static var parameterSummary: some ParameterSummary {
        Summary("Search files for \(\.$query)") { \.$limit }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<[OmniFileEntity]> & ProvidesDialog {
        let files = try await OmniIntentSearch.run(query, limit: limit)
        let dialog: IntentDialog = files.isEmpty ? "Nothing on this Mac matched." : "Found \(files.count) file\(files.count == 1 ? "" : "s")."
        return .result(value: files, dialog: dialog)
    }
}

/// Open Omni's window on a search.
struct OpenSearchIntent: AppIntent {
    static let title: LocalizedStringResource = "Search in Omni"
    static let description = IntentDescription("Opens Omni with a search.", categoryName: "Search")
    static let openAppWhenRun = true

    @Parameter(title: "Query", requestValueDialog: "What are you looking for?")
    var query: String

    static var parameterSummary: some ParameterSummary { Summary("Search Omni for \(\.$query)") }

    @MainActor func perform() async throws -> some IntentResult {
        AppDelegate.showMainWindow()
        guard let model = AppModel.shared else { throw OmniIntentSearch.NotReady() }
        model.applyParsedQuery(query)
        model.search()
        return .result()
    }
}

/// Open Omni on the files most like this one. As a Shortcut set to "Use as Quick Action", it is a
/// Finder context-menu item.
struct FindSimilarIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Similar in Omni"
    static let description = IntentDescription("Opens Omni with the files most like the one given.", categoryName: "Search")
    static let openAppWhenRun = true

    @Parameter(title: "File")
    var file: IntentFile

    static var parameterSummary: some ParameterSummary { Summary("Find files similar to \(\.$file)") }

    @MainActor func perform() async throws -> some IntentResult {
        guard let url = file.fileURL else { throw OmniIntentSearch.NotReady() }
        AppDelegate.showMainWindow()
        guard let model = AppModel.shared else { throw OmniIntentSearch.NotReady() }
        model.searchBySimilar(to: url.path)
        return .result()
    }
}

/// What Spotlight, Siri and the Shortcuts gallery offer without the user building anything.
struct OmniShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: OpenSearchIntent(), phrases: ["Search in \(.applicationName)", "Find files with \(.applicationName)"],
                    shortTitle: "Search", systemImageName: "magnifyingglass")
        AppShortcut(intent: SearchFilesIntent(), phrases: ["Search my files with \(.applicationName)"],
                    shortTitle: "Search Files by Meaning", systemImageName: "doc.text.magnifyingglass")
    }
}
