import Foundation
import OmniKit

/// WHAT OMNI INDEXES, over HTTP: list it, add to it, pause it, remove from it.
///
/// The one rule this file exists to keep: **every operation here is the sidebar's operation.**
/// Adding a folder through the API and dropping one on the sidebar reach the same AppModel method,
/// so an API-added folder is canonicalized, persisted, watched, queued and preempted exactly like a
/// hand-added one, and there is no second code path to keep in step. The closures below are the
/// whole seam - Serving still knows nothing about AppModel, for the same reason ServingBackend
/// exists.
///
/// Mutating routes are deliberately all POST. A DELETE carrying a body is awkward for clients and
/// for the tiny server here, and the four verbs read as what they are: the four things the sidebar
/// can do.

// MARK: - The seam

/// One thing Omni indexes: a folder, or a slice of the Photos library.
struct ServedSource: Sendable {
    let key: String        // absolute folder path, or "photos://<id>"
    let kind: String       // "folder" | "photos"
    let name: String       // what the sidebar shows
    let paused: Bool
    let indexing: Bool     // a pass is working on it right now
    let queued: Bool       // added, waiting its turn
    let indexedFiles: Int  // rows the store holds under it
    let done: Int          // live pass progress; 0/0 when idle
    let total: Int
}

/// A Photos album the caller could add but has not. Returned alongside the sources so an agent can
/// go from "what can I add" to "add it" without a second round trip - the album's identifier is
/// not guessable, so listing it is the only way to name it.
struct ServedAlbum: Sendable {
    let id: String
    let title: String
    let count: Int
    let smart: Bool
}

struct SourcesSnapshot: Sendable {
    let sources: [ServedSource]
    let albums: [ServedAlbum]
    let photosAuthorized: Bool
    let indexing: Bool
    /// What a caller gets before the model is ready - not an error, just nothing indexed yet.
    static let empty = SourcesSnapshot(sources: [], albums: [], photosAuthorized: false, indexing: false)
}

/// The outcome of a mutation. `error` non-nil means nothing changed.
struct SourceMutation: Sendable {
    let key: String?
    let error: String?
    static func ok(_ key: String) -> SourceMutation { .init(key: key, error: nil) }
    static func fail(_ message: String) -> SourceMutation { .init(key: nil, error: message) }
}

/// Async closures supplied by AppModel. Every one hops to the main actor on its own side.
struct SourcesControl: Sendable {
    var snapshot: @Sendable () async -> SourcesSnapshot
    var addFolder: @Sendable (String) async -> SourceMutation
    var addAlbum: @Sendable (String) async -> SourceMutation
    var setPaused: @Sendable (String, Bool) async -> SourceMutation
    var remove: @Sendable (String) async -> SourceMutation
}

// MARK: - HTTP

enum SourcesAdapter {

    static func handle(_ req: HTTPRequest, route: String, _ control: SourcesControl?) async -> HTTPResponse {
        guard let control else {
            return error("indexing control is unavailable (the index is still loading)", status: 503)
        }
        switch (req.method, route) {
        case ("GET", "/v1/sources"):
            return json(snapshotBody(await control.snapshot()))

        case ("POST", "/v1/sources/add"):
            guard let body = jsonBody(req) else { return error("body must be a JSON object") }
            let path = string(body["path"])
            let album = string(body["album"])
            // Exactly one of the two: a request naming both is a caller bug, and guessing which it
            // meant would silently index the wrong thing.
            switch (path, album) {
            case (let p?, nil): return await mutation(await control.addFolder(p), control)
            case (nil, let a?): return await mutation(await control.addAlbum(a), control)
            case (nil, nil): return error("'path' (a folder) or 'album' (\"all\", or an album id from GET /v1/sources) is required")
            default: return error("give 'path' or 'album', not both")
            }

        case ("POST", "/v1/sources/pause"):
            guard let body = jsonBody(req), let key = string(body["key"]) else {
                return error("'key' is required (from GET /v1/sources)")
            }
            // Default true so `{"key": ...}` pauses; resuming is the explicit one.
            let paused = (body["paused"] as? Bool) ?? true
            return await mutation(await control.setPaused(key, paused), control)

        case ("POST", "/v1/sources/remove"):
            guard let body = jsonBody(req), let key = string(body["key"]) else {
                return error("'key' is required (from GET /v1/sources)")
            }
            return await mutation(await control.remove(key), control)

        default:
            return error("no route for \(req.method) \(route)", status: 404)
        }
    }

    /// Routes this adapter owns, so the Router can dispatch without repeating the list.
    static func owns(_ route: String) -> Bool {
        route == "/v1/sources" || route.hasPrefix("/v1/sources/")
    }

    // MARK: Shaping

    private static func jsonBody(_ req: HTTPRequest) -> [String: Any]? {
        guard !req.body.isEmpty,
              let obj = try? JSONSerialization.jsonObject(with: req.body),
              let dict = obj as? [String: Any] else { return nil }
        return dict
    }

    private static func string(_ v: Any?) -> String? {
        guard let s = v as? String else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    private static func snapshotBody(_ s: SourcesSnapshot) -> [String: Any] {
        [
            "indexing": s.indexing,
            "photos_authorized": s.photosAuthorized,
            "sources": s.sources.map { src -> [String: Any] in
                var row: [String: Any] = [
                    "key": src.key,
                    "kind": src.kind,
                    "name": src.name,
                    "paused": src.paused,
                    "indexing": src.indexing,
                    "queued": src.queued,
                    "indexed_files": src.indexedFiles,
                ]
                // Only while a pass is actually counting this source - a stale 0/0 on an idle row
                // reads as "nothing here", which is what the sidebar's own progress rules avoid.
                if src.total > 0 { row["progress"] = ["done": src.done, "total": src.total] }
                return row
            },
            "available_photo_albums": s.albums.map {
                ["id": $0.id, "title": $0.title, "count": $0.count, "smart": $0.smart]
            },
        ]
    }

    /// A mutation answers with the FULL new state, not just an ack: the caller's next question is
    /// always "so what is indexed now", and a source's key is canonicalized on the way in (a
    /// symlinked path, a nested folder absorbed by its parent), so echoing the request back would
    /// often be a lie.
    private static func mutation(_ m: SourceMutation, _ control: SourcesControl) async -> HTTPResponse {
        if let err = m.error { return error(err) }
        var body = snapshotBody(await control.snapshot())
        body["changed"] = m.key ?? ""
        return json(body)
    }

    private static func json(_ body: [String: Any]) -> HTTPResponse { HTTPResponse.json(body) }

    private static func error(_ message: String, status: Int = 400) -> HTTPResponse {
        HTTPResponse.json(["error": ["message": message, "type": "invalid_request_error"]], status: status)
    }
}
