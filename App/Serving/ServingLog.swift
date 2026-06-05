import Foundation

/// One serviced request, rendered live in the Serving tab's log list.
/// Sendable so it can cross from the network queue to the main actor.
struct LogEntry: Identifiable, Sendable {
    let id = UUID()
    let time: Date
    let method: String
    let path: String
    let status: Int
    let ms: Double
    let client: String
}

/// Where the server binds. Persisted as its rawValue string in UserDefaults.
enum ServingScope: String, Sendable {
    case local
    case `public`
}
