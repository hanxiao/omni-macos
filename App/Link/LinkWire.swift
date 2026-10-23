import Foundation
import OmniKit
import OmniLink

/// What two copies of Omni say to each other over the link (docs/notes/link.md). JSON bodies; file
/// bytes and thumbnails travel in the message's blob. Every path in here is a path on the Mac that
/// ANSWERS - the asking Mac turns it into an `omni-link://` key only when it shows it.
enum LinkWire {
    /// The search filter, as the asking Mac's window expresses it. Folders are folders on the
    /// answering Mac (a linked Mac's folder scope), never this Mac's.
    struct Filter: Codable, Sendable, Hashable {
        var kinds: [String]?
        var folders: [String]?
        var ext: String?
        var since: Double?
        var until: Double?
        var filename: String?
        var tags: [String]?
        var excludeTags: [String]?
        var recents: Int?
        /// Optional, so a peer from before `-ext:` decodes the request and ignores it; the asking
        /// side re-applies it to what comes back (see `linkedSearch`).
        var excludeExts: [String]?

        init(_ f: SearchFilter, folders: [String]?) {
            kinds = f.kinds.isEmpty ? nil : f.kinds.sorted()
            self.folders = folders?.isEmpty == false ? folders : nil
            ext = f.ext
            since = f.since
            until = f.until
            filename = f.filenameQuery
            tags = f.tagTerms.isEmpty ? nil : f.tagTerms
            excludeTags = f.tagExcludeTerms.isEmpty ? nil : f.tagExcludeTerms
            recents = f.recentsLimit
            excludeExts = f.excludeExts.isEmpty ? nil : f.excludeExts
        }

        var searchFilter: SearchFilter {
            var f = SearchFilter()
            f.kinds = Set(kinds ?? [])
            f.folderPrefixes = folders ?? []
            f.ext = ext
            f.since = since
            f.until = until
            f.filenameQuery = filename
            f.tagTerms = tags ?? []
            f.tagExcludeTerms = excludeTags ?? []
            f.recentsLimit = recents
            f.excludeExts = excludeExts ?? []
            return f
        }
    }

    struct Hit: Codable, Sendable {
        var path: String
        var score: Float
        var snippet: String
        var kind: String
        var chunk: Int
        var modified: Double
        var w: Int
        var h: Int
        var dur: Double
        var size: Int
        var loc: String
        var chunks: Int
        /// What the answering Mac multiplied the score by (a media hit of a text query, on the text
        /// scale) and the scale its kind scores on there, so this Mac's relevance floor and fold read
        /// the hit as the answering Mac does. Absent from a Mac before 0.17.2 (and for an unlifted
        /// hit): then 1, and the kind's default scale. Optional, so either side ignores what the other
        /// does not send.
        var lift: Double?
        var scale: Double?

        /// Snippets are capped: a row shows a line of one, and a page of 120 long passages would
        /// otherwise be most of the reply.
        static let snippetCap = 600

        init(_ s: SearchHit) {
            path = s.path; score = s.score; snippet = String(s.snippet.prefix(Self.snippetCap)); kind = s.kind
            chunk = s.chunkIndex; modified = s.modified; w = s.width; h = s.height; dur = s.duration
            size = s.size; loc = s.locator; chunks = s.chunkCount
            lift = s.lift == 1 ? nil : s.lift; scale = s.modelScale
        }

        /// The hit as this Mac shows it: the path becomes the peer's key.
        func hit(peer: String) -> SearchHit {
            var out = SearchHit.make(path: RemotePath.make(peer: peer, path: path), score: score, snippet: snippet, kind: kind,
                                     chunkIndex: chunk, modified: modified, width: w, height: h, duration: dur, size: size,
                                     locator: loc, chunkCount: chunks)
            out.lift = lift ?? 1
            out.modelScale = scale
            return out
        }
    }

    struct SearchRequest: Codable, Sendable, Hashable {
        /// Text, embedded by the answering Mac's own model; or a vector, sent only to a Mac in the
        /// same vector space (a file query, Find Similar); or both, a text query with the vector the
        /// asking Mac already holds for it, sent to a Mac whose `Space.fusesText` says it uses the
        /// two together, which then embeds nothing.
        var query: String?
        var vector: [Float]?
        var filter: Filter
        var topK: Int
        /// Paths (on the answering Mac) never to return: the file a Find Similar started from.
        var exclude: [String]?
    }

    struct SearchReply: Codable, Sendable {
        var hits: [Hit]
        var ms: Double
    }

    /// The answering Mac's vector space: the probe its index was stamped with (AppModel's
    /// `space_probe`), so the asking Mac can tell whether scores are comparable.
    struct Space: Codable, Sendable {
        var dim: Int
        var probe: [Float]?
        var model: String
        var files: Int
        /// This Mac's search takes a request's vector AND its text (the lexical and name fusion still
        /// read the text). Absent from older Macs, which would drop the text beside a vector.
        var fusesText: Bool?
    }

    struct PathRequest: Codable, Sendable {
        var path: String
        var px: Int?
        var offset: Int64?
        var length: Int?
        var query: String?
        var vector: [Float]?
    }

    struct FileInfo: Codable, Sendable {
        var size: Int64
        var modified: Double
        var name: String
    }

    struct Source: Codable, Sendable, Hashable, Identifiable {
        var key: String
        var name: String
        var kind: String     // "folder" | "photos"
        var files: Int
        var id: String { key }
    }

    struct Sources: Codable, Sendable { var sources: [Source] }

    struct Child: Codable, Sendable, Hashable {
        var path: String
        var folder: Bool
        var kind: String
        var modified: Double
        var size: Int
        var files: Int
    }

    struct Listing: Codable, Sendable { var children: [Child] }

    struct Passage: Codable, Sendable {
        var chunk: Int
        var score: Float
        var snippet: String
        var loc: String
    }

    struct Passages: Codable, Sendable { var passages: [Passage] }

    struct Similar: Codable, Sendable {
        var hits: [Hit]
        var vector: [Float]?
    }
}
