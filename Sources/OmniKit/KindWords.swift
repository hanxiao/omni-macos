import Foundation

/// The kind a plain query asks for in words - "photo of a penguin", "the video with the zebra",
/// "voice memo about groceries", "pdf about climbing" - in the languages the eval covers. A SOFT
/// signal (VectorStore.fuseKindWords), never a filter: `type:` is the filter, and "a recording of
/// the meeting" may still be best answered by the meeting's notes.
public enum KindWords {
    public struct Request: Equatable, Sendable {
        public let kind: String          // FileKind raw value
        public let screenshot: Bool      // an image that is a screen capture
        public let ext: String?          // "pdf"
    }

    static let image = ["photo", "photos", "picture", "pictures", "pic", "pics", "image", "images", "照片", "图片", "相片",
                        "写真", "画像", "사진", "foto", "fotos", "bild", "bilder", "imagen", "imágenes", "фото",
                        "фотография", "صورة", "صور", "तस्वीर", "फोटो"]
    static let screenshot = ["screenshot", "screenshots", "screen shot", "screen capture", "截图", "截屏", "スクリーンショット",
                             "スクショ", "스크린샷", "bildschirmfoto", "capture d'écran", "captura de pantalla", "скриншот"]
    static let video = ["video", "videos", "movie", "movies", "clip", "clips", "film", "footage", "视频", "影片", "動画",
                        "ビデオ", "동영상", "비디오", "vidéo", "vídeo", "видео", "فيديو", "वीडियो"]
    static let audio = ["audio", "recording", "recordings", "voice memo", "voice memos", "voice note", "voicemail", "podcast",
                        "song", "songs", "录音", "音频", "録音", "音声", "녹음", "음성", "aufnahme", "sprachnachricht",
                        "enregistrement", "grabación", "запись", "تسجيل", "रिकॉर्डिंग"]
    static let pdf = ["pdf", "pdfs"]

    /// The first kind the query names, or nil. Whole words for scripts with spaces; contained for
    /// CJK, which has none.
    public static func requested(in text: String) -> Request? {
        let q = text.lowercased()
        let words = Set(q.split { !$0.isLetter && $0 != "'" }.map(String.init))
        func names(_ list: [String]) -> Bool {
            list.contains { w in
                if w.contains(" ") { return q.contains(w) }
                if w.unicodeScalars.contains(where: { $0.value >= 0x3000 }) { return q.contains(w) }
                return words.contains(w)
            }
        }
        if names(screenshot) { return Request(kind: FileKind.image.rawValue, screenshot: true, ext: nil) }
        if names(pdf) { return Request(kind: FileKind.text.rawValue, screenshot: false, ext: "pdf") }
        if names(video) { return Request(kind: FileKind.video.rawValue, screenshot: false, ext: nil) }
        if names(audio) { return Request(kind: FileKind.audio.rawValue, screenshot: false, ext: nil) }
        if names(image) { return Request(kind: FileKind.image.rawValue, screenshot: false, ext: nil) }
        return nil
    }

    /// A hit is what the request asks for.
    /// By the file, not by the chunk that matched: a screenshot found by its text is a picture.
    public static func matches(_ r: Request, path: String, kind: String) -> Bool {
        if let ext = r.ext { return (path as NSString).pathExtension.lowercased() == ext }
        let fileKind = SearchHit(path: path, score: 0, snippet: "", kind: kind, chunkIndex: 0, modified: 0).fileKind
        guard fileKind == r.kind || (r.kind == FileKind.text.rawValue && fileKind == FileKind.scan.rawValue) else { return false }
        return !r.screenshot || Indexer.isScreenshot(path)
    }
}
