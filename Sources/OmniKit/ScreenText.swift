import Foundation
import Vision
import CoreGraphics

/// The text on a screenshot, read on the device by Vision: the error code, the booking reference,
/// the chat line someone remembers - what a picture's embedding sees only as a picture of text.
/// Measured on the eval's nine screenshots (1280x800, English, Chinese, Japanese): 68-138 ms each,
/// every line read correctly but a few characters of a Swift listing. Accurate recognition, the
/// language detected per image, NO language correction: it rewrote code and codes toward words
/// ("= 0" read as "= O", "+=" as "+ ="), which is what identifier lookup needs verbatim, and
/// cost 19% (omni-verify ocrcost: 827 -> 673 ms for the nine, the other eight read the same).
/// The compute device was measured too: Neural Engine, GPU and CPU all ~120 ms an image.
public enum ScreenText {
    /// The recognized lines, top to bottom, one per line; "" when there is none.
    public static func recognize(_ image: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.automaticallyDetectsLanguage = true
        do { try VNImageRequestHandler(cgImage: image).perform([request]) } catch { return "" }
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }
}
