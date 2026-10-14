import Foundation
import Vision
import CoreGraphics

/// The text on a screenshot, read on the device by Vision: the error code, the booking reference,
/// the chat line someone remembers - what a picture's embedding sees only as a picture of text.
/// Measured on the eval's nine screenshots (1280x800, English, Chinese, Japanese): 68-138 ms each,
/// every line read correctly but a few characters of a Swift listing. Accurate recognition with
/// language correction, the language detected per image.
public enum ScreenText {
    /// The recognized lines, top to bottom, one per line; "" when there is none.
    public static func recognize(_ image: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true
        do { try VNImageRequestHandler(cgImage: image).perform([request]) } catch { return "" }
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }
}
