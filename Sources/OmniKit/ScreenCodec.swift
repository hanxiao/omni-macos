import Foundation
import VideoToolbox
import CoreMedia
import CoreVideo
import ImageIO

/// THE SCREEN HISTORY IS A VIDEO, NOT A PILE OF PICTURES (docs/screenshots.md). Successive shots of
/// one desktop are nearly the same picture, and an image codec pays for the whole of it every time.
/// HEVC on the Mac's own encoder pays for the first frame and then for what moved: a typed line is a
/// few hundred bytes where a HEIC of the same frame is a few hundred thousand.
///
/// One frame in, one frame out, at once: no reordering and no frames held back, so every shot is on
/// disk before the next one is taken and a kill loses nothing but the frame in flight.
public final class ScreenFrameEncoder {
    public struct Frame {
        /// The sample as VideoToolbox emits it: NAL units, each behind a 4-byte length.
        public let data: Data
        public let isKey: Bool
        /// VPS, SPS, PPS. Carried by every key frame, so a key frame decodes with nothing but itself.
        public let parameterSets: [Data]
    }

    public let width: Int
    public let height: Int
    private let session: VTCompressionSession
    private var count: Int64 = 0

    /// `quality` is VideoToolbox's 0...1; `keyInterval` the longest run between key frames, which
    /// is the most frames a jump into the timeline has to decode.
    ///
    /// The Mac's hardware encoder, which gives up its reference whenever another hardware encoder
    /// starts anywhere on the Mac (a call, a recording, a HEIC written by the hardware): the next
    /// shot is then a key frame, correct and the size of one. The software encoder keeps it, and
    /// costs 79 ms of CPU a shot at 1920x1080 where this one costs 2, every shot of every day
    /// (docs/screenshots.md, "A real session").
    public init?(width: Int, height: Int, quality: Float = ScreenCodec.quality,
                 keyInterval: Int = ScreenCodec.keyInterval) {
        self.width = width; self.height = height
        var s: VTCompressionSession?
        let spec: [CFString: Any] = [kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true]
        guard VTCompressionSessionCreate(allocator: nil, width: Int32(width), height: Int32(height),
                                         codecType: kCMVideoCodecType_HEVC, encoderSpecification: spec as CFDictionary,
                                         imageBufferAttributes: nil, compressedDataAllocator: nil,
                                         outputCallback: nil, refcon: nil, compressionSessionOut: &s) == noErr,
              let s else { return nil }
        session = s
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_MaxFrameDelayCount, value: 0 as CFNumber)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, value: keyInterval as CFNumber)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_Quality, value: quality as CFNumber)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_HEVC_Main_AutoLevel)
        VTCompressionSessionPrepareToEncodeFrames(s)
    }

    deinit { VTCompressionSessionInvalidate(session) }

    /// Encode one frame and wait for it. nil when the encoder refused it.
    public func encode(_ pixels: CVPixelBuffer, forceKey: Bool = false) -> Frame? {
        guard CVPixelBufferGetWidth(pixels) == width, CVPixelBufferGetHeight(pixels) == height else { return nil }
        let pts = CMTime(value: count, timescale: 30)
        count += 1
        let props: CFDictionary? = forceKey ? [kVTEncodeFrameOptionKey_ForceKeyFrame: true] as CFDictionary : nil
        let out = Handoff<Frame>()
        let status = VTCompressionSessionEncodeFrame(session, imageBuffer: pixels, presentationTimeStamp: pts,
                                                     duration: CMTime(value: 1, timescale: 30),
                                                     frameProperties: props, infoFlagsOut: nil) { status, _, sample in
            guard status == noErr, let sample, let frame = Self.frame(from: sample) else { return }
            out.set(frame)
        }
        guard status == noErr else { return nil }
        VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: pts)
        return out.take()
    }

    private static func frame(from sample: CMSampleBuffer) -> Frame? {
        guard let block = CMSampleBufferGetDataBuffer(sample) else { return nil }
        let length = CMBlockBufferGetDataLength(block)
        var data = Data(count: length)
        let copied = data.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
        guard copied == noErr else { return nil }
        let attachments = (CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false) as? [[CFString: Any]])?.first
        let isKey = !((attachments?[kCMSampleAttachmentKey_NotSync] as? Bool) ?? false)
        var sets: [Data] = []
        if isKey, let format = CMSampleBufferGetFormatDescription(sample) {
            var n = 0
            CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(format, parameterSetIndex: 0, parameterSetPointerOut: nil,
                                                               parameterSetSizeOut: nil, parameterSetCountOut: &n, nalUnitHeaderLengthOut: nil)
            for i in 0 ..< n {
                var p: UnsafePointer<UInt8>?
                var size = 0
                guard CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(format, parameterSetIndex: i, parameterSetPointerOut: &p,
                                                                         parameterSetSizeOut: &size, parameterSetCountOut: nil,
                                                                         nalUnitHeaderLengthOut: nil) == noErr, let p else { return nil }
                sets.append(Data(bytes: p, count: size))
            }
            guard !sets.isEmpty else { return nil }
        }
        return Frame(data: data, isKey: isKey, parameterSets: sets)
    }
}

/// Decodes a run of frames in order. A key frame (one that comes with parameter sets) starts a run;
/// every other frame needs the one before it to have been decoded by this decoder.
public final class ScreenFrameDecoder {
    private var session: VTDecompressionSession?
    private var format: CMVideoFormatDescription?

    public init() {}
    deinit { if let session { VTDecompressionSessionInvalidate(session) } }

    /// The frame's pixels (BGRA, IOSurface-backed). `parameterSets` is non-empty for a key frame.
    public func decode(_ data: Data, parameterSets: [Data]) -> CVPixelBuffer? {
        if !parameterSets.isEmpty, !start(parameterSets) { return nil }
        guard let session, let format else { return nil }
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: data.count,
                                                 blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
                                                 dataLength: data.count, flags: kCMBlockBufferAssureMemoryNowFlag,
                                                 blockBufferOut: &block) == noErr, let block else { return nil }
        let filled = data.withUnsafeBytes { CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: data.count) }
        guard filled == noErr else { return nil }
        var sample: CMSampleBuffer?
        var size = data.count
        guard CMSampleBufferCreateReady(allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: format,
                                        sampleCount: 1, sampleTimingEntryCount: 0, sampleTimingArray: nil,
                                        sampleSizeEntryCount: 1, sampleSizeArray: &size, sampleBufferOut: &sample) == noErr,
              let sample else { return nil }
        let out = Handoff<CVPixelBuffer>()
        // No asynchronous flag: the handler has run by the time this returns.
        let status = VTDecompressionSessionDecodeFrame(session, sampleBuffer: sample, flags: [], infoFlagsOut: nil) { status, _, image, _, _ in
            if status == noErr, let image { out.set(image) }
        }
        return status == noErr ? out.take() : nil
    }

    private func start(_ sets: [Data]) -> Bool {
        var made: CMVideoFormatDescription?
        let status: OSStatus = withPointers(sets) { pointers, sizes in
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(allocator: kCFAllocatorDefault, parameterSetCount: sets.count,
                                                                parameterSetPointers: pointers, parameterSetSizes: sizes,
                                                                nalUnitHeaderLength: 4, extensions: nil, formatDescriptionOut: &made)
        }
        guard status == noErr, let made else { return false }
        if let session, VTDecompressionSessionCanAcceptFormatDescription(session, formatDescription: made) {
            format = made
            return true
        }
        if let session { VTDecompressionSessionInvalidate(session) }
        session = nil
        let attributes: [CFString: Any] = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                                           kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                                           kCVPixelBufferMetalCompatibilityKey: true]
        var s: VTDecompressionSession?
        guard VTDecompressionSessionCreate(allocator: kCFAllocatorDefault, formatDescription: made, decoderSpecification: nil,
                                           imageBufferAttributes: attributes as CFDictionary, outputCallback: nil,
                                           decompressionSessionOut: &s) == noErr, let s else { return false }
        session = s
        format = made
        return true
    }

    private func withPointers<R>(_ sets: [Data], _ body: (UnsafePointer<UnsafePointer<UInt8>>, UnsafePointer<Int>) -> R) -> R {
        let buffers = sets.map { set -> UnsafeMutablePointer<UInt8> in
            let p = UnsafeMutablePointer<UInt8>.allocate(capacity: max(set.count, 1))
            set.copyBytes(to: p, count: set.count)
            return p
        }
        defer { buffers.forEach { $0.deallocate() } }
        let pointers = buffers.map { UnsafePointer($0) }
        let sizes = sets.map(\.count)
        return pointers.withUnsafeBufferPointer { p in sizes.withUnsafeBufferPointer { s in body(p.baseAddress!, s.baseAddress!) } }
    }
}

/// One value from a VideoToolbox callback to the call that waited for it.
private final class Handoff<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value?
    func set(_ v: Value) { lock.withLock { value = v } }
    func take() -> Value? { lock.withLock { value } }
}

public enum ScreenCodec {
    /// Measured on generated desktop sessions (omni-verify shotbench; docs/screenshots.md).
    public static let quality: Float = 0.7
    public static let keyInterval = 30

    /// A BGRA, IOSurface-backed buffer: what ScreenCaptureKit hands over and what the encoder takes.
    public static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                                           kCVPixelBufferCGImageCompatibilityKey: true,
                                           kCVPixelBufferCGBitmapContextCompatibilityKey: true]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &pb)
        return pb
    }

    /// How a shot's picture file is written. NOT ON THE HARDWARE ENCODER: ImageIO's HEIC starts a
    /// hardware HEVC session per picture, and each start costs the video its reference - the shot
    /// after every indexed one came out a key frame (48 keys in 150 shots where 5 were due, the
    /// segment 5.1 MB for 3.0). The option is exported by ImageIO but not in its headers; with it
    /// the picture is the same size and encodes faster (33 ms against 42 at 1568 wide). If a later
    /// macOS ignores it the history grows and nothing else changes:
    /// testTheShotAfterAnIndexedOneIsStillSmall is what notices.
    public static func stillOptions(quality: Double) -> CFDictionary {
        [kCGImageDestinationLossyCompressionQuality: quality,
         "kCGImageDestinationUseHardwareAcceleration" as CFString: false] as CFDictionary
    }

    public static func cgImage(_ pixels: CVPixelBuffer) -> CGImage? {
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixels, options: nil, imageOut: &image)
        return image
    }
}
