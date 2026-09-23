// recwin <pid> <seconds> <out.mov>: record the largest on-screen window of <pid> with ScreenCaptureKit,
// at the window's backing resolution and 60 fps, cursor hidden. Works while the window is covered.
import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia
import AppKit

_ = NSApplication.shared   // connects to the window server; ScreenCaptureKit asserts without it

let args = CommandLine.arguments
guard args.count >= 4, let pid = Int32(args[1]), let seconds = Double(args[2]) else {
    print("usage: recwin <pid> <seconds> <out.mov> [min-width]"); exit(2)
}
let minWidth = args.count > 4 ? Double(args[4]) ?? 400 : 400
let out = URL(fileURLWithPath: args[3])
try? FileManager.default.removeItem(at: out)

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate {
    let writer: AVAssetWriter
    let input: AVAssetWriterInput
    var started = false
    var frames = 0
    init(url: URL, width: Int, height: Int) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 80_000_000],
        ])
        input.expectsMediaDataInRealTime = true
        writer.add(input)
    }
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sb.isValid,
              let att = (CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first,
              let raw = att[.status] as? Int, SCFrameStatus(rawValue: raw) == .complete else { return }
        if !started {
            writer.startWriting(); writer.startSession(atSourceTime: sb.presentationTimeStamp); started = true
        }
        if input.isReadyForMoreMediaData { input.append(sb); frames += 1 }
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) { print("stopped: \(error)") }
}

let sem = DispatchSemaphore(value: 0)
Task {
    do {
        // Wait (up to 60 s) for the window to exist at its final size: a stream is fixed to the
        // size it started at, and a resize after that stops its frames.
        var found: SCWindow?
        for _ in 0 ..< 240 {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            found = content.windows
                .filter({ $0.owningApplication?.processID == pid && $0.frame.width >= minWidth && $0.windowLayer == 0 })
                .max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height })
            if found != nil { break }
            try await Task.sleep(for: .milliseconds(250))
        }
        guard let win = found else { print("no window for pid \(pid)"); exit(1) }
        try await Task.sleep(for: .milliseconds(400))   // let the resize settle
        let filter = SCContentFilter(desktopIndependentWindow: win)
        let scale = Int(filter.pointPixelScale)
        let cfg = SCStreamConfiguration()
        cfg.width = Int(win.frame.width) * scale
        cfg.height = Int(win.frame.height) * scale
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
        cfg.showsCursor = false
        cfg.queueDepth = 8
        cfg.ignoreShadowsSingleWindow = true
        cfg.shouldBeOpaque = false
        let rec = try Recorder(url: out, width: cfg.width, height: cfg.height)
        let stream = SCStream(filter: filter, configuration: cfg, delegate: rec)
        try stream.addStreamOutput(rec, type: .screen, sampleHandlerQueue: DispatchQueue(label: "rec"))
        try await stream.startCapture()
        print("recording \(win.title ?? "") \(cfg.width)x\(cfg.height)")
        try await Task.sleep(for: .seconds(seconds))
        try await stream.stopCapture()
        rec.input.markAsFinished()
        await rec.writer.finishWriting()
        print("frames \(rec.frames) -> \(out.path)")
    } catch { print("error: \(error)"); exit(1) }
    sem.signal()
}
sem.wait()
