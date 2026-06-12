import Accelerate
@preconcurrency import AVFoundation
import Foundation
import MLX

/// Audio preprocessing for the omni-small audio path: decode any audio file to
/// 16 kHz mono PCM (AVFoundation) and compute the 128-bin Whisper/Qwen2.5-Omni
/// log-mel `input_features` with the SAME parameters the reference fixture uses
/// (`Tools/gen_audio_fixtures.py` via `WhisperFeatureExtractor(feature_size=128)`).
///
/// Pipeline (validated against `Fixtures/audio_ref.safetensors`):
///   decode -> 16 kHz mono float32
///   STFT: Hann periodic window length 400, hop 160, center reflect-pad n_fft//2=200,
///         drop the last frame (Whisper `[..., :-1]`), power spectrum |STFT|^2
///   128 Slaney-normalized triangular mel filters over 201 FFT bins (0..8000 Hz)
///   log10(max(mel, 1e-10)) -> clamp to (max - 8) -> (x + 4) / 4
///
/// Output `inputFeatures` is mel-major `[num_mel_bins=128, total_frames]` Float32
/// (rows = mel bins, columns = time frames) — exactly what `OmniAudioTower.forward`
/// consumes (it transposes internally). `featureLens = [total_frames]` for a single
/// audio. These are the REAL unpadded frames (no Whisper 30s / 3000-frame padding).
public enum OmniAudioPreprocess {
    // Whisper / Qwen2.5-Omni feature-extractor parameters.
    private static let sampleRate: Double = 16000
    private static let nFFT = 400
    private static let hop = 160
    private static let numMelBins = 128
    private static let melFMin: Float = 0.0
    private static let melFMax: Float = 8000.0

    /// Smallest mel-frame count a clip may have and still survive the audio tower. The conv
    /// frontend halves frames (conv2 stride 2) and the per-audio pool then averages frame
    /// PAIRS; a clip with 1-2 mel frames collapses to a single post-conv frame, whose pair-pool
    /// reduces over a zero-size axis -> MLX `reduce` abort (SIGABRT), killing the whole indexing
    /// scan (issue #3). 3 mel frames is the smallest count that survives (verified). Clips below
    /// it (< ~30 ms of audio) carry no searchable content, so we skip them at the source -
    /// deterministically, without feeding a degenerate tensor to the GPU.
    private static let minMelFrames = 3

    // Tables that depend only on the constants above, built once (static let is lazy and
    // thread-safe). Previously rebuilt per audio FILE: the DFT cos/sin matrices alone are
    // 2 x 201 x 400 trig evaluations, and the filterbank another 128 x 201 pass - pure waste
    // in the concurrent decode stage when indexing a large audio folder.
    /// Hann periodic window length nFFT: w[n] = 0.5 - 0.5*cos(2*pi*n/N).
    private static let hannWindow: [Float] = {
        var window = [Float](repeating: 0, count: nFFT)
        let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
        for n in 0 ..< nFFT { window[n] = 0.5 - 0.5 * cosf(twoPiOverN * Float(n)) }
        return window
    }()
    /// Direct-DFT basis matrices [nBins x nFFT] (nFFT=400 is not a vDSP DFT length).
    private static let dftCos: [Float] = {
        let nBins = nFFT / 2 + 1
        var cosM = [Float](repeating: 0, count: nBins * nFFT)
        for b in 0 ..< nBins {
            for n in 0 ..< nFFT {
                cosM[b * nFFT + n] = cosf(-2.0 * Float.pi * Float(b) * Float(n) / Float(nFFT))
            }
        }
        return cosM
    }()
    private static let dftSin: [Float] = {
        let nBins = nFFT / 2 + 1
        var sinM = [Float](repeating: 0, count: nBins * nFFT)
        for b in 0 ..< nBins {
            for n in 0 ..< nFFT {
                sinM[b * nFFT + n] = sinf(-2.0 * Float.pi * Float(b) * Float(n) / Float(nFFT))
            }
        }
        return sinM
    }()

    /// Mel frames per long-audio segment: 24000 frames = 240 s. Matches the indexer's audio
    /// frame budget, so one segment fills exactly one batched-forward budget; the backbone cost
    /// of a segment is the same as the longest clip the budget already allowed.
    public static let segmentMelFrames = 24000
    /// Seconds of audio per segment (240.0).
    public static var segmentSeconds: Double { Double(segmentMelFrames) * Double(hop) / sampleRate }

    /// Decode + log-mel as a plain Float buffer (mel-major `[128*frames]`) + frame count.
    /// CPU-only and Sendable, so it can run in the concurrent decode stage of indexing.
    /// Capped at ONE segment (240 s): a whole-file decode of a multi-hour file overflows
    /// AudioToolbox's 32-bit byte count (issue #7) and its mel would feed the backbone an
    /// O(L^2) sequence no machine survives. Files at or under the cap are byte-identical to
    /// the old whole-file path; longer audio streams segment by segment via AudioSegmentReader.
    public static func melFeatures(url: URL) -> (mel: [Float], frames: Int)? {
        guard let reader = AudioSegmentReader(url: url), let samples = reader.nextSegment() else { return nil }
        return melFrom(samples: samples)
    }

    /// Log-mel of already-decoded 16 kHz mono samples (mel-major `[128*frames]`).
    static func melFrom(samples: [Float]) -> (mel: [Float], frames: Int)? {
        guard !samples.isEmpty else { return nil }
        let nBins = nFFT / 2 + 1   // 201
        let power = stftPower(samples)                  // [nBins, frames] row-major
        let frames = power.count / nBins
        // Skip clips too short to survive the tower's conv + pair-pool (issue #3): < 3 mel
        // frames would otherwise reduce over an empty axis and abort the scan.
        if frames < minMelFrames { return nil }

        let melFB = Self.melFB                           // [nMel, nBins] row-major, cached

        // Mel projection + log10, parallelized across mel bins (rows are independent;
        // the max reduction is deferred to a second pass). This is CPU-bound matmul work
        // that runs off the GPU stage, in the indexer's concurrent decode phase.
        var feat = [Float](repeating: 0, count: numMelBins * frames)
        var rowMax = [Float](repeating: -Float.greatestFiniteMagnitude, count: numMelBins)
        feat.withUnsafeMutableBufferPointer { featBuf in
            rowMax.withUnsafeMutableBufferPointer { rowMaxBuf in
                melFB.withUnsafeBufferPointer { fbBuf in
                    power.withUnsafeBufferPointer { powBuf in
                        let featP = featBuf.baseAddress!
                        let rowMaxP = rowMaxBuf.baseAddress!
                        let fbP = fbBuf.baseAddress!
                        let powP = powBuf.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: numMelBins) { m in
                            let fbRow = m * nBins
                            let outRow = m * frames
                            var localMax: Float = -Float.greatestFiniteMagnitude
                            for t in 0 ..< frames {
                                var acc: Float = 0
                                for b in 0 ..< nBins {
                                    acc += fbP[fbRow + b] * powP[b * frames + t]
                                }
                                let v = log10f(Swift.max(acc, 1e-10))
                                featP[outRow + t] = v
                                if v > localMax { localMax = v }
                            }
                            rowMaxP[m] = localMax
                        }
                    }
                }
            }
        }
        var maxLog: Float = -Float.greatestFiniteMagnitude
        for m in 0 ..< numMelBins where rowMax[m] > maxLog { maxLog = rowMax[m] }
        let floorVal = maxLog - 8.0
        for i in 0 ..< feat.count {
            feat[i] = (Swift.max(feat[i], floorVal) + 4.0) / 4.0
        }
        return (feat, frames)
    }

    /// Decode `url` and compute log-mel features as an MLXArray (mel-major `[128, frames]`).
    public static func features(url: URL) -> (inputFeatures: MLXArray, featureLens: [Int])? {
        guard let (mel, frames) = melFeatures(url: url) else { return nil }
        return (MLXArray(mel).reshaped([numMelBins, frames]), [frames])
    }

    // MARK: - Decode

    /// Sequential decoder: any audio file -> 16 kHz mono Float32 PCM, one bounded segment
    /// (240 s) at a time. Reads the file's native float32 processing format in <= 60 s slices
    /// into ONE reusable buffer, downmixes each slice to mono, and linearly resamples per
    /// segment. Never sizes a buffer by the file's length: the old whole-file
    /// `AVAudioPCMBuffer(frameCapacity: file.length)` made AudioToolbox compute
    /// `frames x bytesPerFrame` in 32 bits, which throws an uncatchable C++ exception past
    /// ~4.29 GB (a >= ~3 h 23 m stereo 44.1 kHz file) and aborted the whole scan - issue #7.
    /// Slice reads are bounded regardless of file length, so the overflow is structurally
    /// impossible. Avoids AVAudioConverter (whose single-shot streaming is brittle for
    /// same-rate passthrough): sequential AVAudioFile reads concatenate to exactly the bytes
    /// one whole-file read returns, so a file that fits in one segment decodes byte-identical
    /// to the old path. Single-consumer; calls must be sequenced (the indexer's prefetch
    /// queue does this, mirroring the scanned-PDF reader).
    public final class AudioSegmentReader: @unchecked Sendable {
        private let file: AVAudioFile
        private let slice: AVAudioPCMBuffer
        private let channelCount: Int
        private let nativeRate: Double
        private let nativeSegmentFrames: Int   // native frames per 240 s segment
        private var done = false

        public init?(url: URL) {
            guard let f = try? AVAudioFile(forReading: url) else { return nil }
            let fmt = f.processingFormat
            guard fmt.sampleRate > 0, fmt.channelCount > 0, f.length > 0 else { return nil }
            // <= 60 s of native audio per read, additionally capped in frames so an exotic
            // high-rate/many-channel format stays far from any 32-bit byte limit.
            let sliceFrames = AVAudioFrameCount(Swift.min(60.0 * fmt.sampleRate, 8_000_000))
            guard sliceFrames > 0, let b = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: sliceFrames) else { return nil }
            file = f
            slice = b
            channelCount = Int(fmt.channelCount)
            nativeRate = fmt.sampleRate
            nativeSegmentFrames = Int(OmniAudioPreprocess.segmentSeconds * fmt.sampleRate)
        }

        /// The next segment as 16 kHz mono samples; nil at end of file.
        public func nextSegment() -> [Float]? {
            if done { return nil }
            var mono: [Float] = []
            mono.reserveCapacity(Swift.min(nativeSegmentFrames, 4_000_000))
            while mono.count < nativeSegmentFrames {
                let want = AVAudioFrameCount(Swift.min(Int(slice.frameCapacity), nativeSegmentFrames - mono.count))
                slice.frameLength = 0
                do { try file.read(into: slice, frameCount: want) } catch { done = true; break }
                let n = Int(slice.frameLength)
                if n == 0 { done = true; break }   // end of file
                guard let chans = slice.floatChannelData else { done = true; break }
                let base = mono.count
                mono.append(contentsOf: repeatElement(0, count: n))
                mono.withUnsafeMutableBufferPointer { m in
                    for c in 0 ..< channelCount {
                        let p = chans[c]
                        for i in 0 ..< n { m[base + i] += p[i] }
                    }
                    if channelCount > 1 {
                        let inv = 1.0 / Float(channelCount)
                        for i in 0 ..< n { m[base + i] *= inv }
                    }
                }
            }
            if mono.isEmpty { return nil }
            if abs(nativeRate - OmniAudioPreprocess.sampleRate) < 0.5 { return mono }
            return OmniAudioPreprocess.resampleLinear(mono, from: nativeRate, to: OmniAudioPreprocess.sampleRate)
        }

        /// The next segment's log-mel features; nil at end of file. Segments too short to
        /// survive the tower (< 3 mel frames, < ~30 ms tail) come back as `([], 0)` so the
        /// caller can skip the segment without mistaking it for end-of-file.
        public func nextMelSegment() -> (mel: [Float], frames: Int)? {
            if let p = pending { pending = nil; return p }
            guard let samples = nextSegment() else { return nil }
            return OmniAudioPreprocess.melFrom(samples: samples) ?? ([], 0)
        }

        /// Hand a segment back so the next `nextMelSegment()` returns it again. The indexer's
        /// decode stage peeks one segment ahead to tell single-segment files (the common case,
        /// which keeps the batched path) from long ones, then returns the peeked segment.
        public func pushBack(_ segment: (mel: [Float], frames: Int)) { pending = segment }
        private var pending: (mel: [Float], frames: Int)?
    }

    /// Linear-interpolation resample. Adequate for mel features; the embedding is robust.
    private static func resampleLinear(_ x: [Float], from: Double, to: Double) -> [Float] {
        guard x.count > 1, from > 0 else { return x }
        let outN = Int((Double(x.count) * to / from).rounded())
        guard outN > 1 else { return x }
        var out = [Float](repeating: 0, count: outN)
        let step = from / to
        for i in 0 ..< outN {
            let pos = Double(i) * step
            let i0 = Int(pos)
            let frac = Float(pos - Double(i0))
            let a = x[Swift.min(i0, x.count - 1)]
            let b = x[Swift.min(i0 + 1, x.count - 1)]
            out[i] = a + (b - a) * frac
        }
        return out
    }

    // MARK: - STFT power spectrum

    /// Center-padded (reflect, n_fft//2) Hann-periodic STFT, power spectrum.
    /// Returns a row-major `[nBins, frames]` flat buffer; frames follows the Whisper
    /// convention of dropping the final frame (`stft(...)[..., :-1]`).
    private static func stftPower(_ samples: [Float]) -> [Float] {
        let pad = nFFT / 2
        // Reflect padding (NumPy 'reflect': edge sample not repeated).
        var x = [Float](repeating: 0, count: samples.count + 2 * pad)
        for i in 0 ..< pad { x[pad - 1 - i] = samples[Swift.min(i + 1, samples.count - 1)] }
        for i in 0 ..< samples.count { x[pad + i] = samples[i] }
        let last = samples.count - 1
        for i in 0 ..< pad { x[pad + samples.count + i] = samples[Swift.max(last - 1 - i, 0)] }

        let nBins = nFFT / 2 + 1
        // Whisper drops the final frame: number of frames = (len - n_fft) / hop + 1, then -1.
        let fullFrames = (x.count - nFFT) / hop + 1
        let frames = Swift.max(fullFrames - 1, 0)
        if frames == 0 { return [] }

        // nFFT=400 is not a vDSP-supported DFT length (vDSP needs f*2^n, f in {1,3,5,15};
        // 400 = 2^4 * 25). Use a direct DFT via the precomputed [nBins x nFFT] cos/sin
        // matrices + vDSP_mmul, preserving the exact 201-bin grid (no zero-padding).
        let window = hannWindow
        let cosM = dftCos
        let sinM = dftSin

        // Per-frame DFT is independent across frames -> parallelize the outer loop.
        // Each worker owns thread-local scratch (frame/re/im) and writes disjoint
        // columns of `out`, so there is no contention. Bit-identical to the serial
        // version (same vDSP_mmul, same arithmetic, deterministic accumulation order).
        var out = [Float](repeating: 0, count: nBins * frames)
        out.withUnsafeMutableBufferPointer { outBuf in
            x.withUnsafeBufferPointer { xBuf in
                window.withUnsafeBufferPointer { winBuf in
                    cosM.withUnsafeBufferPointer { cosBuf in
                        sinM.withUnsafeBufferPointer { sinBuf in
                            let outP = outBuf.baseAddress!
                            let xP = xBuf.baseAddress!
                            let winP = winBuf.baseAddress!
                            let cosP = cosBuf.baseAddress!
                            let sinP = sinBuf.baseAddress!
                            DispatchQueue.concurrentPerform(iterations: frames) { t in
                                var frame = [Float](repeating: 0, count: nFFT)
                                var re = [Float](repeating: 0, count: nBins)
                                var im = [Float](repeating: 0, count: nBins)
                                let base = t * hop
                                for n in 0 ..< nFFT { frame[n] = xP[base + n] * winP[n] }
                                // re = cosM[nBins x nFFT] * frame[nFFT]; im = sinM * frame.
                                vDSP_mmul(cosP, 1, frame, 1, &re, 1, vDSP_Length(nBins), 1, vDSP_Length(nFFT))
                                vDSP_mmul(sinP, 1, frame, 1, &im, 1, vDSP_Length(nBins), 1, vDSP_Length(nFFT))
                                for b in 0 ..< nBins { outP[b * frames + t] = re[b] * re[b] + im[b] * im[b] }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - Mel filterbank (Slaney-normalized triangular, matches transformers)

    /// 128 triangular mel filters over 201 FFT bins (0..8000 Hz), Slaney mel scale +
    /// Slaney area normalization. Row-major `[nMel, nBins]`. Built from scratch to
    /// match `transformers.audio_utils.mel_filter_bank(norm='slaney', mel_scale='slaney')`.
    /// Built once (constants-only); previously recomputed per audio file.
    private static let melFB: [Float] = melFilterbank()

    private static func melFilterbank() -> [Float] {
        let nBins = nFFT / 2 + 1
        // FFT bin center frequencies (Hz).
        var fftFreqs = [Float](repeating: 0, count: nBins)
        let nyquist = Float(sampleRate) / 2
        for b in 0 ..< nBins { fftFreqs[b] = nyquist * Float(b) / Float(nBins - 1) }

        // Mel band edges, linear in Slaney mel space.
        let melMin = hzToMelSlaney(melFMin)
        let melMax = hzToMelSlaney(melFMax)
        var hzPts = [Float](repeating: 0, count: numMelBins + 2)
        for i in 0 ..< (numMelBins + 2) {
            let mel = melMin + (melMax - melMin) * Float(i) / Float(numMelBins + 1)
            hzPts[i] = melToHzSlaney(mel)
        }

        var fb = [Float](repeating: 0, count: numMelBins * nBins)
        for m in 0 ..< numMelBins {
            let l = hzPts[m], c = hzPts[m + 1], r = hzPts[m + 2]
            let enorm: Float = 2.0 / (r - l)   // Slaney area normalization
            for b in 0 ..< nBins {
                let f = fftFreqs[b]
                var w: Float = 0
                if c > l && f >= l && f <= c {
                    w = (f - l) / (c - l)
                } else if r > c && f > c && f <= r {
                    w = (r - f) / (r - c)
                }
                fb[m * nBins + b] = w * enorm
            }
        }
        return fb
    }

    /// Slaney hz->mel: linear below 1000 Hz (200/3 Hz per mel), log above.
    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep = logf(6.4) / 27.0
        if hz >= minLogHz {
            return minLogMel + logf(hz / minLogHz) / logstep
        }
        return hz / fSp
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep = logf(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * expf(logstep * (mel - minLogMel))
        }
        return fSp * mel
    }
}
