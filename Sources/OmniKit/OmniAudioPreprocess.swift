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

    /// Decode `url` and compute log-mel features.
    /// Returns mel-major `[128, total_frames]` and `[total_frames]`, or nil on failure.
    public static func features(url: URL) -> (inputFeatures: MLXArray, featureLens: [Int])? {
        guard let samples = decodeMono16k(url: url), !samples.isEmpty else { return nil }

        let nBins = nFFT / 2 + 1   // 201
        let power = stftPower(samples)                  // [nBins, frames] row-major
        let frames = power.count / nBins
        if frames == 0 { return nil }

        let melFB = melFilterbank()                      // [nMel, nBins] row-major

        // mel = melFB @ power -> [nMel, frames], then log10/clamp/scale.
        // Build directly into a mel-major flat buffer [nMel, frames].
        var feat = [Float](repeating: 0, count: numMelBins * frames)
        var maxLog: Float = -Float.greatestFiniteMagnitude
        // mel[m, t] = sum_b melFB[m, b] * power[b, t]
        for m in 0 ..< numMelBins {
            let fbRow = m * nBins
            for t in 0 ..< frames {
                var acc: Float = 0
                for b in 0 ..< nBins {
                    acc += melFB[fbRow + b] * power[b * frames + t]
                }
                let v = log10f(Swift.max(acc, 1e-10))
                feat[m * frames + t] = v
                if v > maxLog { maxLog = v }
            }
        }
        // clamp to (maxLog - 8), then (x + 4) / 4.
        let floorVal = maxLog - 8.0
        for i in 0 ..< feat.count {
            let v = Swift.max(feat[i], floorVal)
            feat[i] = (v + 4.0) / 4.0
        }

        let arr = MLXArray(feat).reshaped([numMelBins, frames])   // mel-major
        return (arr, [frames])
    }

    // MARK: - Decode

    /// Decode any audio file to 16 kHz mono Float32 PCM via AVAudioFile + AVAudioConverter.
    private static func decodeMono16k(url: URL) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let srcFormat = file.processingFormat
        guard let dstFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false)
        else { return nil }

        guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else { return nil }
        converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue

        let srcFrameCount = AVAudioFrameCount(file.length)
        guard srcFrameCount > 0,
              let inBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrameCount)
        else { return nil }
        do {
            try file.read(into: inBuffer)
        } catch {
            return nil
        }

        // Output capacity estimate with headroom for the resample ratio.
        let ratio = sampleRate / srcFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inBuffer.frameLength) * ratio + 4096)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: outCapacity)
        else { return nil }

        // Single-shot input block: hand the whole buffer once, then end the stream.
        final class Feed: @unchecked Sendable { var done = false }
        let feed = Feed()
        var convError: NSError?
        let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
            if feed.done {
                outStatus.pointee = .endOfStream
                return nil
            }
            feed.done = true
            outStatus.pointee = .haveData
            return inBuffer
        }
        if status == .error || convError != nil { return nil }

        let n = Int(outBuffer.frameLength)
        guard n > 0, let ch = outBuffer.floatChannelData else { return nil }
        let ptr = ch[0]
        return Array(UnsafeBufferPointer(start: ptr, count: n))
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

        // Hann periodic window length nFFT: w[n] = 0.5 - 0.5*cos(2*pi*n/N).
        var window = [Float](repeating: 0, count: nFFT)
        let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
        for n in 0 ..< nFFT { window[n] = 0.5 - 0.5 * cosf(twoPiOverN * Float(n)) }

        // nFFT=400 is not a power of two; vDSP_DFT_zop supports arbitrary even
        // lengths and preserves the 201-bin frequency grid (no zero-padding).
        guard let dft = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
            return []
        }
        defer { vDSP_DFT_DestroySetup(dft) }

        var out = [Float](repeating: 0, count: nBins * frames)
        var reIn = [Float](repeating: 0, count: nFFT)
        var imIn = [Float](repeating: 0, count: nFFT)
        var reOut = [Float](repeating: 0, count: nFFT)
        var imOut = [Float](repeating: 0, count: nFFT)

        for t in 0 ..< frames {
            let base = t * hop
            for n in 0 ..< nFFT { reIn[n] = x[base + n] * window[n] }
            for n in 0 ..< nFFT { imIn[n] = 0 }
            vDSP_DFT_Execute(dft, reIn, imIn, &reOut, &imOut)
            for b in 0 ..< nBins {
                let re = reOut[b], im = imOut[b]
                out[b * frames + t] = re * re + im * im   // power |STFT|^2
            }
        }
        return out
    }

    // MARK: - Mel filterbank (Slaney-normalized triangular, matches transformers)

    /// 128 triangular mel filters over 201 FFT bins (0..8000 Hz), Slaney mel scale +
    /// Slaney area normalization. Row-major `[nMel, nBins]`. Built from scratch to
    /// match `transformers.audio_utils.mel_filter_bank(norm='slaney', mel_scale='slaney')`.
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
