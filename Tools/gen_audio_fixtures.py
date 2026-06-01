"""Generate the audio-embedding reference fixture for the Swift MLX port.

Synthesizes ~3s of deterministic 16kHz mono audio (sum of fixed sine tones, no
randomness), computes the Whisper/Qwen2.5-Omni log-mel `input_features` the
reference `AudioModel` consumes, runs `JinaOmniSmallEmbeddingModel.encode_audio`,
and saves the canonical input_features / feature_lens / input_ids / embedding so
the Swift audio tower can match byte-for-byte.

Run inside the mls venv (mlx + transformers 5.1 + numpy + tokenizers; no torchaudio):
    cd ~/Documents/mls && uv run python ~/Documents/omni-macos/Tools/gen_audio_fixtures.py

Outputs:
    Fixtures/test_audio.wav            deterministic 16kHz mono PCM16 (~3s)
    Fixtures/audio_ref.safetensors     input_features, feature_lens, input_ids, embedding

------------------------------------------------------------------------------
AUDIO INPUT_FEATURES CONTRACT (what AudioModel.__call__ in model.py consumes):

  input_features : float32, shape [num_mel_bins=128, total_frames]  (MEL-MAJOR).
      The tower does `features_tc = input_features.T` to get [total_frames, 128],
      so the array stored on disk is [128, total_frames]: rows=mel bins,
      cols=time frames. total_frames == feature_lens.sum().
      These are the REAL (unpadded) mel frames concatenated across audios;
      there is NO 30s / 3000-frame Whisper padding here.

  feature_lens : int32, shape [num_audios].
      Per-audio number of mel time frames (== the unpadded frame count of that
      audio). For a single audio this is [total_frames]. The tower uses it to
      window the sequence into chunks of n_window*2 = 200 frames before conv.

Mel parameters (Whisper feature extractor, Qwen2.5-Omni settings):
  sampling_rate 16000, n_fft 400, hop_length 160, window = Hann (periodic, length 400),
  center padding (reflect) of n_fft//2 = 200 each side, power spectrogram |STFT|^2,
  128 mel bins, mel scale = "slaney" htk-style? -> Whisper uses mel_filters from
  librosa-equivalent triangular filters (norm="slaney", mel_scale="htk" via
  transformers `mel_filter_bank`), log compression:
      log_spec = log10(max(mel, 1e-10)); log_spec = max(log_spec, log_spec.max()-8);
      log_spec = (log_spec + 4) / 4.
  frame count for an L-sample clip (Whisper, center=True) = 1 + L // hop_length,
  with the final frame dropped by transformers (slices [..., :-1]); the unpadded
  feature_lens then = floor(L / hop_length) ... derived here from the attention mask.

input_ids construction (single audio):
  ids = [audio_start_token_id(151670)] + [audio_token_id(151669)] * N_audio
        + [audio_end_token_id(151671)]
  where N_audio = audio_features.shape[0] = number of 151669 placeholder slots.
  attention_mask = ones([1, L]).

Tower downsampling to N_audio (audio_features.shape[0]):
  conv1 (k3 s1 p1): T unchanged.   conv2 (k3 s2 p1): T -> (T-1)//2 + 1.
  Applied per chunk via feat_extract_output_lengths, then a final factor-2 mean
  pool: seg -> seg[:2*(T//2)].reshape(T//2, 2, d).mean(1). So roughly N_audio ~=
  total_frames / 4. The script asserts N_audio == count of 151669 tokens.

Deviation from model.py: NONE. We feed exactly what encode_audio expects.
------------------------------------------------------------------------------
"""

import importlib.util
import os
import struct
import sys
import wave

import mlx.core as mx
import numpy as np

MODEL_DIR = os.environ.get(
    "OMNI_MODEL_DIR",
    "/Volumes/One Touch/ai-models/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-mlx/snapshots/716c5b684db3f6ba574dd9b4f6b14af3b2eb8bda",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fixtures"
)

# --- Audio / mel constants ---------------------------------------------------
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
DURATION_S = 3.0

# --- Special token ids (from config.json / model.py) -------------------------
AUDIO_START = 151670
AUDIO_TOKEN = 151669
AUDIO_END = 151671


# ---------------------------------------------------------------------------
# 1. Deterministic synthetic audio (no randomness; fixed sine sum).
# ---------------------------------------------------------------------------
def make_test_audio(duration_s: float = DURATION_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sum of a few fixed sine tones with a fixed amplitude envelope.

    Fully reproducible: deterministic math, no np.random anywhere. Returns a
    float32 mono waveform in [-1, 1].
    """
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    # Fixed musical-ish chord plus a low rumble; fixed phases.
    sig = (
        0.45 * np.sin(2.0 * np.pi * 220.0 * t)            # A3
        + 0.30 * np.sin(2.0 * np.pi * 440.0 * t + 0.5)    # A4
        + 0.20 * np.sin(2.0 * np.pi * 660.0 * t + 1.0)    # E5
        + 0.10 * np.sin(2.0 * np.pi * 110.0 * t + 0.25)   # A2 rumble
    )
    # Smooth fixed amplitude envelope (raised-cosine fade in/out).
    fade = int(0.05 * sr)
    env = np.ones(n, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    env[:fade] = ramp
    env[-fade:] = ramp[::-1]
    sig = sig * env
    # Normalize deterministically to 0.9 peak.
    peak = float(np.max(np.abs(sig)))
    sig = (sig / peak) * 0.9
    return sig.astype(np.float32)


def save_wav_pcm16(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write mono PCM16 WAV."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(pcm), *pcm.tolist()))


# ---------------------------------------------------------------------------
# 2. Log-mel input_features via the Whisper feature extractor.
#    Qwen2.5-Omni uses padding="max_length" then crops by the attention mask.
# ---------------------------------------------------------------------------
def compute_input_features(audio: np.ndarray):
    """Return (input_features [128, total_frames] f32, feature_lens [1] i32).

    Uses transformers WhisperFeatureExtractor with the Qwen2.5-Omni audio
    settings (128 mel bins). The extractor pads to 30s; we recover the real
    frame count from the attention mask and crop to the unpadded mel frames,
    exactly mirroring how the Qwen2.5-Omni processor derives feature_lens and
    how the audio tower consumes the concatenated unpadded features.
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(
        feature_size=NUM_MEL_BINS,
        sampling_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        chunk_length=30,
        padding_value=0.0,
        dither=0.0,
    )
    out = fe(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        return_attention_mask=True,
        padding="max_length",
    )
    feats = np.asarray(out["input_features"], dtype=np.float32)  # [1, 128, 3000]
    attn = np.asarray(out["attention_mask"])                     # [1, 3000]
    # Real frame count for this audio (number of valid mel time frames).
    n_frames = int(attn[0].sum())
    feats = feats[0, :, :n_frames]                              # [128, n_frames]
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    feature_lens = np.array([n_frames], dtype=np.int32)
    return feats, feature_lens


# ---------------------------------------------------------------------------
# 3. Load reference model.
# ---------------------------------------------------------------------------
def load_ref():
    spec = importlib.util.spec_from_file_location(
        "jina_omni_utils", os.path.join(MODEL_DIR, "utils.py")
    )
    utils = importlib.util.module_from_spec(spec)
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    spec.loader.exec_module(utils)
    return utils.load_model(MODEL_DIR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Synthesize + save deterministic audio.
    audio = make_test_audio()
    wav_path = os.path.join(OUT_DIR, "test_audio.wav")
    save_wav_pcm16(wav_path, audio)
    print(f"[audio] saved {wav_path} samples={audio.shape[0]} "
          f"({audio.shape[0] / SAMPLE_RATE:.3f}s @ {SAMPLE_RATE}Hz mono PCM16)")

    # 2. Log-mel input_features + feature_lens.
    feats, feature_lens = compute_input_features(audio)
    total_frames = feats.shape[1]
    assert feats.shape[0] == NUM_MEL_BINS, feats.shape
    assert int(feature_lens.sum()) == total_frames, (feature_lens, total_frames)
    print(f"[mel] input_features={feats.shape} (mel-major [num_mel_bins, total_frames]) "
          f"dtype={feats.dtype} feature_lens={feature_lens.tolist()}")

    # 3. Load reference model + retrieval task (merges LoRA into text tower).
    m = load_ref()
    m.switch_task("retrieval")
    base = m.model  # JinaOmniSmallEmbeddingModel

    # 4. Run audio tower alone first to learn N_audio.
    feats_mx = mx.array(feats)
    flens_mx = mx.array(feature_lens)
    audio_hidden = base.audio_tower(feats_mx, flens_mx)
    mx.eval(audio_hidden)
    n_audio = int(audio_hidden.shape[0])
    print(f"[tower] audio_features={tuple(audio_hidden.shape)} N_audio={n_audio}")
    prefix_ids = list(m.tokenizer.encode("Document: ").ids)  # retrieval document prefix (all modalities)

    # 5. input_ids = [Document: ] + [audio_start] + audio_token * N_audio + [audio_end].
    ids = prefix_ids + [AUDIO_START] + [AUDIO_TOKEN] * n_audio + [AUDIO_END]
    L = len(ids)
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.ones((1, L), dtype=mx.int32)
    n_audio_tokens = sum(1 for t in ids if t == AUDIO_TOKEN)
    assert n_audio_tokens == n_audio, (
        f"audio-token count {n_audio_tokens} != N_audio {n_audio}"
    )

    # 6. Encode audio -> [1, 1024], L2-normalized inside _last_token_pool.
    emb = base.encode_audio(
        input_features=feats_mx,
        feature_lens=flens_mx,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    mx.eval(emb)

    norm = float(mx.linalg.norm(emb[0]).item())
    first5 = [float(x) for x in emb[0, :5].tolist()]

    # 7. Save the fixture.
    fixture_path = os.path.join(OUT_DIR, "audio_ref.safetensors")
    mx.save_safetensors(
        fixture_path,
        {
            "input_features": feats_mx.astype(mx.float32),
            "feature_lens": flens_mx.astype(mx.int32),
            "input_ids": input_ids.astype(mx.int32),
            "embedding": emb.astype(mx.float32),
        },
    )

    # 8. Report.
    print("=" * 60)
    print(f"num_mel_bins   = {NUM_MEL_BINS}")
    print(f"total_frames   = {total_frames}")
    print(f"N_audio        = {n_audio}")
    print(f"L (seq len)    = {L}")
    print(f"emb norm       = {norm:.6f}")
    print(f"emb[:5]        = {first5}")
    print(f"input_features shape={feats.shape} dtype={feats.dtype} (mel-major)")
    print(f"feature_lens   = {feature_lens.tolist()} (per-audio frame count)")
    print(f"input_ids      = [{AUDIO_START}] + [{AUDIO_TOKEN}]*{n_audio} + [{AUDIO_END}]")
    print(f"[saved] {fixture_path}")


if __name__ == "__main__":
    main()
