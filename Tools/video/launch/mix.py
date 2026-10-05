# Sound: the score edit, the narration on its marks, the score ducked under the voice.
# usage: mix.py <score.mp3> <vo trim dir> <out.wav>
import sys, subprocess, numpy as np
import timeline as T

SR = 48000

def load(path, channels):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(channels),
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)

def smooth(env, attack, release, rate):
    # one-pole follower at `rate` Hz, separate attack (down) and release (up) times, in seconds
    out = np.empty_like(env); a = np.exp(-1 / (attack * rate)); r = np.exp(-1 / (release * rate)); y = env[0]
    for i, x in enumerate(env):
        k = a if x < y else r; y = k * y + (1 - k) * x; out[i] = y
    return out

score_path, vo_dir, out_path = sys.argv[1:4]
score = load(score_path, 2)
n = int(T.LENGTH * SR) + SR
music = np.zeros((n, 2), np.float32)
XF = int(T.PRE * SR)                                  # equal-power crossfade, over before the kick
up = np.sin(np.linspace(0, np.pi / 2, XF))[:, None].astype(np.float32)
for k, ((a, b), s) in enumerate(zip(T.MUSIC, T.SEG_START)):
    last = k == len(T.MUSIC) - 1
    seg = score[int(a * SR):int(b * SR) + (0 if last else XF)].copy()
    if k: seg[:XF] *= up                               # fades in while the previous one fades out
    if not last: seg[-XF:] *= up[::-1]
    i = int(s * SR); music[i:i + len(seg)] += seg

import os
if os.environ.get("MIX_DEBUG"):                         # the score edit alone, for beat checks
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                    out_path.replace(".wav", "-music.wav")], input=music.tobytes(), check=True)

voice = np.zeros(n, np.float32)
for line, start in T.VO.items():
    v = load(f"{vo_dir}/{line}.wav", 1)[:, 0]
    i = int(start * SR); voice[i:i + len(v)] += v

# duck: -10 dB under speech (gated on the voice's own envelope), -14 dB in the lift, whose lead
# melody sits where a voice does; fast attack, slow release so it breathes between sentences
level = np.abs(voice)
gate = (np.convolve(level, np.ones(int(0.05 * SR)) / int(0.05 * SR), "same") > 0.01).astype(np.float32)
# hold through the gaps inside a sentence
hold = int(0.45 * SR)
gate = np.minimum(1, np.convolve(gate, np.ones(hold), "same")).astype(np.float32)
t = np.arange(n) / SR
depth = np.where((t > T.LIFT) & (t < T.OUTRO), 10 ** (-17 / 20), 10 ** (-13 / 20)).astype(np.float32)
target = 1 - gate * (1 - depth)
# decimate for the follower, then back up
step = 48
g = smooth(target[::step], 0.12, 0.6, SR / step)
g = np.interp(np.arange(n), np.arange(0, n, step), g).astype(np.float32)
mix = music * g[:, None] + voice[:, None] * 1.0
# a gentle fade on the final tail
tail = int((T.LENGTH - 1.2) * SR)
mix[tail:] *= np.linspace(1, 0, n - tail)[:, None] ** 2
mix = mix[:int(T.LENGTH * SR)]
# linear loudness to -16 LUFS (no compressor pumping), then a brickwall only if a peak needs it
import pyloudnorm
meter = pyloudnorm.Meter(SR)
mix *= 10 ** ((-16 - meter.integrated_loudness(mix)) / 20)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                "-af", "alimiter=limit=0.84:attack=1:release=60:level=false", "-ar", str(SR), out_path],
               input=mix.astype(np.float32).tobytes(), check=True)
print(out_path, f"{T.LENGTH:.2f}s")
