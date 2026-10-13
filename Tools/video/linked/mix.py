# Sound for the Linked Macs film: the score whole, the narration on its marks, the score ducked under
# the voice. usage: mix.py <out.wav>
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock as T

SR = 48000
def load(path, channels):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(channels), "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)

def smooth(env, attack, release, rate):
    out = np.empty_like(env); a = np.exp(-1 / (attack * rate)); r = np.exp(-1 / (release * rate)); y = env[0]
    for i, x in enumerate(env):
        k = a if x < y else r; y = k * y + (1 - k) * x; out[i] = y
    return out

out_path = sys.argv[1]
n = int(T.LENGTH * SR)
music = np.zeros((n, 2), np.float32)
sc = load(T.SCORE, 2)[:n]; music[:len(sc)] = sc
voice = np.zeros(n, np.float32)
for line, start in T.VO.items():
    v = load(os.path.join(T.VO_DIR, line + ".wav"), 1)[:, 0]
    i = int(start * SR); voice[i:i + len(v)] += v[:max(0, n - i)]
level = np.abs(voice)
gate = (np.convolve(level, np.ones(int(0.05 * SR)) / int(0.05 * SR), "same") > 0.01).astype(np.float32)
gate = np.minimum(1, np.convolve(gate, np.ones(int(0.45 * SR)), "same")).astype(np.float32)
target = 1 - gate * (1 - 10 ** (-13 / 20))
step = 48
g = smooth(target[::step], 0.12, 0.6, SR / step)
g = np.interp(np.arange(n), np.arange(0, n, step), g).astype(np.float32)
mix = music * g[:, None] + voice[:, None]
tail = int((T.LENGTH - 1.3) * SR); mix[tail:] *= np.linspace(1, 0, n - tail)[:, None] ** 2
import pyloudnorm
mix *= 10 ** ((-16 - pyloudnorm.Meter(SR).integrated_loudness(mix)) / 20)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                "-af", "alimiter=limit=0.84:attack=1:release=60:level=false", "-ar", str(SR), out_path],
               input=mix.astype(np.float32).tobytes(), check=True)
print(out_path, f"{T.LENGTH:.2f}s")
