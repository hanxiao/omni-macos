# Synthesized score for the Omni intro. 116 BPM, laid out on the video's beat grid.
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import wave, sys

SR = 48000
BPM = 116.0
BEAT = 60.0 / BPM
TOTAL_BEATS = 62
DUR = TOTAL_BEATS * BEAT + 2.5
N = int(DUR * SR)
t_all = np.arange(N) / SR
rng = np.random.default_rng(7)

DROP, OUTRO = 6, 52                  # drums from the drop to the outro
TRANSITIONS = [6, 20, 30, 38, 52]    # scene boundaries, in beats

def midi(n): return 440.0 * 2 ** ((n - 69) / 12)
def at(beat): return int(beat * BEAT * SR)
def lp(x, fc, order=2): return sosfilt(butter(order, fc, 'low', fs=SR, output='sos'), x)
def hp(x, fc, order=2): return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)

def env_adsr(n, a, d, s, r, sustain_len):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    hold = max(0, int(sustain_len * SR) - a - d)
    e = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, s, d, endpoint=False),
                        np.full(hold, s), np.linspace(s, 0, r)])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))

def saw(f, n, phase=0.0):
    ph = (phase + f * np.arange(n) / SR) % 1.0
    return 2 * ph - 1

# F major, mellow: Fmaj7, Am7, Dm9, Bbmaj9 - one chord per bar.
CHORDS = [[53, 57, 60, 64], [57, 60, 64, 67], [50, 53, 57, 60, 64], [46, 50, 53, 57, 60]]
ROOTS = [41, 45, 38, 46]

mix = {k: np.zeros(N) for k in ['pad', 'pluck', 'kick', 'clap', 'hat', 'bass', 'fx', 'bell']}

# PAD: detuned saws per chord note, low-passed, slow attack; the whole piece.
for bar in range(TOTAL_BEATS // 4 + 1):
    b0 = bar * 4
    if b0 >= TOTAL_BEATS: break
    chord = CHORDS[bar % 4]
    length = 4 * BEAT + 0.6
    n = int(length * SR); s0 = at(b0)
    if s0 + n > N: n = N - s0
    voice = np.zeros(n)
    for note in chord:
        for det in (-0.08, 0.0, 0.07):
            voice += saw(midi(note + 12) * 2 ** (det / 12), n, rng.random())
    cutoff = 1400 if b0 < DROP else 2600
    voice = lp(voice, cutoff, 4) * env_adsr(n, 0.35, 0.4, 0.8, 0.6, 4 * BEAT) * 0.035
    mix['pad'][s0:s0 + n] += voice

# PLUCK ARP: 16ths over chord tones, short decay; from the drop to the outro (sparse in the intro).
pattern = [0, 2, 1, 3, 2, 1, 3, 2]
for step in range(TOTAL_BEATS * 2):
    b = step / 2
    if b >= OUTRO + 4: break
    if b < DROP and step % 2: continue
    chord = CHORDS[int(b // 4) % 4]
    note = chord[pattern[step % 8] % len(chord)] + 24
    n = int(0.35 * SR); s0 = at(b)
    if s0 + n > N: continue
    tt = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * midi(note) * tt) + 0.35 * np.sin(4 * np.pi * midi(note) * tt)) * np.exp(-tt * 11)
    mix["pluck"][s0:s0 + n] += tone * (0.14 if b >= DROP else 0.09)

# KICK: pitch-swept sine, four on the floor.
for b in range(DROP, OUTRO):
    n = int(0.45 * SR); s0 = at(b); tt = np.arange(n) / SR
    f = 48 + 110 * np.exp(-tt * 30)
    kick = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 7.5)
    kick += 0.3 * np.exp(-tt * 180) * rng.standard_normal(n) * 0.2
    mix['kick'][s0:s0 + n] += kick * 0.55

# CLAP on 2 and 4.
for b in range(DROP, OUTRO):
    if b % 2 == 1:
        n = int(0.25 * SR); s0 = at(b); tt = np.arange(n) / SR
        noise = bp(rng.standard_normal(n), 900, 3500)
        e = np.exp(-tt * 22) + 0.6 * np.exp(-((tt - 0.012) ** 2) / 1e-5)
        mix["clap"][s0:s0 + n] += noise * e * 0.6

# HATS: closed 16ths with accents, open hat on the off-beat.
for step in range(DROP * 4, OUTRO * 4):
    b = step / 4
    open_hat = step % 4 == 2
    n = int((0.18 if open_hat else 0.05) * SR); s0 = at(b); tt = np.arange(n) / SR
    noise = hp(rng.standard_normal(n), 7000)
    vel = [0.5, 0.25, 0.8, 0.3][step % 4]
    mix['hat'][s0:s0 + n] += noise * np.exp(-tt * (18 if open_hat else 70)) * vel * 0.07

# BASS: sub sine + a little saw on the root, 8th-note pulse off the kick.
for step in range(DROP * 2, OUTRO * 2):
    b = step / 2
    if step % 2 == 0: continue            # the off-beat, so it breathes against the kick
    root = ROOTS[int(b // 4) % 4]
    n = int(0.42 * BEAT * 2 * SR); s0 = at(b); tt = np.arange(n) / SR
    tone = np.sin(2 * np.pi * midi(root) * tt) + 0.25 * lp(saw(midi(root), n), 500)
    mix['bass'][s0:s0 + n] += tone * env_adsr(n, 0.005, 0.1, 0.7, 0.05, n / SR - 0.05) * 0.22

# FX: a rising filtered-noise whoosh into every scene change, and a sub drop on the drop.
for tb in TRANSITIONS:
    lead = 1.2 * BEAT * 2
    n = int(lead * SR); s0 = at(tb) - n
    if s0 < 0: continue
    tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    sweep = np.zeros(n)
    for i, (a, bnd) in enumerate(zip(np.linspace(300, 6000, 12)[:-1], np.linspace(300, 6000, 12)[1:])):
        seg = slice(i * n // 11, (i + 1) * n // 11)
        sweep[seg] = bp(noise, a, bnd * 1.5)[seg]
    mix['fx'][s0:s0 + n] += sweep * (tt / lead) ** 2 * 0.09
    m = int(0.9 * SR); s1 = at(tb); tt = np.arange(m) / SR
    if s1 + m < N:
        mix['fx'][s1:s1 + m] += np.sin(2 * np.pi * np.cumsum(70 * np.exp(-tt * 3)) / SR) * np.exp(-tt * 4) * 0.25

# BELL: a final chord at the outro, let ring.
n = N - at(OUTRO); tt = np.arange(n) / SR
for note in [65, 69, 72, 76, 79]:
    mix['bell'][at(OUTRO):] += np.sin(2 * np.pi * midi(note) * tt) * np.exp(-tt * 0.9) * 0.03

# SIDECHAIN: pad, pluck and bass duck on every kick - the pump that makes it feel current.
duck = np.ones(N)
for b in range(DROP, OUTRO):
    n = int(BEAT * SR); s0 = at(b); tt = np.arange(n) / SR
    duck[s0:s0 + n] = 1 - 0.6 * np.exp(-tt * 9)
for k in ('pad', 'pluck', 'bass'): mix[k] *= duck

# REVERB: synthetic IR on the melodic parts.
ir_len = int(2.2 * SR); tt = np.arange(ir_len) / SR
ir = rng.standard_normal(ir_len) * np.exp(-tt * 2.6); ir = lp(ir, 5000); ir /= np.abs(ir).sum() ** 0.5 * 12
wet_src = mix['pad'] + mix['pluck'] + mix['bell'] + mix['clap'] * 0.5 + mix['fx'] * 0.4
wet = fftconvolve(wet_src, ir)[:N]

if '--stems' in sys.argv:
    a, b = at(DROP + 4), at(OUTRO - 4)
    kick_db = 20 * np.log10(np.sqrt(np.mean(mix['kick'][a:b] ** 2)) + 1e-12)
    for k, v in mix.items():
        r = np.sqrt(np.mean(v[a:b] ** 2)) + 1e-12
        print(f"{k:6s} {20 * np.log10(r) - kick_db:+6.1f} dB vs kick")
    sys.exit(0)
out = sum(mix.values()) + wet * 0.8
out = hp(out, 28)
# gentle master: soft clip, fade out, normalise to -1 dBFS
out = np.tanh(out * 1.6) / 1.6
fade = int(2.0 * SR); out[-fade:] *= np.linspace(1, 0, fade)
fade_in = int(0.4 * SR); out[:fade_in] *= np.linspace(0, 1, fade_in)
out *= 10 ** (-1 / 20) / np.abs(out).max()
stereo = np.stack([out, out], axis=1)
# a touch of width: delay the melodic wet signal by 12 ms on the right
d = int(0.012 * SR); stereo[d:, 1] += wet[:-d] * 0.15; stereo[:, 0] += wet * 0.05
stereo *= 10 ** (-1 / 20) / np.abs(stereo).max()
pcm = (stereo * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(f"{DUR:.2f}s beat={BEAT:.4f}s drop={DROP*BEAT:.2f}s outro={OUTRO*BEAT:.2f}s")
