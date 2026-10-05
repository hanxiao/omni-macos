# Render the film. usage:
#   render.py stills <out-dir> t1,t2,...        a PNG per time, for review
#   render.py video <out.mp4> [t0 t1] [--jobs N] [--fast]
import sys, os, subprocess, math
import numpy as np
import skia
from multiprocessing import Pool
import core
from core import W, H, FPS, prog, clamp
import film, timeline as T

SHUTTER = 0.5        # 180 degrees: sub-frames span half a frame

def draw_at(c, t):
    c.clear(skia.Color(20, 16, 13))
    prev = None
    for t0, t1, shot in film.SHOTS:
        if not (t0 <= t < t1):
            continue
        a = 1.0
        if prev is not None and (prev, shot) in film.DISSOLVE:
            d0, d1 = film.DISSOLVE[(prev, shot)]
            a = prog(t, d0, d1)
        if a >= 0.999:
            shot(c, t)
        elif a > 0.001:
            c.saveLayerAlpha(None, int(a * 255)); shot(c, t); c.restore()
        prev = shot
    f = film.fade_to_black(t)
    if f < 0.999:
        c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Color=skia.Color(12, 10, 8, int((1 - f) * 255))))

_surface = None
def frame(t, fast=False):
    global _surface
    if _surface is None:
        _surface = skia.Surface(W, H)
    c = _surface.getCanvas()
    n = 1 if fast else film.subframes(t)
    acc = None
    for k in range(n):
        ts = t + ((k + 0.5) / n - 0.5) * SHUTTER / FPS if n > 1 else t
        draw_at(c, ts)
        a = _surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3].astype(np.float32)
        acc = a if acc is None else acc + a
    out = core.post(acc / n, t)
    return out.astype(np.uint8)

def _chunk(args):
    i, f0, f1, path, fast = args
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium" if not fast else "veryfast",
                            "-crf", "12" if not fast else "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(frame(fi / FPS, fast).tobytes())
    enc.stdin.close(); enc.wait()
    return i

def _frames_in(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True)
    try: return int(r.stdout.strip())
    except ValueError: return -1

def video(out, t0, t1, jobs, fast):
    """Each chunk is its own process, checked by frame count and re-run until it is whole."""
    import time
    tmp = out + ".parts"; os.makedirs(tmp, exist_ok=True)
    f0, f1 = int(t0 * FPS), int(t1 * FPS)
    n = max(1, min(jobs * 3, (f1 - f0) // 30))
    edges = [f0 + (f1 - f0) * k // n for k in range(n + 1)]
    tasks = [(k, edges[k], edges[k + 1], os.path.join(tmp, f"{k:04d}.mp4")) for k in range(n)]
    todo = [tk for tk in tasks if _frames_in(tk[3]) != tk[2] - tk[1]]
    for attempt in range(4):
        running = []
        queue = list(todo)
        while queue or running:
            while queue and len(running) < jobs:
                k, a, b, path = queue.pop(0)
                cmd = [sys.executable, os.path.abspath(__file__), "chunk", str(k), str(a), str(b), path] + (["--fast"] if fast else [])
                running.append((subprocess.Popen(cmd, stderr=subprocess.DEVNULL), (k, a, b, path)))
            time.sleep(0.2)
            for pr, tk in list(running):
                if pr.poll() is not None: running.remove((pr, tk))
        todo = [tk for tk in tasks if _frames_in(tk[3]) != tk[2] - tk[1]]
        print(f"pass {attempt + 1}: {len(tasks) - len(todo)}/{len(tasks)} chunks whole", flush=True)
        if not todo: break
    if todo: raise SystemExit(f"chunks still broken: {[tk[0] for tk in todo]}")
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(tk[3])}'\n" for tk in tasks))
    silent = out + ".silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent], check=True)
    mix = os.path.join(core.WORK, "mix.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-ss", str(t0), "-t", str(t1 - t0), "-i", mix,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", out], check=True)
    print(out)

def _still(args):
    t, outdir, fast = args
    from PIL import Image
    Image.fromarray(frame(t, fast)).save(os.path.join(outdir, f"t{t:07.2f}.png"))
    return t

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "stills":
        outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
        times = [float(x) for x in sys.argv[3].split(",")]
        with Pool(min(len(times), 24)) as p:
            for t in p.imap_unordered(_still, [(t, outdir, "--fast" in sys.argv) for t in times]): print("still", t, flush=True)
    elif mode == "chunk":
        k, a, b, path = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        _chunk((k, a, b, path, "--fast" in sys.argv))
    else:
        out = sys.argv[2]
        argv = sys.argv[3:]
        if "--jobs" in argv: i = argv.index("--jobs"); argv = argv[:i] + argv[i + 2:]
        rest = [a for a in argv if not a.startswith("--")]
        t0, t1 = (float(rest[0]), float(rest[1])) if len(rest) >= 2 else (0.0, T.LENGTH)
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 24
        video(out, t0, t1, jobs, "--fast" in sys.argv)
