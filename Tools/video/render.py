# Omni intro renderer, v2: live-index takes, eased camera, captions in their own band, a feature grid.
# usage: render2.py <out.mp4> [--preview "t1,t2,..."]
import sys, os, subprocess, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

V = os.environ.get("VIDEO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
REPO = os.environ.get("OMNI_REPO", "/Users/hanxiao/Documents/omni-macos")
TAKES_DIR = os.path.join(V, "takesR")
W, H, FPS = 1920, 1080, 60
BEAT = 60 / 116.0
SRC_W, SRC_H = 1800, 960
FONT = "/System/Library/Fonts/SFNS.ttf"
def b(n): return n * BEAT

# ---- timeline, on the music's beat grid ----
T_A, T_B, T_C, T_F, T_D, T_FEAT, T_OUT = b(6), b(20), b(38), b(44), b(52), b(64), b(72)
T_END = b(78) + 0.6
XF = 0.42

# ---- easing ----
def clamp(x, a=0.0, c=1.0): return max(a, min(c, x))
def smooth(x): x = clamp(x); return x * x * x * (x * (6 * x - 15) + 10)
def out_expo(x): x = clamp(x); return 1 if x >= 1 else 1 - 2 ** (-10 * x)
def out_cubic(x): x = clamp(x); return 1 - (1 - x) ** 3
def lerp(a, c, t): return a + (c - a) * t

# ---- framing: (focus in source px, where the focus sits on screen, scale) ----
BASE_S = 0.84
BASE = ((SRC_W / 2, SRC_H / 2), (W / 2, H - SRC_H * BASE_S / 2 - 34), BASE_S)   # leaves a caption band
CAPTION_Y = 118
FIELD = ((1640, 36), (W * 0.76, 150), 1.3)                                    # the search field, readable
GRID = ((SRC_W / 2, SRC_H * 0.46), (W / 2, H / 2), 1.0)
GRID2 = ((SRC_W / 2, SRC_H * 0.46), (W / 2, H / 2), 1.04)
CHIP = (985, 920)

SCENES = [
    dict(name="A", t0=T_A, t1=T_B, caption="Search by meaning.",
         clips=[(0.0, "A", 56.4)],
         cam=[(0, BASE), (1.4, BASE), (2.1, FIELD), (5.4, FIELD), (6.4, GRID), (7.24, GRID2)]),
    dict(name="B", t0=T_B, t1=T_C, caption="Any language.",
         clips=[(0.0, "B", 57.0), (6.1, "B", 68.3)],
         cam=[(0, BASE), (1.4, BASE), (2.0, FIELD), (4.9, FIELD), (5.6, GRID), (6.1, GRID), (6.6, FIELD), (7.2, FIELD),
              (7.9, GRID), (9.3, GRID2)]),
    dict(name="C", t0=T_C, t1=T_F, caption="Find similar.",
         clips=[(0.0, "C", 58.3)],
         cam=[(0, BASE), (1.6, BASE), (3.1, ((SRC_W * 0.55, SRC_H * 0.45), (W / 2, H * 0.52), 0.95))]),
    dict(name="F", t0=T_F, t1=T_D, caption="Feels like Finder.",
         clips=[(0.0, "F", 34.6)],
         cam=[(0, BASE), (1.7, BASE), (4.14, ((520, 250), (W * 0.36, H * 0.42), 1.14))]),
    dict(name="D", t0=T_D, t1=T_FEAT, caption="Reads every page.",
         clips=[(0.0, "D", 49.6)],
         cam=[(0, BASE), (1.7, BASE), (3.4, ((1150, 560), (W * 0.5, H * 0.52), 1.08)),
              # the speed chip sits 40 px above the window's bottom edge: keep that edge at the frame's
              # bottom so the window fills the shot rather than leaving a band of background under it
              (5.4, (CHIP, (W * 0.5, H - 36 - 40 * 1.8), 1.8)), (6.2, (CHIP, (W * 0.5, H - 36 - 40 * 1.86), 1.86))]),
]
# each scene enters from where the previous one left the camera, easing to BASE under its caption
for prev, sc in zip(SCENES, SCENES[1:]):
    exit_cam = prev["cam"][-1][1]
    if exit_cam != BASE:
        sc["cam"] = [(0, exit_cam), (0.8, BASE)] + [k for k in sc["cam"] if k[0] > 0.8]

def cam_track(keys, t):
    if t <= keys[0][0]: return keys[0][1]
    for (ta, ca), (tb, cb) in zip(keys, keys[1:]):
        if t <= tb:
            u = smooth((t - ta) / (tb - ta))
            return ((lerp(ca[0][0], cb[0][0], u), lerp(ca[0][1], cb[0][1], u)),
                    (lerp(ca[1][0], cb[1][0], u), lerp(ca[1][1], cb[1][1], u)), lerp(ca[2], cb[2], u))
    return keys[-1][1]

# ---- assets ----
def font(size, weight=600):
    f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([100, min(96, max(17, size)), 400, weight]); return f

def make_wash(t):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.ones((H, W, 3), np.float32) * np.array([251, 251, 253], np.float32)
    drift = [(math.sin(t * 0.21) * 0.03, math.cos(t * 0.17) * 0.03), (math.cos(t * 0.19) * 0.03, math.sin(t * 0.23) * 0.03),
             (math.sin(t * 0.13) * 0.03, math.cos(t * 0.29) * 0.02)]
    for (cx, cy, sx, sy, col, a), (dx, dy) in zip(
            [(0.2, 0.42, 0.30, 0.44, (90, 160, 255), 0.22), (0.82, 0.34, 0.28, 0.42, (255, 150, 190), 0.17),
             (0.56, 0.86, 0.34, 0.40, (120, 220, 200), 0.17)], drift):
        w = np.exp(-(((xx - (cx + dx) * W) / (sx * W)) ** 2 + ((yy - (cy + dy) * H) / (sy * H)) ** 2) * 1.6)[..., None] * a
        img = img * (1 - w) + np.array(col, np.float32) * w
    return img

WASH_KEYS = MOLE = WIN_ALPHA = SHADOW = TAKES = None
SYMS = {}
SH_PAD = 160
FEATURES = [("macwindow", "Native as Finder"), ("cpu", "Metal on the GPU"), ("wifi.slash", "Fully airgapped"),
            ("arrow.triangle.2.circlepath", "Always in sync"), ("doc.text.viewfinder", "OCR built in"),
            ("chevron.left.forwardslash.chevron.right", "Apache 2.0")]

def wash(t):
    k = (t / T_END) * (len(WASH_KEYS) - 1); i = min(int(k), len(WASH_KEYS) - 2); u = k - i
    return WASH_KEYS[i] * (1 - u) + WASH_KEYS[i + 1] * u

def aa_rounded_mask(w, h, r, inset):
    """Supersampled so the edge is antialiased; inset and radius sized to keep the capture's dark
    corner fringe (its own antialiased edge over black) outside the mask."""
    s = 4
    m = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(m).rounded_rectangle([inset * s, inset * s, (w - inset) * s - 1, (h - inset) * s - 1], r * s, fill=255)
    return m.resize((w, h), Image.LANCZOS)

def traffic_lights(frame):
    img = Image.fromarray(frame); d = ImageDraw.Draw(img)
    bg = tuple(int(v) for v in frame[27, 100])
    d.rounded_rectangle([8, 10, 92, 44], 12, fill=bg)
    for i, col in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx, cy = 28 + i * 20, 27; d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
    return img

class Takes:
    def __init__(self):
        self.mm = {}
        for t in "ABCDF":
            meta = json.load(open(os.path.join(TAKES_DIR, f"{t}.json")))
            self.mm[t] = (np.memmap(os.path.join(TAKES_DIR, f"{t}.rgb"), np.uint8, "r",
                                    shape=(meta["n"], SRC_H, SRC_W, 3)), meta["n"], meta.get("t0", 0))
    def frame(self, take, s):
        arr, n, t0 = self.mm[take]
        i = int(round(clamp((s - t0) * FPS, 0, n - 1)))
        return traffic_lights(np.array(arr[i]))

def window_image(take, s):
    img = TAKES.frame(take, s).convert("RGBA"); img.putalpha(WIN_ALPHA); return img

def place_window(canvas, win, cam, alpha=1.0, blur=0.0):
    (fx, fy), (px, py), s = cam
    layer = win.transform((W, H), Image.AFFINE, (1 / s, 0, fx - px / s, 0, 1 / s, fy - py / s), resample=Image.BICUBIC)
    sh = SHADOW.transform((W, H), Image.AFFINE, (1 / s, 0, fx - px / s + SH_PAD, 0, 1 / s, fy - (py - 26 * s) / s + SH_PAD),
                          resample=Image.BILINEAR)
    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha))); sh = sh.point(lambda v: int(v * alpha))
    if blur > 0.3: layer = layer.filter(ImageFilter.GaussianBlur(blur))
    shadow = Image.new("RGBA", (W, H), (30, 30, 60, 0)); shadow.putalpha(sh)
    canvas.alpha_composite(shadow); canvas.alpha_composite(layer)

def draw_text(canvas, text, size, weight, center, color, alpha, rise=0.0, blur=0.0, tracking=-0.02):
    if alpha <= 0: return
    f = font(size, weight)
    widths = [f.getlength(ch) for ch in text]
    total = sum(widths) + tracking * size * (len(text) - 1); pad = int(size * 0.6)
    layer = Image.new("RGBA", (int(total) + 2 * pad, int(size * 1.6) + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer); x = pad
    for ch, w_ in zip(text, widths):
        d.text((x, pad), ch, font=f, fill=tuple(color) + (int(255 * alpha),)); x += w_ + tracking * size
    x0, y0, x1, y1 = layer.getchannel("A").getbbox() or (0, 0, layer.width, layer.height)
    if blur > 0.3: layer = layer.filter(ImageFilter.GaussianBlur(blur))
    cx, cy = center
    canvas.alpha_composite(layer, (int(cx - (x0 + x1) / 2), int(cy - (y0 + y1) / 2 + rise)))

def caption(canvas, text, tl):
    a_in = out_cubic(tl / 0.45); a = clamp(min(a_in, 1 - smooth((tl - 1.45) / 0.4)))
    draw_text(canvas, text, 64, 650, (W / 2, CAPTION_Y), (29, 29, 31), a, rise=(1 - a_in) * 18, blur=(1 - a_in) * 8)

def mole_layer(height, alpha):
    w = int(MOLE.width * height / MOLE.height)
    m = MOLE.resize((max(1, w), max(1, int(height))), Image.LANCZOS)
    if alpha < 1: m.putalpha(m.getchannel("A").point(lambda v: int(v * alpha)))
    return m

def feature_grid(canvas, tl, dur):
    cols, rows, tw, th, gap = 3, 2, 470, 200, 30
    gw, gh = cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap
    x0, y0 = (W - gw) / 2, (H - gh) / 2
    out = smooth((tl - (dur - 0.5)) / 0.45)
    bg = None
    for i, (sym, title) in enumerate(FEATURES):
        r, c = divmod(i, cols)
        u = out_expo((tl - 0.12 - i * 0.09) / 0.7)
        a = clamp(u) * (1 - out)
        if a <= 0: continue
        x = x0 + c * (tw + gap); y = y0 + r * (th + gap) + (1 - u) * 34 - out * 20
        if bg is None: bg = canvas.filter(ImageFilter.GaussianBlur(30))
        # glass: the wash behind, blurred and lifted toward white, a hairline edge and a soft shadow
        box = (int(x), int(y), int(x + tw), int(y + th))
        tile = bg.crop(box); tile = Image.blend(tile, Image.new("RGBA", tile.size, (255, 255, 255, 255)), 0.62)
        mask = Image.new("L", tile.size, 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], 30, fill=int(255 * a))
        shadow = Image.new("L", (tw + 120, th + 120), 0)
        ImageDraw.Draw(shadow).rounded_rectangle([60, 72, 60 + tw, 72 + th], 30, fill=int(40 * a))
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        sh = Image.new("RGBA", shadow.size, (40, 40, 80, 0)); sh.putalpha(shadow)
        canvas.alpha_composite(sh, (box[0] - 60, box[1] - 60))
        canvas.paste(tile, box[:2], mask)
        edge = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        ImageDraw.Draw(edge).rounded_rectangle([0, 0, tw - 1, th - 1], 30, outline=(255, 255, 255, int(230 * a)), width=2)
        canvas.alpha_composite(edge, box[:2])
        icon = SYMS[sym]; ih = 58
        icon = icon.resize((int(icon.width * ih / icon.height), ih), Image.LANCZOS)
        tint = Image.new("RGBA", icon.size, (0, 113, 227, 0)); tint.putalpha(icon.getchannel("A").point(lambda v: int(v * a)))
        canvas.alpha_composite(tint, (int(x + 40), int(y + 44)))
        draw_text(canvas, title, 40, 620, (x + 40 + 0, 0), (29, 29, 31), 0)  # measure only
        f = font(40, 620); tw_ = f.getlength(title)
        draw_text(canvas, title, 40, 620, (x + 40 + tw_ / 2, y + 150), (29, 29, 31), a)

def render_frame(fi):
    t = fi / FPS
    canvas = Image.fromarray(wash(t).astype(np.uint8), "RGB").convert("RGBA")

    # intro: the mole, the name, then the window grows out of the mole
    if t < T_A + 0.2:
        grow = out_expo(t / 0.9); morph = smooth((t - 1.7) / (T_A - 1.5))
        ma = clamp(grow) * (1 - morph)
        if ma > 0:
            m = mole_layer(330 * (0.86 + 0.14 * grow - 0.5 * morph), ma)
            canvas.alpha_composite(m, (int(W / 2 - m.width / 2), int(H * 0.40 - m.height / 2 - 40 * morph)))
        na = out_cubic((t - 0.45) / 0.6) * (1 - smooth((t - 1.6) / 0.5))
        draw_text(canvas, "Omni", 104, 700, (W / 2, H * 0.40 + 165 + 70), (29, 29, 31), na,
                  rise=(1 - out_cubic((t - 0.45) / 0.6)) * 22, blur=(1 - out_cubic((t - 0.45) / 0.6)) * 10)
        if morph > 0:
            s = lerp(0.12, BASE_S, morph)
            cam = ((SRC_W / 2, SRC_H / 2), (W / 2, lerp(H * 0.40, BASE[1][1], morph)), s)
            place_window(canvas, window_image("A", SCENES[0]["clips"][0][2]), cam, alpha=clamp(morph * 1.6))

    for i, sc in enumerate(SCENES):
        t0, t1 = sc["t0"], sc["t1"]
        last = i == len(SCENES) - 1
        if not (t0 - (XF if i else 0) <= t < t1 + (0.9 if last else 0)): continue
        if i == 0 and t < T_A: continue
        tl = t - t0
        clip = [c for c in sc["clips"] if c[0] <= max(tl, 0)][-1]
        take, src = clip[1], clip[2] + (max(tl, 0) - clip[0])
        cam = cam_track(sc["cam"], max(tl, 0))
        alpha, blur, k = 1.0, 0.0, 1.0
        if tl < 0:
            u = (tl + XF) / XF; alpha = smooth(u); blur = math.sin(math.pi * u) * 5; k = 1 + 0.025 * math.sin(math.pi * u)
        if last and t >= t1:     # the last scene gives way to the feature grid: pulls back and dissolves
            u = smooth((t - t1) / 0.9); alpha = 1 - u; blur = u * 14; k = lerp(1.0, 0.9, u)
        (f_, p_, s_) = cam
        place_window(canvas, window_image(take, src), (f_, p_, s_ * k), alpha=alpha, blur=blur)
        for (ct, ctake, csrc) in sc["clips"][1:]:
            if 0 <= tl - ct < 0.25:     # a content crossfade at an in-scene cut
                prev = [c for c in sc["clips"] if c[0] < ct][-1]
                place_window(canvas, window_image(prev[1], prev[2] + (tl - prev[0])), cam, alpha=1 - smooth((tl - ct) / 0.25))
        if tl >= 0: caption(canvas, sc["caption"], tl)

    if T_FEAT <= t < T_OUT + 0.4:
        feature_grid(canvas, t - T_FEAT, T_OUT - T_FEAT + 0.4)

    if t >= T_OUT:
        tl = t - T_OUT
        outer, layer = canvas, Image.new("RGBA", (W, H), (0, 0, 0, 0))
        grow = out_expo((tl - 0.3) / 0.9)
        if grow > 0:
            m = mole_layer(250 * (0.86 + 0.14 * grow), clamp(grow))
            layer.alpha_composite(m, (int(W / 2 - m.width / 2), int(H * 0.36 - m.height / 2)))
            ta = out_cubic((tl - 0.6) / 0.6)
            draw_text(layer, "Omni", 96, 700, (W / 2, H * 0.36 + 190), (29, 29, 31), ta, rise=(1 - ta) * 18, blur=(1 - ta) * 8)
            ua = out_cubic((tl - 0.95) / 0.6)
            draw_text(layer, "Search your Mac by meaning.", 38, 500, (W / 2, H * 0.36 + 272), (110, 110, 118), ua,
                      rise=(1 - ua) * 12, blur=(1 - ua) * 6, tracking=-0.01)
            va = out_cubic((tl - 1.3) / 0.6)
            draw_text(layer, "hanxiao.io/omni", 32, 600, (W / 2, H * 0.36 + 334), (0, 113, 227), va,
                      rise=(1 - va) * 10, blur=(1 - va) * 5, tracking=0.0)
            kk = 1 + 0.03 * smooth(tl / (T_END - T_OUT))
            cx, cy = W / 2, H * 0.45
            layer = layer.transform((W, H), Image.AFFINE, (1 / kk, 0, cx - cx / kk, 0, 1 / kk, cy - cy / kk), resample=Image.BICUBIC)
            outer.alpha_composite(layer)
    return np.asarray(canvas.convert("RGB")).tobytes()

def init():
    global TAKES, WIN_ALPHA, SHADOW, MOLE, WASH_KEYS
    TAKES = Takes()
    WIN_ALPHA = aa_rounded_mask(SRC_W, SRC_H, 28, 1)
    sh = Image.new("L", (SRC_W + 2 * SH_PAD, SRC_H + 2 * SH_PAD), 0)
    ImageDraw.Draw(sh).rounded_rectangle([SH_PAD, SH_PAD, SH_PAD + SRC_W, SH_PAD + SRC_H], 28, fill=70)
    SHADOW = sh.resize((sh.width // 4, sh.height // 4)).filter(ImageFilter.GaussianBlur(14)).resize(sh.size, Image.BILINEAR)
    src = Image.open(os.path.join(REPO, "App/AppIcon.icon/Assets/foreground.png")).convert("RGBA")
    MOLE = src.crop(src.getbbox())
    WASH_KEYS = np.load(os.path.join(V, "wash2.npy"))
    for sym, _ in FEATURES:
        SYMS[sym] = Image.open(os.path.join(V, "sym", f"{sym}.png")).convert("RGBA")

if __name__ == "__main__":
    out = sys.argv[1]
    if not os.path.exists(os.path.join(V, "wash2.npy")):
        np.save(os.path.join(V, "wash2.npy"), np.stack([make_wash(T_END * k / 7) for k in range(8)]).astype(np.float32))
    n = int(T_END * FPS)
    if "--preview" in sys.argv:
        init()
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            Image.frombytes("RGB", (W, H), render_frame(int(float(s) * FPS))).save(f"{out}-{float(s):05.2f}.png")
        sys.exit(0)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", os.path.join(V, "music2.wav"),
                           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-profile:v", "high",
                           "-movflags", "+faststart", "-af", f"afade=t=out:st={T_END - 1.8:.2f}:d=1.8", "-t", f"{T_END:.3f}",
                           "-c:a", "aac", "-b:a", "192k", out], stdin=subprocess.PIPE)
    with Pool(16, initializer=init) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            ff.stdin.write(buf)
            if k % 600 == 0: print(f"frame {k}/{n}", flush=True)
    ff.stdin.close(); ff.wait(); print("done", out)
