# Omni intro renderer: recorded app takes + eased camera + captions, on the site's light wash.
# usage: render.py <out.mp4> [--preview "t1,t2,..."]  (preview writes stills instead of a video)
import sys, os, subprocess, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

V = os.environ.get("VIDEO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H, FPS = 1920, 1080, 60
BEAT = 60 / 116.0
SRC_W, SRC_H = 1800, 960
FONT = "/System/Library/Fonts/SFNS.ttf"

def b(n): return n * BEAT

# ---- timeline (seconds), on the music's beat grid ----
T_INTRO, T_A, T_B, T_C, T_D, T_OUT = 0, b(6), b(20), b(30), b(38), b(52)
T_END = b(58) + 1.0
XF = 0.42          # crossfade between scenes

# scene: take, source start, list of (scene_t, source_t) cut points for B
SCENES = [
    dict(name="A", t0=T_A, t1=T_B, take="A", src=2.2, caption="Search by meaning."),
    dict(name="B", t0=T_B, t1=T_C, take="B", src=1.9, caption="Any language.",
         cut=(2.8, "B", 9.35)),
    dict(name="C", t0=T_C, t1=T_D, take="C", src=3.0, caption="Find similar."),
    dict(name="D", t0=T_D, t1=T_OUT, take="D", src=5.0, caption="Reads every page."),
]

# ---- easing ----
def clamp(x, a=0.0, b_=1.0): return max(a, min(b_, x))
def smooth(x):  # smootherstep: zero velocity and acceleration at both ends - no zig-zag
    x = clamp(x); return x * x * x * (x * (6 * x - 15) + 10)
def out_expo(x): x = clamp(x); return 1 if x >= 1 else 1 - 2 ** (-10 * x)
def out_cubic(x): x = clamp(x); return 1 - (1 - x) ** 3
def lerp(a, b_, t): return a + (b_ - a) * t

# ---- camera: (focus in source px, focus on screen, scale); keyframes eased with smootherstep ----
BASE_S = 0.9
BASE = ((SRC_W / 2, SRC_H / 2), (W / 2, H / 2 + 50), BASE_S)   # every scene starts and ends here
SEARCH = (1640, 36)          # search field in the source frame

def cam_track(keys, t):
    """keys: list of (time, cam). Between keys: smootherstep. Before first/after last: hold."""
    if t <= keys[0][0]: return keys[0][1]
    for (ta, ca), (tb, cb) in zip(keys, keys[1:]):
        if t <= tb:
            u = smooth((t - ta) / (tb - ta))
            return ((lerp(ca[0][0], cb[0][0], u), lerp(ca[0][1], cb[0][1], u)),
                    (lerp(ca[1][0], cb[1][0], u), lerp(ca[1][1], cb[1][1], u)),
                    lerp(ca[2], cb[2], u))
    return keys[-1][1]

ZOOM_SEARCH = (SEARCH, (W * 0.76, 150), 1.3)   # the field high and right: the window fills the frame
CAMS = {   # (scene time, framing); the caption owns the first ~2.3 s, so moves start after it
    "A": [(0.0, BASE), (0.9, BASE), (1.9, ZOOM_SEARCH), (3.0, ZOOM_SEARCH),
          (4.6, ((SRC_W / 2, SRC_H * 0.45), (W / 2, H / 2), 1.0)), (5.9, ((SRC_W / 2, SRC_H * 0.45), (W / 2, H / 2), 1.03)),
          (7.24, BASE)],
    "B": [(0.0, BASE), (0.35, ZOOM_SEARCH), (1.4, ZOOM_SEARCH), (2.5, BASE), (2.8, BASE),
          (3.3, ZOOM_SEARCH), (4.0, ZOOM_SEARCH), (5.17, BASE)],
    "C": [(0.0, BASE), (2.2, BASE), (3.4, ((SRC_W * 0.58, SRC_H * 0.4), (W / 2, H / 2), 1.1)), (4.14, BASE)],
    "D": [(0.0, BASE), (2.2, BASE), (4.0, ((1260, 330), (W * 0.52, H * 0.46), 1.22)),
          (6.3, ((1260, 470), (W * 0.52, H * 0.48), 1.25)), (7.24, BASE)],
}

# ---- assets ----
def font(size, weight=600, opsz=None):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([100, opsz or min(96, max(17, size)), 400, weight])
    return f

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

WASH_KEYS = None
def wash(t):
    # 8 precomputed washes, interpolated: the drift is slow, so this is indistinguishable and cheap
    k = (t / T_END) * (len(WASH_KEYS) - 1)
    i = min(int(k), len(WASH_KEYS) - 2); u = k - i
    return WASH_KEYS[i] * (1 - u) + WASH_KEYS[i + 1] * u

MOLE = None

def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255); return m

# source-size window alpha (rounded corners) and shadow, transformed with the window each frame
WIN_R = 26
WIN_ALPHA = None
SHADOW = None; SH_PAD = 160

def traffic_lights(frame):
    """macOS draws its recording badge where the traffic lights are; paint the lights back."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    bg = tuple(int(v) for v in frame[27, 100])
    d.rounded_rectangle([8, 10, 92, 44], 12, fill=bg)
    for i, col in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx, cy = 28 + i * 20, 27
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
    return img

class Takes:
    def __init__(self):
        self.mm = {}
        for t in "ABCD":
            meta = json.load(open(os.path.join(V, "takes", f"{t}.json")))
            self.mm[t] = (np.memmap(os.path.join(V, "takes", f"{t}.rgb"), np.uint8, "r",
                                    shape=(meta["n"], SRC_H, SRC_W, 3)), meta["n"])
    def frame(self, take, s):
        arr, n = self.mm[take]
        i = int(round(clamp(s * FPS, 0, n - 1)))
        return traffic_lights(np.array(arr[i]))

TAKES = None

def affine_for(cam):
    (fx, fy), (px, py), s = cam
    # screen = P + s * (src - F)  ->  inverse for PIL: src = (screen - P)/s + F
    return (1 / s, 0, fx - px / s, 0, 1 / s, fy - py / s)

def place_window(canvas, win_rgba, cam, alpha=1.0, blur=0.0):
    a = affine_for(cam)
    layer = win_rgba.transform((W, H), Image.AFFINE, a, resample=Image.BICUBIC)
    # shadow: same transform, offset down, pre-blurred at source size
    (fx, fy), (px, py), s = cam
    sa = (1 / s, 0, fx - px / s + SH_PAD, 0, 1 / s, fy - (py - 26 * s) / s + SH_PAD)
    sh = SHADOW.transform((W, H), Image.AFFINE, sa, resample=Image.BILINEAR)
    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
        sh = sh.point(lambda v: int(v * alpha))
    if blur > 0.3:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    shadow_rgba = Image.new("RGBA", (W, H), (30, 30, 60, 0)); shadow_rgba.putalpha(sh)
    canvas.alpha_composite(shadow_rgba); canvas.alpha_composite(layer)

def window_image(take, s):
    img = TAKES.frame(take, s).convert("RGBA"); img.putalpha(WIN_ALPHA); return img

def draw_text(canvas, text, size, weight, center, color, alpha, rise=0.0, blur=0.0, tracking=-0.02):
    f = font(size, weight)
    # tracked text, drawn glyph by glyph for display-size tightening
    widths = [f.getlength(ch) for ch in text]
    total = sum(widths) + tracking * size * (len(text) - 1)
    pad = int(size * 0.6)
    layer = Image.new("RGBA", (int(total) + 2 * pad, int(size * 1.6) + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer); x = pad
    for ch, w_ in zip(text, widths):
        d.text((x, pad), ch, font=f, fill=color + (int(255 * alpha),)); x += w_ + tracking * size
    x0, y0, x1, y1 = layer.getchannel("A").getbbox() or (0, 0, layer.width, layer.height)
    if blur > 0.3: layer = layer.filter(ImageFilter.GaussianBlur(blur))
    cx, cy = center   # the GLYPHS' center, not the layer's: a layer box leaves text sitting high
    canvas.alpha_composite(layer, (int(cx - (x0 + x1) / 2), int(cy - (y0 + y1) / 2 + rise)))

def caption(canvas, text, t_local, dur):
    # in: rise + sharpen, out: fade; the text sits above the window
    a_in = out_cubic(t_local / 0.55)
    a_out = 1 - smooth((t_local - 1.85) / 0.4)
    a = clamp(min(a_in, a_out))
    if a <= 0: return
    draw_text(canvas, text, 60, 640, (W / 2, 84), (29, 29, 31), a, rise=(1 - a_in) * 18, blur=(1 - a_in) * 8)

INTRO_MOLE_H = 330
def mole_layer(scale, alpha):
    h = int(INTRO_MOLE_H * scale); w = int(MOLE.width * h / MOLE.height)
    m = MOLE.resize((max(1, w), max(1, h)), Image.LANCZOS)
    if alpha < 1: m.putalpha(m.getchannel("A").point(lambda v: int(v * alpha)))
    return m

def render_frame(fi):
    t = fi / FPS
    canvas = Image.fromarray(wash(t).astype(np.uint8), "RGB").convert("RGBA")

    # ---- intro: the mole, the name, then the mole opens into the window ----
    if t < T_A + 0.2:
        grow = out_expo(t / 0.9)
        morph = smooth((t - 1.7) / (T_A - 1.7 + 0.2))
        m_alpha = clamp(grow) * (1 - morph)
        if m_alpha > 0:
            m = mole_layer(0.86 + 0.14 * grow - 0.5 * morph, m_alpha)
            canvas.alpha_composite(m, (int(W / 2 - m.width / 2), int(H * 0.42 - m.height / 2 - 40 * morph)))
        name_a = out_cubic((t - 0.45) / 0.6) * (1 - smooth((t - 1.6) / 0.5))
        if name_a > 0:
            draw_text(canvas, "Omni", 104, 700, (W / 2, H * 0.42 + INTRO_MOLE_H / 2 + 96), (29, 29, 31), name_a,
                      rise=(1 - out_cubic((t - 0.45) / 0.6)) * 22, blur=(1 - out_cubic((t - 0.45) / 0.6)) * 10)
        if morph > 0:
            # the window grows out of the mole: small, centered on it, rounding off into the base frame
            s = lerp(0.12, BASE_S, morph)
            py = lerp(H * 0.44, BASE[1][1], morph)
            cam = ((SRC_W / 2, SRC_H / 2), (W / 2, py), s)
            place_window(canvas, window_image("A", SCENES[0]["src"]), cam, alpha=clamp(morph * 1.6))

    # ---- scenes ----
    for i, sc in enumerate(SCENES):
        t0, t1 = sc["t0"], sc["t1"]
        if not (t0 - XF <= t < t1): continue
        tl = t - t0
        take, src = sc["take"], sc["src"] + max(tl, 0)
        if "cut" in sc and tl >= sc["cut"][0]:
            take, src = sc["cut"][1], sc["cut"][2] + (tl - sc["cut"][0])
        cam = cam_track(CAMS[sc["name"]], max(tl, 0))
        alpha, blur, sscale = 1.0, 0.0, 1.0
        # A TRANSITION IS A MORPH OF CONTENT IN ONE FRAME: both scenes sit at BASE across the
        # boundary, so only what is inside the window changes, through a short blur dip and a
        # breath of scale shared by both layers.
        if tl < 0 and i > 0:
            u = (tl + XF) / XF; alpha = smooth(u); blur = math.sin(math.pi * u) * 5
            sscale = 1 + 0.025 * math.sin(math.pi * u)
        if i == 0 and t < T_A: continue                # the intro morph owns the frame before A
        (f_, p_, s_) = cam
        place_window(canvas, window_image(take, src), (f_, p_, s_ * sscale), alpha=alpha, blur=blur)
        # the B cut: a quick crossfade at the cut point so the take change reads as a morph
        if "cut" in sc and 0 <= tl - sc["cut"][0] < 0.25:
            u = (tl - sc["cut"][0]) / 0.25
            place_window(canvas, window_image(sc["take"], sc["src"] + tl), cam, alpha=1 - smooth(u))
        caption(canvas, sc["caption"], tl, t1 - t0)

    # ---- outro: the window folds back into the mole; name and address ----
    if t >= T_OUT - XF:
        tl = t - T_OUT
        fold = smooth(tl / 1.1)
        if fold < 1:
            last = SCENES[-1]
            cam0 = cam_track(CAMS["D"], T_OUT - last["t0"])
            s = lerp(cam0[2], 0.1, fold)
            f_ = (lerp(cam0[0][0], SRC_W / 2, fold), lerp(cam0[0][1], SRC_H / 2, fold))
            p_ = (lerp(cam0[1][0], W / 2, fold), lerp(cam0[1][1], H * 0.37, fold))
            if tl >= 0:
                place_window(canvas, window_image("D", last["src"] + (T_OUT - last["t0"]) + tl), (f_, p_, s),
                             alpha=1 - smooth((tl - 0.5) / 0.6))
        grow = out_expo((tl - 0.6) / 0.9)
        if grow > 0:
            outer, canvas = canvas, Image.new("RGBA", (W, H), (0, 0, 0, 0))
            m = mole_layer(0.62 + 0.14 * grow, clamp(grow))
            canvas.alpha_composite(m, (int(W / 2 - m.width / 2), int(H * 0.37 - m.height / 2)))
            ta = out_cubic((tl - 0.95) / 0.6)
            draw_text(canvas, "Omni", 96, 700, (W / 2, H * 0.37 + 196), (29, 29, 31), ta,
                      rise=(1 - ta) * 18, blur=(1 - ta) * 8)
            ua = out_cubic((tl - 1.35) / 0.6)
            draw_text(canvas, "Search your Mac by meaning.", 38, 500, (W / 2, H * 0.37 + 284), (110, 110, 118), ua,
                      rise=(1 - ua) * 12, blur=(1 - ua) * 6, tracking=-0.01)
            va = out_cubic((tl - 1.75) / 0.6)
            draw_text(canvas, "hanxiao.io/omni", 32, 600, (W / 2, H * 0.37 + 346), (0, 113, 227), va,
                      rise=(1 - va) * 10, blur=(1 - va) * 5, tracking=0.0)
            # a slow push-in, so the last seconds are not a still frame
            k = 1 + 0.035 * smooth((tl - 0.6) / (T_END - T_OUT - 0.6))
            cx, cy = W / 2, H * 0.45
            canvas = canvas.transform((W, H), Image.AFFINE, (1 / k, 0, cx - cx / k, 0, 1 / k, cy - cy / k),
                                      resample=Image.BICUBIC)
            outer.alpha_composite(canvas); canvas = outer
    # fade the very end to the wash
    return np.asarray(canvas.convert("RGB")).tobytes()

def init():
    global TAKES, WIN_ALPHA, SHADOW, MOLE, WASH_KEYS
    TAKES = Takes()
    WIN_ALPHA = rounded_mask(SRC_W, SRC_H, WIN_R)
    sh = Image.new("L", (SRC_W + 2 * SH_PAD, SRC_H + 2 * SH_PAD), 0)
    ImageDraw.Draw(sh).rounded_rectangle([SH_PAD, SH_PAD, SH_PAD + SRC_W, SH_PAD + SRC_H], WIN_R, fill=70)
    SHADOW = sh.resize((sh.width // 4, sh.height // 4)).filter(ImageFilter.GaussianBlur(14)).resize(sh.size, Image.BILINEAR)
    src = Image.open(os.path.join(REPO, "App/AppIcon.icon/Assets/foreground.png")).convert("RGBA")
    MOLE = src.crop(src.getbbox())
    WASH_KEYS = np.load(os.path.join(V, "wash.npy"))

if __name__ == "__main__":
    out = sys.argv[1]
    if not os.path.exists(os.path.join(V, "wash.npy")):
        np.save(os.path.join(V, "wash.npy"), np.stack([make_wash(T_END * k / 7) for k in range(8)]).astype(np.float32))
    n = int(T_END * FPS)
    if "--preview" in sys.argv:
        init()
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            fi = int(float(s) * FPS)
            Image.frombytes("RGB", (W, H), render_frame(fi)).save(f"{out}-{float(s):05.2f}.png")
        sys.exit(0)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", os.path.join(V, "music.wav"),
                           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-movflags", "+faststart",
                           "-af", f"afade=t=out:st={T_END - 1.8:.2f}:d=1.8", "-t", f"{T_END:.3f}",
                           "-c:a", "aac", "-b:a", "192k", out], stdin=subprocess.PIPE)
    with Pool(16, initializer=init) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            ff.stdin.write(buf)
            if k % 300 == 0: print(f"frame {k}/{n}", flush=True)
    ff.stdin.close(); ff.wait()
    print("done", out)
