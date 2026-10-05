# Renderer core: easing, sprites with paper shadows, textured paper shapes, text, footage, post.
import os, math, functools
import numpy as np
import skia
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..", "work", "launch")
ASSETS = os.path.join(WORK, "assets")
W, H, FPS = 1920, 1080, 60

# ---------- palette ----------
def rgb(h, a=1.0):
    h = h.lstrip("#"); return skia.Color4f(int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)
CREAM, KRAFT, SOIL, DEEP, INK = "#F3EDE2", "#C9A27A", "#6B4A33", "#3A2A20", "#1D1D1F"
TEAL, BLUE, MUTED = "#7FD8D0", "#0A84FF", "#7A6A5C"
TEAL_INK = "#2E8F86"                      # teal dark enough to read as text on cream

# ---------- time ----------
def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def lerp(a, b, t): return a + (b - a) * t
def prog(t, a, b): return clamp((t - a) / (b - a)) if b != a else float(t >= a)
def smooth(x): x = clamp(x); return x * x * x * (x * (6 * x - 15) + 10)
def ease_out(x, p=3): x = clamp(x); return 1 - (1 - x) ** p
def ease_in(x, p=3): x = clamp(x); return x ** p
def ease_io(x): x = clamp(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def expo_out(x): x = clamp(x); return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)
def expo_io(x):
    x = clamp(x)
    if x in (0, 1): return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2
def back_out(x, s=1.7): x = clamp(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def spring(x, freq=2.2, damp=6.0):
    """0 -> 1 with one soft overshoot; x in seconds-ish units of the move."""
    if x <= 0: return 0.0
    return 1 - math.exp(-damp * x) * math.cos(2 * math.pi * freq * x)
def keys(t, ks, ease=smooth):
    """Piecewise interpolation over [(time, value), ...]; values may be tuples."""
    if t <= ks[0][0]: return ks[0][1]
    for (ta, va), (tb, vb) in zip(ks, ks[1:]):
        if t <= tb:
            u = ease((t - ta) / (tb - ta)) if tb > ta else 1.0
            if isinstance(va, tuple): return tuple(lerp(a, b, u) for a, b in zip(va, vb))
            return lerp(va, vb, u)
    return ks[-1][1]

# ---------- images ----------
SAMPLE = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)

@functools.lru_cache(maxsize=None)
def img(name, max_side=None):
    path = name if os.path.isabs(name) else os.path.join(ASSETS, name)
    im = Image.open(path).convert("RGBA")
    if max_side and max(im.size) > max_side:
        k = max_side / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    a = np.asarray(im)
    return skia.Image.fromarray(a, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType).withDefaultMipmaps()

@functools.lru_cache(maxsize=None)
def sprite(name, max_side=1400):
    """A cut-out sprite trimmed to its alpha bounds."""
    path = os.path.join(ASSETS, name)
    im = Image.open(path).convert("RGBA")
    im = im.crop(im.getchannel("A").getbbox())
    if max(im.size) > max_side:
        k = max_side / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return skia.Image.fromarray(np.asarray(im), colorType=skia.kRGBA_8888_ColorType,
                                alphaType=skia.kUnpremul_AlphaType).withDefaultMipmaps()

def shadow_filter(scale=1.0, strength=1.0, dx=7, dy=12, blur=11):
    return skia.ImageFilters.DropShadow(dx * scale, dy * scale, blur * scale, blur * scale,
                                        skia.Color(46, 30, 16, int(95 * strength)))

def draw_image(c, im, cx, cy, w=None, h=None, rot=0.0, alpha=1.0, shadow=0.0, anchor=(0.5, 0.5), flip=False):
    """Draw `im` with its anchor at (cx, cy), w/h in px (one may be None to keep aspect)."""
    iw, ih = im.width(), im.height()
    if w is None and h is None: w, h = iw, ih
    elif w is None: w = h * iw / ih
    elif h is None: h = w * ih / iw
    if alpha <= 0.002 or w < 0.5 or h < 0.5: return
    c.save(); c.translate(cx, cy)
    if rot: c.rotate(rot)
    if flip: c.scale(-1, 1)
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(clamp(alpha))
    if shadow > 0: p.setImageFilter(shadow_filter(max(w, h) / 900, shadow))
    c.drawImageRect(im, skia.Rect.MakeXYWH(-anchor[0] * w, -anchor[1] * h, w, h), SAMPLE, p)
    c.restore()

def draw_bg(c, name, cx, cy, scale, alpha=1.0):
    """A backdrop plate centered at its own pixel (cx, cy) on screen center, at `scale`."""
    im = img(name)
    c.save(); c.translate(W / 2, H / 2); c.scale(scale, scale); c.translate(-cx, -cy)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha)
    c.drawImage(im, 0, 0, SAMPLE, p); c.restore()

# ---------- paper shapes ----------
@functools.lru_cache(maxsize=None)
def texture_shader(name, scale=0.6):
    return img(name).makeShader(skia.TileMode.kMirror, skia.TileMode.kMirror, SAMPLE,
                                skia.Matrix.Scale(scale, scale))

def paper_rect(c, x, y, w, h, tex="tex-cream.png", radius=6, alpha=1.0, shadow=1.0, tint=None, edge=True, rot=0.0):
    """A card of cut cardstock: textured fill, soft contact shadow, a lit top edge and a darker cut edge."""
    if alpha <= 0.002 or w <= 0 or h <= 0: return
    c.save()
    if rot:
        c.translate(x + w / 2, y + h / 2); c.rotate(rot); c.translate(-(x + w / 2), -(y + h / 2))
    r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), radius, radius)
    if shadow > 0:
        sp = skia.Paint(AntiAlias=True, Color=skia.Color(46, 30, 16, int(70 * shadow * alpha)))
        sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 7 + h * 0.02))
        c.drawRRect(r.makeOffset(5 + w * 0.004, 9 + h * 0.01), sp)
    # The texture is drawn through a clip, not set as a shader: skia-python copies a shader into every
    # Paint by serializing it, image and all, which cost ~16 MB per rectangle.
    tx = img(tex); k = 1 / 0.6
    sw, sh = min(w * k, tx.width()), min(h * k, tx.height())
    sx = (x * 7.3) % max(1, tx.width() - sw); sy = (y * 5.1) % max(1, tx.height() - sh)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha)
    if tint is not None:
        p.setColorFilter(skia.ColorFilters.Blend(tint, skia.BlendMode.kModulate))
    c.save(); c.clipRRect(r, True)
    c.drawImageRect(tx, skia.Rect.MakeXYWH(sx, sy, sw, sh), skia.Rect.MakeXYWH(x, y, w, h), SAMPLE, p)
    c.restore()
    if edge:
        e = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.2,
                       Color=skia.Color(255, 255, 255, int(110 * alpha)))
        c.save(); c.clipRect(skia.Rect.MakeXYWH(x - 2, y - 2, w + 4, h * 0.5)); c.drawRRect(r, e); c.restore()
        e.setColor(skia.Color(60, 40, 20, int(60 * alpha)))
        c.save(); c.clipRect(skia.Rect.MakeXYWH(x - 2, y + h * 0.5, w + 4, h * 0.5 + 2)); c.drawRRect(r, e); c.restore()
    c.restore()

def line(c, x0, y0, x1, y1, color=INK, width=2.0, alpha=1.0, cap=True):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, Color4f=rgb(color, alpha))
    if cap: p.setStrokeCap(skia.Paint.kRound_Cap)
    c.drawLine(x0, y0, x1, y1, p)

def circle(c, x, y, r, color, alpha=1.0, stroke=0.0):
    p = skia.Paint(AntiAlias=True, Color4f=rgb(color, alpha))
    if stroke: p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(stroke)
    c.drawCircle(x, y, r, p)

def glow(c, x, y, r, color=TEAL, alpha=0.5):
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeRadial(skia.Point(x, y), r, [rgb(color, alpha).toColor(), rgb(color, 0).toColor()]))
    p.setBlendMode(skia.BlendMode.kScreen)
    c.drawCircle(x, y, r, p)

# ---------- text (Pillow FreeType: SF's variable axes are honoured there) ----------
SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
SERIF = "/System/Library/Fonts/NewYork.ttf"

@functools.lru_cache(maxsize=None)
def _pil_font(kind, size, weight):
    if kind == "mono":
        f = ImageFont.truetype(MONO, size)
        w = max(300, min(900, weight)); f.set_variation_by_axes([w, w])   # both axes span 295..900
        return f
    if kind == "serif":
        f = ImageFont.truetype(SERIF, size)
        try: f.set_variation_by_axes([min(256, max(12, size)), weight, 0])
        except Exception: pass
        return f
    f = ImageFont.truetype(SANS, size)
    f.set_variation_by_axes([100, min(96, max(17, size)), 400, weight])
    return f

@functools.lru_cache(maxsize=4096)
def text_image(s, size, weight=600, color=INK, kind="sans", tracking=None):
    """A string as a tight RGBA skia image (2x supersampled), plus its baseline offset."""
    ss = 2
    f = _pil_font(kind, size * ss, weight)
    if tracking is None: tracking = -0.022 if (kind == "sans" and size >= 40) else 0.0
    tr = tracking * size * ss
    asc, desc = f.getmetrics()
    widths = [f.getlength(ch) for ch in s]
    total = f.getlength(s) + tr * max(0, len(s) - 1)
    pad = int(size * ss * 0.25)
    im = Image.new("RGBA", (int(total + 2 * pad), asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    if tr == 0:
        d.text((pad, pad), s, font=f, fill=col + (255,))
    else:
        x = pad
        for i, ch in enumerate(s):
            # kerning-aware advance: the pair length minus the next char alone
            adv = f.getlength(s[i:i + 2]) - f.getlength(s[i + 1:i + 2]) if i + 1 < len(s) else widths[i]
            d.text((x, pad), ch, font=f, fill=col + (255,)); x += adv + tr
    sk = skia.Image.fromarray(np.asarray(im), colorType=skia.kRGBA_8888_ColorType,
                              alphaType=skia.kUnpremul_AlphaType).withDefaultMipmaps()
    return sk, pad / ss, (pad + asc) / ss, total / ss

def text_width(s, size, weight=600, kind="sans", tracking=None):
    return text_image(s, size, weight, INK, kind, tracking)[3]

def draw_text(c, s, x, y, size, weight=600, color=INK, alpha=1.0, align="left", kind="sans",
              tracking=None, rot=0.0, scale=1.0):
    """Baseline-anchored text. align: left|center|right."""
    if alpha <= 0.002 or not s: return 0
    im, padx, base, wid = text_image(s, size, weight, color, kind, tracking)
    w, h = im.width() / 2, im.height() / 2
    ox = {"left": 0, "center": wid / 2, "right": wid}[align]
    c.save(); c.translate(x, y)
    if rot: c.rotate(rot)
    if scale != 1: c.scale(scale, scale)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(clamp(alpha))
    c.drawImageRect(im, skia.Rect.MakeXYWH(-ox - padx, -base, w, h), SAMPLE, p)
    c.restore()
    return wid

def reveal_words(c, s, x, y, size, t, t0, stagger=0.07, dur=0.55, weight=600, color=INK, rise=0.45, kind="sans",
                 alpha=1.0, align="left", tracking=None):
    """Words rise into place from behind a mask line, one after another (the Swiss-poster reveal)."""
    words = s.split(" ")
    sp = text_width(" ", size, weight, kind, tracking)
    total = sum(text_width(w_, size, weight, kind, tracking) for w_ in words) + sp * (len(words) - 1)
    cx = x - {"left": 0, "center": total / 2, "right": total}[align]
    for i, w_ in enumerate(words):
        u = expo_out(prog(t, t0 + i * stagger, t0 + i * stagger + dur))
        ww = text_width(w_, size, weight, kind, tracking)
        if u > 0:
            c.save()
            c.clipRect(skia.Rect.MakeLTRB(cx - size * 0.2, y - size * 1.2, cx + ww + size * 0.3, y + size * 0.32))
            draw_text(c, w_, cx, y + (1 - u) * size * rise * 2.2, size, weight, color, alpha * clamp(u * 1.4), "left", kind, tracking)
            c.restore()
        cx += ww + sp
    return total

# ---------- footage ----------
class Footage:
    """Frames of a recorded take, decoded once to a raw file and memory-mapped."""
    def __init__(self, name, t0, t1, w=1280, h=800):
        self.w, self.h = w, h
        src = os.path.join(WORK, "takes", name + ".mov")
        raw = os.path.join(WORK, "takes", f"{name}-{t0:.2f}-{t1:.2f}.rgba")
        if not os.path.exists(raw):
            import subprocess
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-to", str(t1), "-i", src,
                            "-vf", f"fps=60,scale={w}:{h}", "-pix_fmt", "rgba", "-f", "rawvideo", raw], check=True)
        self.data = np.memmap(raw, np.uint8, "r").reshape(-1, h, w, 4)
        self.n = len(self.data)
        self.alpha = self._corner_alpha(np.asarray(self.data[self.n // 2]))
    @staticmethod
    def _corner_alpha(f):
        """ScreenCaptureKit fills the window's rounded corners with black. Flood each corner over the
        near-black pixels that touch it, make them transparent, and feather the edge by a pixel."""
        from PIL import Image as _I, ImageFilter as _F
        h, w = f.shape[:2]
        dark = f[..., :3].astype(np.int32).sum(2) < 60
        a = np.full((h, w), 255, np.uint8)
        R = 40
        for ys, xs, y0, x0 in [(range(R), range(R), 0, 0), (range(R), range(w - 1, w - R - 1, -1), 0, w - 1),
                               (range(h - 1, h - R - 1, -1), range(R), h - 1, 0),
                               (range(h - 1, h - R - 1, -1), range(w - 1, w - R - 1, -1), h - 1, w - 1)]:
            stack, seen = [(y0, x0)], set()
            while stack:
                y, x = stack.pop()
                if (y, x) in seen or not (0 <= y < h and 0 <= x < w) or abs(y - y0) >= R or abs(x - x0) >= R or not dark[y, x]:
                    continue
                seen.add((y, x)); a[y, x] = 0
                stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
        # The flood is only a measurement: a corner outside a quarter circle of radius r has area
        # r^2 (1 - pi/4). The mask itself is an analytic rounded rectangle, antialiased, inset one
        # pixel to drop the dark hairline the capture leaves on the window's edge.
        n = (a == 0).sum() / 4
        r = math.sqrt(n / (1 - math.pi / 4)) if n > 0 else 0
        surf = skia.Surface(w * 4, h * 4)
        cc = surf.getCanvas(); cc.clear(skia.Color(0, 0, 0, 0))
        cc.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(4, 4, w * 4 - 8, h * 4 - 8), r * 4, r * 4),
                     skia.Paint(AntiAlias=True, Color=skia.Color(255, 255, 255, 255)))
        big = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., 3]
        return big.reshape(h, 4, w, 4).mean(axis=(1, 3)).astype(np.uint8)

    def frame(self, t):
        i = int(clamp(t * 60, 0, self.n - 1))
        f = np.array(self.data[i]); f[..., 3] = self.alpha
        return skia.Image.fromarray(f, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)

def draw_window(c, im, cx, cy, scale, alpha=1.0, radius=12, shadow=1.0):
    """An app window: the frame with rounded corners and a soft macOS shadow."""
    w, h = im.width() * scale, im.height() * scale
    r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(cx - w / 2, cy - h / 2, w, h), radius * scale, radius * scale)
    if shadow > 0:
        sp = skia.Paint(AntiAlias=True, Color=skia.Color(40, 28, 18, int(80 * shadow * alpha)))
        sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 30 * scale))
        c.drawRRect(r.makeOffset(0, 22 * scale), sp)
    c.save(); c.clipRRect(r, True)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha)
    c.drawImageRect(im, skia.Rect.MakeXYWH(cx - w / 2, cy - h / 2, w, h), SAMPLE, p)
    c.restore()

# ---------- post ----------
_grain = None
def post(frame_f, t):
    """Film grain (animated, luminance-weighted) and a soft vignette, on a float HxWx3 frame in 0..255."""
    global _grain
    if _grain is None:
        rng = np.random.default_rng(7)
        _grain = rng.normal(0, 1, (8, H, W)).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        post.vig = (1 - 0.10 * np.clip(d - 0.15, 0, None)).astype(np.float32)[..., None]
    g = _grain[int(t * 24) % 8][..., None]
    lum = frame_f.mean(axis=2, keepdims=True) / 255
    out = frame_f * post.vig + g * (2.2 + 2.0 * (1 - np.abs(lum - 0.5) * 2))
    return np.clip(out, 0, 255)
