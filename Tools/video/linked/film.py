# The Linked Macs film: nine shots on the score's clock and the narration's word timings
# (storyboard.md). The launch film's world and drawing kit, loaded from ../launch.
import importlib.util, json, math, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.path.join(HERE, "..", "launch")
sys.path.insert(0, LAUNCH)
import skia
from core import *
_spec = importlib.util.spec_from_file_location("launchfilm", os.path.join(LAUNCH, "film.py"))
L = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L)
headline, caption, chip, ribbon = L.headline, L.caption, L.chip, L.ribbon
sys.path.insert(0, HERE)
import clock as T

LW = os.path.join(HERE, "..", "work", "linked")
A_ = lambda n: os.path.join(LW, "assets", n)           # this film's own sprites and plates
TK = lambda n: os.path.join(LW, "takes", n)

WORDS = json.load(open(os.path.join(LW, "words.json")))
def word(line, i): return WORDS[line][i][1]
def at(line, text, nth=0):
    hits = [w for w in WORDS[line] if w[0].strip(".,?!-").lower() == text.lower()]
    return hits[nth][1]

KRAFT_INK = "#9A6E44"

# ---------------------------------------------------------------- the two-burrow plate
PLATE, PW = A_("bg-two.png"), 2752
B_HOLE, A_HOLE = (667, 1160), (2108, 1160)     # MacBook Pro (left) and Mac Studio (right), plate px
B_DESK, A_DESK = (680, 430), (2050, 445)
def cam(fx, fy, zoom):
    s = W / PW * zoom
    return s, (lambda x, y: (W / 2 + (x - fx) * s, H / 2 + (y - fy) * s))

def tunnel_path(P, u=1.0):
    (x0, y0), (x1, y1) = P(A_HOLE[0] - 250, A_HOLE[1] + 40), P(B_HOLE[0] + 250, B_HOLE[1] + 40)
    p = skia.Path(); p.moveTo(x0, y0)
    p.cubicTo(x0 - (x0 - x1) * 0.3, y0 + 70, x1 + (x0 - x1) * 0.3, y1 + 70, x1, y1)
    if u >= 0.999: return p
    m = skia.PathMeasure(p, False); seg = skia.Path(); m.getSegment(0, m.getLength() * max(0.001, u), seg, True)
    return seg

def along(P, u):
    """A point on the tunnel, 0 at Mac Studio, 1 at MacBook Pro."""
    p = tunnel_path(P); m = skia.PathMeasure(p, False)
    pos, _ = m.getPosTan(m.getLength() * clamp(u))
    return pos.x(), pos.y()

def moles(c, P, s, t, b_kind="mole-listen-cut.png", a_kind="mole-dig-cut.png", b_alpha=1.0, b_pop=None):
    for (hx, hy), kind, al in [(B_HOLE, b_kind, b_alpha), (A_HOLE, a_kind, 1.0)]:
        x, y = P(hx, hy + 190)
        k, sq = 1.0, 1.0
        if b_pop is not None and (hx, hy) == B_HOLE:
            d = t - b_pop; k = spring(max(0, d) * 0.9, 1.3, 4.5); sq = 1 + 0.12 * math.exp(-5 * max(0, d)) * math.sin(max(0, d) * 18)
        c.save(); c.translate(x, y); c.scale(1 / sq, sq)
        path = A_(kind) if kind.startswith("mole-sleep") else kind
        h = 300 * s / 0.62 if kind.startswith("mole-sleep") else 400 * s / 0.62
        draw_image(c, sprite(path, 1000), 0, (1 - k) * 200 * s, h=h * 0.62, anchor=(0.5, 0.95), alpha=al, shadow=0.6)
        c.restore()

def tunnel(c, P, s, t, t0, t1, thread=1.0):
    u = ease_io(prog(t, t0, t1))
    if u <= 0: return
    ribbon(c, tunnel_path(P, u), 52 * s / 0.6)
    if thread < 1:   # the thread dims while a Mac sleeps
        c.save(); c.drawPath(tunnel_path(P, u), skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                     StrokeWidth=6, Color=skia.Color(58, 42, 32, int(255 * (1 - thread))))); c.restore()

def courier(c, P, s, t, t0, dur, sprite_name, back=False, h=110, label=None):
    """Something carried along the tunnel: Mac Studio -> MacBook Pro, or back."""
    u = ease_io(prog(t, t0, t0 + dur))
    if not (0 < u < 1): return
    x, y = along(P, 1 - u if back else u)
    glow(c, x, y, 120 * s / 0.6, TEAL, 0.35)
    if sprite_name:
        draw_image(c, sprite(sprite_name, 500), x, y - 10, h=h * s / 0.6, rot=math.sin(t * 6) * 6, shadow=0.5)
    if label:
        w = text_width(label, 26, 600, "mono") + 36
        paper_rect(c, x - w / 2, y - 26, w, 48, "tex-teal.png", radius=24, shadow=0.6)
        draw_text(c, label, x, y + 8, 26, 600, INK, 1, "center", "mono")

# ---------------------------------------------------------------- 1. files on more than one Mac
S1_END = 4.45
def s_many(c, t):
    leave = expo_io(prog(t, 4.0, 4.9))
    c.save(); c.translate(0, leave * H * 1.05)
    zoom = 1.04 + 0.02 * prog(t, 0, 4.5)
    s, P = cam(1376, 760, zoom)
    L.plate(c, PLATE, 1376, 760, zoom, src_w=PW)
    rnd = random.Random(2)
    kinds = ["doc-cut.png", "photo-cut.png", "folder-cut.png"]
    for i in range(12):
        side = i % 2
        t0 = word("many", 1) + i * 0.13 + side * 0.07
        d = t - t0
        if d < 0: continue
        dx, dy = (B_DESK if side == 0 else A_DESK)
        tx, ty = P(dx + rnd.uniform(-170, 170), dy - 30 + rnd.uniform(-25, 15))
        drop = spring(d * 1.1, 1.5, 5.0)
        draw_image(c, sprite(kinds[i % 3], 500), tx, ty - (1 - drop) * 520, h=62 * s / 0.6,
                   rot=rnd.uniform(-25, 25) * (1 - 0.4 * drop), shadow=0.5)
    moles(c, P, s, t)
    headline(c, "Your files live on more than one Mac.", W / 2, 84, 56, t, word("many", 0) - 0.05, INK, align="center")
    bx, by = P(B_DESK[0], B_DESK[1] + 180); ax, ay = P(A_DESK[0], A_DESK[1] + 180)
    chip(c, "MacBook Pro", bx - 110, by, t, at("many", "one") - 0.1, 28, "tex-cream.png", INK)
    chip(c, "Mac Studio", ax - 100, ay, t, at("many", "Mac") - 0.05, 28, "tex-cream.png", INK)
    c.restore()

# ---------------------------------------------------------------- 2. link: the real sheets
LAP_SCREEN = (171, 72, 1110, 660)       # laptop-open-cut, trimmed px at max side 1400
DESK_SCREEN = (54, 51, 791, 658)        # desktop-cut, trimmed px
B_FOCUS = (584, 394, 600)               # q1's sheet: centre and the width of capture shown
A_FOCUS = (700, 450, 640)               # n2 / n3's sheet
FIELD = (394, 374, 773, 415)            # q1's code field, capture px; placeholder centred at x 584, baseline 405

@functools.lru_cache(maxsize=None)
def capture(name):
    return img(TK(name))

def screen(c, im, focus, rect_xywh, alpha=1.0):
    """Draw the part of capture `im` around `focus` = (cx, cy, width) into a screen rectangle: the
    sheet, close enough to read, with the window around it."""
    x, y, w, h = rect_xywh
    fcx, fcy, cw = focus; ch = cw * h / w
    src = skia.Rect.MakeXYWH(fcx - cw / 2, fcy - ch / 2, cw, ch)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha)
    c.save(); c.clipRect(skia.Rect.MakeXYWH(x, y, w, h), True)
    c.drawImageRect(im, src, skia.Rect.MakeXYWH(x, y, w, h), SAMPLE, p); c.restore()
    return src

_rounded = {}
def typed(s, size):
    key = (s, size)
    if key not in _rounded:
        from PIL import Image as _I, ImageDraw as _D, ImageFont as _F
        f = _F.truetype("/System/Library/Fonts/SFNSRounded.ttf", size * 2)
        try: f.set_variation_by_axes([600])
        except Exception: pass
        wd = int(f.getlength(s)) + 8; asc, desc = f.getmetrics()
        im = _I.new("RGBA", (wd, asc + desc + 8), (0, 0, 0, 0))
        _D.Draw(im).text((4, 4), s, font=f, fill=(29, 29, 31, 255))
        _rounded[key] = (skia.Image.fromarray(np.asarray(im), colorType=skia.kRGBA_8888_ColorType,
                                              alphaType=skia.kUnpremul_AlphaType), (asc + 4) / 2, wd / 2)
    return _rounded[key]

CODE = "175554"
S2_0 = 4.0
def s_link(c, t):
    enter = expo_io(prog(t, S2_0, S2_0 + 0.9))
    leave = expo_io(prog(t, 12.2, 13.1))
    c.save(); c.translate(0, (1 - enter) * -H * 1.05 + leave * -H * 1.05)
    L.plate(c, "bg-paper.png", 2752, 1536, 1.05)
    # MacBook Pro (laptop, left) and Mac Studio (display, right)
    lap = sprite("laptop-open-cut.png", 1400); dsk = sprite(A_("desktop-cut.png"), 1400)
    rise_b = spring(max(0, t - word("link", 0) - 0.1) * 0.9, 1.2, 4.6)
    rise_a = spring(max(0, t - word("link", 0) - 0.35) * 0.9, 1.2, 4.6)
    lw = 900; ls = lw / lap.width()
    lx, ly = 560, 620 + (1 - rise_b) * 700
    draw_image(c, lap, lx, ly, w=lw, shadow=0.7)
    lsx, lsy = lx - lw / 2 + LAP_SCREEN[0] * ls, ly - lap.height() * ls / 2 + LAP_SCREEN[1] * ls
    lsw, lsh = (LAP_SCREEN[2] - LAP_SCREEN[0]) * ls, (LAP_SCREEN[3] - LAP_SCREEN[1]) * ls
    dh = 680; dsc = dh / dsk.height(); dw = dsk.width() * dsc
    dx, dy = 1440, 580 + (1 - rise_a) * 700
    draw_image(c, dsk, dx, dy, h=dh, shadow=0.7)
    dsx, dsy = dx - dw / 2 + DESK_SCREEN[0] * dsc, dy - dh / 2 + DESK_SCREEN[1] * dsc
    dsw, dsh = (DESK_SCREEN[2] - DESK_SCREEN[0]) * dsc, (DESK_SCREEN[3] - DESK_SCREEN[1]) * dsc
    on = prog(t, at("link", "Open") - 0.2, at("link", "Open") + 0.4)
    linked = at("link", "other") + 0.35
    if on > 0:
        # Mac Studio shows the code; once linked, its confirmation
        a_img = capture("n3-1826.png" if t > linked else "n2-1826.png")
        screen(c, a_img, A_FOCUS, (dsx, dsy, dsw, dsh), alpha=on)
        # MacBook Pro asks for it; the digits are typed in as they are spoken
        src = screen(c, capture("q1-1818.png"), B_FOCUS, (lsx, lsy, lsw, lsh), alpha=on)
        k = lsw / src.width()
        X = lambda px: lsx + (px - src.x()) * k; Y = lambda py: lsy + (py - src.y()) * k
        t_type = at("link", "six")
        n = int(clamp((t - t_type) / 1.6) * 6.999) if t > t_type else 0
        if n > 0:
            paper = skia.Paint(AntiAlias=True, Color=skia.Color(255, 255, 255))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(X(FIELD[0] + 2), Y(FIELD[1] + 2), X(FIELD[2] - 2), Y(FIELD[3] - 2)), 4 * k, 4 * k), paper)
            s_ = CODE[:n][:3] + (" " + CODE[3:n] if n > 3 else "")
            im, base, wd = typed(s_, 28)
            iw, ih = im.width() / 2 * k, im.height() / 2 * k
            c.drawImageRect(im, skia.Rect.MakeXYWH(X(584) - iw / 2, Y(405) - base * k, iw, ih), SAMPLE, skia.Paint(AntiAlias=True))
        if t > linked:   # linked: the request is gone, a quiet check over the screen
            u = expo_out(prog(t, linked, linked + 0.5))
            c.drawRect(skia.Rect.MakeXYWH(lsx, lsy, lsw, lsh), skia.Paint(Color=skia.Color(243, 237, 226, int(150 * u))))
            chip(c, "Linked", lsx + lsw / 2 - 70, lsy + lsh / 2 + 14, t, linked, 34, "tex-teal.png", INK, "sans", 650)
    # the six digits on Mac Studio's screen, ringed while they are spoken
    rt = at("link", "six") - 0.15
    ra = prog(t, rt, rt + 0.3) * (1 - prog(t, linked - 0.4, linked))
    if ra > 0 and on > 0:
        cx, cy = dsx + dsw * 0.5, dsy + dsh * 0.47
        pp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=5, Color4f=rgb(TEAL_INK, ra))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(cx - dsw * 0.27, cy - dsh * 0.1, dsw * 0.54, dsh * 0.2), 16, 16), pp)
    chip(c, "MacBook Pro", lx - 105, ly + 340, t, word("link", 0) + 0.2, 28, "tex-cream.png", INK)
    chip(c, "Mac Studio", dx - 95, dy + 375, t, word("link", 0) + 0.45, 28, "tex-cream.png", INK)
    headline(c, "Open Link a Mac on both.", 120, 150, 66, t, at("link", "Open") - 0.05, INK)
    caption(c, "Type the six digits one shows into the other.", 120, 214, t, at("link", "type") - 0.1, 36, "#5E4B3A", weight=550)
    c.restore()

# ---------------------------------------------------------------- 3. proof: the tunnel, no server
UNDER = (1388, 880, 1.18)    # fy keeps the frame on the plate (its bottom edge is 1536)               # the underground framing: both burrows, the tunnel between
def s_proof(c, t):
    fx, fy, zoom = UNDER
    s, P = cam(fx, fy, zoom)
    L.plate(c, PLATE, fx, fy, zoom, src_w=PW)
    t_dig = word("proof", 1) - 0.1
    tunnel(c, P, s, t, t_dig, t_dig + 1.6)
    moles(c, P, s, t)
    # the two Macs talk: pulses both ways
    for k in range(6):
        t0 = at("proof", "talking") + k * 0.33
        courier(c, P, s, t, t0, 0.9, None, back=k % 2 == 1)
    # the seal: nothing in between
    st = at("proof", "nothing") + 0.1
    d = t - st
    if d > 0:
        mx, my = along(P, 0.5)
        kk = 1 + 1.4 * math.exp(-9 * d)
        draw_image(c, sprite("seal-cut.png", 700), mx, my - 12, h=150 * kk, rot=-10, alpha=clamp(d * 6), shadow=0.8)
        chip(c, "TLS 1.3 · keys pinned at pairing", mx - 290, my + 140, t, st + 0.35, 32, "tex-ink.png", CREAM,
             alpha=1 - prog(t, at("proof", "No") - 0.4, at("proof", "No")))
    # no server: the cloud drifts in and is sent away
    ct = at("proof", "server") - 0.6
    if t > ct - 1.0:
        u_in = ease_out(prog(t, ct - 1.0, ct))
        u_out = ease_in(prog(t, at("proof", "involved") + 0.1, at("proof", "involved") + 1.1), 2.4)
        cx = lerp(W + 420, W / 2 + 120, u_in); cy = 250 - u_out * 600
        draw_image(c, sprite("cloud-cut.png", 1200), cx, cy, w=620, alpha=1 - prog(u_out, 0.7, 1.0), shadow=0.5)
    # your network, or your VPN: a dashed thread between the desks
    nt = at("proof", "network") - 0.2
    if t > nt:
        u = ease_io(prog(t, nt, nt + 0.9))
        (x0, y0), (x1, y1) = P(A_DESK[0] - 40, A_DESK[1] - 70), P(B_DESK[0] + 40, B_DESK[1] - 70)
        pth = skia.Path(); pth.moveTo(x0, y0); pth.quadTo((x0 + x1) / 2, max(y0, y1) + 90, x1, y1)
        m = skia.PathMeasure(pth, False); seg = skia.Path(); m.getSegment(0, m.getLength() * u, seg, True)
        pp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color4f=rgb(TEAL_INK, 0.95))
        pp.setPathEffect(skia.DashPathEffect.Make([16, 12], -t * 40)); c.drawPath(seg, pp)
        chip(c, "your network", (x0 + x1) / 2 - 220, max(y0, y1) + 120, t, nt + 0.6, 30, "tex-teal.png", INK)
        chip(c, "or your VPN", (x0 + x1) / 2 + 40, max(y0, y1) + 120, t, at("proof", "VPN") - 0.1, 30, "tex-cream.png", INK)
    a1 = 1 - prog(t, at("proof", "No") - 0.4, at("proof", "No") - 0.05)
    headline(c, "Only the two Macs, nothing in between.", W / 2, 1040, 58, t, word("proof", 0) - 0.05, CREAM, alpha=a1, align="center")
    a2 = 1.0
    headline(c, "No server.", W / 2, 1040, 58, t, at("proof", "No") - 0.05, CREAM, alpha=a2, align="center")

# ---------------------------------------------------------------- 4. one search, every Mac
APP0 = at("search", "and") - 0.4              # the push into the app, as results "join yours"
def s_search_under(c, t):
    fx, fy, zoom = UNDER
    s, P = cam(fx, fy, zoom)
    L.plate(c, PLATE, fx, fy, zoom, src_w=PW)
    tunnel(c, P, s, t, -10, -9)
    q0 = at("search", "search") - 0.2
    work = at("search", "searches") - 0.2
    digging = t > work
    moles(c, P, s, t, "mole-dig-cut.png" if digging else "mole-listen-cut.png", "mole-dig-cut.png")
    courier(c, P, s, t, q0, 1.3, None, label="fjord at sunset")
    for (hx, hy) in (B_HOLE, A_HOLE):
        if digging:
            x, y = P(hx, hy - 40)
            glow(c, x, y, 260 * s / 0.6, TEAL, 0.28 * (0.6 + 0.4 * math.sin((t - work) * 6)))
    for i, (hx, hy) in enumerate((B_HOLE, A_HOLE)):
        x, y = P(hx, hy + 300)
        chip(c, "own index · own GPU", x - 175, y, t, at("search", "index") - 0.1 + i * 0.2, 26, "tex-ink.png", CREAM)
    back = at("search", "results") - 0.1
    for k in range(5):
        courier(c, P, s, t, back + k * 0.18, 1.2, "photo-cut.png", back=True, h=95)
    headline(c, "One search covers every linked Mac.", W / 2, 1040, 58, t, at("search", "one") - 0.05, CREAM, align="center")

K1 = None
def k1():
    global K1
    if K1 is None: K1 = Footage(TK("K1c"), 3.0, 17.9)
    return K1

def app_window(c, t, foot, tt, t_in, t_out=None, zoom=1.24, focus=(0.5, 0.5), caption_s=None, cap_t=0):
    bg = img("bg-blur.png"); c.drawImageRect(bg, skia.Rect.MakeWH(W, H), SAMPLE, skia.Paint())
    frame = foot.frame(tt)
    u = expo_io(prog(t, t_in, t_in + 1.0))
    v = expo_io(prog(t, t_out, t_out + 0.8)) if t_out else 0.0
    sc = lerp(0.62, zoom, u) * (1 - 0.25 * v)
    cx = W / 2 + (0.5 - focus[0]) * frame.width() * sc * u * 0.0
    draw_window(c, frame, cx, H / 2 + 10 + (1 - u) * 120, sc, 1.0 - v, radius=25)
    if caption_s and u > 0.9:
        chip(c, caption_s, 110, 1012, t, cap_t, 36, "tex-cream.png", INK, "sans", 650, 1 - v)
    return sc

def s_search_app(c, t):
    tt = 4.2 + (t - APP0) * 0.95
    sc = app_window(c, t, k1(), tt, APP0, caption_s="MacBook Pro's photos, among this Mac's", cap_t=APP0 + 1.0)
    # point at the first result's badge: the computer marks a linked Mac's file
    r = prog(t, APP0 + 1.6, APP0 + 2.2) * (1 - prog(t, T.BREAK - 0.6, T.BREAK - 0.2))
    if r > 0:
        fr = k1().frame(tt); wx = W / 2 - fr.width() * sc / 2; wy = H / 2 + 10 - fr.height() * sc / 2
        bx, by = wx + 321 * sc, wy + 89 * sc
        pp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color4f=rgb(TEAL_INK, r))
        c.drawCircle(bx, by, 28 + 6 * math.sin(t * 5), pp)

# ---------------------------------------------------------------- 5. asleep
def s_away(c, t):
    fx, fy, zoom = UNDER
    s, P = cam(fx, fy, zoom)
    L.plate(c, PLATE, fx, fy, zoom, src_w=PW)
    tunnel(c, P, s, t, -10, -9, thread=0.35)
    moles(c, P, s, t, "mole-sleep-cut.png", "mole-dig-cut.png")
    bx, by = P(B_HOLE[0] + 60, B_HOLE[1] - 90)
    for k in range(3):
        d = (t - T.VO["away"] - k * 0.6) % 1.8
        draw_text(c, "z", bx + 40 + d * 40, by - d * 80, 34 + k * 8, 650, CREAM, clamp(1 - d / 1.8) * prog(t, T.BREAK, T.BREAK + 0.6), "center")
    go = at("away", "search") - 0.1
    x, y = P(A_HOLE[0], A_HOLE[1] - 40)
    if t > go: glow(c, x, y, 260 * s / 0.6, TEAL, 0.3)
    for k in range(4):
        d = t - (go + 0.3 + k * 0.25)
        if d > 0:
            u = ease_out(clamp(d / 0.8))
            draw_image(c, sprite("photo-cut.png", 500), x + 175 + (k - 1.5) * 42, y + 40 - u * 190, h=80 * s / 0.6, rot=(k - 1.5) * 8,
                       alpha=clamp(d * 4), shadow=0.5)
    headline(c, "A Mac asleep is simply left out.", W / 2, 1040, 58, t, word("away", 0) - 0.05, CREAM, align="center")

# ---------------------------------------------------------------- 6. open; the sidebar
K2 = None
def k2():
    global K2
    if K2 is None: K2 = Footage(TK("K2"), 2.0, 15.0)
    return K2

def s_open_under(c, t):
    fx, fy, zoom = UNDER
    s, P = cam(fx, fy, zoom)
    L.plate(c, PLATE, fx, fy, zoom, src_w=PW)
    wake = word("open", 0) - 0.3
    thread = 0.35 + 0.65 * prog(t, wake, wake + 0.6)
    tunnel(c, P, s, t, -10, -9, thread=thread)
    if t < wake:
        moles(c, P, s, t, "mole-sleep-cut.png", "mole-dig-cut.png")
    else:
        moles(c, P, s, t, "mole-pop-cut.png", "mole-listen-cut.png", b_pop=wake)
    courier(c, P, s, t, at("open", "comes") - 0.3, 1.4, "photo-cut.png", back=True, h=130)
    headline(c, "Open, preview or drag a result.", W / 2, 1000, 58, t, word("open", 0) - 0.05, CREAM, align="center")
    caption(c, "The file comes across the link.", W / 2, 1052, t, at("open", "file") - 0.1, 32, "#E3D2BA", align="center", weight=550)

SIDEBAR0 = at("open", "folders") - 0.15
def s_open_app(c, t):
    tt = 6.4 + (t - SIDEBAR0) * 1.1
    sc = app_window(c, t, k2(), tt, SIDEBAR0, caption_s="Its folders are in your sidebar", cap_t=SIDEBAR0 + 0.8)
    r = prog(t, SIDEBAR0 + 1.0, SIDEBAR0 + 1.5)
    if r > 0:
        fr = k2().frame(tt); wx = W / 2 - fr.width() * sc / 2; wy = H / 2 + 10 - fr.height() * sc / 2
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(wx + 14 * sc, wy + 470 * sc, 238 * sc, 60 * sc), 10, 10)
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color4f=rgb(TEAL_INK, r)))

# ---------------------------------------------------------------- 7. agents
CLI = [("$ omni search \"snowy mountains by a lake\" --kinds image --top-k 2", CREAM),
       ("1. …/out_vae/orig/meadow_photo.png  (image, 50%)", "#CDBBA3"),
       ("2. …/out_vae/fix/meadow_photo.png  (image, 48%)", "#CDBBA3"),
       ("On MacBook Pro:", TEAL),
       ("1. ~/Pictures/Iceland/IMG_4161.JPG  (image, 49%)", "#CDBBA3"),
       ("2. ~/Pictures/Iceland/IMG_4190.JPG  (image, 46%)", "#CDBBA3")]
AG0 = word("agents", 0) - 0.9
def s_agents(c, t):
    L.plate(c, "bg-strata.png", 2752, 1700, 1.25)
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Color=skia.Color(25, 16, 10, 110)))
    u = back_out(prog(t, AG0 + 0.2, AG0 + 0.8), 1.3)
    cw, chh = 1320, 430; x0, y0 = (W - cw) / 2, 330 + (1 - u) * 80
    paper_rect(c, x0, y0, cw, chh, "tex-ink.png", radius=16, alpha=clamp(u), shadow=0.9)
    draw_text(c, "agent · MCP · omni", x0 + 36, y0 + 52, 24, 600, "#9FD8D2", clamp(u), "left", "mono")
    cmd, col = CLI[0]
    t_cmd = word("agents", 0)
    n = int(len(cmd) * prog(t, t_cmd, t_cmd + 1.4))
    draw_text(c, cmd[:n], x0 + 36, y0 + 112, 27, 500, col, clamp(u), "left", "mono")
    for i, (ln, col) in enumerate(CLI[1:]):
        ti = t_cmd + 1.6 + i * 0.22
        a = prog(t, ti, ti + 0.25)
        draw_text(c, ln, x0 + 36 + (0 if ln.startswith("On") else 24), y0 + 172 + i * 46, 27, 650 if ln.startswith("On") else 500, col, a, "left", "mono")
    headline(c, "Local agents get the same reach.", W / 2, 200, 66, t, word("agents", 0) - 0.05, CREAM, align="center")

# ---------------------------------------------------------------- 8. the app, held
MONT0 = WORDS["agents"][-1][2] + 0.6
def s_montage(c, t):
    tt = 9.5 + (t - MONT0) * 0.9
    app_window(c, t, k1(), tt, MONT0, zoom=1.28, caption_s="One search. Every Mac.", cap_t=MONT0 + 1.0)

# ---------------------------------------------------------------- 9. the end
END0 = word("end", 0) - 0.9
def s_end(c, t):
    L.plate(c, "bg-paper.png", 2752, 1536, 1.03 + 0.012 * prog(t, END0, T.LENGTH))
    t0 = word("end", 0)
    d = t - (t0 - 0.3)
    if d > 0:
        k = spring(d * 0.8, 1.2, 4.0)
        bob = math.sin(max(0, t - t0 - 1.5) * 2.2) * 4
        draw_image(c, sprite("mole-wave-cut.png", 1000), 520, 840 + (1 - k) * 600 + bob, h=600, anchor=(0.5, 0.95), shadow=0.7)
    draw_text(c, "Linked Macs", 780, 520, 150, 720, INK, expo_out(prog(t, t0, t0 + 0.8)), "left",
              scale=1 + 0.06 * math.exp(-6 * max(0, t - t0)))
    reveal_words(c, "New in Omni 0.16.", 786, 610, 46, t, at("end", "new") - 0.1, 0.12, 0.6, 550, MUTED)
    caption(c, "hanxiao.io/omni", 786, 690, t, at("end", "Omni") + 0.2, 34, TEAL_INK, kind="mono", weight=600)
    caption(c, "macOS 14+ · Apple silicon · free and open source", 786, 746, t, T.LAST_HIT - 0.6, 28, MUTED, kind="mono")

# ---------------------------------------------------------------- the edit
UNDER0 = 12.4
SEARCH_U0 = word("search", 0) - 0.6
SHOTS = [
    (0.0, S1_END + 0.5, s_many),
    (S2_0, 13.2, s_link),
    (UNDER0, SEARCH_U0 + 0.4, s_proof),
    (SEARCH_U0, APP0 + 1.0, s_search_under),
    (APP0, T.BREAK + 0.5, s_search_app),
    (T.BREAK, word("open", 0) - 0.4, s_away),
    (word("open", 0) - 0.8, SIDEBAR0 + 1.0, s_open_under),
    (SIDEBAR0, AG0 + 0.6, s_open_app),
    (AG0 - 0.4, MONT0 + 0.5, s_agents),
    (MONT0, END0 + 0.8, s_montage),
    (END0, T.LENGTH + 0.1, s_end),
]
DISSOLVE = {(s_proof, s_search_under): (SEARCH_U0, SEARCH_U0 + 0.4),
            (s_search_app, s_away): (T.BREAK, T.BREAK + 0.5),
            (s_away, s_open_under): (word("open", 0) - 0.8, word("open", 0) - 0.4),
            (s_open_app, s_agents): (AG0 - 0.4, AG0 + 0.6),
            (s_agents, s_montage): (MONT0, MONT0 + 0.5),
            (s_montage, s_end): (END0, END0 + 0.8)}
# where the shots carry their own move (the app pushes in over the shot before it), no dissolve:
# s_search_under -> s_search_app and s_open_under -> s_open_app overlap and the window scales in.
DISSOLVE[(s_search_under, s_search_app)] = (APP0, APP0 + 0.35)
DISSOLVE[(s_open_under, s_open_app)] = (SIDEBAR0, SIDEBAR0 + 0.35)
DISSOLVE[(s_link, s_proof)] = (12.4, 13.0)

def fade_to_black(t):
    return 1 - prog(t, T.LENGTH - 1.3, T.LENGTH)

WHIP = [(3.9, 5.0), (12.1, 13.2)]
BLUR = [(APP0, APP0 + 1.1), (SIDEBAR0, SIDEBAR0 + 1.1), (MONT0, MONT0 + 1.1)]
def subframes(t):
    if any(a <= t <= b for a, b in WHIP): return 12
    return 5 if any(a <= t <= b for a, b in BLUR) else 1
