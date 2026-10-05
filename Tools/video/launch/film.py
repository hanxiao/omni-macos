# The film: ten shots on the score's clock and the narration's word timings.
import json, math, os, random
import skia
from core import *
import timeline as T

WORDS = json.load(open(os.path.join(WORK, "words.json")))
def word(line, i):
    """Film time a narration word starts: WORDS[line][i] (negative i from the end)."""
    return WORDS[line][i][1]
def at(line, text, nth=0):
    hits = [w for w in WORDS[line] if w[0].strip(".,?!").lower() == text.lower()]
    return hits[nth][1]

KRAFT_INK = "#9A6E44"            # kraft dark enough to read as type on cream
BAR = T.BAR

# ---------------------------------------------------------------- shared pieces
def headline(c, s, x, y, size, t, t0, color=INK, weight=650, alpha=1.0, align="left"):
    return reveal_words(c, s, x, y, size, t, t0, stagger=0.075, dur=0.6, weight=weight, color=color,
                        alpha=alpha, align=align)

def caption(c, s, x, y, t, t0, size=30, color=MUTED, alpha=1.0, align="left", kind="sans", weight=500):
    u = expo_out(prog(t, t0, t0 + 0.6))
    if u <= 0: return
    draw_text(c, s, x, y + (1 - u) * 18, size, weight, color, alpha * u, align, kind)

def chip(c, s, x, y, t, t0, size=26, tex="tex-cream.png", color=INK, kind="mono", weight=500, alpha=1.0, pad=18):
    """A small paper label that drops onto the page."""
    u = back_out(prog(t, t0, t0 + 0.45), 1.4)
    if u <= 0: return
    w = text_width(s, size, weight, kind) + 2 * pad; h = size * 1.9
    a = alpha * clamp(prog(t, t0, t0 + 0.2))
    c.save(); c.translate(x, y + (1 - u) * -26)
    paper_rect(c, 0, -h * 0.68, w, h, tex, radius=6, alpha=a, shadow=0.8)
    draw_text(c, s, pad, 0, size, weight, color, a, "left", kind)
    c.restore()

def counter(t, t0, t1, v0, v1):
    return v0 + (v1 - v0) * ease_out(prog(t, t0, t1), 4)

def paper_bg(c, tex="tex-cream.png", alpha=1.0):
    tx = img(tex); p = skia.Paint(); p.setAlphaf(alpha)
    c.drawImageRect(tx, skia.Rect.MakeXYWH(0, (tx.height() - tx.width() * H / W) / 2, tx.width(), tx.width() * H / W),
                    skia.Rect.MakeWH(W, H), SAMPLE, p)

def plate(c, name, fx, fy, zoom, alpha=1.0, src_w=5504):
    """A 16:9 plate scaled to fill the frame at zoom 1, focused on its own source pixel (fx, fy)."""
    im = img(name, 4096); k = im.width() / src_w
    s = W / src_w * zoom
    c.save(); c.translate(W / 2, H / 2); c.scale(s / k, s / k); c.translate(-fx * k, -fy * k)
    p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha)
    c.drawImage(im, 0, 0, SAMPLE, p); c.restore()
    return s

# the tall plate (3072 x 5504 source px, drawn from a 4096-max copy): ground at 2650, chamber at (1550, 4950)
TALL_W = 3072
class Tall:
    def __init__(self, cy, zoom=1.0, cx=1536):
        self.s = W / TALL_W * zoom; self.cx, self.cy = cx, cy
    def P(self, x, y): return (W / 2 + (x - self.cx) * self.s, H / 2 + (y - self.cy) * self.s)
    def draw(self, c, alpha=1.0):
        im = img("bg-tall.png", 4096); k = im.width() / TALL_W
        c.save(); c.translate(W / 2, H / 2); c.scale(self.s / k, self.s / k); c.translate(-self.cx * k, -self.cy * k)
        p = skia.Paint(AntiAlias=True); p.setAlphaf(alpha); c.drawImage(im, 0, 0, SAMPLE, p); c.restore()

GROUND, CHAMBER = 2650, (1550, 4950)

# ---------------------------------------------------------------- 1. it is not new
def s_old(c, t):
    exit_u = expo_io(prog(t, 6.55, 7.65))
    c.save(); c.translate(0, exit_u * H * 1.05)
    plate(c, "bg-paper.png", 2752 + t * 6, 1536, 1.04 + t * 0.004)
    headline(c, "Semantic search", 160, 470, 124, t, word("old", 0) - 0.05)
    headline(c, "has existed for decades.", 160, 610, 124, t, at("old", "has") - 0.05, KRAFT_INK)
    # the timeline: a cotton string, tags clipped on as the years go by
    u = ease_io(prog(t, 3.55, 4.6))
    if u > 0:
        line(c, 150, 812, 150 + 1620 * u, 812 + 10 * math.sin(u * 3), "#8B7765", 2.2)
    tags = [("1990", "LSI"), ("2003", "LDA"), ("2013", "word2vec"), ("2018", "BERT"), ("2023", "vector DBs")]
    tg = sprite("tag-cut.png", 700)
    t0s = [word("old", 2), word("old", 3), word("old", 4), word("old", 5), word("old", 5) + 0.35]
    for i, ((yr, nm), t0) in enumerate(zip(tags, t0s)):
        x = 270 + i * 330
        d = t - t0
        if d < 0: continue
        drop = spring(d * 1.0, 1.6, 5.0)
        sw = math.exp(-2.4 * d) * math.sin(d * 9) * 9
        c.save(); c.translate(x, 812 - (1 - drop) * 120); c.rotate(sw)
        draw_image(c, tg, 0, -14, h=250, anchor=(0.62, 0.0), shadow=0.6, alpha=clamp(d * 5))
        draw_text(c, yr, -22, 165, 34, 650, INK, clamp(d * 5), "center", "mono")
        draw_text(c, nm, -22, 202, 25, 600, "#4A3A2C", clamp(d * 5), "center", "mono")
        c.restore()
    c.restore()

# ---------------------------------------------------------------- 2. it became a service
CARDS = ["Embeddings API", "Vector DB", "Search API", "RAG platform"]
def s_cloud(c, t):
    enter = expo_io(prog(t, 6.55, 7.65))
    leave = expo_io(prog(t, 15.0, 16.3))
    dy = (1 - enter) * -H * 1.05 - leave * H * 0.9
    c.save(); c.translate(0, dy)
    plate(c, "bg-paper.png", 2752, 1536, 1.06)
    bob = math.sin(t * 0.9) * 6
    cl = sprite("cloud-cut.png", 1400)
    # files rising into the cloud on threads
    kinds = ["doc-cut.png", "photo-cut.png", "folder-cut.png"]
    rnd = random.Random(3)
    for i in range(16):
        t0 = at("cloud", "files") + i * 0.22
        d = t - t0
        if d < 0: continue
        sx = 700 + rnd.random() * 1100; sy = 1180
        ex = 1150 + rnd.random() * 560; ey = 380
        u = ease_in(clamp(d / 2.6), 2.2)
        x = lerp(sx, ex, u) + math.sin(d * 2 + i) * 14; y = lerp(sy, ey, u)
        a = clamp(1 - (u - 0.75) / 0.25) if u > 0.75 else 1.0
        line(c, x, y - 40, ex, ey, "#BBAA96", 1.0, 0.6 * a)
        draw_image(c, sprite(kinds[i % 3], 500), x, y, h=122, rot=math.sin(d * 1.5 + i) * 10, alpha=a, shadow=0.5)
    # service cards hanging from the cloud
    cd = sprite("card-cut.png", 700)
    for i, name in enumerate(CARDS):
        t0 = word("cloud", 2) + i * 0.24
        d = t - t0
        if d < 0: continue
        x = 1090 + i * 225; top = 430 + bob
        drop = spring(d, 1.4, 4.0)
        sw = math.exp(-1.8 * d) * math.sin(d * 7 + i) * 6
        y = top + 40 + drop * 120
        line(c, x, top, x, y + 4, "#8B7765", 1.4)
        c.save(); c.translate(x, y); c.rotate(sw)
        draw_image(c, cd, 0, 0, h=270, anchor=(0.5, 0.06), shadow=0.7)
        for j, part in enumerate(name.split(" ")):
            draw_text(c, part, 0, 130 + j * 32, 25, 600, INK, 1, "center", "mono")
        c.restore()
    draw_image(c, cl, 1420, 250 + bob, w=1060, shadow=0.5)
    headline(c, "Today it is mostly", 160, 420, 80, t, word("cloud", 0) - 0.05)
    headline(c, "a cloud service.", 160, 510, 80, t, at("cloud", "cloud") - 0.05)
    reveal_words(c, "Your files are uploaded to remote servers.", 160, 585, 40, t, at("cloud", "files") - 0.05,
                 weight=500, color=MUTED)
    # the upload, as a paper progress chip
    if t > at("cloud", "uploaded") - 0.3:
        up0 = at("cloud", "uploaded") - 0.3; a = clamp((t - up0) * 4)
        n = int(counter(t, up0 + 0.2, up0 + 3.2, 0, 2431))
        paper_rect(c, 160, 640, 470, 74, "tex-cream.png", radius=10, alpha=a)
        draw_text(c, f"Uploading {n:,} files", 190, 686, 26, 500, INK, a, "left", "mono")
        paper_rect(c, 190, 700, 410 * clamp((t - up0 - 0.2) / 3.0), 5, "tex-teal.png", radius=2, alpha=a, shadow=0, edge=False)
    c.restore()

# ---------------------------------------------------------------- 3. what if they never had to leave
def tall_cam(t):
    return keys(t, [(15.0, 600.0), (16.2, 900.0), (18.6, 1500.0), (20.2, GROUND - 260.0), (21.6, GROUND - 160.0),
                    (23.4, 3700.0), (25.4, 4560.0), (28.2, 4625.0), (30.7, 4640.0)], smooth)

def s_down(c, t):
    cy = tall_cam(t)
    w = Tall(cy)
    enter = expo_io(prog(t, 15.0, 16.3))
    c.save(); c.translate(0, (1 - enter) * H)
    w.draw(c)
    s = w.s
    # files falling back to earth, slipping under the ground line
    rnd = random.Random(11)
    kinds = ["doc-cut.png", "photo-cut.png", "folder-cut.png"]
    for i in range(10):
        t0 = word("down", 1) + i * 0.18
        d = t - t0
        if d < 0: continue
        x0 = 700 + rnd.random() * 1700; y0 = 900 + rnd.random() * 300
        x1 = 900 + rnd.random() * 1300; y1 = 3050 + rnd.random() * 900
        u = ease_io(clamp(d / 3.0))
        x, y = w.P(lerp(x0, x1, u), lerp(y0, y1, u))
        gy = w.P(0, GROUND - 20)[1]
        c.save()
        if lerp(y0, y1, u) > GROUND - 60: c.clipRect(skia.Rect.MakeLTRB(0, gy + 30 * s, W, H))
        draw_image(c, sprite(kinds[i % 3], 500), x, y, h=150 * s, rot=(1 - u) * 40 * (1 if i % 2 else -1),
                   alpha=0.95, shadow=0.4)
        c.restore()
    # fossils: files resting in the strata
    rnd = random.Random(5)
    for i in range(9):
        x, y = w.P(250 + rnd.random() * 2600, 2950 + rnd.random() * 1200)
        draw_image(c, sprite(kinds[i % 3], 500), x, y, h=(110 + rnd.random() * 60) * s,
                   rot=rnd.uniform(-35, 35), alpha=0.92, shadow=0.35)
    # the desk, the laptop, and the cable to the sky
    dx, dy = w.P(2380, GROUND + 6)
    cut = at("down", "server") + 0.12                # "no server": snip
    lt = (dx - 40 * s, dy - 520 * s)                 # top of the laptop on the desk
    cab = sprite("cable-cut.png", 1400)
    cw = 30 * s                                      # the cable's thickness on screen
    snip_y = lt[1] - 150 * s
    def cable_piece(y0, y1, rot=0.0, pivot=None):
        # the cable sprite runs plug-right; stood on end it runs plug-down into the laptop
        L = y1 - y0
        if L <= 2: return
        c.save()
        if pivot: c.translate(*pivot); c.rotate(rot); c.translate(-pivot[0], -pivot[1])
        c.clipRect(skia.Rect.MakeLTRB(lt[0] - 60, y0, lt[0] + 60, y1))
        c.translate(lt[0], lt[1]); c.rotate(-90)
        draw_image(c, cab, 0, 0, w=max(L, 900 * s) + 400, h=cw * 2.2, anchor=(0.02, 0.5), shadow=0.5)
        c.restore()
    if t < cut:
        cable_piece(-80, lt[1])
    else:
        d = t - cut
        up = spring(d * 0.8, 0.9, 3.0)
        c.save(); c.translate(0, -up * 1100); cable_piece(-80, snip_y); c.restore()
        cable_piece(snip_y + 6, lt[1], -min(1, d * 2.2) * 40, (lt[0], lt[1]))
        if d < 0.8:
            rnd2 = random.Random(4)
            for i in range(10):
                ang = rnd2.uniform(0, 6.28); sp = rnd2.uniform(80, 220)
                circle(c, lt[0] + math.cos(ang) * sp * d, snip_y + math.sin(ang) * sp * d + 300 * d * d,
                       rnd2.uniform(3, 6), CREAM, 1 - d / 0.8)
    if cut - 0.7 < t < cut + 0.9:
        k = prog(t, cut - 0.7, cut - 0.05)
        sx = lerp(W + 300, lt[0] + 210 * s, ease_out(k, 4))
        a = 1 - prog(t, cut + 0.4, cut + 0.9)
        draw_image(c, sprite("scissors-cut.png", 900), sx, snip_y, h=300 * s, alpha=a, shadow=0.7,
                   rot=-10 + 16 * prog(t, cut - 0.12, cut))
    draw_image(c, sprite("desk-cut.png", 900), dx, dy, h=560 * s, anchor=(0.5, 0.98), shadow=0.6)
    # the claims, stamped as they are spoken
    gx = 160
    gy = w.P(0, GROUND - 560)[1]
    if t < 24.5 and gy > -400:
        headline(c, "Everything stays on your Mac.", gx, gy, 68, t, word("down", 0) + 0.2)
        headline(c, "It needs no server and no network.", gx, gy + 88, 68, t, at("down", "with") - 0.05, color=TEAL_INK)
        pass
    # the mole surfaces in its chamber
    mx, my = w.P(CHAMBER[0], CHAMBER[1] + 420)
    pop = word("omni", 0) + 0.2
    d = t - pop
    if d > 0:
        k = spring(d * 0.9, 1.3, 4.5)
        sq = 1 + 0.12 * math.exp(-5 * d) * math.sin(d * 18)
        c.save(); c.translate(mx, my); c.scale(1 / sq, sq)
        draw_image(c, sprite("mole-pop-cut.png", 1000), 0, (1 - k) * 360 * s, h=760 * s, anchor=(0.5, 0.92), shadow=0.6)
        c.restore()
        rnd = random.Random(9)
        for i in range(22):
            ang = rnd.uniform(-2.6, -0.5); sp = rnd.uniform(280, 620) * s
            px = mx + math.cos(ang) * sp * d * 2; py = my - 240 * s + math.sin(ang) * sp * d * 2 + 900 * s * d * d
            if d < 1.4:
                circle(c, px, py, rnd.uniform(5, 11) * s, rnd.choice([SOIL, DEEP, KRAFT]), 1 - d / 1.4)
    c.restore()
    title(c, t)

# ---------------------------------------------------------------- 4. the easy part (title, then the GPU)
def title(c, t):
    d = t - T.GROOVE
    if d < 0 or t > 30.2: return
    a = clamp(d * 8) * (1 - prog(t, 29.75, 30.15))
    sc = 1 + 0.18 * math.exp(-7 * d)
    draw_text(c, "Omni", W / 2, 250, 190, 720, CREAM, a, "center", scale=sc)
    caption(c, "Semantic search that runs on your Mac.", W / 2, 322, t, T.GROOVE + 0.25, 34, "#F3E7D3", a, "center", weight=550)

def gpu(c, t, ox, oy, scale=1.0, alpha=1.0, cols=16, rows=8):
    tile, gap = 54 * scale, 10 * scale
    for r in range(rows):
        for q in range(cols):
            x = ox + q * (tile + gap); y = oy + r * (tile + gap)
            phase = (q + r) * 0.11
            lit = 0.5 + 0.5 * math.sin((t - 31.2) * 5.0 - phase * 6)
            on = prog(t, 31.0 + (q + r) * 0.03, 31.4 + (q + r) * 0.03)
            v = on * clamp(lit * 1.3 - 0.15)
            paper_rect(c, x, y, tile, tile, "tex-soil.png", radius=5 * scale, alpha=alpha, shadow=0.5, edge=True)
            if v > 0.02:
                paper_rect(c, x, y, tile, tile, "tex-teal.png", radius=5 * scale, alpha=alpha * v, shadow=0, edge=False)
    return cols * (tile + gap) - gap, rows * (tile + gap) - gap

def easy_panel(c, t):
    """The GPU panel, full frame. The messy shot shrinks this very drawing into its encoder slot."""
    plate(c, "bg-under.png", 1900, 2000, 1.6)   # focus keeps the zoomed plate covering the frame
    only = word("only", 0)
    a = clamp((t - 30.4) * 2)
    gw, gh = gpu(c, t, 880, 330, 0.88, a)
    al = a * (1 - prog(t, only - 0.4, only))
    n = int(counter(t, 31.5, 34.5, 0, 83105))
    draw_text(c, f"{n:,} tokens/s", 880, 330 + gh + 92, 60, 500, CREAM, al, "left", "mono")
    draw_text(c, "indexing throughput on an M3 Ultra", 880, 330 + gh + 140, 30, 500, "#CDBBA3", al, "left", "mono")
    la = 1 - prog(t, only - 0.4, only)
    headline(c, "Running the model locally", 160, 190, 72, t, word("easy", 0) - 0.05, CREAM, alpha=la)
    headline(c, "is straightforward.", 160, 275, 72, t, at("easy", "straightforward") - 0.3, TEAL, alpha=la)
    x = 160
    for wd, key in [("Swift", "Swift"), ("Metal", "Metal"), ("No Python", "Python")]:
        tt = at("easy", key) - (0.35 if key == "Python" else 0.05)
        chip(c, wd, x, 470, t, tt, 46, "tex-cream.png", INK, "sans", 650, la)
        x += text_width(wd, 46, 650) + 80
    headline(c, "Inference is only one part", 160, 190, 72, t, word("only", 0) - 0.05, CREAM)
    headline(c, "of the problem.", 160, 275, 72, t, at("only", "part") - 0.05, TEAL)

def s_easy(c, t):
    easy_panel(c, t)

ENC_R = (585, 150, 250, 141)        # where the shrunk panel lives in the files shot: the encoder

def encoder(c, t, shrink=1.0, alpha=1.0):
    """The easy panel scaled from the full frame into ENC_R."""
    x, y, w, h = ENC_R
    k = lerp(1.0, w / W, shrink)
    ox, oy = lerp(0, x, shrink), lerp(0, y, shrink)
    r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(ox, oy, W * k, H * k), 10 * shrink, 10 * shrink)
    if shrink > 0.02:
        sp = skia.Paint(AntiAlias=True, Color=skia.Color(30, 20, 12, int(110 * shrink * alpha)))
        sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 12))
        c.drawRRect(r.makeOffset(6, 12), sp)
    c.save(); c.clipRRect(r, True); c.translate(ox, oy); c.scale(k, k)
    if alpha < 1: c.saveLayerAlpha(None, int(alpha * 255))
    easy_panel(c, t)
    if alpha < 1: c.restore()
    c.restore()

# ---------------------------------------------------------------- 5. messy files
VERS = [("report_v1.pages", "ABCDEF"), ("report_v2.pages", "ABCDEG"), ("report_final.pages", "ABCHEG"),
        ("report_final_FINAL.pages", "ABCHEG")]
PASSAGE_TEX = {"A": "tex-cream.png", "B": "tex-kraft.png", "C": "tex-white.png", "D": "tex-teal.png",
               "E": "tex-kraft.png", "F": "tex-teal.png", "G": "tex-cream.png", "H": "tex-teal.png", "E2": "tex-teal.png"}
PASSAGE_W = {"A": 0.92, "B": 0.78, "C": 0.86, "D": 0.7, "E": 0.82, "F": 0.64, "G": 0.9, "H": 0.74, "E2": 0.82}
SLOTS = list("ABCDEFGH")

# shot anchors, from the narration: each shot starts a beat before its line
MESSY0 = WORDS["only"][-1][2] + 0.95          # the GPU panel shrinks once "inference" is said
ASIDE0 = word("aside", 0) - 1.0
FUNNEL0 = word("funnel", 0) - 0.6
AGENTS0 = word("agents", 0) - 0.8
PAPER0 = word("paper", 0) - 0.5
END0 = word("end", 0) - 0.9
def doc_xy(i): return 430 + i * 260, 590

def s_messy(c, t):
    plate(c, "bg-strata.png", 2752, 1545, 1.02 + 0.01 * prog(t, MESSY0, ASIDE0))   # 1545: covers top and bottom
    # the GPU panel from the last shot shrinks into the cavity: the encoder everything passes through
    shrink = expo_io(prog(t, MESSY0, MESSY0 + 1.25))
    ea = 1 - prog(t, ASIDE0 - 0.6, ASIDE0)
    encoder(c, t, shrink, ea)
    ENC = (ENC_R[0] + 40, ENC_R[1] + 30)
    if shrink > 0.95:
        chip(c, "inference", ENC_R[0] + 60, ENC_R[1] + ENC_R[3] + 46, t, MESSY0 + 1.2, 24, "tex-ink.png", CREAM, alpha=ea)
    # 1) the pile tumbles in
    rnd = random.Random(21)
    piles = []
    gather = expo_io(prog(t, at("messy", "Version") - 0.3, at("messy", "Version") + 0.6))
    land_t = [at("messy", "Version", 0), at("messy", "Version", 1), at("messy", "Final", 0), at("messy", "Final", 1)]
    for i in range(14):
        t0 = word("messy", 0) + i * 0.09
        d = t - t0
        if d < 0: continue
        x1 = 760 + rnd.uniform(-330, 330); y1 = 650 + rnd.uniform(-120, 90); r1 = rnd.uniform(-40, 40)
        u = ease_out(clamp(d / 0.9), 3)
        y = lerp(-200, y1, u) + (1 - spring(d, 2.0, 5)) * 0
        a = 1 - gather
        if a > 0: draw_image(c, sprite("doc-cut.png", 600), x1, y, h=230, rot=r1 * (1 - u * 0.3), alpha=a, shadow=0.6)
    # 2) copies of copies: one document fans into four
    copies = expo_out(prog(t, at("messy", "folders"), at("messy", "folders") + 0.9))
    # 3) the four versions, each a stack of passages
    fly = expo_io(prog(t, at("once", "each") - 0.2, at("once", "once") + 0.3))
    edit_t = at("once", "edit")
    read_t = at("once", "only")
    for i, (name, ps) in enumerate(VERS):
        if gather <= 0 and copies <= 0: break
        x, y = doc_xy(i)
        g = back_out(prog(t, land_t[i] - 0.18, land_t[i] + 0.42), 1.5)
        px, py = 760 + i * 26 * copies, 640 - i * 18 * copies
        cx = lerp(px, x, g); cy = lerp(py, y, g) - math.sin(clamp(g) * math.pi) * 40
        rot = lerp(-6 + i * 4, 0, clamp(g))
        a = clamp(copies * 4 - i) if gather <= 0 else 1.0
        draw_image(c, sprite("doc-cut.png", 700), cx, cy, h=360, rot=rot, alpha=a, shadow=0.7)
        if g > 0.5:
            la = prog(t, land_t[i], land_t[i] + 0.3)
            draw_text(c, name, cx, cy + 226 + (i % 2) * 28, 19, 500, INK, la, "center", "mono")
            # passages: strips on the page
            for j, p in enumerate(ps):
                edited = (i == 3 and p == "E" and t > edit_t)
                key = "E2" if edited else p
                sx = cx - 74; sy = cy - 118 + j * 38
                ww = 150 * PASSAGE_W[key]
                strip_a = prog(t, land_t[i] + 0.2, land_t[i] + 0.6)
                if fly > 0 and not (edited and t > edit_t):
                    # the strip leaves for its slot; an outline stays behind as the reference
                    slot = SLOTS.index(p)
                    tx, ty = 1520, 248 + slot * 70
                    sxx = lerp(sx, tx, fly); syy = lerp(sy, ty, fly)
                    paper_rect(c, sx, sy, ww, 24, "tex-kraft.png", radius=3, alpha=0.45 * strip_a, shadow=0, edge=False)
                    if fly < 0.999:
                        paper_rect(c, sxx, syy, lerp(ww, 300, fly), lerp(24, 50, fly), PASSAGE_TEX[p], radius=4,
                                   alpha=strip_a, shadow=0.5)
                else:
                    flash = 1 + 0.25 * math.sin(clamp((t - edit_t) * 3) * math.pi) if edited else 1
                    paper_rect(c, sx, sy, ww, 24 * flash, PASSAGE_TEX[key], radius=3, alpha=strip_a, shadow=0.4)
    # 4) the shelf of contents: every passage once
    if fly > 0:
        sa = clamp(fly * 3)
        chip(c, "stored once", 1520, 216, t, at("once", "each"), 26, "tex-ink.png", CREAM)
        for k, p in enumerate(SLOTS):
            ty = 248 + k * 70
            land = fly > 0.98
            paper_rect(c, 1520, ty, 300, 50, PASSAGE_TEX[p], radius=4, alpha=sa if land else sa * 0.25, shadow=0.6 if land else 0)
            refs = sum(ps.count(p) for _, ps in VERS)
            if land and refs > 1:
                draw_text(c, f"×{refs}", 1802, ty + 36, 30, 650, INK, prog(t, at("once", "once") + 0.3 + k * 0.05, at("once", "once") + 0.6 + k * 0.05), "right", "mono")
        if fly > 0.98:
            caption(c, "Four files with 24 passages need 8 vectors.", 160, 975, t, at("once", "once") + 0.35, 44, CREAM, weight=600)
            caption(c, "A copy adds no vectors.", 160, 1030, t, at("once", "copy") - 0.1, 30, "#E3D2BA", kind="mono")
    # 5) the edit: one new passage goes through the encoder, and only it
    if t > read_t - 0.3:
        x, y = doc_xy(3); sx = x - 92; sy = y - 120 + 4 * 38
        k1 = expo_io(prog(t, read_t - 0.2, read_t + 0.9))       # to the encoder
        k2 = expo_io(prog(t, read_t + 1.1, read_t + 2.0))       # to the shelf, new slot
        ex, ey = ENC[0] + 50, ENC[1] + 30
        px = lerp(lerp(sx, ex, k1), 1520, k2); py = lerp(lerp(sy, ey, k1), 248 + 8 * 70, k2)
        glow(c, ENC[0] + 80, ENC[1] + 40, 160, TEAL, 0.55 * math.sin(clamp((t - read_t - 0.7) * 2.2) * math.pi))
        paper_rect(c, px, py, lerp(150, 300, k2), lerp(24, 50, k2), "tex-teal.png", radius=4, shadow=0.6)
        caption(c, "Saving an edit takes 8.5 ms.", 1060, 975, t, read_t + 1.6, 44, CREAM, weight=600)
        caption(c, "median on an M3 Ultra", 1060, 1030, t, read_t + 1.8, 30, "#E3D2BA", kind="mono")
    # the words
    hw = 1 - prog(t, at("messy", "folders") - 0.35, at("messy", "folders") - 0.05)
    headline(c, "The harder part is keeping the index current.", 160, 1010, 60, t, word("messy", 0) - 0.05, CREAM, alpha=hw)
    if fly < 0.05:
        headline(c, "Folders hold many versions of one document.", 160, 1010, 60, t, at("messy", "folders") - 0.05, "#E8C9A0", alpha=1 - fly * 20)

# ---------------------------------------------------------------- 6. sharing the machine
def s_aside(c, t):
    # surface: the desk with someone typing; under it the mole at work
    w = Tall(keys(t, [(ASIDE0, GROUND - 700.0), (ASIDE0 + 1.2, GROUND + 120.0), (FUNNEL0 + 0.4, GROUND + 160.0)], ease_io), 1.55, 1650)
    w.draw(c)
    s = w.s
    lx, ly = w.P(2260, GROUND + 4)
    draw_image(c, sprite("laptop-open-cut.png", 1000), lx, ly, h=330 * s / 1.55 * 1.55, anchor=(0.5, 0.97), shadow=0.6)
    hands = sprite("hands-cut.png", 1000)
    typing = at("aside", "While")
    jig = math.sin(t * 38) * 2.5 if t > typing else 0
    draw_image(c, hands, lx + 120 * s, ly - 40 * s + jig, h=150 * s, anchor=(0.5, 0.9), shadow=0.5)
    # keystrokes ripple down into the ground
    for k in range(10):
        t0 = typing + k * 0.42
        d = t - t0
        if 0 < d < 1.6:
            r = d * 520
            p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3, Color4f=rgb(TEAL_INK, 0.55 * (1 - d / 1.6)))
            c.save(); c.clipRect(skia.Rect.MakeLTRB(0, ly + 10, W, H))
            c.drawArc(skia.Rect.MakeXYWH(lx + 80 * s - r, ly - r * 0.55, 2 * r, 1.1 * r), 20, 140, False, p); c.restore()
    # the mole's work: big bricks that split into small ones while you type
    gy = ly + 300
    steps_t = at("aside", "smaller")
    split = expo_out(prog(t, typing + 0.2, steps_t))
    mole = sprite("mole-dig-cut.png", 900)
    mx = 520 + (t - 63) * 6
    draw_image(c, mole, mx, gy + 70, h=230, anchor=(0.5, 0.95), shadow=0.6)
    bx = mx + 120
    chip(c, "indexing", bx + 20, gy - 150, t, ASIDE0 + 1.4, 30, "tex-ink.png", CREAM)
    n_big = 4
    for b in range(n_big):
        x0 = bx + b * 250
        if split < 0.02:
            paper_rect(c, x0, gy - 110, 230, 150, "tex-kraft.png", radius=6, shadow=0.7)
        else:
            for q in range(4):
                ox = (q % 2) * (115 + 18 * split); oy = (q // 2) * (75 + 14 * split)
                paper_rect(c, x0 + ox, gy - 110 + oy, 112, 72, "tex-kraft.png", radius=5, shadow=0.7)
    # a query slips through the gaps, down and back up
    q0 = steps_t + 0.1
    if q0 < t < q0 + 1.6:
        u = prog(t, q0, q0 + 1.6)
        path_y = ly + 20 + math.sin(u * math.pi) * (gy - ly + 30)
        path_x = lx + 60 * s + u * 700
        glow(c, path_x, path_y, 70, TEAL, 0.6)
        paper_rect(c, path_x - 26, path_y - 13, 52, 26, "tex-teal.png", radius=13, shadow=0.5)
        draw_text(c, "your search", path_x + 40, path_y - 18, 30, 650, TEAL_INK if path_y < ly + 40 else CREAM, 1.0, "left", "mono")
    headline(c, "Indexing shares the machine with you.", 160, 150, 58, t, word("aside", 0) - 0.05, INK,
             alpha=1 - prog(t, FUNNEL0 - 1.0, FUNNEL0 - 0.2))
    caption(c, "While you type, the indexer works in smaller units.", 160, 222, t, typing, 42, INK, 1 - prog(t, FUNNEL0 - 1.0, FUNNEL0 - 0.2), weight=600)
    caption(c, "The longest wait for a query drops from 3.1 s to 1.4 s.", 160, 276, t, steps_t + 0.4, 30, "#5E4B3A",
            1 - prog(t, FUNNEL0 - 1.0, FUNNEL0 - 0.2), kind="mono")
    # the memory you allow: a paper gauge that never overflows
    mem_t = at("aside", "memory") - 0.4
    if t > mem_t:
        a = clamp((t - mem_t) * 3) * (1 - prog(t, FUNNEL0 - 0.2, FUNNEL0 + 0.4))
        gx, gy0, gh = 1680, 560, 420
        paper_rect(c, gx, gy0, 150, gh, "tex-soil.png", radius=10, alpha=a, shadow=0.7)
        lim = lerp(0.82, 0.58, ease_io(prog(t, mem_t + 0.6, mem_t + 1.8)))
        fill = min(lim, 0.3 + (t - mem_t) * 0.35)
        for k in range(int(fill * 12)):
            paper_rect(c, gx + 14, gy0 + gh - 14 - (k + 1) * (gh - 28) / 12, 122, (gh - 28) / 12 - 6, "tex-kraft.png",
                       radius=3, alpha=a, shadow=0.3)
        ly2 = gy0 + gh * (1 - lim)
        line(c, gx - 30, ly2, gx + 180, ly2, TEAL_INK, 4, a)
        draw_text(c, "your memory limit", gx - 40, ly2 + 10, 30, 600, CREAM, a, "right", "mono")

# ---------------------------------------------------------------- 6b. the funnel
def s_funnel(c, t):
    plate(c, "bg-paper.png", 2752, 1536, 1.05 + 0.02 * prog(t, FUNNEL0, AGENTS0 + 0.4))
    t0 = word("funnel", 0)
    # one exact vector vs its 1-bit replica
    a = clamp((t - t0) * 2.5) * (1 - prog(t, at("funnel", "copy"), at("funnel", "copy") + 0.6))
    if a > 0:
        paper_rect(c, 600, 260, 70, 560, "tex-kraft.png", radius=6, alpha=a)
        paper_rect(c, 900, 260 + 525, 70, 35, "tex-teal.png", radius=4, alpha=a * prog(t, at("funnel", "one"), at("funnel", "one") + 0.4))
        draw_text(c, "exact vector", 635, 880, 30, 500, INK, a, "center", "mono")
        draw_text(c, "1-bit copy", 935, 880, 30, 600, TEAL_INK, a * prog(t, at("funnel", "one"), at("funnel", "one") + 0.4), "center", "mono")
    # the index as rows of 1-bit strips, scanned by a light
    scan0 = at("funnel", "copy")
    cols, rows = 24, 16
    if t > scan0 - 0.2:
        ga = clamp((t - scan0 + 0.2) * 3)
        sweep = prog(t, scan0 + 0.4, at("funnel", "and") - 0.1)
        top = sorted(range(cols * rows), key=lambda i: (i * 7919) % 389)[:10]
        lift = expo_io(prog(t, at("funnel", "rescores") - 0.2, at("funnel", "rescores") + 0.6))
        grow = expo_io(prog(t, at("funnel", "candidates") - 0.1, at("funnel", "exactly") + 0.5))
        for i in range(cols * rows):
            q, r = i % cols, i // cols
            x = 230 + q * 62; y = 300 + r * 30
            hot = i in top
            if hot and lift > 0:
                rank = top.index(i)
                tx = 230 + rank * 150; ty = 300
                x = lerp(x, tx, lift); y = lerp(y, ty, lift)
                hh = lerp(14, 14 * 16, grow); ww = lerp(52, 120, lift)
                paper_rect(c, x, y, ww, hh, "tex-kraft.png" if grow > 0.5 else "tex-teal.png", radius=4, alpha=ga, shadow=0.6)
                if grow > 0.6:
                    for ln in range(9):
                        line(c, x + 16, y + 22 + ln * hh / 10, x + ww - 16 - ((ln * 37) % 30), y + 22 + ln * hh / 10, "#6B4A33", 3, 0.5 * prog(grow, 0.6, 1))
                    draw_text(c, f"#{rank + 1}", x + ww / 2, y + hh + 40, 30, 650, INK, prog(grow, 0.6, 1), "center", "mono")
            else:
                lit = clamp(1 - abs(q / cols - sweep) * 8) if sweep < 1 else 0.0
                aa = ga * (1 - lift * 0.75)
                paper_rect(c, x, y, 52, 14, "tex-teal.png" if lit > 0.5 else "tex-kraft.png", radius=3,
                           alpha=aa * (1.0 if lit > 0.5 else 0.7), shadow=0.25, edge=False)
        if 0 < sweep < 1:
            sx = 230 + sweep * cols * 62
            glow(c, sx, 540, 260, TEAL, 0.35)
    headline(c, "There is no vector database.", 160, 170, 68, t, t0 - 0.05, INK)
    caption(c, "Each query scans a 1-bit copy, 16× smaller.", 160, 228, t, at("funnel", "copy") - 0.1, 32, "#5E4B3A", kind="mono")
    headline(c, "The best candidates are rescored exactly.", 160, 985, 54, t, at("funnel", "rescores") - 0.1, TEAL_INK)
    caption(c, "It recovers 95.8% of the exact top 10.", 160, 1038, t, at("funnel", "exactly"), 30, "#5E4B3A",
            kind="mono")

# ---------------------------------------------------------------- 7. agents
_rib = {}
def ribbon(c, path, width):
    """A cut-paper ribbon along `path`: contact shadow, kraft band with a lit edge, a teal thread."""
    sh = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width + 6, Color=skia.Color(15, 9, 5, 150))
    sh.setStrokeCap(skia.Paint.kRound_Cap); sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 9))
    c.save(); c.translate(6, 11); c.drawPath(path, sh); c.restore()
    if "k" not in _rib:   # small swatches: a shader is copied into every Paint, so keep them light
        _rib["k"] = img("tex-kraft-s.png").makeShader(skia.TileMode.kMirror, skia.TileMode.kMirror, SAMPLE, skia.Matrix.Scale(0.6, 0.6))
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width)
    p.setStrokeCap(skia.Paint.kRound_Cap); p.setShader(_rib["k"]); c.drawPath(path, p)
    e = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, Color=skia.Color(255, 245, 225, 40))
    e.setStrokeCap(skia.Paint.kRound_Cap); c.save(); c.translate(-1.5, -2.5); c.drawPath(path, e); c.restore()
    t = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=5, Color4f=rgb(TEAL, 0.95))
    t.setStrokeCap(skia.Paint.kRound_Cap); c.drawPath(path, t)

AGENTS = [("search_files", "“q3 board deck”", -1), ("find_similar", "“logo_final.png”", 1),
          ("search_files", "“Rechnung März”", -1)]
def s_agents(c, t):
    w = Tall(keys(t, [(AGENTS0, CHAMBER[1] - 200.0), (PAPER0 + 0.4, CHAMBER[1] - 420.0)], ease_io), keys(t, [(AGENTS0, 1.25), (PAPER0 + 0.4, 0.95)], ease_io),
             CHAMBER[0])
    w.draw(c)
    s = w.s
    mx, my = w.P(CHAMBER[0], CHAMBER[1] + 380)
    draw_image(c, sprite("mole-listen-cut.png", 900), mx, my, h=640 * s, anchor=(0.5, 0.95), shadow=0.6)
    t0 = at("agents", "agents")
    ends = [(170, 300), (1750, 250), (1700, 820)]
    for k, ((fn, arg, side), (ex, ey)) in enumerate(zip(AGENTS, ends)):
        tk = t0 + k * 0.45
        u = ease_io(prog(t, tk, tk + 1.1))
        if u <= 0: continue
        # a tunnel: a thick soil path with a lit inner edge, drawn as it is dug
        sx, sy = w.P(CHAMBER[0] + side * 520, CHAMBER[1] - 200 + k * 160)
        path = skia.Path(); path.moveTo(sx, sy)
        mx2 = (sx + ex) / 2; path.cubicTo(mx2, sy + 120, mx2, ey - 120, ex, ey)
        meas = skia.PathMeasure(path, False); seg = skia.Path(); meas.getSegment(0, meas.getLength() * u, seg, True)
        ribbon(c, seg, 40)
        # the agent at the end of it: a small terminal card
        ca = prog(t, tk + 0.9, tk + 1.3)
        if ca > 0:
            cw, chh = 580, 156
            x0 = ex - cw / 2 if k != 0 else ex - 60
            x0 = min(max(x0, 40), W - cw - 40)
            paper_rect(c, x0, ey - chh / 2, cw, chh, "tex-ink.png", radius=12, alpha=ca, shadow=0.9)
            draw_text(c, "agent · MCP", x0 + 24, ey - chh / 2 + 40, 24, 600, "#9FD8D2", ca, "left", "mono")
            call = f"{fn}({arg})"
            n = int(len(call) * prog(t, tk + 1.1, tk + 1.9))
            draw_text(c, call[:n], x0 + 24, ey + 10, 27, 500, CREAM, ca, "left", "mono")
            res = ["12 results · 9 ms", "40 similar · 6 ms", "7 results · 10 ms"][k]
            draw_text(c, res, x0 + 24, ey + 52, 24, 500, "#CDBBA3", prog(t, tk + 2.0, tk + 2.3) * ca, "left", "mono")
    # the boundary: all of it inside the machine
    bt = at("agents", "never") - 0.2
    if t > bt:
        u = ease_io(prog(t, bt, bt + 1.2))
        r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(30, 30, W - 60, H - 60), 28, 28)
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3, Color4f=rgb(TEAL, 0.9))
        p.setPathEffect(skia.DashPathEffect.Make([14, 12], -t * 30))
        pth = skia.Path(); pth.addRRect(r)
        meas = skia.PathMeasure(pth, False); seg = skia.Path(); meas.getSegment(0, meas.getLength() * u, seg, True)
        c.drawPath(seg, p)
        chip(c, "served on localhost only", W / 2 - 230, H - 50, t, bt + 0.8, 30, "tex-teal.png", INK)
    headline(c, "Local agents search the same index.", W / 2, 150, 72, t, word("agents", 0) - 0.05, CREAM, align="center")

# ---------------------------------------------------------------- 8. the paper
def s_paper(c, t):
    plate(c, "bg-strata.png", 2752, 1700, 1.25)
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Color=skia.Color(25, 16, 10, 120)))
    t0 = word("paper", 0) - 0.4
    u = expo_out(prog(t, t0, t0 + 1.1))
    pg = img(os.path.join(WORK, "paper", "p1-01.png"), 1600)
    ph = 900; pw = ph * pg.width() / pg.height()
    cx = 640; cy = lerp(H + ph / 2, 560, u)
    rot = lerp(9, -2.5, u)
    c.save(); c.translate(cx, cy); c.rotate(rot)
    sp = skia.Paint(AntiAlias=True, Color=skia.Color(46, 30, 16, 90))
    sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 22))
    c.drawRect(skia.Rect.MakeXYWH(-pw / 2 + 14, -ph / 2 + 22, pw, ph), sp)
    c.drawImageRect(pg, skia.Rect.MakeXYWH(-pw / 2, -ph / 2, pw, ph), SAMPLE, skia.Paint(AntiAlias=True))
    # the seal stamps onto the page
    st = at("paper", "NeurIPS") - 0.15
    d = t - st
    if d > 0:
        k = 1 + 1.4 * math.exp(-9 * d)
        draw_image(c, sprite("seal-cut.png", 700), pw / 2 - 40, ph / 2 - 120, h=240 * k, rot=-12, alpha=clamp(d * 6), shadow=0.8)
    c.restore()
    headline(c, "The design is published.", 1100, 300, 76, t, word("paper", 0), CREAM)
    caption(c, "NeurIPS 2026", 1100, 400, t, at("paper", "NeurIPS") + 0.1, 52, TEAL, weight=650)
    caption(c, "Workshop on On-Device Intelligence", 1100, 452, t, at("paper", "workshop"), 34, "#E3D2BA", weight=500)
    stats = [("9.7 ms", "a text query"), ("8.5 ms", "saving one edit"), ("83,105", "tokens/s indexing"), ("5 Macs", "M2 to M3 Ultra")]
    for i, (big, small) in enumerate(stats):
        ts = word("paper", -1) + 0.2 + i * 0.32
        x = 1100 + (i % 2) * 360; y = 600 + (i // 2) * 150
        chip_u = expo_out(prog(t, ts, ts + 0.5))
        if chip_u > 0:
            draw_text(c, big, x, y + (1 - chip_u) * 20, 60, 650, CREAM, chip_u, "left", "mono")
            draw_text(c, small, x, y + 44 + (1 - chip_u) * 20, 28, 500, "#E3D2BA", chip_u, "left", "mono")

# ---------------------------------------------------------------- 9. the app
CLIPS = None
def clips():
    global CLIPS
    if CLIPS is None:
        CLIPS = {"T1": Footage("T1-mountain", 9.0, 21.0), "T2": Footage("T2-moon", 13.4, 21.0),
                 "T3": Footage("T3-cats", 11.8, 22.0), "T4": Footage("T4-browse", 9.6, 19.0),
                 "T5": Footage("T5-history", 9.6, 17.0)}
    return CLIPS

L0 = T.LIFT
CUTS = [L0, L0 + 2 * BAR, L0 + 4 * BAR, L0 + 6 * BAR, L0 + 8 * BAR, L0 + 10 * BAR]   # 103.6 .. 127.2
LAP_SCREEN = (171, 72, 1110, 660)          # the laptop sprite's screen panel, in its trimmed pixels

def montage_frame(t):
    """(clip, take-time, caption, focus) for film time t inside the app montage."""
    if t < CUTS[1]:
        u = t - L0
        return "T1", (1.6 + u * 0.6) if u < 1.6 else (2.56 + (u - 1.6) * 2.2) if u < 3.4 else (6.52 + (u - 3.4)), "Search by meaning.", (0.5, 0.5)
    if t < CUTS[2]:   return "T2", 0.6 + (t - CUTS[1]), "Search in any language.", (0.62, 0.42)
    if t < CUTS[3]:
        u = t - CUTS[2]
        return "T3", (0.9 + u * 0.55) if u < 1.6 else (3.6 + (u - 1.6) * 1.0), "Find similar files.", (0.55, 0.45)
    if t < CUTS[4]:   return "T4", 1.0 + (t - CUTS[3]) * 1.15, "Browse folders as in Finder.", (0.5, 0.45)
    return "T5", 0.8 + (t - CUTS[4]), "Replay past searches.", (0.45, 0.5)

def s_app(c, t):
    bg = img("bg-blur.png")
    c.drawImageRect(bg, skia.Rect.MakeWH(W, H), SAMPLE, skia.Paint())
    cl = clips()
    clip, tt, cap, focus = montage_frame(t)
    frame = cl[clip].frame(tt)
    lap = sprite("laptop-open-cut.png", 1400)
    # 0 = the paper laptop on the desk, 1 = its screen fills the frame; in at the lift, out at the end
    inn = expo_io(prog(t, L0 + 0.15, L0 + 1.9))
    out = expo_io(prog(t, CUTS[5] - 0.6, CUTS[5] + 1.3))
    push = inn * (1 - out)
    breathe = 1 + 0.02 * ((t - L0) % (2 * BAR)) / (2 * BAR)
    full = 1.22 * breathe                                      # window scale when it fills the frame
    scr_w = LAP_SCREEN[2] - LAP_SCREEN[0]
    small_lw = 900                                             # laptop width on the desk
    lw = lerp(small_lw, frame.width() * full * lap.width() / scr_w, push)
    sc = lw / lap.width()
    sx0 = (LAP_SCREEN[0] + LAP_SCREEN[2]) / 2 * sc; sy0 = (LAP_SCREEN[1] + LAP_SCREEN[3]) / 2 * sc
    lx = W / 2 - sx0 + lw / 2; ly = lerp(H / 2 + 70, H / 2, push) - sy0 + lw * lap.height() / lap.width() / 2
    if push < 0.999:
        draw_image(c, lap, lx, ly, w=lw, alpha=1 - prog(push, 0.8, 1.0), shadow=0.7)
    s_win = scr_w * sc / frame.width()
    # punch-in: (start, end) in film time, field centre in window pixels, zoom
    pz = 0.0
    for a0, a1 in [(L0 + 1.9, L0 + 3.9), (CUTS[1] - 0.05, CUTS[1] + 1.5)]:
        if a0 - 0.5 < t < a1 + 0.6:
            pz = max(pz, expo_io(prog(t, a0 - 0.5, a0)) * (1 - expo_io(prog(t, a1, a1 + 0.6))))
    if pz > 0:
        fx, fy = 1130, 26
        z = 1 + 1.25 * pz
        c.save(); c.translate(W / 2, H / 2); c.scale(z, z)
        c.translate(-W / 2 - (fx - frame.width() / 2) * s_win * pz * 0.9, -H / 2 - (fy - frame.height() / 2) * s_win * pz * 0.75)
    lh = lw * lap.height() / lap.width()
    draw_window(c, frame, lx - lw / 2 + sx0, ly - lh / 2 + sy0, s_win, 1.0, radius=25, shadow=push)
    if pz > 0: c.restore()
    if push > 0.9:
        ci = next((i for i in range(5) if t < CUTS[i + 1]), 4)
        chip(c, cap, 110, 1012, t, CUTS[ci] + (0.5 if ci == 0 else 0.1), 38, "tex-cream.png", INK, "sans", 650,
             prog(push, 0.9, 1.0))

# ---------------------------------------------------------------- 10. the end
def s_end(c, t):
    plate(c, "bg-paper.png", 2752, 1536, 1.03 + 0.012 * prog(t, 128, 145))
    t0 = word("end", 0)
    d = t - (t0 - 0.3)
    if d > 0:
        k = spring(d * 0.8, 1.2, 4.0)
        bob = math.sin(max(0, t - t0 - 1.5) * 2.2) * 4
        draw_image(c, sprite("mole-wave-cut.png", 1000), 560, 840 + (1 - k) * 600 + bob, h=600, anchor=(0.5, 0.95), shadow=0.7)
    draw_text(c, "Omni", 820, 520, 200, 720, INK, expo_out(prog(t, t0, t0 + 0.8)), "left",
              scale=1 + 0.06 * math.exp(-6 * max(0, t - t0)))
    reveal_words(c, "Free and open source.", 826, 610, 46, t, at("end", "free") - 0.1, 0.12, 0.6, 550, MUTED)
    caption(c, "hanxiao.io/omni", 826, 690, t, T.CHORD - 0.1, 34, TEAL_INK, kind="mono", weight=600)
    caption(c, "macOS 14+ · Apple silicon · Apache 2.0", 826, 746, t, T.CHORD + 0.3, 28, MUTED, kind="mono")

# ---------------------------------------------------------------- the edit
SHOTS = [
    (0.0, 7.7, s_old),
    (6.5, 16.4, s_cloud),
    (15.0, 30.7, s_down),
    (30.0, MESSY0, s_easy),
    (MESSY0, ASIDE0 + 0.4, s_messy),
    (ASIDE0, FUNNEL0 + 0.4, s_aside),
    (FUNNEL0, AGENTS0 + 0.4, s_funnel),
    (AGENTS0, PAPER0 + 0.4, s_agents),
    (PAPER0, L0 + 0.15, s_paper),
    (L0 - 0.3, END0 + 0.8, s_app),
    (END0, T.LENGTH + 0.1, s_end),
]
# cross-dissolves where shots overlap and do not carry their own transition
DISSOLVE = {(s_down, s_easy): (30.05, 30.55), (s_messy, s_aside): (ASIDE0, ASIDE0 + 0.4),
            (s_aside, s_funnel): (FUNNEL0, FUNNEL0 + 0.4), (s_funnel, s_agents): (AGENTS0, AGENTS0 + 0.4),
            (s_agents, s_paper): (PAPER0, PAPER0 + 0.4), (s_paper, s_app): (L0 - 0.3, L0 + 0.15),
            (s_app, s_end): (END0, END0 + 0.8)}

def fade_to_black(t):
    return prog(t, 0.0, 0.7) * (1 - prog(t, T.LENGTH - 1.3, T.LENGTH))

# shots that move fast enough to need motion blur
BLUR = [(6.5, 7.8), (15.0, 16.5), (18.0, 26.0), (MESSY0, MESSY0 + 1.4),
        (at("once", "each") - 0.3, at("once", "once") + 0.5), (at("once", "only") - 0.4, at("once", "only") + 2.2),
        (L0 - 0.3, L0 + 2.1), (L0 + 1.3, L0 + 2.0), (L0 + 3.8, L0 + 4.6), (CUTS[1] - 0.6, CUTS[1] + 0.1),
        (CUTS[1] + 1.4, CUTS[1] + 2.2), (CUTS[5] - 0.7, CUTS[5] + 1.4)]

# the whips: a frame there moves 100+ px, which 5 sub-frames render as visible copies
WHIP = [(6.5, 7.8), (15.0, 16.5), (MESSY0, MESSY0 + 1.4), (L0 - 0.3, L0 + 2.1), (CUTS[5] - 0.7, CUTS[5] + 1.4),
        (18.3, 20.3), (21.3, 25.6)]
def subframes(t):
    if any(a <= t <= b for a, b in WHIP): return 16
    return 5 if any(a <= t <= b for a, b in BLUR) else 1
