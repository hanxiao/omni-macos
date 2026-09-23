# Build a simplified, symbol-style Omni mark from the icon's silhouette plus a few drawn features.
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFilter
REPO = sys.argv[1]; OUT = sys.argv[2]
src = Image.open(f"{REPO}/App/AppIcon.icon/Assets/foreground.png").convert("RGBA"); m = src.crop(src.getbbox()); m.thumbnail((512, 512), Image.LANCZOS)
K = 4; W, H = m.size[0] * K, m.size[1] * K
A = m.getchannel("A").resize((W, H), Image.LANCZOS)
def smooth(mask, r):   # blur + threshold: removes fur and pebbles, keeps the outline
    return mask.filter(ImageFilter.GaussianBlur(r * K)).point(lambda v: 255 if v > 127 else 0)
sil = smooth(A.point(lambda v: 255 if v > 128 else 0), 7)
y = np.arange(H)[:, None] / K
body_raw = np.array(sil) > 0
body_raw &= (y < 318)
body = smooth(Image.fromarray((body_raw * 255).astype(np.uint8)), 10)
mound = np.array(sil) > 0
mound &= (y > 262)
mound = smooth(Image.fromarray((mound * 255).astype(np.uint8)), 16)
gap = body.filter(ImageFilter.MaxFilter(1 + 2 * 7 * K // 2 * 2 + 1)) if False else body.filter(ImageFilter.GaussianBlur(7 * K)).point(lambda v: 255 if v > 8 else 0)
mound_arr = (np.array(mound) > 0) & ~(np.array(gap) > 0)
body_arr = np.array(body) > 0
# features cut out of the body, as clean shapes
feat = Image.new("L", (W, H), 0); d = ImageDraw.Draw(feat)
def circ(cx, cy, r): d.ellipse([(cx - r) * K, (cy - r) * K, (cx + r) * K, (cy + r) * K], fill=255)
circ(176, 110, 17); circ(295, 115, 17)                                   # eyes
d.ellipse([196 * K, 138 * K, 246 * K, 170 * K], fill=255)                 # nose
d.rounded_rectangle([218 * K, 176 * K, 232 * K, 196 * K], 3 * K, fill=255) # teeth
feat_arr = np.array(feat) > 0
alpha = np.zeros((H, W), np.float32)
alpha[mound_arr] = 0.55
alpha[body_arr] = 1.0
alpha[body_arr & feat_arr] = 0.0
img = Image.fromarray((alpha * 255).astype(np.uint8)).resize((W // K, H // K), Image.LANCZOS)
out = Image.new("RGBA", img.size, (0, 0, 0, 0)); out.putalpha(img)
out.save(OUT)
