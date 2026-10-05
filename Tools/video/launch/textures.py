# Paper swatches for procedural shapes, all from ONE generated sheet so every colour shares its grain.
# The fiber detail is the sheet's luminance over its own blur; each swatch is a flat ink times it.
import os
import numpy as np
from PIL import Image, ImageFilter

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "launch", "assets")
INKS = {"cream": (243, 237, 226), "kraft": (201, 162, 122), "teal": (127, 216, 208),
        "soil": (107, 74, 51), "deep": (58, 42, 32), "ink": (40, 38, 36), "white": (250, 248, 243)}

src = Image.open(os.path.join(A, "bg-paper.png")).convert("L")
w, h = src.size
crop = src.crop((w // 2 - 1024, h // 2 - 1024, w // 2 + 1024, h // 2 + 1024))
lum = np.asarray(crop, np.float32)
low = np.asarray(crop.filter(ImageFilter.GaussianBlur(24)), np.float32)
detail = lum / np.maximum(low, 1)                       # ~1.0, the fiber
detail = 1 + (detail - 1) * 2.2                        # a touch more tooth than the photo
# tile seamlessly: mirror-blend the edges
detail = (detail + detail[:, ::-1] + detail[::-1, :] + detail[::-1, ::-1]) / 4
for name, rgb in INKS.items():
    out = np.clip(detail[..., None] * np.array(rgb, np.float32), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(A, f"tex-{name}.png"))
# the dark backdrop: deep soil at full frame with the same fiber and a soft vignette
full = np.asarray(src, np.float32)
fl = np.asarray(src.filter(ImageFilter.GaussianBlur(24)), np.float32)
d = 1 + (full / np.maximum(fl, 1) - 1) * 2.2
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
vig = 1 - 0.35 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2.2)
dark = np.clip(d[..., None] * vig[..., None] * np.array(INKS["deep"], np.float32), 0, 255).astype(np.uint8)
Image.fromarray(dark).save(os.path.join(A, "bg-dark.png"))
print("ok", list(INKS))
