"""Generate the hard-page corpus the OCR port is graded on.

The certified six/seven pages are clean synthetic type. That corpus cannot discriminate between
builds - easy pages agree whatever you do to the weights - and it says nothing about the inputs a
document tool actually meets. This makes the pages that DO discriminate:

  handwriting      script/marker faces, which the model has to read as glyphs rather than as type
  spreadsheet      dense ruled numeric grids, thousands separators, parenthesised negatives
  receipt          narrow thermal-printer layout, faded, monospaced
  form             boxed fields with handwritten entries
  degraded         a clean page put through JPEG, blur, noise, skew, shadow and bleed-through
  photo            perspective warp plus an illumination gradient, i.e. a phone snap of paper

Every page carries CHECKABLE content - unique row ids, exact totals - so a transcription can be
scored on whether the facts survived, not only on whether it matches a reference string.

Handwriting here is font-rendered, not real pen strokes. That is stated rather than glossed: it
is a reproducible proxy for cursive/marker glyph shapes, and the degradation pipeline is what
makes these pages genuinely hard.

    python Tools/ocr/make_hard_pages.py --out bench/hard2
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = {
    "hand_bradley": "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "hand_chalk": "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "hand_snell": "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "hand_comic": "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "mono": "/System/Library/Fonts/Menlo.ttc",
    "serif": "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "sans": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "sans_bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS[name]
    if path.endswith(".ttc"):
        return ImageFont.truetype(path, size, index=0)
    return ImageFont.truetype(path, size)


# --------------------------------------------------------------------------- degradation


def degrade(img: Image.Image, *, jpeg: int | None = 25, blur: float = 0.8, noise: int = 12,
            skew_deg: float = 0.0, shadow: bool = False, bleed: Image.Image | None = None,
            contrast: float = 1.0, seed: int = 0) -> Image.Image:
    """Put a clean render through a scanner's worth of damage, in the order a scanner does it.

    Order matters and is not arbitrary: bleed-through and skew happen on the physical page,
    illumination and blur happen in the optics, noise in the sensor, and JPEG last in the file.
    Applying JPEG before blur would smooth away exactly the artefacts it is there to introduce.
    """
    rng = random.Random(seed)
    out = img.convert("RGB")

    if bleed is not None:
        # Ink from the reverse side, mirrored and faint - the classic thin-paper scan artefact.
        back = bleed.convert("RGB").resize(out.size).transpose(Image.FLIP_LEFT_RIGHT)
        out = Image.blend(out, Image.blend(out, back, 0.5), 0.22)

    if skew_deg:
        out = out.rotate(skew_deg, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    if shadow:
        # A soft illumination gradient across the page, as if lit from one side.
        w, h = out.size
        grad = Image.new("L", (w, h))
        gd = ImageDraw.Draw(grad)
        for x in range(0, w, 4):
            v = int(255 - 90 * (x / w) ** 1.6)
            gd.rectangle([x, 0, x + 4, h], fill=v)
        grad = grad.filter(ImageFilter.GaussianBlur(40))
        out = Image.composite(out, Image.new("RGB", out.size, (0, 0, 0)), grad)

    if contrast != 1.0:
        from PIL import ImageEnhance
        out = ImageEnhance.Contrast(out).enhance(contrast)

    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))

    if noise:
        px = out.load()
        w, h = out.size
        for y in range(h):
            for x in range(w):
                n = rng.randint(-noise, noise)
                r, g, b = px[x, y]
                px[x, y] = (min(255, max(0, r + n)), min(255, max(0, g + n)), min(255, max(0, b + n)))

    if jpeg:
        import io
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=jpeg)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    return out


def perspective(img: Image.Image, strength: float = 0.06) -> Image.Image:
    """A mild keystone, the shape a phone photo of a page has."""
    w, h = img.size
    dx = int(w * strength)
    dy = int(h * strength * 0.4)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx, dy // 2), (w - dx // 2, 0), (w, h - dy), (dx // 2, h)]

    # Solve the 8-coefficient projective transform mapping dst -> src (PIL wants the inverse).
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    import numpy as np
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(a, b)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                         fillcolor=(255, 255, 255))


# --------------------------------------------------------------------------- pages


def page_handwriting() -> Image.Image:
    """A handwritten meeting note: cursive body, marker headings, checkable figures."""
    img = Image.new("RGB", (1240, 1754), (252, 250, 244))
    d = ImageDraw.Draw(img)
    # Ruled paper.
    for y in range(220, 1700, 52):
        d.line([(90, y), (1150, y)], fill=(206, 216, 232), width=1)
    d.line([(150, 120), (150, 1700)], fill=(232, 190, 190), width=2)

    d.text((170, 130), "Site Visit Notes - 14 March", font=font("hand_bradley", 44), fill=(24, 32, 60))
    lines = [
        ("hand_snell", 36, "Batch A-117 arrived at 09:40, seal intact."),
        ("hand_snell", 36, "Counted 48 crates, 3 damaged (crates 12, 29, 41)."),
        ("hand_snell", 36, "Moisture reading 12.4 percent, within tolerance."),
        ("hand_snell", 36, "Reject rate this run: 6.25 percent."),
        ("hand_snell", 36, "Supervisor: M. Okonkwo. Signed off 11:15."),
        ("hand_comic", 32, "Follow up: reorder 3 crates before 21 March."),
        ("hand_comic", 32, "Invoice ref INV-2026-0417, total EUR 8,412.60"),
    ]
    y = 240
    for name, size, text in lines:
        d.text((175, y), text, font=font(name, size), fill=(28, 36, 72))
        y += 104

    d.text((175, y + 40), "TOTALS", font=font("hand_bradley", 40), fill=(140, 30, 30))
    y += 120
    for label, value in (("Crates", "48"), ("Damaged", "3"), ("Net", "45"),
                         ("Value EUR", "8,412.60")):
        d.text((200, y), label, font=font("hand_snell", 34), fill=(28, 36, 72))
        d.text((640, y), value, font=font("hand_bradley", 34), fill=(28, 36, 72))
        y += 62
    return img


def page_spreadsheet() -> Image.Image:
    """A ruled financial grid: right-aligned numerics, separators, parenthesised negatives."""
    img = Image.new("RGB", (1600, 1200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    head = font("sans_bold", 20)
    body = font("sans", 19)
    mono = font("mono", 18)

    d.text((40, 28), "Quarterly Reconciliation - Region NORTH", font=font("sans_bold", 26),
           fill=(0, 0, 0))

    cols = [40, 300, 560, 820, 1080, 1340, 1560]
    headers = ["Account", "Q1", "Q2", "Q3", "Q4", "Total"]
    rows = [
        ("4010 Revenue", "1,204,880", "1,318,455", "1,190,002", "1,402,771", "5,116,108"),
        ("4020 Returns", "(48,220)", "(51,004)", "(39,880)", "(60,115)", "(199,219)"),
        ("5010 COGS", "(602,440)", "(659,228)", "(595,001)", "(701,386)", "(2,558,055)"),
        ("6100 Salaries", "(310,500)", "(310,500)", "(318,750)", "(318,750)", "(1,258,500)"),
        ("6200 Rent", "(48,000)", "(48,000)", "(48,000)", "(48,000)", "(192,000)"),
        ("6310 Utilities", "(11,204)", "(9,880)", "(8,455)", "(12,901)", "(42,440)"),
        ("6400 Travel", "(22,118)", "(31,006)", "(18,772)", "(27,540)", "(99,436)"),
        ("7010 Interest", "(4,010)", "(4,010)", "(3,880)", "(3,880)", "(15,780)"),
        ("8000 Tax", "(38,900)", "(45,110)", "(36,220)", "(48,004)", "(168,234)"),
        ("Net Result", "119,488", "159,717", "121,044", "182,195", "582,444"),
    ]
    y = 80
    d.rectangle([cols[0], y, cols[-1], y + 34], fill=(232, 236, 242))
    for i, h in enumerate(headers):
        x = cols[i] + 8 if i == 0 else cols[i + 1] - 12 - d.textlength(h, font=head)
        d.text((x, y + 8), h, font=head, fill=(0, 0, 0))
    y += 34
    for r, row in enumerate(rows):
        if r == len(rows) - 1:
            d.line([(cols[0], y), (cols[-1], y)], fill=(0, 0, 0), width=2)
        bg = (247, 249, 251) if r % 2 else (255, 255, 255)
        d.rectangle([cols[0], y, cols[-1], y + 32], fill=bg)
        d.text((cols[0] + 8, y + 7), row[0], font=body, fill=(0, 0, 0))
        for i, cell in enumerate(row[1:]):
            x = cols[i + 2] - 12 - d.textlength(cell, font=mono)
            d.text((x, y + 7), cell, font=mono, fill=(0, 0, 0))
        y += 32
    for c in cols:
        d.line([(c, 80), (c, y)], fill=(190, 198, 210), width=1)
    d.line([(cols[0], y), (cols[-1], y)], fill=(190, 198, 210), width=1)

    d.text((40, y + 24), "Checksum: rows 10, columns 5, grand total 582,444",
           font=font("sans", 18), fill=(60, 60, 60))
    return img


def page_receipt() -> Image.Image:
    """Thermal receipt: narrow, monospaced, faded, the aspect ratio that stresses tiling."""
    img = Image.new("RGB", (620, 1500), (250, 249, 245))
    d = ImageDraw.Draw(img)
    mono = font("mono", 20)
    small = font("mono", 17)
    d.text((150, 40), "HAFENMARKT GmbH", font=font("mono", 26), fill=(40, 40, 40))
    d.text((160, 78), "Speicherstadt 14", font=small, fill=(60, 60, 60))
    d.text((150, 104), "20457 Hamburg  DE", font=small, fill=(60, 60, 60))
    d.text((40, 150), "-" * 44, font=small, fill=(90, 90, 90))
    items = [
        ("Roggenbrot 750g", "1", "3.49"), ("Bio Milch 1L", "3", "4.47"),
        ("Kaffee Bohnen 500g", "1", "12.99"), ("Tomaten lose kg", "0.842", "2.94"),
        ("Kaese Gouda 200g", "2", "5.98"), ("Olivenoel 750ml", "1", "9.49"),
        ("Apfelsaft 1L", "4", "5.16"), ("Nudeln 500g", "3", "3.87"),
    ]
    y = 178
    for name, qty, total in items:
        d.text((40, y), name[:24], font=mono, fill=(35, 35, 35))
        d.text((400, y), qty, font=small, fill=(70, 70, 70))
        d.text((500, y), total, font=mono, fill=(35, 35, 35))
        y += 34
    d.text((40, y + 8), "-" * 44, font=small, fill=(90, 90, 90))
    y += 44
    for label, value in (("ZWISCHENSUMME", "48.39"), ("MWST 7%", "2.64"),
                         ("MWST 19%", "3.21"), ("SUMME EUR", "48.39")):
        d.text((40, y), label, font=mono, fill=(35, 35, 35))
        d.text((470, y), value, font=mono, fill=(35, 35, 35))
        y += 34
    d.text((40, y + 20), "Beleg 2026-03-14 17:42", font=small, fill=(70, 70, 70))
    d.text((40, y + 48), "Kasse 03  Bon-Nr 004917", font=small, fill=(70, 70, 70))
    d.text((40, y + 76), "* * * VIELEN DANK * * *", font=small, fill=(70, 70, 70))
    return img


def page_form() -> Image.Image:
    """A boxed form with handwritten entries - printed labels, cursive values."""
    img = Image.new("RGB", (1240, 1600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    label = font("sans", 20)
    hand = font("hand_bradley", 30)
    d.text((60, 50), "CUSTOMS DECLARATION - FORM C-88", font=font("sans_bold", 28), fill=(0, 0, 0))
    d.line([(60, 92), (1180, 92)], fill=(0, 0, 0), width=2)

    fields = [
        ("1. Consignor", "Baltic Freight OU"), ("2. Consignee", "Meridian Imports Ltd"),
        ("3. Country of origin", "Estonia"), ("4. Gross weight kg", "1,284.5"),
        ("5. Packages", "48"), ("6. Invoice value EUR", "8,412.60"),
        ("7. Commodity code", "8471 30 00"), ("8. Transport doc", "CMR-2026-88417"),
        ("9. Declarant", "M. Okonkwo"), ("10. Date", "14 March 2026"),
    ]
    y = 130
    for name, value in fields:
        d.rectangle([60, y, 1180, y + 108], outline=(0, 0, 0), width=1)
        d.text((72, y + 8), name, font=label, fill=(40, 40, 40))
        d.text((110, y + 42), value, font=hand, fill=(20, 30, 90))
        y += 118
    d.text((72, y + 16), "Signature", font=label, fill=(40, 40, 40))
    d.text((300, y + 6), "M. Okonkwo", font=font("hand_snell", 40), fill=(20, 30, 90))
    return img


def page_twocolumn_small() -> Image.Image:
    """Two dense columns of 7pt mixed-script body text - the legibility floor."""
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 50), "Proceedings, Volume 12", font=font("serif", 26), fill=(0, 0, 0))
    body = font("serif", 15)
    para = [
        "The encoder projects each window into a shared latent space before the",
        "router assigns tokens to experts. We observe that routing entropy falls",
        "monotonically with depth, from 3.81 nats at layer 1 to 1.02 at layer 12.",
        "Table 4 reports the ablation over group sizes 32, 64 and 128 with the",
        "corresponding character error rates 0.014, 0.021 and 0.038 respectively.",
        "Multilingual segments: 文档识别 and 日本語テキスト and Ελληνικά.",
        "Numeric spans such as 1,204,880 and (48,220) are preserved verbatim.",
    ]
    for col in (0, 1):
        y = 110
        for rep in range(14):
            for line in para:
                d.text((60 + col * 590, y), line, font=body, fill=(0, 0, 0))
                y += 20
            y += 6
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/hard2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean_hand = page_handwriting()
    clean_sheet = page_spreadsheet()
    clean_receipt = page_receipt()
    clean_form = page_form()
    clean_cols = page_twocolumn_small()

    pages = {
        # Clean renders first: they separate "the model cannot read this style" from
        # "the model cannot read this damage".
        "hand_clean": clean_hand,
        "sheet_clean": clean_sheet,
        "receipt_clean": clean_receipt,
        "form_clean": clean_form,
        "cols7pt_clean": clean_cols,
        # Then the same content, damaged.
        "hand_scan": degrade(clean_hand, jpeg=22, blur=1.1, noise=14, skew_deg=1.4,
                             shadow=True, seed=1),
        "sheet_scan": degrade(clean_sheet, jpeg=18, blur=0.9, noise=16, skew_deg=-0.8,
                              bleed=clean_cols, seed=2),
        "receipt_faded": degrade(clean_receipt, jpeg=30, blur=1.3, noise=10, contrast=0.55,
                                 seed=3),
        "form_photo": degrade(perspective(clean_form, 0.05), jpeg=26, blur=0.7, noise=12,
                              shadow=True, seed=4),
        "cols7pt_scan": degrade(clean_cols, jpeg=20, blur=1.0, noise=15, skew_deg=0.6, seed=5),
    }
    for name, img in pages.items():
        path = out / f"{name}.png"
        img.save(path)
        print(f"  {path}  {img.size[0]}x{img.size[1]}")
    print(f"{len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()
