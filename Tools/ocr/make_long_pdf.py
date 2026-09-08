"""Build a long scanned-looking PDF for the OCR page pipeline.

Every page carries a UNIQUE, checkable marker (`PAGE 007 OF 040`, a per-page token, a per-page
total) so a run can be scored on whether all pages came back in the right order - not just on
whether it produced plausible text. A repeating page would make coverage unmeasurable, which is
a mistake this project has already made once.

Pages alternate between a ruled table, a prose block and a handwritten note so the document is
not uniformly easy, and everything goes through the same scan degradation as the hard corpus.

    python Tools/ocr/make_long_pdf.py --pages 40 --out bench/long_scan.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hard_pages import degrade, font  # noqa: E402


def page_table(n: int, total: int) -> Image.Image:
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((70, 60), f"PAGE {n:03d} OF {total:03d}", font=font("sans_bold", 30), fill=(0, 0, 0))
    d.text((70, 105), f"Ledger extract, section {n}", font=font("sans", 22), fill=(0, 0, 0))
    head = font("sans_bold", 19)
    mono = font("mono", 18)
    cols = [70, 460, 700, 940, 1170]
    y = 160
    d.rectangle([cols[0], y, cols[-1], y + 32], fill=(233, 237, 243))
    for i, h in enumerate(["Item", "Units", "Rate", "Amount"]):
        d.text((cols[i] + 8, y + 7), h, font=head, fill=(0, 0, 0))
    y += 32
    checksum = 0
    for r in range(22):
        units = (n * 37 + r * 11) % 97 + 3
        rate = (n * 13 + r * 7) % 89 + 11
        amount = units * rate
        checksum += amount
        d.rectangle([cols[0], y, cols[-1], y + 30], fill=(248, 250, 252) if r % 2 else (255, 255, 255))
        d.text((cols[0] + 8, y + 6), f"P{n:03d}-ITEM-{r:02d}", font=mono, fill=(0, 0, 0))
        for i, cell in enumerate([str(units), f"{rate}.00", f"{amount}.00"]):
            x = cols[i + 2] - 12 - d.textlength(cell, font=mono)
            d.text((x, y + 6), cell, font=mono, fill=(0, 0, 0))
        y += 30
    d.line([(cols[0], y), (cols[-1], y)], fill=(0, 0, 0), width=2)
    d.text((cols[0] + 8, y + 8), "PAGE TOTAL", font=head, fill=(0, 0, 0))
    tot = f"{checksum}.00"
    d.text((cols[-1] - 12 - d.textlength(tot, font=mono), y + 8), tot, font=mono, fill=(0, 0, 0))
    for c in cols:
        d.line([(c, 160), (c, y)], fill=(190, 198, 210), width=1)
    return img


def page_prose(n: int, total: int) -> Image.Image:
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((70, 60), f"PAGE {n:03d} OF {total:03d}", font=font("sans_bold", 30), fill=(0, 0, 0))
    d.text((70, 110), f"Section {n}. Findings", font=font("serif", 26), fill=(0, 0, 0))
    body = font("serif", 19)
    lines = [
        f"The measurement for section {n} was repeated {n % 5 + 3} times under identical",
        f"conditions. Mean throughput was {100 + n * 7}.{n % 10} units per second with a",
        f"standard deviation of {n % 9}.{(n * 3) % 10}. The reference identifier for this",
        f"run is REF-{n:04d}-{(n * 17) % 997:03d}, recorded at 1{n % 10}:{(n * 7) % 60:02d}.",
        "",
        "Observations were consistent with the model described in section 2. No",
        "anomalies were recorded in the control channel. The apparatus was",
        f"recalibrated after trial {n % 4 + 1} and the offset fell to {n % 3}.{n % 7} mm.",
        "",
        f"Conclusion for page {n}: within tolerance, no action required.",
    ]
    y = 170
    for _ in range(3):
        for line in lines:
            d.text((70, y), line, font=body, fill=(0, 0, 0))
            y += 30
        y += 14
    return img


def page_hand(n: int, total: int) -> Image.Image:
    img = Image.new("RGB", (1240, 1754), (252, 250, 244))
    d = ImageDraw.Draw(img)
    for yy in range(210, 1700, 54):
        d.line([(90, yy), (1150, yy)], fill=(208, 218, 234), width=1)
    d.text((100, 60), f"PAGE {n:03d} OF {total:03d}", font=font("sans_bold", 30), fill=(0, 0, 0))
    d.text((110, 120), f"Field notes - entry {n}", font=font("hand_bradley", 38), fill=(24, 32, 60))
    notes = [
        f"Sample {n} collected at depth {n % 12 + 1}.{n % 10} metres.",
        f"Temperature {12 + n % 9}.{n % 10} C, pressure {990 + n % 30} hPa.",
        f"Container ref C-{n:03d}-{(n * 23) % 89:02d}, seal verified.",
        f"Observer initials: {'ABCDEFGH'[n % 8]}.{'JKLMNPQR'[n % 8]}.",
        f"Total volume recorded: {n * 14 + 37} mL.",
    ]
    y = 250
    for line in notes:
        d.text((115, y), line, font=font("hand_snell", 34), fill=(28, 36, 72))
        y += 108
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", action="store_true", help="skip the scan degradation")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    makers = [page_table, page_prose, page_hand]
    images = []
    for n in range(1, args.pages + 1):
        img = makers[(n - 1) % len(makers)](n, args.pages)
        if not args.clean:
            img = degrade(img, jpeg=26, blur=0.9, noise=12,
                          skew_deg=(0.7 if n % 2 else -0.6), shadow=(n % 3 == 0), seed=n)
        images.append(img.convert("RGB"))
        print(f"  page {n}/{args.pages}", end="\r", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], resolution=args.dpi)
    print(f"\n{args.pages} pages -> {out}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
