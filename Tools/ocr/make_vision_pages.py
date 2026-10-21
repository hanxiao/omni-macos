"""Pages with KNOWN text, for grading Apple Vision's two document readers against each other.

make_hard_pages.py grades the OCR model against the model's own oracle; here the reference is the
text drawn on the page, so any reader can be scored by character error rate. Eight kinds of page
(prose, two columns, a numeric table, a receipt, accented Latin, Chinese, Japanese, small dense
type), each clean, scanned and photographed through make_hard_pages' damage.

    uv run --with pillow --with numpy python Tools/ocr/make_vision_pages.py --out <dir>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from make_hard_pages import degrade, font, perspective  # noqa: E402

W, H = 1700, 2200
CJK = {"zh": "/System/Library/Fonts/Supplemental/Songti.ttc", "ja": "/System/Library/Fonts/Hiragino Sans GB.ttc"}

PROSE = ("The committee met on Tuesday to review the quarterly budget and the hiring plan for the "
         "Berlin office. Revenue grew by 12 percent while costs rose by 4 percent, mostly from cloud "
         "infrastructure. The board asked for a revised forecast before the end of November, with a "
         "separate line for travel and a contingency of 80,000 euros. Maria will circulate the draft "
         "minutes on Friday; comments are due by the following Wednesday at noon.")
GERMAN = ("Sehr geehrte Frau Müller, wir bestätigen den Eingang Ihrer Kündigung vom 14. März. Die "
          "Kaution in Höhe von 2.400 Euro wird nach der Übergabe der Wohnung überwiesen. Bitte "
          "beachten Sie, dass die Zählerstände für Strom und Wasser bei der Übergabe gemeinsam "
          "abgelesen werden. Mit freundlichen Grüßen, Ihre Hausverwaltung. Le délai de préavis est "
          "de trois mois; veuillez nous indiquer votre nouvelle adresse à l'avance.")
CHINESE = ("本季度销售额比去年同期增长百分之十二，主要来自欧洲市场的新客户。成本上升百分之四，"
           "其中大部分来自云计算基础设施。董事会要求在十一月底之前提交修订后的预测，并为差旅单独列出预算。"
           "会议纪要将于周五发出，意见请于下周三中午之前反馈。")
JAPANESE = ("今週の会議では四半期の予算と採用計画について話し合いました。売上は前年比で十二パーセント増加し、"
            "費用は主にクラウド基盤のために四パーセント増えました。修正した見通しを十一月末までに提出してください。"
            "議事録は金曜日に配布します。")


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int, cjk: bool = False) -> list[str]:
    units = list(text) if cjk else text.split(" ")
    lines, cur = [], ""
    for u in units:
        trial = cur + u if cjk else (cur + " " + u if cur else u)
        if draw.textlength(trial, font=f) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = u
    if cur:
        lines.append(cur)
    return lines


def render(lines: list[str], f: ImageFont.FreeTypeFont, *, x: int = 120, y: int = 140, step: int | None = None,
           size: tuple[int, int] = (W, H), img: Image.Image | None = None) -> Image.Image:
    img = img or Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    step = step or int(f.size * 1.45)
    for ln in lines:
        d.text((x, y), ln, fill="black", font=f)
        y += step
    return img


def page(kind: str) -> tuple[Image.Image, str]:
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    if kind == "letter":
        f = font("sans", 34)
        lines = wrap(d, PROSE, f, W - 240) + [""] + wrap(d, PROSE.replace("Berlin", "Lisbon").replace("Maria", "Tomas"), f, W - 240)
        return render(lines, f), "\n".join(lines)
    if kind == "columns":
        f = font("serif", 26)
        colw = (W - 300) // 2
        left = wrap(d, PROSE + " " + PROSE.replace("Tuesday", "Thursday"), f, colw)
        right = wrap(d, PROSE.replace("Berlin", "Vienna") + " " + PROSE.replace("budget", "roadmap"), f, colw)
        img = render(left, f, x=120)
        img = render(right, f, x=120 + colw + 60, img=img)
        return img, "\n".join(left + right)
    if kind == "table":
        f = font("mono", 28)
        rng = random.Random(4)
        lines = ["ID      Region     Units    Revenue"]
        for i in range(28):
            lines.append(f"R-{1000 + i:04d}  {rng.choice(['North', 'South', 'East', 'West']):9s}  {rng.randint(10, 999):5d}  {rng.randint(1000, 99999):9,d}")
        return render(lines, f), "\n".join(lines)
    if kind == "receipt":
        f = font("mono", 30)
        lines = ["CORNER MARKET", "Store 0412  Till 3", "Milk 1L           1.29", "Eggs 12          3.49",
                 "Spinach          2.19", "Bread rye        2.80", "Coffee beans     8.95", "SUBTOTAL        18.72",
                 "VAT 7%           1.31", "TOTAL           20.03", "Card ****4821", "Thank you"]
        img = render(lines, f, x=520, y=300, size=(W, H))
        return img, "\n".join(lines)
    if kind == "accents":
        f = font("serif", 32)
        lines = wrap(d, GERMAN, f, W - 240)
        return render(lines, f), "\n".join(lines)
    if kind in ("zh", "ja"):
        f = ImageFont.truetype(CJK[kind], 36, index=0)
        lines = wrap(d, CHINESE if kind == "zh" else JAPANESE, f, W - 240, cjk=True)
        return render(lines, f), "\n".join(lines)
    if kind == "dense":
        f = font("sans", 20)
        lines = wrap(d, " ".join([PROSE, GERMAN.split(" Le délai")[0], PROSE.replace("Tuesday", "Monday")]) * 2, f, W - 240)
        return render(lines, f), "\n".join(lines)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for kind in ["letter", "columns", "table", "receipt", "accents", "zh", "ja", "dense"]:
        clean, text = page(kind)
        variants = {
            "clean": clean,
            "scan": degrade(clean, jpeg=30, blur=0.9, noise=10, skew_deg=0.7, seed=n),
            "photo": degrade(perspective(clean), jpeg=40, blur=1.1, noise=8, shadow=True, seed=n + 1),
        }
        for name, img in variants.items():
            stem = f"{kind}_{name}"
            img.save(out / f"{stem}.png")
            (out / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
            n += 1
    print(f"{n} pages -> {out}")


if __name__ == "__main__":
    main()
