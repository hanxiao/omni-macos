"""Generate the "profiling-v1" dataset used to benchmark Omni indexing speed.

1000 files mixed across the four indexer kinds (text/image/audio/video). Content
is PLACEHOLDER but metadata (resolution, duration, sample rate, byte size, file
type, filename tokens) is real and varied so the indexer does real work.

Low-entropy content (lorem, solid colors, silence, flat video) so the whole set
zips to well under GitHub Pages' 100MB limit. The script measures the final zip
and exits non-zero if it exceeds 95MB.

Run with uv (Pillow + ffmpeg required):
    uv run --with pillow python Tools/gen_profiling_v1.py
    uv run --with pillow python Tools/gen_profiling_v1.py --count 200 --out /tmp/prof

Outputs (under --out, default ./out):
    profiling-v1/        the generated files
    profiling-v1.zip     the packaged dataset (< 100MB)
    profiling-v1.json    manifest with counts, zipBytes, md5
"""

import argparse
import hashlib
import io
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

# Canonical mix: 1000 files, 40/30/16/14 across the kinds. (Was 5000; trimmed so a profiling run is
# quick on every Mac while keeping modality diversity and the same distribution.)
FULL_MIX = {"text": 400, "image": 300, "audio": 160, "video": 140}
TOTAL = sum(FULL_MIX.values())

ZIP_FAIL_BYTES = 95 * 1024 * 1024  # fail loudly above this
ZIP_LIMIT_BYTES = 100 * 1024 * 1024  # GitHub Pages hard limit

# Real-but-fake metadata tokens for filenames.
TOPICS = [
    "invoice", "report", "meeting-notes", "vacation", "product-demo", "tutorial",
    "interview", "sunset", "portrait", "landscape", "podcast", "lecture",
    "standup", "roadmap", "budget", "screenshot", "diagram", "voice-memo",
    "field-recording", "drone-footage", "unboxing", "review", "keynote", "webinar",
]
DATES = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in (3, 11, 19, 27)]

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
    "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
)


def tok(i: int) -> str:
    return f"{DATES[i % len(DATES)]}_{TOPICS[i % len(TOPICS)]}"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
TEXT_EXTS = ["txt", "md", "json", "csv"]


def text_payload(ext: str, target_bytes: int, idx: int) -> str:
    if ext == "json":
        rows = []
        i = 0
        while True:
            rows.append(
                f'  {{"id": {i}, "topic": "{TOPICS[i % len(TOPICS)]}", '
                f'"date": "{DATES[i % len(DATES)]}", "note": "{LOREM}"}}'
            )
            body = "[\n" + ",\n".join(rows) + "\n]\n"
            if len(body.encode()) >= target_bytes:
                return body
            i += 1
    if ext == "csv":
        lines = ["id,date,topic,note"]
        i = 0
        while True:
            lines.append(f'{i},{DATES[i % len(DATES)]},{TOPICS[i % len(TOPICS)]},"{LOREM.strip()}"')
            body = "\n".join(lines) + "\n"
            if len(body.encode()) >= target_bytes:
                return body
            i += 1
    # txt / md
    head = f"# {tok(idx)}\n\n" if ext == "md" else ""
    n = max(1, (target_bytes - len(head.encode())) // len(LOREM.encode()) + 1)
    return head + (LOREM * n)


def gen_text(args):
    idx, out_dir = args
    ext = TEXT_EXTS[idx % len(TEXT_EXTS)]
    # spread sizes 5KB..120KB
    target = 5 * 1024 + (idx * 1117) % (115 * 1024)
    body = text_payload(ext, target, idx)
    path = out_dir / f"text_{tok(idx)}_{idx:05d}.{ext}"
    path.write_text(body, encoding="utf-8")
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Images (Pillow, solid color + occasional gradient)
# ---------------------------------------------------------------------------
IMG_EXTS = ["png", "jpg"]
IMG_RES = [
    (640, 480), (800, 600), (1024, 768), (1280, 720),
    (1280, 960), (1600, 900), (1920, 1080), (1920, 1440),
]


def gen_image(args):
    idx, out_dir = args
    ext = IMG_EXTS[idx % len(IMG_EXTS)]
    w, h = IMG_RES[idx % len(IMG_RES)]
    r = (idx * 53) % 256
    g = (idx * 97) % 256
    b = (idx * 151) % 256
    if idx % 12 == 0:
        # a few real gradients (still compresses reasonably)
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            t = y / max(1, h - 1)
            draw.line([(0, y), (w, y)], fill=(int(r * (1 - t)), int(g * t), b))
    else:
        img = Image.new("RGB", (w, h), (r, g, b))
    path = out_dir / f"image_{tok(idx)}_{w}x{h}_{idx:05d}.{ext}"
    if ext == "jpg":
        img.save(path, format="JPEG", quality=85)
    else:
        img.save(path, format="PNG", optimize=True)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Audio (ffmpeg silence / single tone)
# ---------------------------------------------------------------------------
AUDIO_EXTS = ["mp3", "m4a"]
AUDIO_RATES = [22050, 32000, 44100, 48000]
AUDIO_DURS = [10, 20, 30, 45, 60, 75, 90]


def gen_audio(args):
    idx, out_dir = args
    ext = AUDIO_EXTS[idx % len(AUDIO_EXTS)]
    rate = AUDIO_RATES[idx % len(AUDIO_RATES)]
    dur = AUDIO_DURS[idx % len(AUDIO_DURS)]
    path = out_dir / f"audio_{tok(idx)}_{dur}s_{rate}hz_{idx:05d}.{ext}"
    if idx % 3 == 0:
        # single tone
        src = f"sine=frequency={220 + (idx % 8) * 55}:sample_rate={rate}:duration={dur}"
    else:
        # silence
        src = f"anullsrc=channel_layout=mono:sample_rate={rate}:duration={dur}"
    codec = ["-c:a", "libmp3lame", "-b:a", "64k"] if ext == "mp3" else ["-c:a", "aac", "-b:a", "64k"]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", src, "-t", str(dur),
        *codec, str(path),
    ]
    subprocess.run(cmd, check=True)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Video (ffmpeg flat color, all-intra libx264 high CRF)
# ---------------------------------------------------------------------------
VIDEO_EXTS = ["mp4", "mov"]
VIDEO_RES = [(1280, 720), (1920, 1080)]
VIDEO_DURS = [5, 8, 10, 12, 15, 18, 20]


def gen_video(args):
    idx, out_dir = args
    ext = VIDEO_EXTS[idx % len(VIDEO_EXTS)]
    w, h = VIDEO_RES[idx % len(VIDEO_RES)]
    dur = VIDEO_DURS[idx % len(VIDEO_DURS)]
    r = (idx * 53) % 256
    g = (idx * 97) % 256
    b = (idx * 151) % 256
    color = f"0x{r:02X}{g:02X}{b:02X}"
    path = out_dir / f"video_{tok(idx)}_{h}p_{dur}s_{idx:05d}.{ext}"
    src = f"color=c={color}:s={w}x{h}:r=24:d={dur}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", src,
        "-c:v", "libx264", "-preset", "veryfast",
        "-x264-params", "keyint=1",  # all-intra
        "-crf", "40", "-pix_fmt", "yuv420p",
        "-t", str(dur), str(path),
    ]
    subprocess.run(cmd, check=True)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def scaled_mix(count: int) -> dict:
    if count >= TOTAL:
        return dict(FULL_MIX)
    mix = {k: max(1, round(v * count / TOTAL)) for k, v in FULL_MIX.items()}
    # adjust to hit exactly `count`
    diff = count - sum(mix.values())
    order = sorted(mix, key=lambda k: -FULL_MIX[k])
    i = 0
    while diff != 0:
        k = order[i % len(order)]
        if diff > 0:
            mix[k] += 1
            diff -= 1
        elif mix[k] > 1:
            mix[k] -= 1
            diff += 1
        i += 1
    return mix


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=TOTAL, help="total files (proportional sample)")
    ap.add_argument("--out", type=str, default="out", help="output directory")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    data_dir = out_root / "profiling-v1"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    mix = scaled_mix(args.count)
    print(f"[profiling-v1] generating {sum(mix.values())} files: {mix}", flush=True)

    sizes = {}

    # Text + images: cheap, do single-process (Pillow releases little; still fast).
    sizes["text"] = [gen_text((i, data_dir)) for i in range(mix["text"])]
    print(f"  text done ({len(sizes['text'])})", flush=True)
    sizes["image"] = [gen_image((i, data_dir)) for i in range(mix["image"])]
    print(f"  images done ({len(sizes['image'])})", flush=True)

    # Audio + video: ffmpeg CPU-bound -> multiprocessing pool.
    nproc = max(1, mp.cpu_count() - 1)
    with mp.Pool(nproc) as pool:
        sizes["audio"] = pool.map(gen_audio, [(i, data_dir) for i in range(mix["audio"])])
        print(f"  audio done ({len(sizes['audio'])})", flush=True)
        sizes["video"] = pool.map(gen_video, [(i, data_dir) for i in range(mix["video"])])
        print(f"  video done ({len(sizes['video'])})", flush=True)

    file_count = sum(len(v) for v in sizes.values())

    # Build the zip.
    zip_path = out_root / "profiling-v1.zip"
    if zip_path.exists():
        zip_path.unlink()
    # report compressed bytes per modality by zipping in passes
    comp_by_mod = {k: 0 for k in mix}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(data_dir.rglob("*")):
            if not f.is_file():
                continue
            arc = f.relative_to(out_root)
            zf.write(f, arc)
        for info in zf.infolist():
            base = Path(info.filename).name
            for mod in mix:
                if base.startswith(mod + "_"):
                    comp_by_mod[mod] += info.compress_size
                    break

    zip_bytes = zip_path.stat().st_size
    md5 = md5_of(zip_path)

    # Manifest.
    manifest = {
        "version": "profiling-v1",
        "fileCount": file_count,
        "modalityCounts": {k: len(v) for k, v in sizes.items()},
        "zipBytes": zip_bytes,
        "md5": md5,
        "generatedNote": "placeholder content; metadata realistic",
    }
    import json

    manifest_path = out_root / "profiling-v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Report.
    print("\n[profiling-v1] summary", flush=True)
    print(f"  files: {file_count}", flush=True)
    for mod in ("text", "image", "audio", "video"):
        n = len(sizes[mod])
        raw = sum(sizes[mod])
        comp = comp_by_mod[mod]
        if n:
            print(
                f"  {mod:6s} n={n:5d}  raw={raw/1e6:8.2f}MB  zip={comp/1e6:8.2f}MB  "
                f"zip/file={comp/n:9.1f}B",
                flush=True,
            )
    print(f"  zip total: {zip_bytes/1e6:.2f}MB  md5={md5}", flush=True)
    print(f"  manifest:  {manifest_path}", flush=True)
    print(f"  zip:       {zip_path}", flush=True)

    if zip_bytes > ZIP_FAIL_BYTES:
        print(
            f"\nFAIL: zip {zip_bytes/1e6:.2f}MB exceeds {ZIP_FAIL_BYTES/1e6:.0f}MB "
            f"safety threshold (GitHub Pages limit {ZIP_LIMIT_BYTES/1e6:.0f}MB). Re-tune.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    print(f"\nPASS: zip under {ZIP_FAIL_BYTES/1e6:.0f}MB threshold.", flush=True)


if __name__ == "__main__":
    main()
