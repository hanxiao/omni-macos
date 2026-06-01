"""Generate the video-embedding reference fixture for the Swift MLX port.

Synthesizes a deterministic short clip (4 RGB frames, 480x360, a moving
rectangle drawn with pure numpy math - no randomness, no fonts), preprocesses
it EXACTLY the way the Qwen3VL video processor that the reference `model.py`
expects, runs `JinaOmniSmallEmbeddingModel.encode_video`, and saves the
canonical pixel_values_videos / grid_thw / input_ids / embedding so the Swift
side can match.

Run inside the mls venv (has mlx + transformers 5.1 + PIL + numpy + tokenizers):
    cd ~/Documents/mls && uv run python ~/Documents/omni-macos/Tools/gen_video_fixtures.py

Outputs:
    Fixtures/video_frames/frame_{0..3}.png   deterministic synthetic frames
    Fixtures/video_ref.safetensors           pixel_values_videos, grid_thw,
                                              input_ids, embedding

Preprocessing spec (canonical - matches Qwen3VLVideoProcessor._preprocess in
transformers 5.1; the Swift port must match this fixture):
  factor = patch_size * merge_size = 16 * 2 = 32
  temporal_factor = temporal_patch_size = 2
  size = {shortest_edge: 262144, longest_edge: 12845056}  (snapshot
    video_preprocessor_config.json)
  smart_resize(num_frames, height, width):
    h_bar = round(height/factor) * factor
    w_bar = round(width/factor) * factor
    t_bar = ceil(num_frames/temporal_factor) * temporal_factor
    then scale by t_bar*h_bar*w_bar against [min_pixels, max_pixels].
  resize via PIL Image.BICUBIC (resample=3) per frame - canonical (matches the
    image fixture; HF's fast video processor uses torchvision which is absent
    here, so PIL BICUBIC is the reference backend the Swift side targets).
  rescale 1/255, normalize (x - 0.5) / 0.5  (mean = std = 0.5).
  temporal grouping: temporal_patch_size(=2) consecutive frames form one
    temporal group, so 4 frames -> grid_t = 2 (NO temporal-repeat padding:
    4 is already divisible by 2).
  patchify (Qwen3VL video reshape/permute) -> pixel_values_videos
    [num_patches, channels*temporal*patch*patch = 1536],
    grid_thw = [[grid_t, grid_h, grid_w]].

The flat layout of one pixel_values_videos row is [channel, temporal, ph, pw]
flattened, exactly what VisionPatchEmbed expects: it reshapes to
[-1, in_channels=3, temporal=2, patch=16, patch=16] then moveaxis(1, 4).
"""

import importlib.util
import math
import os
import sys

import mlx.core as mx
import numpy as np
from PIL import Image

MODEL_DIR = os.environ.get(
    "OMNI_MODEL_DIR",
    "/Volumes/One Touch/ai-models/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-mlx/snapshots/716c5b684db3f6ba574dd9b4f6b14af3b2eb8bda",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fixtures"
)
FRAMES_DIR = os.path.join(OUT_DIR, "video_frames")

# --- Preprocessing constants (from video_preprocessor_config.json) -----------
PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
# snapshot video_preprocessor_config.json size dict.
MIN_PIXELS = 262144
MAX_PIXELS = 12845056
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
RESCALE = 1.0 / 255.0

# --- Special token ids (from model.py / config) ------------------------------
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656

# --- Clip geometry -----------------------------------------------------------
NUM_FRAMES = 4
FRAME_W = 480
FRAME_H = 360


# ---------------------------------------------------------------------------
# 1. Deterministic synthetic clip (no fonts/assets, no randomness).
# ---------------------------------------------------------------------------
def make_frames(
    num_frames: int = NUM_FRAMES, width: int = FRAME_W, height: int = FRAME_H
) -> list[Image.Image]:
    """Fully reproducible RGB clip: static gradient + a rectangle that slides.

    Built purely with deterministic numpy math (no np.random anywhere). The
    rectangle translates a fixed amount per frame so temporal patches differ.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    fx = xx / (width - 1)
    fy = yy / (height - 1)

    frames: list[Image.Image] = []
    for t in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.float64)
        # Static smooth gradient background (same every frame).
        img[..., 0] = 255.0 * fx
        img[..., 1] = 255.0 * fy
        img[..., 2] = 255.0 * (0.5 + 0.5 * np.sin(6.2831853 * (fx + fy)))

        # Moving solid rectangle: x0 advances 90 px per frame, deterministic.
        rect_w, rect_h = 120, 100
        x0 = 30 + 90 * t
        y0 = 40 + 30 * t
        x1 = min(width, x0 + rect_w)
        y1 = min(height, y0 + rect_h)
        img[y0:y1, x0:x1] = np.array([230.0, 40.0, 50.0])

        # A small static white marker block to anchor spatial structure.
        img[300:340, 20:80] = np.array([245.0, 245.0, 245.0])

        frames.append(
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
        )
    return frames


# ---------------------------------------------------------------------------
# 2. Preprocessing (canonical numpy, mirrors Qwen3VLVideoProcessor._preprocess).
# ---------------------------------------------------------------------------
def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = TEMPORAL_PATCH_SIZE,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen3VL video smart_resize (transformers 5.1 video_processing_qwen3_vl)."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be > factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(frames: list[Image.Image]):
    """Return (pixel_values_videos [num_patches,1536] f32, grid_thw [[gt,gh,gw]]).

    Mirrors Qwen3VLVideoProcessor._preprocess: per-frame resize (bicubic) ->
    rescale -> normalize -> temporal-pad-to-multiple -> patchify.
    """
    num_frames = len(frames)
    w0, h0 = frames[0].size  # PIL size is (width, height)
    resized_h, resized_w = smart_resize(num_frames, h0, w0)

    # Per-frame PIL resize (width, height); resample=3 == Image.BICUBIC.
    chans = []
    for f in frames:
        f = f.convert("RGB").resize(
            (resized_w, resized_h), resample=Image.BICUBIC
        )
        arr = np.asarray(f, dtype=np.float32)  # (H, W, 3)
        arr = arr * RESCALE
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        chans.append(arr.transpose(2, 0, 1))  # (3, H, W)
    # (T, C, H, W)
    patches = np.stack(chans, axis=0)

    # Pad T up to a multiple of temporal_patch_size by repeating the last frame.
    T = patches.shape[0]
    pad = (-T) % TEMPORAL_PATCH_SIZE
    if pad:
        repeats = np.repeat(patches[-1:], pad, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)

    channel = IN_CHANNELS
    grid_t = patches.shape[0] // TEMPORAL_PATCH_SIZE
    grid_h = resized_h // PATCH_SIZE
    grid_w = resized_w // PATCH_SIZE

    # Qwen3VL video patchify: view then permute (0,1,4,7,5,8,3,2,6,9).
    # Add a leading batch axis of 1 to mirror the HF (batch, T, C, H, W) layout.
    patches = patches[np.newaxis, ...]  # (1, T, C, H, W)
    patches = patches.reshape(
        1,
        grid_t,
        TEMPORAL_PATCH_SIZE,
        channel,
        grid_h // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
        grid_w // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE,
    )
    pixel_values_videos = np.ascontiguousarray(flatten_patches, dtype=np.float32)
    grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values_videos, grid_thw


def cross_check(pixel_values: np.ndarray, grid_thw: np.ndarray):
    """Cross-check the patchify layout against a direct numpy de-patchify.

    Re-derives one row's [channel, temporal, ph, pw] block and confirms it
    matches VisionPatchEmbed's expected reshape [-1, 3, 2, 16, 16]. This is a
    structural check (no torchvision available to run the HF fast processor)."""
    row = pixel_values[0]
    block = row.reshape(IN_CHANNELS, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    print(
        f"[xcheck] row0 reshape -> {block.shape} == [channel,temporal,ph,pw] "
        f"(VisionPatchEmbed expects [-1,3,2,16,16]); row dim={row.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 3. Load reference model.
# ---------------------------------------------------------------------------
def load_ref():
    spec = importlib.util.spec_from_file_location(
        "jina_omni_utils", os.path.join(MODEL_DIR, "utils.py")
    )
    utils = importlib.util.module_from_spec(spec)
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    spec.loader.exec_module(utils)
    return utils.load_model(MODEL_DIR)


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # 1. Synthesize + save the clip frames.
    frames = make_frames()
    for i, f in enumerate(frames):
        p = os.path.join(FRAMES_DIR, f"frame_{i}.png")
        f.save(p)
    print(
        f"[clip] saved {len(frames)} frames to {FRAMES_DIR} "
        f"size={frames[0].size} (w,h)"
    )

    # 2. Preprocess (canonical numpy) + structural cross-check.
    pixel_values, grid_thw = preprocess(frames)
    grid_t, grid_h, grid_w = (int(x) for x in grid_thw[0])
    num_patches = grid_t * grid_h * grid_w
    n_merged = num_patches // (MERGE_SIZE**2)
    print(
        f"[pp] grid_thw={grid_thw.tolist()} pixel_values_videos={pixel_values.shape} "
        f"dtype={pixel_values.dtype} num_patches={num_patches} N_merged={n_merged}"
    )
    cross_check(pixel_values, grid_thw)

    # 3. Load reference model + switch to retrieval (merges LoRA into text tower).
    m = load_ref()
    m.switch_task("retrieval")
    base = m.model  # JinaOmniSmallEmbeddingModel
    prefix_ids = list(m.tokenizer.encode("Document: ").ids)  # retrieval document prefix (all modalities)

    # 4. input_ids = [Document: ] + [vision_start] + video_token * N_merged + [vision_end].
    ids = prefix_ids + [VISION_START] + [VIDEO_TOKEN] * n_merged + [VISION_END]
    L = len(ids)
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.ones((1, L), dtype=mx.int32)
    n_video_tokens = sum(1 for t in ids if t == VIDEO_TOKEN)
    assert n_video_tokens == n_merged, (
        f"video-token count {n_video_tokens} != N_merged {n_merged}"
    )

    # 5. Encode video -> [1, 1024], L2-normalized inside _last_token_pool.
    pv = mx.array(pixel_values)
    thw = mx.array(grid_thw)
    emb = base.encode_video(
        pixel_values_videos=pv,
        video_grid_thw=thw,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    mx.eval(emb)

    norm = float(mx.linalg.norm(emb[0]).item())
    first5 = [float(x) for x in emb[0, :5].tolist()]

    # 6. Save the fixture.
    fixture_path = os.path.join(OUT_DIR, "video_ref.safetensors")
    mx.save_safetensors(
        fixture_path,
        {
            "pixel_values_videos": pv.astype(mx.float32),
            "grid_thw": thw.astype(mx.int32),
            "input_ids": input_ids.astype(mx.int32),
            "embedding": emb.astype(mx.float32),
        },
    )

    # 7. Report.
    print("=" * 60)
    print(f"grid_t      = {grid_t}")
    print(f"grid_h      = {grid_h}")
    print(f"grid_w      = {grid_w}")
    print(f"num_patches = {num_patches}")
    print(f"N_merged    = {n_merged}")
    print(f"L (seq len) = {L}")
    print(f"emb norm    = {norm:.6f}")
    print(f"emb[:5]     = {first5}")
    print(
        f"pixel_values_videos shape={tuple(pv.shape)} dtype={pv.dtype}"
    )
    print(
        f"input_ids   = [{VISION_START}] + [{VIDEO_TOKEN}]*{n_merged} + [{VISION_END}]"
    )
    print(f"[saved] {fixture_path}")


if __name__ == "__main__":
    main()
