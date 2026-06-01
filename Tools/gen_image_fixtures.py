"""Generate the image-embedding reference fixture for the Swift MLX port.

Produces a deterministic synthetic RGB test image, preprocesses it EXACTLY the
way the Qwen2VL/Qwen3VL image processor that the reference `model.py` expects,
runs `JinaOmniSmallEmbeddingModel.encode_image`, and saves the canonical
pixel_values / grid_thw / input_ids / embedding so the Swift side can match.

Run inside the mls venv (has mlx + transformers 5.1 + PIL + numpy + tokenizers):
    cd ~/Documents/mls && uv run python ~/Documents/omni-macos/Tools/gen_image_fixtures.py

Outputs:
    Fixtures/test_image.png            deterministic synthetic test image
    Fixtures/image_ref.safetensors     pixel_values, grid_thw, input_ids, embedding

Preprocessing spec (canonical - what model.py consumes; the Swift port must match):
  factor = patch_size * merge_size = 16 * 2 = 32
  min_pixels = 262144, max_pixels = 1310720
  smart_resize: round each side to a multiple of `factor`, then scale the whole
    image so that h*w lands in [min_pixels, max_pixels] (Qwen2VL algorithm).
  resize via PIL Image.BICUBIC (resample=3) on the RGB uint8 image.
  rescale 1/255, normalize (x - 0.5) / 0.5  (mean=std=0.5).
  temporal repeat: stack the single frame temporal_patch_size(=2) times.
  patchify (Qwen2VL processor reshape/transpose) -> pixel_values
    [num_patches, channels*temporal*patch*patch = 1536], grid_thw = [[1, gh, gw]].

The flat layout of one pixel_values row is [channel, temporal, ph, pw] flattened,
which is exactly what VisionPatchEmbed expects: it reshapes to
[-1, in_channels=3, temporal=2, patch=16, patch=16] then moveaxis(1, 4).
"""

import importlib.util
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

# --- Preprocessing constants (from preprocessor_config.json) -----------------
PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
MIN_PIXELS = 262144
MAX_PIXELS = 1310720
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
RESCALE = 1.0 / 255.0

# --- Special token ids (from model.py / config) ------------------------------
VISION_START = 151652
VISION_END = 151653
IMAGE_TOKEN = 151655


# ---------------------------------------------------------------------------
# 1. Deterministic synthetic test image (no fonts/assets, no randomness).
# ---------------------------------------------------------------------------
def make_test_image(width: int = 600, height: int = 440) -> Image.Image:
    """Fully reproducible RGB image: gradients + colored rectangles + a circle.

    Built purely with deterministic numpy math (no np.random anywhere).
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    fx = xx / (width - 1)   # 0..1 across columns
    fy = yy / (height - 1)  # 0..1 down rows

    img = np.zeros((height, width, 3), dtype=np.float64)
    # Base smooth gradient background.
    img[..., 0] = 255.0 * fx                      # red ramps left->right
    img[..., 1] = 255.0 * fy                      # green ramps top->bottom
    img[..., 2] = 255.0 * (0.5 + 0.5 * np.sin(6.2831853 * (fx + fy)))  # blue wave

    # Solid colored rectangles (deterministic coords).
    img[40:160, 60:260] = np.array([220.0, 30.0, 40.0])     # red block
    img[200:360, 320:560] = np.array([30.0, 200.0, 90.0])   # green block
    img[260:420, 40:220] = np.array([40.0, 70.0, 230.0])    # blue block

    # A bright yellow filled circle.
    cy, cx, r = 120.0, 460.0, 80.0
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[circle] = np.array([250.0, 230.0, 20.0])

    # A diagonal cyan stripe (deterministic line band).
    stripe = np.abs((xx - yy) - 80.0) < 12.0
    img[stripe] = np.array([20.0, 220.0, 220.0])

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# 2. Preprocessing (canonical numpy implementation).
# ---------------------------------------------------------------------------
def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen2VL smart_resize: nearest multiple of `factor`, clamped to pixel budget."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        h_bar = max(factor, int(np.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(np.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = int(np.ceil(height * beta / factor)) * factor
        w_bar = int(np.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def preprocess(img: Image.Image):
    """Return (pixel_values [num_patches,1536] float32, grid_thw [[1,gh,gw]] int).

    Matches Qwen2VLImageProcessor exactly:
      resize (bicubic) -> rescale -> normalize -> temporal repeat -> patchify.
    """
    img = img.convert("RGB")
    w0, h0 = img.size  # PIL size is (width, height)
    resized_h, resized_w = smart_resize(h0, w0)

    # PIL resize takes (width, height); resample=3 == Image.BICUBIC.
    img_resized = img.resize((resized_w, resized_h), resample=Image.BICUBIC)
    arr = np.asarray(img_resized, dtype=np.float32)  # (H, W, 3)

    # rescale + normalize.
    arr = arr * RESCALE
    arr = (arr - IMAGE_MEAN) / IMAGE_STD

    # to channels-first (C, H, W).
    patches = arr.transpose(2, 0, 1)  # (3, H, W)

    # temporal repeat: single frame -> grid_t = temporal_patch_size frames.
    # Qwen2VL repeats the image temporal_patch_size times along a new T axis.
    patches = patches[np.newaxis, :, :, :]  # (1, 3, H, W)
    patches = np.tile(patches, (TEMPORAL_PATCH_SIZE, 1, 1, 1))  # (T=2, 3, H, W)

    channel = IN_CHANNELS
    grid_t = 1  # one image -> one temporal group after dividing by temporal_patch_size
    grid_h = resized_h // PATCH_SIZE
    grid_w = resized_w // PATCH_SIZE

    # Qwen2VL patchify reshape.
    patches = patches.reshape(
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
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE,
    )
    pixel_values = np.ascontiguousarray(flatten_patches, dtype=np.float32)
    grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, grid_thw


def cross_check(img: Image.Image, pixel_values: np.ndarray, grid_thw: np.ndarray):
    """Cross-check our numpy preprocessing against transformers' processor."""
    try:
        from transformers import Qwen2VLImageProcessor
    except Exception as e:  # pragma: no cover
        print(f"[xcheck] skipped (no Qwen2VLImageProcessor): {e}")
        return
    proc = Qwen2VLImageProcessor(
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        image_mean=[IMAGE_MEAN] * 3,
        image_std=[IMAGE_STD] * 3,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        patch_size=PATCH_SIZE,
        merge_size=MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        resample=3,
    )
    out = proc(images=img, return_tensors="np")
    ref_pv = np.asarray(out["pixel_values"], dtype=np.float32)
    ref_thw = np.asarray(out["image_grid_thw"])
    if ref_pv.shape != pixel_values.shape:
        print(
            f"[xcheck] SHAPE MISMATCH ours={pixel_values.shape} hf={ref_pv.shape}; "
            "using ours (canonical)."
        )
        return
    max_abs = float(np.abs(ref_pv - pixel_values).max())
    thw_ok = np.array_equal(ref_thw, grid_thw)
    print(
        f"[xcheck] Qwen2VLImageProcessor: max|dpv|={max_abs:.3e} grid_match={thw_ok} "
        f"(hf grid={ref_thw.tolist()})"
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
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Synthesize + save the test image.
    img = make_test_image()
    img_path = os.path.join(OUT_DIR, "test_image.png")
    img.save(img_path)
    print(f"[img] saved {img_path} size={img.size} (w,h)")

    # 2. Preprocess (canonical numpy) + cross-check.
    pixel_values, grid_thw = preprocess(img)
    grid_t, grid_h, grid_w = (int(x) for x in grid_thw[0])
    n_merged = (grid_h * grid_w) // (MERGE_SIZE ** 2)
    print(
        f"[pp] grid_thw={grid_thw.tolist()} pixel_values={pixel_values.shape} "
        f"dtype={pixel_values.dtype} N_merged={n_merged}"
    )
    cross_check(img, pixel_values, grid_thw)

    # 3. Load reference model + switch to retrieval (merges LoRA into text tower).
    m = load_ref()
    m.switch_task("retrieval")
    base = m.model  # JinaOmniSmallEmbeddingModel
    # The "Query: "/"Document: " prefix applies to EVERY modality (official model
    # card). Media is indexed as documents, so prepend the document prefix ids.
    prefix_ids = list(m.tokenizer.encode("Document: ").ids)

    # 4. input_ids = [Document: ] + [vision_start] + image_token * N_merged + [vision_end].
    ids = prefix_ids + [VISION_START] + [IMAGE_TOKEN] * n_merged + [VISION_END]
    L = len(ids)
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.ones((1, L), dtype=mx.int32)
    n_image_tokens = sum(1 for t in ids if t == IMAGE_TOKEN)
    assert n_image_tokens == n_merged, (
        f"image-token count {n_image_tokens} != N_merged {n_merged}"
    )

    # 5. Encode image -> [1, 1024], L2-normalized inside _last_token_pool.
    pv = mx.array(pixel_values)
    thw = mx.array(grid_thw)
    emb = base.encode_image(
        pixel_values=pv,
        image_grid_thw=thw,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    mx.eval(emb)

    norm = float(mx.linalg.norm(emb[0]).item())
    first5 = [float(x) for x in emb[0, :5].tolist()]

    # 6. Save the fixture.
    fixture_path = os.path.join(OUT_DIR, "image_ref.safetensors")
    mx.save_safetensors(
        fixture_path,
        {
            "pixel_values": pv.astype(mx.float32),
            "grid_thw": thw.astype(mx.int32),
            "input_ids": input_ids.astype(mx.int32),
            "embedding": emb.astype(mx.float32),
        },
    )

    # 7. Report.
    print("=" * 60)
    print(f"grid_h      = {grid_h}")
    print(f"grid_w      = {grid_w}")
    print(f"N_merged    = {n_merged}")
    print(f"L (seq len) = {L}")
    print(f"emb norm    = {norm:.6f}")
    print(f"emb[:5]     = {first5}")
    print(f"pixel_values shape={tuple(pv.shape)} dtype={pv.dtype}")
    print(f"input_ids   = [{VISION_START}] + [{IMAGE_TOKEN}]*{n_merged} + [{VISION_END}]")
    print(f"[saved] {fixture_path}")


if __name__ == "__main__":
    main()
