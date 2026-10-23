"""The weight file of a built Neural Engine program, as a RECIPE the app replays on the user's Mac.

A Core ML program is its graph (model.mlmodel, ~2 MB, shipped in the app) and its weights
(weights/weight.bin, the model's size, NOT shipped). The app writes weight.bin itself from the weights
it already downloaded (OmniKit.ANEProgram): this script says, for every blob in the order the graph
references them, which tensor it is and what was done to it - so the file the app writes is the one
built and measured here, byte for byte, or the script fails.

    uv run --python 3.12 --with coremltools --with mlx Tools/ane/extract_recipe.py \
        <weights.safetensors (dumpbackbone, the app's own bf16)> <nano_text.mlpackage> <out.json>

Transforms (Tools/ane/build_nano_text.py): `id`, `q` (x 1/sqrt(head_dim), folded scale), `qP` (the
signed rotate-half permutation of q), `kP` (of k). Every blob is fp16, row-major [out, in].
"""
import hashlib, json, struct, sys
import numpy as np
import mlx.core as mx

weights, package, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = {k: np.array(v.astype(mx.float32), copy=False) for k, v in mx.load(weights).items()}
hidden = W["language_model.norm.weight"].shape[0]
small = hidden == 1024                             # Small: 16 x 128 (q), 8 x 128 (kv), q/k norms
head_dim = 128 if small else 64
q_heads, kv_heads = (16, 8) if small else (12, 12)


def rot(m):
    """P @ m, P the signed rotate-half per head: row j of a head <- -row (j + half), and back."""
    o = np.empty_like(m)
    half = head_dim // 2
    for h in range(m.shape[0] // head_dim):
        a = h * head_dim
        o[a:a + half] = -m[a + half:a + head_dim]
        o[a + half:a + head_dim] = m[a:a + half]
    return o


def perm(v):
    """A tiled norm weight as the rotated channels see it: halves of each head swapped, no sign."""
    o = np.empty_like(v); half = head_dim // 2
    for h in range(v.shape[0] // head_dim):
        a = h * head_dim
        o[a:a + half] = v[a + half:a + head_dim]; o[a + half:a + head_dim] = v[a:a + half]
    return o


def indicator(heads):
    I = np.zeros((heads, heads * head_dim), dtype=np.float32)
    for h in range(heads): I[h, h * head_dim:(h + 1) * head_dim] = 1
    return I


cands = {}
def add(name, op, arr, **extra):
    cands[hashlib.sha256(arr.astype(np.float16).tobytes()).hexdigest()] = dict(tensor=name, op=op, **extra)

for k, v in W.items():
    if not k.startswith("language_model.") or k.endswith("embed_tokens.weight"):
        continue
    add(k, "id", v)
    if small:
        # Small: the projections plain, their rotations, and the per-head norm weights tiled over the
        # heads (times 1/sqrt(d) for q), each also as the rotated channels see it (build_small_text.py).
        if k.endswith("q_proj.weight"): add(k, "rot", rot(v))
        if k.endswith("k_proj.weight"): add(k, "rot", rot(v))
        if k.endswith("q_norm.weight"):
            t = np.tile(v, q_heads) * head_dim ** -0.5
            add(k, "tile", t, heads=q_heads, scaled=True); add(k, "tileP", perm(t), heads=q_heads, scaled=True)
        if k.endswith("k_norm.weight"):
            t = np.tile(v, kv_heads)
            add(k, "tile", t, heads=kv_heads, scaled=False); add(k, "tileP", perm(t), heads=kv_heads, scaled=False)
        continue
    if k.endswith("q_proj.weight"):
        q = v * head_dim ** -0.5
        add(k, "q", q); add(k, "qP", rot(q))
    if k.endswith("k_proj.weight"):
        add(k, "kP", rot(v))
# Constants of the graph, not of the weights: the head indicators (a head's channels summed, or its
# scale sent back out), named by shape, never embedded.
for heads in sorted({q_heads, kv_heads}):
    I = indicator(heads)
    add("", "indicator", I / head_dim, heads=heads, transpose=False, value=1 / head_dim)
    add("", "indicator", I, heads=heads, transpose=False, value=1.0)
    add("", "indicator", I.T.copy(), heads=heads, transpose=True, value=1.0)
    for eps in (1e-5, 1e-4):   # an epsilon the compiler broadcast over the heads
        add("", "fill", np.full(heads, eps, dtype=np.float32), count=heads, value=eps)

blob = open(f"{package}/Data/com.apple.CoreML/weights/weight.bin", "rb").read()
count, version = struct.unpack("<II", blob[:8])
assert version == 2, version
recipe, p = [], 64
for i in range(count):
    sentinel, dtype, size, offset = struct.unpack("<IIQQ", blob[p:p + 24])
    assert sentinel == 0xDEADBEEF and dtype == 1 and offset == p + 64, (i, hex(sentinel), dtype, offset, p)
    h = hashlib.sha256(blob[offset:offset + size]).hexdigest()
    if h not in cands:
        vals = np.frombuffer(blob[offset:offset + size], dtype=np.float16)
        sys.exit(f"blob {i} ({size} bytes at {offset}) matches no tensor: the recipe would be wrong. First values {vals[:8]}")
    recipe.append(dict(cands[h], bytes=size))
    p = (offset + size + 63) // 64 * 64
assert p == len(blob), (p, len(blob))
json.dump({"format": 2, "weightBytes": len(blob), "headDim": head_dim, "blobs": recipe,
           "sha256": hashlib.sha256(blob).hexdigest()}, open(out, "w"), indent=0)
print(f"{count} blobs, {len(blob)} bytes -> {out}")
