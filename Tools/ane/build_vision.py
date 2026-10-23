"""The vision tower (Qwen3-VL ViT + the 2x2 patch merger) for the Neural Engine as BLOCK programs:
no weights inside (anelib.py), three functions per patch-count bucket N, all sharing the app's
weights, which it passes in:
  E{N}  patch embedding: px (1, 1536, 1, N) -> W (C, 1536) @ + b + pos (1, C, 1, N). W is the
        checkpoint's [C, t, p, p, c] weight with its columns in the raw pixel order (c, t, p, p), so
        OmniVisionPreprocess's patches go in as they are.
  B{N}  one ViT block, called once per layer: LayerNorm (eps 1e-6); q, k with 2D RoPE folded in
        (rope(Wx + b) = (Wx + b) * cos + (PWx + Pb) * sin, P the signed rotate-half per head, so the
        app passes W and PW) and 1/sqrt(d) folded into q; v; attention over the bucket with a key
        mask for its padding; proj; LayerNorm, fc1, GELU (tanh), fc2.
  M{N}  the merger: LayerNorm, the 2x2 merge (the patches arrive merge-block-major, so four
        neighbours are four consecutive positions: (C, N) -> (4C, N/4) in fc1's (j, c) order), fc1,
        GELU (erf), fc2 -> (1, D, 1, N/4), the image's tokens for the backbone.
Read the GPU path first (Sources/OmniKit/OmniVisionTower.swift); the app side is ANEVisionBackend.

  uv run --python 3.12 --with coremltools python Tools/ane/build_vision.py --config <config.json> \\
      --out <dir> [--name nano_vision] [--buckets ...]
"""
import argparse, json
from coremltools.converters.mil import Builder as mb
from anelib import spec, program, lin, layer_norm, attention, save_multifunction

# Images are 1024 ... 5120 patches (OmniVisionPreprocess's pixel range); a video's temporal slices
# can be smaller, and at the default 6 frames of 1568 px larger: 16:9 5,488, 4:3 7,252, square 8,100.
BUCKETS = "256,576,1024,1600,2304,3136,4096,5120,5632,7296,8192"


def embed(N, C, K):
    return program(dict(px=spec(1, K, 1, N), pos=spec(1, C, 1, N), w=spec(C, K), b=spec(1, C, 1, 1)),
                   lambda px, pos, w, b: mb.add(x=lin(px, w, b, C, N, "pe"), y=pos, name="out"))


def block(N, C, H, F):
    vec = lambda c=C: spec(1, c, 1, 1)
    specs = dict(h=spec(1, C, 1, N), cos=spec(1, C, 1, N), sin=spec(1, C, 1, N), mask=spec(1, N, 1, 1),
                 g1=vec(), b1=vec(), wq=spec(C, C), bq=vec(), wqp=spec(C, C), bqp=vec(), wk=spec(C, C), bk=vec(),
                 wkp=spec(C, C), bkp=vec(), wv=spec(C, C), bv=vec(), wo=spec(C, C), bo=vec(), g2=vec(), b2=vec(),
                 w1=spec(F, C), bf1=vec(F), w2=spec(C, F), bf2=vec())

    def body(h, cos, sin, mask, g1, b1, wq, bq, wqp, bqp, wk, bk, wkp, bkp, wv, bv, wo, bo, g2, b2, w1, bf1, w2, bf2):
        xn = layer_norm(h, g1, b1, 1e-6, "n1")
        roped = lambda w, b, wp, bp, t: mb.add(x=mb.mul(x=lin(xn, w, b, C, N, t), y=cos, name=f"{t}_c"),
                                               y=mb.mul(x=lin(xn, wp, bp, C, N, t + "p"), y=sin, name=f"{t}_s"), name=f"{t}_r")
        q, k = roped(wq, bq, wqp, bqp, "q"), roped(wk, bk, wkp, bkp, "k")
        v = lin(xn, wv, bv, C, N, "v")
        h = mb.add(x=h, y=lin(attention(q, k, v, mask, H, N, "at"), wo, bo, C, N, "o"), name="r1")
        g = mb.gelu(x=lin(layer_norm(h, g2, b2, 1e-6, "n2"), w1, bf1, F, N, "f1"), mode="TANH_APPROXIMATION", name="g")
        return mb.add(x=h, y=lin(g, w2, bf2, C, N, "f2"), name="out")

    return program(specs, body)


def merger(N, C, M, D):
    T = N // 4

    def body(h, g, b, w1, b1, w2, b2):
        x = layer_norm(h, g, b, 1e-6, "mn")
        x = mb.reshape(x=x, shape=[C, T, 4], name="m3")
        x = mb.reshape(x=mb.transpose(x=x, perm=[2, 0, 1], name="mt"), shape=[1, 4 * C, 1, T], name="m4")
        x = mb.gelu(x=lin(x, w1, b1, M, T, "f1"), mode="EXACT", name="g")
        return mb.identity(x=lin(x, w2, b2, D, T, "f2"), name="out")

    return program(dict(h=spec(1, C, 1, N), g=spec(1, C, 1, 1), b=spec(1, C, 1, 1), w1=spec(M, 4 * C),
                        b1=spec(1, M, 1, 1), w2=spec(D, M), b2=spec(1, D, 1, 1)), body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="nano_vision"); ap.add_argument("--buckets", default=BUCKETS)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    v = cfg["vision_config"]
    C, H, F = v["hidden_size"], v["num_heads"], v["intermediate_size"]
    K = v["in_channels"] * v["temporal_patch_size"] * v["patch_size"] ** 2
    # the merger's output is the text model's width (Nano's config says out_hidden_size 1024; its
    # linear_fc2 is [768, 3072])
    M, D = C * v["spatial_merge_size"] ** 2, cfg["text_config"]["hidden_size"]
    parts = {}
    for N in map(int, a.buckets.split(",")):
        parts[f"E{N}"], parts[f"B{N}"], parts[f"M{N}"] = embed(N, C, K), block(N, C, H, F), merger(N, C, M, D)
        print("built N", N, flush=True)
    print("saved", save_multifunction(parts, f"{a.out}/{a.name}.mlpackage"))
