"""Nano's text backbone as a Core ML query encoder for the Neural Engine, one program per bucket L.

ANE-shaped from the start (ml-ane-transformers recipe), because the 2026-09 Small port showed a
reshape-to-(H, D, L) attention falls off the ANE:
  - layout (1, C, 1, L), linears as 1x1 convs;
  - heads by `split` on channels (keys transposed to (1, L, 1, C) first), attention per head as
    the two einsums Apple's recipe uses, softmax over the key axis;
  - RoPE folded into the projection: q = conv(x, Wq) * cos + conv(x, P Wq) * sin, where P is the
    rotate-half permutation with signs, so no slicing or concat per head; the 1/sqrt(d) scale is
    folded into q's tables;
  - RMSNorm without an fp16 overflow: the l2 norm of x/64, then rescaled (LayerNorm of [x, -x],
    ANEMLL's form, measured the same: median cosine 0.99903 to fp32 either way);
  - inputs are the embedded tokens (the host gathers rows from the app's own table), an additive
    key mask (1, L, 1, 1) and a one-hot pick (1, 1, 1, L) for the last real token. Output is the
    final-normed pooled hidden (1, C, 1, 1); the host L2-normalizes in fp32.
Weights: `omni-verify dumpbackbone` with OMNI_BACKBONE_DTYPE=fp16 (the app's merged tensors).
  uv run --python 3.12 --with coremltools --with mlx Tools/ane/build_nano_query.py <weights> --out <dir>
then `omni-verify anequery <nano> <dir> <imageDir>` (index.md, "The query that waited out the hold").
"""
import argparse
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

EPS, THETA, HEADS, HDIM, LAYERS, HID = 1e-5, 1_000_000.0, 12, 64, 12, 768
PRECISION = "fp16"


def rope_tables(L, scale):
    half = HDIM // 2
    inv = 1.0 / (THETA ** (np.arange(half, dtype=np.float64) * 2.0 / HDIM))
    ang = np.outer(inv, np.arange(L, dtype=np.float64))                     # [half, L]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], 0)                     # [HDIM, L]
    sin = np.concatenate([np.sin(ang), np.sin(ang)], 0)
    cos = np.tile(cos, (HEADS, 1)) * scale                                  # [C, L]
    sin = np.tile(sin, (HEADS, 1)) * scale
    return cos[None, :, None, :].astype(np.float16), sin[None, :, None, :].astype(np.float16)


def rotate_half_matrix():
    # rotate_half(x) = [-x2, x1] per head, as y = P x
    P = np.zeros((HID, HID), dtype=np.float32)
    half = HDIM // 2
    for h in range(HEADS):
        o = h * HDIM
        for j in range(half):
            P[o + j, o + half + j] = -1.0
            P[o + half + j, o + j] = 1.0
    return P


def build(W, L):
    f32 = {k: np.array(v.astype(mx.float32), copy=False) for k, v in W.items()}
    P = rotate_half_matrix()
    cq, sq = rope_tables(L, HDIM ** -0.5)
    ck, sk = rope_tables(L, 1.0)

    def cw(m):                                  # [out, in] -> (out, in, 1, 1) fp16
        return m.astype(np.float16)[:, :, None, None]

    def nw(name):
        return f32[name].astype(np.float16).reshape(1, HID, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HID, 1, L), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, L, 1, 1), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, 1, 1, L), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x, mask, pick):
        def rms(t, wname, tag):
            n = mb.reduce_l2_norm(x=mb.mul(x=t, y=np.float16(1 / 64), name=f"{tag}_s"), axes=[1],
                                  keep_dims=True, name=f"{tag}_l2")
            inv = mb.real_div(x=np.float16(np.sqrt(HID) / 64), y=mb.add(x=n, y=np.float16(1e-4), name=f"{tag}_e"),
                              name=f"{tag}_inv")
            return mb.mul(x=mb.mul(x=t, y=inv, name=f"{tag}_n"), y=nw(wname), name=f"{tag}_w")

        h = x
        for i in range(LAYERS):
            p = f"language_model.layers.{i}."
            t = f"l{i}"
            xn = rms(h, p + "input_layernorm.weight", f"{t}_in")
            wq, wk = f32[p + "self_attn.q_proj.weight"], f32[p + "self_attn.k_proj.weight"]
            q = mb.add(x=mb.mul(x=mb.conv(x=xn, weight=cw(wq), name=f"{t}_q"), y=cq, name=f"{t}_qc"),
                       y=mb.mul(x=mb.conv(x=xn, weight=cw(P @ wq), name=f"{t}_qp"), y=sq, name=f"{t}_qs"),
                       name=f"{t}_qr")
            k = mb.add(x=mb.mul(x=mb.conv(x=xn, weight=cw(wk), name=f"{t}_k"), y=ck, name=f"{t}_kc"),
                       y=mb.mul(x=mb.conv(x=xn, weight=cw(P @ wk), name=f"{t}_kp"), y=sk, name=f"{t}_ks"),
                       name=f"{t}_kr")
            v = mb.conv(x=xn, weight=cw(f32[p + "self_attn.v_proj.weight"]), name=f"{t}_v")
            qs = mb.split(x=q, num_splits=HEADS, axis=1, name=f"{t}_qsplit")
            kT = mb.transpose(x=k, perm=[0, 3, 2, 1], name=f"{t}_kT")                 # (1, L, 1, C)
            ks = mb.split(x=kT, num_splits=HEADS, axis=3, name=f"{t}_ksplit")
            vs = mb.split(x=v, num_splits=HEADS, axis=1, name=f"{t}_vsplit")
            outs = []
            for hd in range(HEADS):
                s = mb.einsum(values=(ks[hd], qs[hd]), equation="bkhc,bchq->bkhq", name=f"{t}_h{hd}_qk")
                s = mb.add(x=s, y=mask, name=f"{t}_h{hd}_m")
                a = mb.softmax(x=s, axis=1, name=f"{t}_h{hd}_sm")
                outs.append(mb.einsum(values=(vs[hd], a), equation="bchk,bkhq->bchq", name=f"{t}_h{hd}_av"))
            ctx = mb.concat(values=outs, axis=1, name=f"{t}_ctx")
            h = mb.add(x=h, y=mb.conv(x=ctx, weight=cw(f32[p + "self_attn.o_proj.weight"]), name=f"{t}_o"),
                       name=f"{t}_r1")
            xn2 = rms(h, p + "post_attention_layernorm.weight", f"{t}_post")
            g = mb.conv(x=xn2, weight=cw(f32[p + "mlp.gate_proj.weight"]), name=f"{t}_g")
            u = mb.conv(x=xn2, weight=cw(f32[p + "mlp.up_proj.weight"]), name=f"{t}_u")
            act = mb.mul(x=mb.mul(x=g, y=mb.sigmoid(x=g, name=f"{t}_sig"), name=f"{t}_silu"), y=u, name=f"{t}_gu")
            h = mb.add(x=h, y=mb.conv(x=act, weight=cw(f32[p + "mlp.down_proj.weight"]), name=f"{t}_d"),
                       name=f"{t}_r2")
        pooled = mb.reduce_sum(x=mb.mul(x=h, y=pick, name="pick"), axes=[3], keep_dims=True, name="pool")
        return rms(pooled, "language_model.norm.weight", "final")

    return ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT32 if PRECISION == "fp32" else ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("--lengths", default="16,32,64"); ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="fp16", help="fp32: the same program in fp32, a maths check on the CPU")
    a = ap.parse_args()
    PRECISION = a.precision
    W = mx.load(a.weights)
    for L in map(int, a.lengths.split(",")):
        m = build(W, L)
        path = f"{a.out}/nano_query_L{L}.mlpackage"
        m.save(path)
        print("saved", path)
