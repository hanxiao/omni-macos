"""Build Omni's text tower as a CoreML program for the Neural Engine.

Layout is (1, C, 1, L): channels first, tokens on W. Linears become 1x1 convolutions, which is
the spelling the ANE is fastest at (measured in `ocr-verify --probe-ane`). Attention drops to
(H, D, L) for the score/!value matmuls and comes back.

Weights come from `omni-verify dumpbackbone`, i.e. the app's own LoRA-merged fp16 tensors, so
any mismatch against Tools/ane/tower.py is this file's fault and not the weights'.
"""
from __future__ import annotations
import argparse
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

EPS, THETA = 1e-6, 3_500_000.0
FP16_NORM = False
HEADS, KV, HDIM, LAYERS, HID = 16, 8, 128, 28, 1024


def rope_tables(L: int):
    half = HDIM // 2
    inv = 1.0 / (THETA ** (np.arange(half, dtype=np.float64) * 2.0 / HDIM))
    ang = np.outer(np.arange(L, dtype=np.float64), inv)          # [L, half]
    # (1, half, L) so it broadcasts over heads in the (H, D, L) layout
    return (np.cos(ang).T[None].astype(np.float16),
            np.sin(ang).T[None].astype(np.float16))


def causal_mask(L: int):
    m = np.zeros((1, L, L), dtype=np.float16)
    m[0][np.triu_indices(L, k=1)] = -65504.0                      # fp16 min, not -inf
    return m


def build(weights: dict, L: int):
    W = {k: np.array(v.astype(mx.float16), copy=False) for k, v in weights.items()}
    cosT, sinT = rope_tables(L)
    maskC = causal_mask(L)

    def conv_w(name):                      # [out, in] -> (out, in, 1, 1)
        return W[name][:, :, None, None]

    def norm_w(name, c):                   # [C] -> (1, C, 1, 1)
        return W[name].reshape(1, c, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HID, 1, L), dtype=types.fp16)],
                opset_version=ct.target.iOS16)
    def prog(x):
        def rms(t, wname, c, tag):
            if FP16_NORM:
                sq = mb.mul(x=t, y=t, name=f"{tag}_sq16")
                mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{tag}_mean16")
                inv = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS), name=f"{tag}_eps16"),
                               name=f"{tag}_rs16")
                return mb.mul(x=mb.mul(x=t, y=inv, name=f"{tag}_nrm16"),
                              y=norm_w(wname, c), name=f"{tag}_w16")
            f = mb.cast(x=t, dtype="fp32", name=f"{tag}_f32")
            sq = mb.mul(x=f, y=f, name=f"{tag}_sq")
            mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{tag}_mean")
            inv = mb.rsqrt(x=mb.add(x=mean, y=np.float32(EPS), name=f"{tag}_eps"), name=f"{tag}_rs")
            nrm = mb.mul(x=f, y=inv, name=f"{tag}_nrm")
            out = mb.mul(x=nrm, y=norm_w(wname, c).astype(np.float32), name=f"{tag}_w")
            return mb.cast(x=out, dtype="fp16", name=f"{tag}_out")

        def rope(t, heads, tag):
            # (heads, HDIM, L) -> split the channel axis in half (non-traditional / half-split)
            a, b = mb.split(x=t, num_splits=2, axis=1, name=f"{tag}_split")
            ac = mb.mul(x=a, y=cosT, name=f"{tag}_ac"); bs = mb.mul(x=b, y=sinT, name=f"{tag}_bs")
            bc = mb.mul(x=b, y=cosT, name=f"{tag}_bc"); as_ = mb.mul(x=a, y=sinT, name=f"{tag}_as")
            return mb.concat(values=[mb.sub(x=ac, y=bs, name=f"{tag}_lo"),
                                     mb.add(x=bc, y=as_, name=f"{tag}_hi")],
                             axis=1, name=f"{tag}_rope")

        h = x
        for i in range(LAYERS):
            p = f"language_model.layers.{i}."
            t = f"l{i}"
            xn = rms(h, p + "input_layernorm.weight", HID, f"{t}_in")

            q = mb.conv(x=xn, weight=conv_w(p + "self_attn.q_proj.weight"), name=f"{t}_q")
            k = mb.conv(x=xn, weight=conv_w(p + "self_attn.k_proj.weight"), name=f"{t}_k")
            v = mb.conv(x=xn, weight=conv_w(p + "self_attn.v_proj.weight"), name=f"{t}_v")

            q = mb.reshape(x=q, shape=(HEADS, HDIM, L), name=f"{t}_qh")
            k = mb.reshape(x=k, shape=(KV, HDIM, L), name=f"{t}_kh")
            v = mb.reshape(x=v, shape=(KV, HDIM, L), name=f"{t}_vh")

            # per-head RMSNorm over HDIM, which is axis 1 in this layout
            def head_norm(t_in, wname, tag):
                if FP16_NORM:
                    sq = mb.mul(x=t_in, y=t_in, name=f"{tag}_sq16")
                    mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{tag}_mean16")
                    inv = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS), name=f"{tag}_eps16"),
                                   name=f"{tag}_rs16")
                    return mb.mul(x=mb.mul(x=t_in, y=inv, name=f"{tag}_nrm16"),
                                  y=W[wname].reshape(1, HDIM, 1), name=f"{tag}_w16")
                f = mb.cast(x=t_in, dtype="fp32", name=f"{tag}_f32")
                sq = mb.mul(x=f, y=f, name=f"{tag}_sq")
                mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{tag}_mean")
                inv = mb.rsqrt(x=mb.add(x=mean, y=np.float32(EPS), name=f"{tag}_eps"), name=f"{tag}_rs")
                nrm = mb.mul(x=f, y=inv, name=f"{tag}_nrm")
                out = mb.mul(x=nrm, y=W[wname].reshape(1, HDIM, 1).astype(np.float32), name=f"{tag}_w")
                return mb.cast(x=out, dtype="fp16", name=f"{tag}_out")

            if p + "self_attn.q_norm.weight" in W:
                q = head_norm(q, p + "self_attn.q_norm.weight", f"{t}_qn")
                k = head_norm(k, p + "self_attn.k_norm.weight", f"{t}_kn")
            q = rope(q, HEADS, f"{t}_qr")
            k = rope(k, KV, f"{t}_kr")

            # GQA: each kv head serves HEADS/KV query heads, contiguously
            k = mb.reshape(x=mb.tile(x=mb.reshape(x=k, shape=(KV, 1, HDIM * L), name=f"{t}_kf"),
                                     reps=(1, HEADS // KV, 1), name=f"{t}_kt"),
                           shape=(HEADS, HDIM, L), name=f"{t}_kg")
            v = mb.reshape(x=mb.tile(x=mb.reshape(x=v, shape=(KV, 1, HDIM * L), name=f"{t}_vf"),
                                     reps=(1, HEADS // KV, 1), name=f"{t}_vt"),
                           shape=(HEADS, HDIM, L), name=f"{t}_vg")

            scores = mb.matmul(x=q, y=k, transpose_x=True, transpose_y=False, name=f"{t}_qk")
            scores = mb.mul(x=scores, y=np.float16(HDIM ** -0.5), name=f"{t}_scale")
            scores = mb.add(x=scores, y=maskC, name=f"{t}_mask")
            probs = mb.softmax(x=scores, axis=-1, name=f"{t}_sm")
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=True, name=f"{t}_av")
            ctx = mb.reshape(x=mb.transpose(x=ctx, perm=[0, 2, 1], name=f"{t}_ct"),
                             shape=(1, HEADS * HDIM, 1, L), name=f"{t}_cr")
            h = mb.add(x=h, y=mb.conv(x=ctx, weight=conv_w(p + "self_attn.o_proj.weight"),
                                      name=f"{t}_o"), name=f"{t}_res1")

            xn2 = rms(h, p + "post_attention_layernorm.weight", HID, f"{t}_post")
            g = mb.conv(x=xn2, weight=conv_w(p + "mlp.gate_proj.weight"), name=f"{t}_g")
            u = mb.conv(x=xn2, weight=conv_w(p + "mlp.up_proj.weight"), name=f"{t}_u")
            act = mb.mul(x=mb.sigmoid(x=g, name=f"{t}_sig"), y=g, name=f"{t}_silu")
            h = mb.add(x=h, y=mb.conv(x=mb.mul(x=act, y=u, name=f"{t}_gu"),
                                      weight=conv_w(p + "mlp.down_proj.weight"), name=f"{t}_d"),
                       name=f"{t}_res2")

        return rms(h, "language_model.norm.weight", HID, "final")

    return ct.convert(prog, minimum_deployment_target=ct.target.macOS14,
                      compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16-norm", action="store_true",
                    help="RMSNorm in fp16: no cast ops, which is what fragments ANE residency")
    a = ap.parse_args()
    globals()['FP16_NORM'] = a.fp16_norm
    m = build(mx.load(a.weights), a.length)
    m.save(a.out)
    print(f"saved {a.out}  L={a.length}")
