"""Small's text backbone (Qwen3: 28 layers, hidden 1024, 16 query / 8 key-value heads of 128, per-head
q/k RMSNorm, causal) as one Core ML program for the Neural Engine, one fixed-shape function per
sequence length sharing one copy of the weights - the same contract as build_nano_text.py, so the
app's ANETextBackend runs either.

What Small adds over Nano, each kept on convolutions the Neural Engine runs natively (a reshape to
(H, D, L) is what pushed the 2026-09 port off it):
  - PER-HEAD q/k NORM by indicator convolutions: the sum of a head's squares is a 1x1 conv against a
    (heads, C) matrix of ones, and the per-head scale goes back out through its transpose.
  - RoPE through the norm: rotate-half permutes channels inside a head, which leaves the head's RMS
    alone, so rope(norm(q)) = (q * (w * cos) + (P q) * (w[perm] * sin)) / rms(q), with P q its own
    conv (P W) and w, w[perm] the norm weight tiled over heads (x 1/sqrt(d) for q).
  - GROUPED-QUERY ATTENTION: query head h reads key-value head h // 2.
  - CAUSAL: an input mask (1, L, 1, L) of 0 and -30000, keys on axis 1. Right padding needs no mask
    of its own under it, and the last real token is picked.
Inputs x (1, C, 1, L), cos, sin (1, 128-periodic over C... per q channel layout), cmask, pick.

  uv run --python 3.12 --with coremltools --with mlx Tools/ane/build_small_text.py <weights> --out <dir>
"""
import argparse, os, shutil, tempfile
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from anelib import query_block

EPS, HEADS, KV, HDIM, LAYERS, HID = 1e-6, 16, 8, 128, 28, 1024
QC, KC = HEADS * HDIM, KV * HDIM


def rotate_half_rows(m, heads):
    o = np.empty_like(m); half = HDIM // 2
    for h in range(heads):
        a = h * HDIM
        o[a:a + half] = -m[a + half:a + HDIM]
        o[a + half:a + HDIM] = m[a:a + half]
    return o


def perm(v, heads):
    """w[perm]: the norm weight as the rotated channels see it (no sign: the sign is in P W)."""
    o = np.empty_like(v); half = HDIM // 2
    for h in range(heads):
        a = h * HDIM
        o[a:a + half] = v[a + half:a + HDIM]
        o[a + half:a + HDIM] = v[a:a + half]
    return o


def indicator(heads):
    I = np.zeros((heads, heads * HDIM), dtype=np.float32)
    for h in range(heads): I[h, h * HDIM:(h + 1) * HDIM] = 1
    return I


def build(W, L):
    f32 = {k: np.array(v.astype(mx.float32), copy=False) for k, v in W.items()}
    cw = lambda m: m.astype(np.float16)[:, :, None, None]
    col = lambda v: v.astype(np.float16).reshape(1, -1, 1, 1)
    Iq, Ik = indicator(HEADS), indicator(KV)
    scale = HDIM ** -0.5

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HID, 1, L), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, QC, 1, L), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, QC, 1, L), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, L, 1, L), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, 1, 1, L), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x, cos, sin, cmask, pick):
        def scaled(t, tag, axis_split=None):
            """t / max|t| per token (per head when `axis_split` heads): every square in [0, 1]."""
            if axis_split is None:
                m = mb.reduce_max(x=mb.abs(x=t, name=f"{tag}_abs"), axes=[1], keep_dims=True, name=f"{tag}_max")
                return mb.real_div(x=t, y=mb.add(x=m, y=np.float16(1e-4), name=f"{tag}_s"), name=f"{tag}_y"), None
            heads, I = axis_split
            parts = mb.split(x=mb.abs(x=t, name=f"{tag}_abs"), num_splits=heads, axis=1, name=f"{tag}_hs")
            m = mb.concat(values=[mb.reduce_max(x=pt, axes=[1], keep_dims=True, name=f"{tag}_m{j}") for j, pt in enumerate(parts)],
                          axis=1, name=f"{tag}_mh")                                                       # (1, heads, 1, L)
            inv = mb.real_div(x=np.float16(1), y=mb.add(x=m, y=np.float16(1e-4), name=f"{tag}_s"), name=f"{tag}_is")
            invc = mb.conv(x=inv, weight=I.T.copy().astype(np.float16)[:, :, None, None], name=f"{tag}_isc")
            return mb.mul(x=t, y=invc, name=f"{tag}_y"), invc

        def rms(t, w, c, tag):
            # SCALE OUT FIRST (measured: Small's attention sink, token 0, carries ~23,900 from layer 3
            # on, so a sum of (x/64)^2 reached 139,000 and overflowed fp16). RMSNorm does not see the
            # scale: x / rms(x) = y / sqrt(mean(y^2)) for y = x / max|x|.
            y, _ = scaled(t, tag)
            ms = mb.reduce_mean(x=mb.mul(x=y, y=y, name=f"{tag}_sq"), axes=[1], keep_dims=True, name=f"{tag}_ms")
            inv = mb.rsqrt(x=mb.add(x=ms, y=np.float16(1e-5), name=f"{tag}_e"), name=f"{tag}_inv")
            return mb.mul(x=mb.mul(x=y, y=inv, name=f"{tag}_n"), y=col(w), name=f"{tag}_w")

        def head_rope(t, tP, I, wn, heads, s, tag):
            # per head: y = t / max|t|, the same scale for P t (P permutes inside a head), then
            # 1/rms from the per-head mean of y^2 (an indicator conv), out through I^T.
            y, invs = scaled(t, tag, (heads, I))
            yP = mb.mul(x=tP, y=invs, name=f"{tag}_yp")
            ss = mb.conv(x=mb.mul(x=y, y=y, name=f"{tag}_sq"), weight=(I / HDIM).astype(np.float16)[:, :, None, None], name=f"{tag}_ss")
            inv = mb.rsqrt(x=mb.add(x=ss, y=np.float16(1e-5), name=f"{tag}_e"), name=f"{tag}_inv")
            invc = mb.conv(x=inv, weight=I.T.copy().astype(np.float16)[:, :, None, None], name=f"{tag}_invc")
            c = cos if heads == HEADS else mb.slice_by_size(x=cos, begin=[0, 0, 0, 0], size=[-1, KC, -1, -1], name=f"{tag}_cs")
            si = sin if heads == HEADS else mb.slice_by_size(x=sin, begin=[0, 0, 0, 0], size=[-1, KC, -1, -1], name=f"{tag}_ss2")
            wt = np.tile(wn, heads) * s
            num = mb.add(x=mb.mul(x=mb.mul(x=y, y=col(wt), name=f"{tag}_tw"), y=c, name=f"{tag}_tc"),
                         y=mb.mul(x=mb.mul(x=yP, y=col(perm(wt, heads)), name=f"{tag}_pw"), y=si, name=f"{tag}_ps"), name=f"{tag}_num")
            return mb.mul(x=num, y=invc, name=f"{tag}_rot")

        # Queries in blocks of 256 (build_nano_text.py), the causal mask cut into the same blocks once.
        NB = L // query_block(L, 256)
        masks = [cmask] if NB == 1 else mb.split(x=cmask, num_splits=NB, axis=3, name="cmask_qb")
        h = x
        for i in range(LAYERS):
            p = f"language_model.layers.{i}."; t = f"l{i}"
            xn = rms(h, f32[p + "input_layernorm.weight"], HID, f"{t}_in")
            wq, wk = f32[p + "self_attn.q_proj.weight"], f32[p + "self_attn.k_proj.weight"]
            q = head_rope(mb.conv(x=xn, weight=cw(wq), name=f"{t}_q"), mb.conv(x=xn, weight=cw(rotate_half_rows(wq, HEADS)), name=f"{t}_qp"),
                          Iq, f32[p + "self_attn.q_norm.weight"], HEADS, scale, f"{t}_qr")
            k = head_rope(mb.conv(x=xn, weight=cw(wk), name=f"{t}_k"), mb.conv(x=xn, weight=cw(rotate_half_rows(wk, KV)), name=f"{t}_kp"),
                          Ik, f32[p + "self_attn.k_norm.weight"], KV, 1.0, f"{t}_kr")
            v = mb.conv(x=xn, weight=cw(f32[p + "self_attn.v_proj.weight"]), name=f"{t}_v")
            qs = mb.split(x=q, num_splits=HEADS, axis=1, name=f"{t}_qsplit")
            ks = mb.split(x=mb.transpose(x=k, perm=[0, 3, 2, 1], name=f"{t}_kT"), num_splits=KV, axis=3, name=f"{t}_ksplit")
            vs = mb.split(x=v, num_splits=KV, axis=1, name=f"{t}_vsplit")
            outs = []
            for hd in range(HEADS):
                g = hd // (HEADS // KV)
                parts = []
                for j, qp in enumerate([qs[hd]] if NB == 1 else mb.split(x=qs[hd], num_splits=NB, axis=3, name=f"{t}_h{hd}_qb")):
                    s_ = mb.einsum(values=(ks[g], qp), equation="bkhc,bchq->bkhq", name=f"{t}_h{hd}_{j}_qk")
                    a = mb.softmax(x=mb.add(x=s_, y=masks[j], name=f"{t}_h{hd}_{j}_m"), axis=1, name=f"{t}_h{hd}_{j}_sm")
                    parts.append(mb.einsum(values=(vs[g], a), equation="bchk,bkhq->bchq", name=f"{t}_h{hd}_{j}_av"))
                outs.append(parts[0] if NB == 1 else mb.concat(values=parts, axis=3, name=f"{t}_h{hd}_cat"))
            ctx = mb.concat(values=outs, axis=1, name=f"{t}_ctx")
            h = mb.add(x=h, y=mb.conv(x=ctx, weight=cw(f32[p + "self_attn.o_proj.weight"]), name=f"{t}_o"), name=f"{t}_r1")
            xn2 = rms(h, f32[p + "post_attention_layernorm.weight"], HID, f"{t}_post")
            g_ = mb.conv(x=xn2, weight=cw(f32[p + "mlp.gate_proj.weight"]), name=f"{t}_g")
            u = mb.conv(x=xn2, weight=cw(f32[p + "mlp.up_proj.weight"]), name=f"{t}_u")
            act = mb.mul(x=mb.mul(x=g_, y=mb.sigmoid(x=g_, name=f"{t}_sig"), name=f"{t}_silu"), y=u, name=f"{t}_gu")
            h = mb.add(x=h, y=mb.conv(x=act, weight=cw(f32[p + "mlp.down_proj.weight"]), name=f"{t}_d"), name=f"{t}_r2")
        # Every position final-normed (an image's tags are scored on its rows), and the picked one.
        hn = rms(h, f32["language_model.norm.weight"], HID, "final")
        pooled = mb.reduce_sum(x=mb.mul(x=hn, y=pick, name="pick"), axes=[3], keep_dims=True, name="pooled")
        return pooled, mb.identity(x=hn, name="hidden")

    return ct.convert(prog, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="32,64,128,256,512,1024,2048")
    a = ap.parse_args()
    W = mx.load(a.weights)
    lengths = list(map(int, a.lengths.split(",")))
    tmp = tempfile.mkdtemp()
    desc = ct.utils.MultiFunctionDescriptor()
    for L in lengths:
        part = os.path.join(tmp, f"L{L}.mlpackage")
        build(W, L).save(part)
        desc.add_function(part, src_function_name="main", target_function_name=f"L{L}")
        print("built L", L, flush=True)
    desc.default_function_name = f"L{lengths[-1]}"
    path = f"{a.out}/small_text.mlpackage"
    if os.path.exists(path): shutil.rmtree(path)
    ct.utils.save_multifunction(desc, path)
    shutil.rmtree(tmp)
    print("saved", path)
