"""Nano's text backbone as ONE Core ML program for the Neural Engine (ANETextBackend), one
fixed-shape function per sequence length, sharing one copy of the weights.

ANE-shaped (ml-ane-transformers): (B, C, 1, L), linears as 1x1 convs, heads split on channels (keys
transposed to (B, L, 1, C) first), attention per head as the two einsum forms coremltools accepts,
softmax over the key axis. RoPE folded into the projections - q = conv(x, Wq) * cos + conv(x, P Wq)
* sin, P the signed rotate-half permutation - with cos and sin INPUTS (positions 0 ..< L, written by
the backend), and 1/sqrt(d) folded into Wq (a power of two: exact in fp16). RMSNorm as the l2 norm
of x/64 (no fp16 overflow). Inputs x (B, C, 1, L), cos, sin (1, C, 1, L), mask (B, L, 1, 1), pick
(B, 1, 1, L); output the final-normed hidden at the picked token (B, C, 1, 1).

Weights: `omni-verify dumpbackbone` with OMNI_BACKBONE_DTYPE=fp16 (the app's merged tensors).
  uv run --python 3.12 --with coremltools --with mlx Tools/ane/build_nano_text.py <weights> --out <dir>
"""
import argparse
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

EPS, THETA, HEADS, HDIM, LAYERS, HID = 1e-5, 1_000_000.0, 12, 64, 12, 768
PRECISION = "fp16"


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


def build(W, L, B=1):
    f32 = {k: np.array(v.astype(mx.float32), copy=False) for k, v in W.items()}
    P = rotate_half_matrix()
    Ls = L

    def cw(m):                                  # [out, in] -> (out, in, 1, 1) fp16
        return m.astype(np.float16)[:, :, None, None]

    def nw(name):
        return f32[name].astype(np.float16).reshape(1, HID, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(B, HID, 1, Ls), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, HID, 1, Ls), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, HID, 1, Ls), dtype=types.fp16),
                             mb.TensorSpec(shape=(B, Ls, 1, 1), dtype=types.fp16),
                             mb.TensorSpec(shape=(B, 1, 1, Ls), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x, cos, sin, mask, pick):
        cq, sq, ck, sk = cos, sin, cos, sin
        def rms(t, wname, tag):
            # The l2 norm of x/64: Nano has no activation large enough to overflow it. Scaling by
            # max|x| first (Small's fix) was measured here too: the same parity, 32 tokens 30% slower.
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
            wq, wk = f32[p + "self_attn.q_proj.weight"] * HDIM ** -0.5, f32[p + "self_attn.k_proj.weight"]
            q = mb.add(x=mb.mul(x=mb.conv(x=xn, weight=cw(wq), name=f"{t}_q"), y=cq, name=f"{t}_qc"),
                       y=mb.mul(x=mb.conv(x=xn, weight=cw(P @ wq), name=f"{t}_qp"), y=sq, name=f"{t}_qs"),
                       name=f"{t}_qr")
            k = mb.add(x=mb.mul(x=mb.conv(x=xn, weight=cw(wk), name=f"{t}_k"), y=ck, name=f"{t}_kc"),
                       y=mb.mul(x=mb.conv(x=xn, weight=cw(P @ wk), name=f"{t}_kp"), y=sk, name=f"{t}_ks"),
                       name=f"{t}_kr")
            v = mb.conv(x=xn, weight=cw(f32[p + "self_attn.v_proj.weight"]), name=f"{t}_v")
            qs = mb.split(x=q, num_splits=HEADS, axis=1, name=f"{t}_qsplit")
            kT = mb.transpose(x=k, perm=[0, 3, 2, 1], name=f"{t}_kT")                 # (B, L, 1, C)
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
    ap.add_argument("weights"); ap.add_argument("--lengths", default="32,64,128,256,512,1024,2048"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1, help="sequences per call (indexing shapes)")
    ap.add_argument("--precision", default="fp16", help="fp32: the same program in fp32, a maths check on the CPU")
    a = ap.parse_args()
    PRECISION = a.precision
    W = mx.load(a.weights)
    # ONE FUNCTION PER LENGTH, ONE COPY OF THE WEIGHTS. Enumerated shapes in one function were
    # measured and rejected: 51.5 s to load and every length at about half its fixed-shape speed
    # (E5RT: "tensor_buffer has known strides while the model has FlexibleShapeInfo"). A
    # multifunction package holds fixed-shape functions that share their weight blobs.
    import os, shutil, tempfile
    lengths = list(map(int, a.lengths.split(",")))
    tmp = tempfile.mkdtemp()
    desc = ct.utils.MultiFunctionDescriptor()
    for L in lengths:
        part = os.path.join(tmp, f"L{L}.mlpackage")
        build(W, L, a.batch).save(part)
        desc.add_function(part, src_function_name="main", target_function_name=f"L{L}")
    desc.default_function_name = f"L{lengths[-1]}"
    path = f"{a.out}/nano_text.mlpackage"
    if os.path.exists(path): shutil.rmtree(path)
    ct.utils.save_multifunction(desc, path)
    shutil.rmtree(tmp)
    print("saved", path)
