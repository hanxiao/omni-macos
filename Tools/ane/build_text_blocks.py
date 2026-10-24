"""The text backbone past its baked program's longest length (2,048), for the long sequences a video
segment makes (a 1080p clip ~4,130 tokens, square ~6,090): ONE LAYER as a block program, no weights
inside (anelib.py), called once per layer with that layer's weights, which ANETextBackend reads from
the baked program's own weight file (the same fp16 tensors, already scaled and rotated: no copy).
The layer is the baked one's (build_nano_text.py, build_small_text.py) with every weight an input
and the linears as matmuls; rotate-half its grouped conv, a constant of the graph. Output `out`, the layer's hidden (1, C, 1, L); the final norm and the
pick are done on the CPU.

Measured at 6,144 tokens (index.md, "The long sequences"): baked, Nano took 742 s to compile and
1.2 s a call, Small at 4,608 took 74 min and 3.0 s. As a block, queries in blocks of 128: Nano 17 s
and 52 ms a layer (0.63 s a call); Small, its causal attention reading only the keys a block can
see, 30 s and 77 ms a layer (2.2 s). Blocks of 256 ran 127 / 222 ms a layer; of 64, 51 / 67 ms for
three times the compile.

  uv run --python 3.12 --with coremltools python Tools/ane/build_text_blocks.py <nano|small> --out <dir>
"""
import argparse
import numpy as np
from coremltools.converters.mil import Builder as mb
from anelib import spec, program, lin, attention, query_block, save_multifunction

# The segment shapes past 2,048 at OmniVideoPreprocess's 6 frames (3 temporal slices) and the
# wrapper's tokens: 720p 4:3 ~2,085, 720p 16:9 ~2,775, 1080p 16:9 ~4,131, 3:2 ~4,866, 4:3 ~5,454,
# square ~6,090; the pixel budget caps any clip at 6,272 media tokens.
LENGTHS = "2560,3072,4224,4992,5504,6400"
QBLOCK = 128


def rotate_groups(heads, d):
    """rotate_half per head as a grouped 1x1 conv's weight: the baked programs' (exact)."""
    R = np.zeros((heads * d, d, 1, 1), dtype=np.float16); half = d // 2
    for h in range(heads):
        for j in range(half):
            R[h * d + j, half + j] = -1
            R[h * d + half + j, j] = 1
    return R


def nano_layer(L, C=768, H=12, F=3072):
    """build_nano_text.py's layer: RMSNorm as the l2 norm of x/64, RoPE folded into q and k (Wq
    carries 1/sqrt(d)), bidirectional attention with a key mask for the padding."""
    vec = lambda c=C: spec(1, c, 1, 1)
    specs = dict(h=spec(1, C, 1, L), cos=spec(1, C, 1, L), sin=spec(1, C, 1, L), mask=spec(1, L, 1, 1),
                 g1=vec(), wq=spec(C, C), wk=spec(C, C), wv=spec(C, C), wo=spec(C, C),
                 g2=vec(), wg=spec(F, C), wu=spec(F, C), wd=spec(C, F))
    R = rotate_groups(H, C // H)

    def rms(t, g, tag):
        n = mb.reduce_l2_norm(x=mb.mul(x=t, y=np.float16(1 / 64), name=f"{tag}_s"), axes=[1], keep_dims=True, name=f"{tag}_l2")
        inv = mb.real_div(x=np.float16(np.sqrt(C) / 64), y=mb.add(x=n, y=np.float16(1e-4), name=f"{tag}_e"), name=f"{tag}_inv")
        return mb.mul(x=mb.mul(x=t, y=inv, name=f"{tag}_n"), y=g, name=f"{tag}_w")

    def body(h, cos, sin, mask, g1, wq, wk, wv, wo, g2, wg, wu, wd):
        xn = rms(h, g1, "n1")

        def rope(w, t):
            y = lin(xn, w, None, C, L, t)
            return mb.add(x=mb.mul(x=y, y=cos, name=f"{t}_c"),
                          y=mb.mul(x=mb.conv(x=y, weight=R, groups=H, name=f"{t}p"), y=sin, name=f"{t}_s"), name=f"{t}_r")
        q, k, v = rope(wq, "q"), rope(wk, "k"), lin(xn, wv, None, C, L, "v")
        h = mb.add(x=h, y=lin(attention(q, k, v, mask, H, L, "at", most=QBLOCK), wo, None, C, L, "o"), name="r1")
        xn2 = rms(h, g2, "n2")
        g = lin(xn2, wg, None, F, L, "g")
        act = mb.mul(x=mb.mul(x=g, y=mb.sigmoid(x=g, name="sig"), name="silu"), y=lin(xn2, wu, None, F, L, "u"), name="gu")
        return mb.add(x=h, y=lin(act, wd, None, C, L, "d"), name="out")

    return program(specs, body)


def small_layer(L, C=1024, H=16, KV=8, D=128, F=3072):
    """build_small_text.py's layer: RMSNorm scaled by max|x| first (the attention sink), per-head
    q/k norm with RoPE through it (qn/qnp, kn/knp: the norm weight tiled and permuted, q's times
    1/sqrt(d)), grouped-query causal attention. Right padding needs no mask under causal."""
    QC, KC = H * D, KV * D
    vec = lambda c=C: spec(1, c, 1, 1)

    def indicator(heads):
        I = np.zeros((heads, heads * D), dtype=np.float32)
        for j in range(heads): I[j, j * D:(j + 1) * D] = 1
        return I
    Iq, Ik = indicator(H), indicator(KV)
    specs = dict(h=spec(1, C, 1, L), cos=spec(1, QC, 1, L), sin=spec(1, QC, 1, L),
                 g1=vec(), wq=spec(QC, C), wk=spec(KC, C), wv=spec(KC, C),
                 qn=vec(QC), qnp=vec(QC), kn=vec(KC), knp=vec(KC), wo=spec(C, QC),
                 g2=vec(), wg=spec(F, C), wu=spec(F, C), wd=spec(C, F))

    def scaled(t, tag, split=None):
        if split is None:
            m = mb.reduce_max(x=mb.abs(x=t, name=f"{tag}_abs"), axes=[1], keep_dims=True, name=f"{tag}_max")
            return mb.real_div(x=t, y=mb.add(x=m, y=np.float16(1e-4), name=f"{tag}_s"), name=f"{tag}_y"), None
        heads, I = split
        parts = mb.split(x=mb.abs(x=t, name=f"{tag}_abs"), num_splits=heads, axis=1, name=f"{tag}_hs")
        m = mb.concat(values=[mb.reduce_max(x=pt, axes=[1], keep_dims=True, name=f"{tag}_m{j}") for j, pt in enumerate(parts)],
                      axis=1, name=f"{tag}_mh")
        inv = mb.real_div(x=np.float16(1), y=mb.add(x=m, y=np.float16(1e-4), name=f"{tag}_s"), name=f"{tag}_is")
        invc = mb.conv(x=inv, weight=I.T.copy().astype(np.float16)[:, :, None, None], name=f"{tag}_isc")
        return mb.mul(x=t, y=invc, name=f"{tag}_y"), invc

    def rms(t, g, tag):
        y, _ = scaled(t, tag)
        ms = mb.reduce_mean(x=mb.mul(x=y, y=y, name=f"{tag}_sq"), axes=[1], keep_dims=True, name=f"{tag}_ms")
        inv = mb.rsqrt(x=mb.add(x=ms, y=np.float16(1e-5), name=f"{tag}_e"), name=f"{tag}_inv")
        return mb.mul(x=mb.mul(x=y, y=inv, name=f"{tag}_n"), y=g, name=f"{tag}_w")

    def head_rope(t, tP, I, wn, wnp, heads, cos, sin, tag):
        y, invs = scaled(t, tag, (heads, I))
        yP = mb.mul(x=tP, y=invs, name=f"{tag}_yp")
        ss = mb.conv(x=mb.mul(x=y, y=y, name=f"{tag}_sq"), weight=(I / D).astype(np.float16)[:, :, None, None], name=f"{tag}_ss")
        inv = mb.rsqrt(x=mb.add(x=ss, y=np.float16(1e-5), name=f"{tag}_e"), name=f"{tag}_inv")
        invc = mb.conv(x=inv, weight=I.T.copy().astype(np.float16)[:, :, None, None], name=f"{tag}_invc")
        c = cos if heads == H else mb.slice_by_size(x=cos, begin=[0, 0, 0, 0], size=[-1, KC, -1, -1], name=f"{tag}_cs")
        s = sin if heads == H else mb.slice_by_size(x=sin, begin=[0, 0, 0, 0], size=[-1, KC, -1, -1], name=f"{tag}_ss2")
        num = mb.add(x=mb.mul(x=mb.mul(x=y, y=wn, name=f"{tag}_tw"), y=c, name=f"{tag}_tc"),
                     y=mb.mul(x=mb.mul(x=yP, y=wnp, name=f"{tag}_pw"), y=s, name=f"{tag}_ps"), name=f"{tag}_num")
        return mb.mul(x=num, y=invc, name=f"{tag}_rot")

    Rq, Rk = rotate_groups(H, D), rotate_groups(KV, D)

    def body(h, cos, sin, g1, wq, wk, wv, qn, qnp, kn, knp, wo, g2, wg, wu, wd):
        bq = query_block(L, QBLOCK)
        NB = L // bq
        # A query block sees the keys before it unmasked and its own block under the triangle.
        tri = np.where(np.arange(bq)[:, None] <= np.arange(bq)[None, :], 0, -30000).astype(np.float16).reshape(1, bq, 1, bq)
        xn = rms(h, g1, "n1")
        q0, k0 = lin(xn, wq, None, QC, L, "q"), lin(xn, wk, None, KC, L, "k")
        q = head_rope(q0, mb.conv(x=q0, weight=Rq, groups=H, name="qp"), Iq, qn, qnp, H, cos, sin, "qr")
        k = head_rope(k0, mb.conv(x=k0, weight=Rk, groups=KV, name="kp"), Ik, kn, knp, KV, cos, sin, "kr")
        v = lin(xn, wv, None, KC, L, "v")
        qs = mb.split(x=q, num_splits=H, axis=1, name="qsplit")
        ks = mb.split(x=mb.transpose(x=k, perm=[0, 3, 2, 1], name="kT"), num_splits=KV, axis=3, name="ksplit")
        vs = mb.split(x=v, num_splits=KV, axis=1, name="vsplit")
        outs, seen = [], {}
        for hd in range(H):
            g = hd // (H // KV)
            parts = []
            for j, qp in enumerate([qs[hd]] if NB == 1 else mb.split(x=qs[hd], num_splits=NB, axis=3, name=f"h{hd}_qb")):
                if (g, j) not in seen:
                    K = (j + 1) * bq
                    seen[(g, j)] = (mb.slice_by_size(x=ks[g], begin=[0, 0, 0, 0], size=[-1, K, -1, -1], name=f"k{g}_{j}"),
                                    mb.slice_by_size(x=vs[g], begin=[0, 0, 0, 0], size=[-1, -1, -1, K], name=f"v{g}_{j}"))
                kj, vj = seen[(g, j)]
                s = mb.einsum(values=(kj, qp), equation="bkhc,bchq->bkhq", name=f"h{hd}_{j}_qk")
                if j == 0:
                    s = mb.add(x=s, y=tri, name=f"h{hd}_{j}_m")
                else:
                    before, own = mb.split(x=s, split_sizes=[j * bq, bq], axis=1, name=f"h{hd}_{j}_sp")
                    s = mb.concat(values=[before, mb.add(x=own, y=tri, name=f"h{hd}_{j}_m")], axis=1, name=f"h{hd}_{j}_mc")
                a = mb.softmax(x=s, axis=1, name=f"h{hd}_{j}_sm")
                parts.append(mb.einsum(values=(vj, a), equation="bchk,bkhq->bchq", name=f"h{hd}_{j}_av"))
            outs.append(parts[0] if NB == 1 else mb.concat(values=parts, axis=3, name=f"h{hd}_cat"))
        h = mb.add(x=h, y=lin(mb.concat(values=outs, axis=1, name="ctx"), wo, None, C, L, "o"), name="r1")
        xn2 = rms(h, g2, "n2")
        gg = lin(xn2, wg, None, F, L, "g")
        act = mb.mul(x=mb.mul(x=gg, y=mb.sigmoid(x=gg, name="sig"), name="silu"), y=lin(xn2, wu, None, F, L, "u"), name="gu")
        return mb.add(x=h, y=lin(act, wd, None, C, L, "d"), name="out")

    return program(specs, body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["nano", "small"]); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default=LENGTHS)
    a = ap.parse_args()
    make = nano_layer if a.model == "nano" else small_layer
    parts = {}
    for L in map(int, a.lengths.split(",")):
        parts[f"B{L}"] = make(L)
        print("built L", L, flush=True)
    print("saved", save_multifunction(parts, f"{a.out}/{a.model}_textblocks.mlpackage"))
