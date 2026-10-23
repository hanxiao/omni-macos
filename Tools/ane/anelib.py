"""The pieces every Neural Engine block program is built from (build_vision.py, build_audio.py).

A block program carries NO weights: each weight is an input, so one program per input size serves
every layer, and the Neural Engine compiles one layer instead of all of them (a 12-layer vision tower
at 5,120 patches: 15.8 s against 192 s baked; index.md, "The Neural Engine, all or nothing").
Measured and kept:
  - a linear as matmul(W, x) on x reshaped to (C, N) - 38.7 ms a photo-size vision layer, where a
    1x1 convolution with the weight as an input took 45.1 ms (the whole tower 499 against 580 ms);
  - LayerNorm as the fused op, gamma and beta applied after it (the same parity as by hand: on a
    photo the tokens' cosine to fp32 is ~0.978 in either form, from fp16 attention over ~5,000 keys);
  - attention per head (ml-ane-transformers' split einsums) with the queries in blocks of at most
    128 that divide N: 400 ms for the 5,120-patch tower against 750 at 256. All heads in one einsum
    compiled 4x faster and ran 2.6x slower; the fused scaled_dot_product_attention op and weights
    held as Core ML state are both refused by the M3's Neural Engine compiler.
"""
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def query_block(N, most=128):
    """Queries per attention block: the largest multiple of 32 that divides N, at most `most`."""
    return max([b for b in range(32, min(most, N) + 1, 32) if N % b == 0] or [N])


def spec(*shape):
    return mb.TensorSpec(shape=shape, dtype=types.fp16)


def program(specs, body):
    """An fp16 Core ML program whose inputs are `specs` (name -> TensorSpec, in order) and whose
    graph is `body(**inputs)`. mb.program reads parameter names off the function, so spell them out."""
    names = list(specs)
    ns = {"body": body}
    exec(f"def prog({', '.join(names)}): return body({', '.join(f'{n}={n}' for n in names)})", ns)
    prog = mb.program(input_specs=[specs[k] for k in names], opset_version=ct.target.macOS15)(ns["prog"])
    return ct.convert(prog, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)


def lin(x, w, b, cout, n, tag):
    """(B, Cin, 1, n) -> W (Cout, Cin) @ -> (B, Cout, 1, n), plus b (1, Cout, 1, 1) unless None."""
    B, cin = x.shape[0], x.shape[1]
    if B == 1:
        y = mb.matmul(x=w, y=mb.reshape(x=x, shape=[cin, n], name=f"{tag}_r"), name=f"{tag}_mm")
    else:
        y = mb.matmul(x=w, y=mb.reshape(x=x, shape=[B, cin, n], name=f"{tag}_r"), name=f"{tag}_mm")
    y = mb.reshape(x=y, shape=[B, cout, 1, n], name=f"{tag}_rb")
    return y if b is None else mb.add(x=y, y=b, name=f"{tag}_b")


def layer_norm(x, g, b, eps, tag):
    """LayerNorm over channels, x scaled by max|x| first (fp16 statistics of outlier activations;
    LayerNorm does not see the scale), gamma and beta as (1, C, 1, 1) inputs."""
    m = mb.add(x=mb.reduce_max(x=mb.abs(x=x, name=f"{tag}_abs"), axes=[1], keep_dims=True, name=f"{tag}_max"),
               y=np.float16(1e-4), name=f"{tag}_s")
    y = mb.real_div(x=x, y=m, name=f"{tag}_y")
    # the fused op (its statistics are not fp16 reductions), gamma and beta applied after it
    n = mb.layer_norm(x=y, axes=[1], epsilon=np.float16(eps), name=f"{tag}_n")
    return mb.add(x=mb.mul(x=n, y=g, name=f"{tag}_g"), y=b, name=f"{tag}_out")


def attention(q, k, v, mask, heads, n, tag, most=128):
    """Bidirectional attention per head over (B, C, 1, n) q, k, v (1/sqrt(d) already in q); `mask`
    (B, n, 1, 1) is 0 for a real key and -30000 for padding. Queries in blocks (query_block)."""
    nb = n // query_block(n, most)
    qs = mb.split(x=q, num_splits=heads, axis=1, name=f"{tag}_qs")
    ks = mb.split(x=mb.transpose(x=k, perm=[0, 3, 2, 1], name=f"{tag}_kT"), num_splits=heads, axis=3, name=f"{tag}_ks")
    vs = mb.split(x=v, num_splits=heads, axis=1, name=f"{tag}_vs")
    outs = []
    for hd in range(heads):
        parts = []
        for j, qp in enumerate([qs[hd]] if nb == 1 else mb.split(x=qs[hd], num_splits=nb, axis=3, name=f"{tag}_h{hd}_qb")):
            s = mb.einsum(values=(ks[hd], qp), equation="bkhc,bchq->bkhq", name=f"{tag}_h{hd}_{j}_qk")
            p = mb.softmax(x=mb.add(x=s, y=mask, name=f"{tag}_h{hd}_{j}_m"), axis=1, name=f"{tag}_h{hd}_{j}_sm")
            parts.append(mb.einsum(values=(vs[hd], p), equation="bchk,bkhq->bchq", name=f"{tag}_h{hd}_{j}_av"))
        outs.append(parts[0] if nb == 1 else mb.concat(values=parts, axis=3, name=f"{tag}_h{hd}_cat"))
    return mb.concat(values=outs, axis=1, name=f"{tag}_ctx")


def save_multifunction(parts, path):
    """{function name: converted model} -> one .mlpackage, then compiled next to it (.mlmodelc)."""
    import os, shutil, subprocess, tempfile
    tmp = tempfile.mkdtemp()
    desc = ct.utils.MultiFunctionDescriptor()
    for name, model in parts.items():
        p = os.path.join(tmp, f"{name}.mlpackage")
        model.save(p)
        desc.add_function(p, src_function_name="main", target_function_name=name)
    desc.default_function_name = list(parts)[-1]
    if os.path.exists(path): shutil.rmtree(path)
    ct.utils.save_multifunction(desc, path)
    shutil.rmtree(tmp)
    out = os.path.dirname(os.path.abspath(path))
    subprocess.run(["xcrun", "coremlcompiler", "compile", path, out], check=True, capture_output=True)
    return path[:-len(".mlpackage")] + ".mlmodelc"
