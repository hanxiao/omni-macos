"""Emit CoreML fixtures for `ocr-verify --probe-ane`. Python builds the model file; every
measurement is native Swift.

Layout is (1, C, 1, M): the token axis goes to W, so an MLMultiArray over this shape can be
backed by a CVPixelBuffer of width M and height C, which is the documented zero-copy path to
the ANE. A (1, C, M, 1) layout would need a 1-pixel-wide surface instead.

  python Tools/ane/emit_fixtures.py --out build/ane
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

H, FFN = 1024, 3072          # jina-embeddings-v5-omni-small text tower


def chain(name, M, pairs, depth, out: Path):
    rng = np.random.default_rng(0)
    Ws = [[(rng.standard_normal((n, k, 1, 1)) * 0.02).astype(np.float16) for k, n in pairs]
          for _ in range(depth)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, H, 1, M), dtype=types.fp16)],
                opset_version=ct.target.iOS16)
    def prog(x):
        h = x
        for i, group in enumerate(Ws):
            for j, W in enumerate(group):
                h = mb.conv(x=h, weight=W, strides=[1, 1], pad_type="valid", name=f"c{i}_{j}")
                # relu so consecutive 1x1 convs cannot collapse into one matmul
                h = mb.relu(x=h, name=f"r{i}_{j}")
        return h

    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS14,
                   compute_precision=ct.precision.FLOAT16,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    p = out / f"{name}.mlpackage"
    m.save(str(p))
    flops = depth * M * sum(2 * k * n for k, n in pairs)
    print(f"{p.name:<22} M={M:<6} depth={depth:<3} {flops/1e9:8.2f} GFLOP")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/ane")
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chain("attn", a.tokens, [(H, H)], 8, out)
    chain("mlp", a.tokens, [(H, FFN), (FFN, H)], 6, out)
