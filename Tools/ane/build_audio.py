"""The audio tower (Qwen2.5-Omni's encoder, then pool, ln_post and the projector) for the Neural
Engine as BLOCK programs: no weights inside (anelib.py), three functions per chunk count B, all
sharing the app's weights, which it passes in.

Read the GPU path first (Sources/OmniKit/OmniAudioTower.swift). A window's mel is cut into chunks of
200 frames, and EVERYTHING up to the pool is per chunk: the conv frontend, the positions (rows
0 ..< 100 of the sinusoid table for every chunk), the attention (each chunk attends to itself). So
the chunks are the BATCH here, (B, C, 1, L), and no sequence is longer than 100. Only a last,
partial chunk is masked (its padded frames zeroed after conv1, its padded keys out of every softmax).
  F{B}  conv1 (k3 s1 p1) and conv2 (k3 s2 p1) as (1, 3) convolutions with the kernels as inputs
        ((out, in, 1, 3), the checkpoint's [out, k, in] transposed), GELU (erf) after each, + pos.
        Inputs melx (B, 128, 1, 200), fmask (B, 1, 1, 200) of 0/1, pos (1, 1280, 1, 100).
  B{B}  one encoder layer, called once per layer: LayerNorm (eps 1e-5), q (bias, 1/sqrt(64)
        folded: exact), k (no bias), v (bias), attention per head with kmask (B, 100, 1, 1) of
        0/-30000, out_proj; LayerNorm, fc1, GELU (erf), fc2.
  P{B}  the factor-2 mean pool as an average pool of stride 2, ln_post, the projector -> (B, D, 1, 50).
The app side is ANEAudioBackend.

  uv run --python 3.12 --with coremltools python Tools/ane/build_audio.py --config <config.json> \\
      --out <dir> [--name nano_audio] [--batches 1,2,4,8,16,32]
"""
import argparse, json
from coremltools.converters.mil import Builder as mb
from anelib import spec, program, lin, layer_norm, attention, save_multifunction


def frontend(B, mel, C, T):
    def body(melx, fmask, pos, w1, b1, w2, b2):
        h = mb.conv(x=melx, weight=w1, pad_type="custom", pad=[0, 0, 1, 1], name="c1")
        h = mb.mul(x=mb.gelu(x=mb.add(x=h, y=b1, name="c1b"), mode="EXACT", name="c1g"), y=fmask, name="c1m")
        h = mb.conv(x=h, weight=w2, strides=[1, 2], pad_type="custom", pad=[0, 0, 1, 1], name="c2")
        return mb.add(x=mb.gelu(x=mb.add(x=h, y=b2, name="c2b"), mode="EXACT", name="c2g"), y=pos, name="out")

    return program(dict(melx=spec(B, mel, 1, T), fmask=spec(B, 1, 1, T), pos=spec(1, C, 1, T // 2),
                        w1=spec(C, mel, 1, 3), b1=spec(1, C, 1, 1), w2=spec(C, C, 1, 3), b2=spec(1, C, 1, 1)), body)


def block(B, C, H, F, L):
    vec = lambda c=C: spec(1, c, 1, 1)

    def body(h, kmask, g1, b1, wq, bq, wk, wv, bv, wo, bo, g2, b2, w1, bf1, w2, bf2):
        xn = layer_norm(h, g1, b1, 1e-5, "n1")
        q, k, v = lin(xn, wq, bq, C, L, "q"), lin(xn, wk, None, C, L, "k"), lin(xn, wv, bv, C, L, "v")
        h = mb.add(x=h, y=lin(attention(q, k, v, kmask, H, L, "at"), wo, bo, C, L, "o"), name="r1")
        g = mb.gelu(x=lin(layer_norm(h, g2, b2, 1e-5, "n2"), w1, bf1, F, L, "f1"), mode="EXACT", name="g")
        return mb.add(x=h, y=lin(g, w2, bf2, C, L, "f2"), name="out")

    return program(dict(h=spec(B, C, 1, L), kmask=spec(B, L, 1, 1), g1=vec(), b1=vec(), wq=spec(C, C), bq=vec(),
                        wk=spec(C, C), wv=spec(C, C), bv=vec(), wo=spec(C, C), bo=vec(), g2=vec(), b2=vec(),
                        w1=spec(F, C), bf1=vec(F), w2=spec(C, F), bf2=vec()), body)


def post(B, C, D, L):
    def body(h, g, b, w, bias):
        x = mb.avg_pool(x=h, kernel_sizes=[1, 2], strides=[1, 2], pad_type="valid", name="pool")
        return mb.identity(x=lin(layer_norm(x, g, b, 1e-5, "post"), w, bias, D, L // 2, "proj"), name="out")

    return program(dict(h=spec(B, C, 1, L), g=spec(1, C, 1, 1), b=spec(1, C, 1, 1), w=spec(D, C), bias=spec(1, D, 1, 1)), body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="nano_audio"); ap.add_argument("--batches", default="1,2,4,8,16,32")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    au = cfg["audio_config"]
    C, H, F, mel, T = au["d_model"], au["encoder_attention_heads"], au["encoder_ffn_dim"], au["num_mel_bins"], 2 * au["n_window"]
    D = cfg["text_config"]["hidden_size"]                      # the projector's output, the text width
    parts = {}
    for B in map(int, a.batches.split(",")):
        parts[f"F{B}"], parts[f"B{B}"], parts[f"P{B}"] = frontend(B, mel, C, T), block(B, C, H, F, T // 2), post(B, C, D, T // 2)
        print("built B", B, flush=True)
    print("saved", save_multifunction(parts, f"{a.out}/{a.name}.mlpackage"))
