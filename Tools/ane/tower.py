"""Reference implementation of Omni's text tower, built from the weights the app itself
exports (`omni-verify dumpbackbone`). Its only job is to be provably the same maths as
Sources/OmniKit/Qwen3Backbone.swift, so the CoreML port can be checked against it rather
than against a guess.

Verified by scoring the shipped fixtures: cosine against `passage_embedding` must clear 0.999.
"""
from __future__ import annotations
import json, sys
import mlx.core as mx

EPS, THETA = 1e-6, 3_500_000.0
HEADS, KV, HDIM, LAYERS = 16, 8, 128, 28


def rms(x, w, eps=EPS):
    xf = x.astype(mx.float32)
    v = mx.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * mx.rsqrt(v + eps) * w.astype(mx.float32)).astype(x.dtype)


def lin(x, w):                      # Swift: matmul(x, w.transposed(1,0)), w is [out, in]
    return x @ w.T


def tower(W, ids, dtype=mx.float16):
    h = W["language_model.embed_tokens.weight"][mx.array(ids)][None].astype(dtype)
    L = h.shape[1]
    scale = HDIM ** -0.5
    for i in range(LAYERS):
        p = f"language_model.layers.{i}."
        xn = rms(h, W[p + "input_layernorm.weight"])
        q = lin(xn, W[p + "self_attn.q_proj.weight"]).reshape(1, L, HEADS, HDIM).transpose(0, 2, 1, 3)
        k = lin(xn, W[p + "self_attn.k_proj.weight"]).reshape(1, L, KV, HDIM).transpose(0, 2, 1, 3)
        v = lin(xn, W[p + "self_attn.v_proj.weight"]).reshape(1, L, KV, HDIM).transpose(0, 2, 1, 3)
        if p + "self_attn.q_norm.weight" in W:
            q = rms(q, W[p + "self_attn.q_norm.weight"])
            k = rms(k, W[p + "self_attn.k_norm.weight"])
        q = mx.fast.rope(q, HDIM, traditional=False, base=THETA, scale=1.0, offset=0)
        k = mx.fast.rope(k, HDIM, traditional=False, base=THETA, scale=1.0, offset=0)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(1, L, HEADS * HDIM)
        h = h + lin(o, W[p + "self_attn.o_proj.weight"])
        xn2 = rms(h, W[p + "post_attention_layernorm.weight"])
        gate = mx.sigmoid(lin(xn2, W[p + "mlp.gate_proj.weight"])) * lin(xn2, W[p + "mlp.gate_proj.weight"])
        up = lin(xn2, W[p + "mlp.up_proj.weight"])
        h = h + lin(gate * up, W[p + "mlp.down_proj.weight"])
    return rms(h, W["language_model.norm.weight"])


def embed(W, ids, dtype=mx.float16):
    hidden = tower(W, ids, dtype)
    pooled = hidden[0, -1].astype(mx.float32)                 # last-token pool
    return pooled / mx.linalg.norm(pooled)


if __name__ == "__main__":
    W = mx.load(sys.argv[1])
    recs = json.load(open(sys.argv[2]))["records"]
    worst = 1.0
    for r in recs:
        got = embed(W, r["passage_token_ids"])
        ref = mx.array(r["passage_embedding"], dtype=mx.float32)
        ref = ref / mx.linalg.norm(ref)
        cos = float(mx.sum(got * ref).item())
        worst = min(worst, cos)
        print(f"  cos={cos:.5f}  {r['text'][:44]!r}")
    print(f"worst cosine {worst:.5f}   {'PASS' if worst >= 0.999 else 'FAIL'}")
