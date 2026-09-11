"""Score the CoreML tower against the shipped fixtures. Correctness only - throughput is
measured natively by `ocr-verify --probe-ane`."""
import json, sys
import numpy as np, mlx.core as mx, coremltools as ct

W = mx.load(sys.argv[1])
recs = json.load(open(sys.argv[2]))["records"]
L = int(sys.argv[3])
model = ct.models.MLModel(sys.argv[4], compute_units=ct.ComputeUnit.CPU_AND_NE)
emb = W["language_model.embed_tokens.weight"]

worst = 1.0
for r in recs:
    ids = r["passage_token_ids"]
    n = len(ids)
    if n > L:
        continue
    # Right-pad to the built length. Attention is causal, so a real token never attends to a
    # pad and its hidden state is unchanged; pooling just reads index n-1 instead of -1.
    padded = ids + [0] * (L - n)
    x = np.array(emb[mx.array(padded)].astype(mx.float16), copy=False)
    x = np.ascontiguousarray(x.T[None, :, None, :])
    out = list(model.predict({"x": x}).values())[0]          # (1, C, 1, L)
    got = np.asarray(out, dtype=np.float32)[0, :, 0, n - 1]
    got /= np.linalg.norm(got)
    ref = np.asarray(r["passage_embedding"], dtype=np.float32)
    ref /= np.linalg.norm(ref)
    cos = float(got @ ref)
    worst = min(worst, cos)
    print(f"  cos={cos:.5f}  {r['text'][:44]!r}")
print(f"worst cosine {worst:.5f}   {'PASS' if worst >= 0.999 else 'FAIL'}")
