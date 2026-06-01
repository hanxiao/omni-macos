"""Generate numeric fixtures from the reference jina-embeddings-v5-omni-small-mlx model.

The Swift MLX port is validated against these fixtures (token ids + reference
embeddings). The Swift engine loads the SAME original HF files and merges the
retrieval LoRA at load time, so its output must match the reference here to
high precision.

Run inside the mls venv (has mlx + tokenizers):
    cd ~/Documents/mls && uv run python ~/Documents/omni-macos/Tools/gen_fixtures.py

Outputs:
    Fixtures/text_fixtures.json   token ids + query/passage embeddings per string
    Fixtures/meta.json            model dir, dims, dtype notes
"""

import importlib.util
import json
import os
import sys

import mlx.core as mx

MODEL_DIR = os.environ.get(
    "OMNI_MODEL_DIR",
    "/Volumes/One Touch/ai-models/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-mlx/snapshots/716c5b684db3f6ba574dd9b4f6b14af3b2eb8bda",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fixtures")

# Diverse probe strings: short, long, multilingual, code, punctuation.
TEXTS = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "semantic search over local files on macOS",
    "Die Hauptstadt von Deutschland ist Berlin.",
    "今天天气很好，我们去公园散步吧。",
    "def cosine(a, b):\n    return sum(x*y for x, y in zip(a, b))",
    "The quarterly revenue report shows a 12% increase over last year.",
    "a",
]


def load_ref():
    spec = importlib.util.spec_from_file_location("jina_omni_utils", os.path.join(MODEL_DIR, "utils.py"))
    utils = importlib.util.module_from_spec(spec)
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    spec.loader.exec_module(utils)
    return utils.load_model(MODEL_DIR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_ref()
    model.switch_task("retrieval")
    tok = model.tokenizer

    records = []
    for t in TEXTS:
        q = model.encode([t], task_type="retrieval.query")
        p = model.encode([t], task_type="retrieval.passage")
        mx.eval(q, p)
        q = q[0].tolist()
        p = p[0].tolist()
        # Token ids the reference actually fed (with the task prefix prepended).
        q_ids = tok.encode("Query: " + t).ids
        p_ids = tok.encode("Document: " + t).ids
        records.append(
            {
                "text": t,
                "query_token_ids": q_ids,
                "passage_token_ids": p_ids,
                "query_embedding": q,
                "passage_embedding": p,
            }
        )
        # quick self-check norm
        qn = sum(x * x for x in q) ** 0.5
        print(f"[ok] len(q_ids)={len(q_ids)} dim={len(q)} |q|={qn:.4f}  {t[:40]!r}")

    with open(os.path.join(OUT_DIR, "text_fixtures.json"), "w") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(
            {
                "model_dir": MODEL_DIR,
                "dim": len(records[0]["query_embedding"]),
                "task": "retrieval",
                "query_prefix": "Query: ",
                "passage_prefix": "Document: ",
                "lora_alpha": 32,
                "lora_r": 32,
                "notes": "language_model upcast to fp32; vision/merger bf16; audio dropped",
            },
            f,
            indent=2,
        )
    print(f"wrote {len(records)} fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
