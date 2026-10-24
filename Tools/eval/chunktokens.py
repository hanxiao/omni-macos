"""How much of each chunk the model sees: the chunks `omni-verify chunkdump` wrote, tokenized as the
app does (prefix "Document: ", special tokens) without the tokenizer.json's 512 cut, against a cap.

  uv run --with tokenizers python Tools/eval/chunktokens.py <model dir> <chunks.jsonl> [cap] [sample]

`sample`: that many chunks drawn uniformly (reservoir), for a dump too large to tokenize whole.
"""
import json, random, sys
from collections import defaultdict
from tokenizers import Tokenizer

model, path = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 512
sample = int(sys.argv[4]) if len(sys.argv) > 4 else 0
t = Tokenizer.from_file(f"{model}/tokenizer.json"); t.no_truncation(); t.no_padding()
rows, rng = [], random.Random(0)
for i, line in enumerate(open(path)):
    if not sample or len(rows) < sample: rows.append(line)
    elif (j := rng.randrange(i + 1)) < sample: rows[j] = line
rows = [json.loads(l) for l in rows]
n = [len(e.ids) for e in t.encode_batch(["Document: " + r["text"] for r in rows])]
over = [k for k in n if k > cap]
seen = sum(min(k, cap) for k in n)
by = defaultdict(lambda: [0, 0])
for r, k in zip(rows, n):
    ext = r["path"].rsplit(".", 1)[-1].lower() if "." in r["path"] else "-"
    by[ext][0] += 1; by[ext][1] += k > cap
s = sorted(n)
print(f"{len(rows)} chunks: tokens p50 {s[len(s)//2]}, p90 {s[len(s)*9//10]}, max {s[-1]}")
print(f"past {cap}: {len(over)} chunks ({100*len(over)/len(rows):.1f}%); tokens never embedded {100*(1-seen/sum(n)):.1f}%")
files = defaultdict(int)
for r, k in zip(rows, n): files[r["path"]] += k > cap
print(f"files with a cut chunk: {sum(1 for v in files.values() if v)} of {len(files)}")
for ext, (c, o) in sorted(by.items(), key=lambda x: -x[1][1])[:8]:
    print(f"  .{ext}: {o} of {c} chunks past {cap}")
