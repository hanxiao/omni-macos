"""jina-ocr-v1 HuggingFace checkpoint -> the sharded MLX checkpoint OmniKit loads.

Two jobs.

1. FUSION / RE-LAYOUT. Every `nn.Linear` becomes MLX's `(in, out)`; q/k/v fuse into one
   `wqkv`; gate/up fuse into one `gate_up` and stack over the 64 experts; SAM's stride-16 patch
   convolution flattens into a matmul. The key names produced here are the contract with
   `Sources/OmniKit/OCR/*`.

2. DYNAMIC QUANTIZATION. Bit width is chosen PER TENSOR by a named policy and recorded in
   `quant_map` metadata, because a single global `(bits, group_size)` cannot describe a mixed
   file - and a quantized matmul given the wrong group size returns plausible garbage rather
   than an error.

Which tensors may be narrowed is not a matter of taste; it was measured on this model:
  * MoE ROUTED expert stacks are the sensitive ones. Quantizing them to 4-bit uniformly costs
    CER 0.082 on dense pages AND the ability to stop (every 4-bit run ends at the token cap
    instead of EOS). Keeping the shared expert wide does NOT rescue it - a build differing from
    plain 4-bit only in shared-expert precision diverged at the identical character.
  * MoE ROUTERS (`mlp.gate`, 11 tensors, ~3.6 MB total) are never quantized. They choose 6 of 64
    experts per token, so one flipped comparison reroutes the whole token.
  * `lm_head` is outlier-dominated, which is why llama.cpp and mlx-lm leave it wide by default.
  * Vision runs once per page, never per decoded token, so narrowing it buys no decode speed.

Output is sharded under 1.9 GB per file: a GitHub release asset is capped at 2 GiB, and the
whole point of this artifact is that it can be hosted there.

  python Tools/ocr/convert.py --src model_hf --out build/jina-ocr-v1-mlx-dyn4 --policy dyn-a
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx

SHARD_LIMIT = 1_900_000_000

# --------------------------------------------------------------------------- policies
# A policy maps a tensor ROLE to (bits, group_size) or None for "keep at the base dtype".
# Roles, not names, so a policy reads as a decision about the model rather than a regex.
ROLES = ("routed_early", "routed_late", "routed_gate_up", "routed_down",
         "shared_expert", "dense_mlp", "attention",
         "lm_head", "mtp", "vision", "router", "embedding")

# Layers 1..EARLY_LAST keep the wider grid under the split policies. Layer 0 is dense (no
# routed experts at all), so the split starts at 1.
POLICIES: dict[str, dict] = {
    # Control: nothing quantized. The fidelity oracle for every other row.
    "bf16": {r: None for r in ROLES},

    # Control: the known-good conservative point - 8-bit routed experts, everything else wide.
    "q8": {**{r: None for r in ROLES}, "routed_early": (8, 64), "routed_late": (8, 64)},

    # Control: uniform 4-bit over everything that is normally quantized. Included so the ladder
    # contains the configuration this project is expected to beat, measured on the same pages.
    "q4": {**{r: None for r in ROLES}, "routed_early": (4, 64), "routed_late": (4, 64),
           "attention": (4, 64), "lm_head": (4, 64)},

    # Dynamic candidates. All keep routers, vision, embeddings and lm_head wide; they differ in
    # how much of the routed-expert stack is allowed down to 4 bits.
    "dyn-a": {**{r: None for r in ROLES},
              "routed_early": (4, 32), "routed_late": (4, 32), "attention": (4, 32)},
    "dyn-b": {**{r: None for r in ROLES},
              "routed_early": (8, 64), "routed_late": (4, 32), "attention": (4, 32)},
    "dyn-c": {**{r: None for r in ROLES},
              "routed_early": (8, 64), "routed_late": (4, 32), "attention": (4, 32),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},

    # The two single-factor arms. dyn-b changes routed-late AND attention at once, so on its own
    # it cannot say which one costs the quality; these separate them.
    "dyn-d": {**{r: None for r in ROLES},          # routed split only, attention left wide
              "routed_early": (8, 64), "routed_late": (4, 32)},
    "dyn-e": {**{r: None for r in ROLES},          # attention narrowed only, routed all 8-bit
              "routed_early": (8, 64), "routed_late": (8, 64), "attention": (4, 32)},

    # The 4-bit arms that spend their budget where the BYTES are (the routed expert stacks are
    # ~2.7 GB of a 6.7 GB model) and leave attention wide, since the factorial above says
    # attention is where 4 bits costs the most transcription quality per byte saved.
    "dyn-f": {**{r: None for r in ROLES},
              "routed_early": (4, 32), "routed_late": (4, 32),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},
    "dyn-g": {**{r: None for r in ROLES},
              "routed_early": (4, 64), "routed_late": (4, 64),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},

    # Split the routed stack by projection. gate_up is ~2/3 of the routed bytes, so if IT is the
    # half that tolerates 4 bits, dyn-h is the cheap dynamic build; if `down` is, dyn-i is.
    "dyn-h": {**{r: None for r in ROLES},          # gate_up narrow, down wide
              "routed_early": (8, 64), "routed_late": (8, 64),
              "routed_gate_up": (4, 32),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},
    "dyn-i": {**{r: None for r in ROLES},          # down narrow, gate_up wide
              "routed_early": (8, 64), "routed_late": (8, 64),
              "routed_down": (4, 32),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},

    # Attribution arm: q8 plus 8-bit shared expert and dense MLP, with NOTHING at 4 bits. If this
    # is as fast as the 4-bit arms, the speed was never coming from 4 bits.
    "dyn-k": {**{r: None for r in ROLES},
              "routed_early": (8, 64), "routed_late": (8, 64),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},

    # dyn-k plus 4-bit attention: the routed experts stay at 8 bits (where quality lives) and the
    # 4 bits go to attention (where the remaining decode speed lives).
    "dyn-l": {**{r: None for r in ROLES},
              "routed_early": (8, 64), "routed_late": (8, 64),
              "shared_expert": (8, 64), "dense_mlp": (8, 64), "attention": (4, 32)},

    # The shipped compact build: dyn-i's expert split (which the ladder shows is the half that
    # tolerates 4 bits) plus 4-bit attention, which is where the remaining decode speed is.
    "dyn-j": {**{r: None for r in ROLES},
              "routed_early": (8, 64), "routed_late": (8, 64),
              "routed_down": (4, 32), "attention": (4, 32),
              "shared_expert": (8, 64), "dense_mlp": (8, 64)},
}

EARLY_LAST_DEFAULT = 5


class Sink:
    def __init__(self, policy: dict, early_last: int, base_dtype,
                 mtp_self_contained: bool = False):
        self.mtp_self_contained = mtp_self_contained
        self.out: dict = {}
        self.quant_map: dict[str, list[int]] = {}
        self.policy = policy
        self.early_last = early_last
        self.base_dtype = base_dtype

    def role_of(self, key: str) -> str:
        if key.startswith(("sam.", "clip.", "projector.")):
            return "vision"
        if key.startswith("mtp."):
            return "mtp"
        if key in ("embed_tokens", "mtp.embed_tokens", "image_newline", "view_seperator"):
            return "embedding"
        if key in ("lm_head", "mtp.head"):
            return "lm_head"
        m = re.match(r"layers\.(\d+)\.", key)
        if m:
            layer = int(m.group(1))
            if ".attn." in key:
                return "attention"
            if key.endswith(".mlp.gate"):
                return "router"
            if ".mlp.shared." in key:
                return "shared_expert"
            if ".mlp." in key:
                if layer == 0:
                    return "dense_mlp"
                # A policy may split the routed stack by PROJECTION as well as by layer. The two
                # halves are not the same size or the same sensitivity: gate_up is twice the
                # bytes of down, so which one can take 4 bits decides both the artifact size and
                # whether the build survives dense pages.
                if "routed_gate_up" in self.policy and self.policy["routed_gate_up"] is not None \
                        and key.endswith(".mlp.gate_up"):
                    return "routed_gate_up"
                if "routed_down" in self.policy and self.policy["routed_down"] is not None \
                        and key.endswith(".mlp.down"):
                    return "routed_down"
                return "routed_early" if layer <= self.early_last else "routed_late"
        return "embedding"

    def plain(self, key: str, value: mx.array):
        self.out[key] = value if isinstance(value, mx.array) else mx.array(value)

    def linear(self, key: str, w: mx.array):
        """Store a matmul weight laid out `(in, out)[, ...]`, quantizing per the policy."""
        w = mx.contiguous(w)
        spec = self.policy.get(self.role_of(key))
        if spec is None:
            self.out[key] = w
            return
        bits, gs = spec
        # Quantize from the FP32 source, never by re-packing an already-lossy tensor: unpacking a
        # 4-bit file and re-packing it at 8 bits cannot restore what the first pass threw away.
        q, sc, bi = mx.quantize(w.astype(mx.float32), group_size=gs, bits=bits)
        self.out[f"{key}.qweight"] = q
        self.out[f"{key}.scales"] = sc
        self.out[f"{key}.biases"] = bi
        self.quant_map[key] = [gs, bits]


def lin(a: mx.array) -> mx.array:
    """nn.Linear (out, in) -> MLX (in, out)."""
    return mx.contiguous(a.transpose(1, 0))


def conv_nhwc(a: mx.array) -> mx.array:
    """torch (O, I, kH, kW) -> MLX (O, kH, kW, I)."""
    return mx.contiguous(a.transpose(0, 2, 3, 1))


def patch_flat(a: mx.array) -> mx.array:
    """torch conv (O, C, kH, kW) with stride == kernel -> (kH*kW*C, O)."""
    return mx.contiguous(a.transpose(2, 3, 1, 0).reshape(-1, a.shape[0]))


def cat1(parts) -> mx.array:
    return mx.contiguous(mx.concatenate(list(parts), axis=1))


def stack0(parts) -> mx.array:
    return mx.contiguous(mx.stack(list(parts), axis=0))


def load_hf(src: Path) -> dict:
    index = src / "model.safetensors.index.json"
    if index.exists():
        parts = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        parts = ["model.safetensors"]
    w: dict = {}
    for p in parts:
        w.update(mx.load(str(src / p)))
    return w


def _convert_mtp(hf, S, mp, self_contained: bool = False):
    """FastMTP draft head: one transformer block, ~70 MB. Worth every byte.

    The snapshot ALSO ships `mtp_embed_tokens`, `shared_head.local_head` and
    `shared_head.norm`, which the format calls "self-contained". They are not emitted here
    because they are BIT-IDENTICAL duplicates of `model.embed_tokens`, `lm_head` and
    `model.norm` - verified element-wise, max|d| = 0.000e+00 on all three - and carrying them
    would add 662 MB to the artifact for nothing. Measured end to end: draft acceptance with and
    without them is identical to the individual count (0.71 / 0.53 / 0.51, by-position
    [213, 169, 140] either way).

    `--mtp-self-contained` emits them anyway, for a consumer that insists on the fuller format.
    """
    S.plain("mtp.enorm", hf[f"{mp}.enorm.weight"])
    S.plain("mtp.hnorm", hf[f"{mp}.hnorm.weight"])
    S.linear("mtp.eh_proj", lin(hf[f"{mp}.eh_proj.weight"]))
    S.plain("mtp.block.input_layernorm", hf[f"{mp}.mtp_block.input_layernorm.weight"])
    S.plain("mtp.block.post_attention_layernorm",
            hf[f"{mp}.mtp_block.post_attention_layernorm.weight"])
    S.linear("mtp.block.attn.wqkv",
             cat1([lin(hf[f"{mp}.mtp_block.self_attn.q_proj.weight"]),
                   lin(hf[f"{mp}.mtp_block.self_attn.k_proj.weight"]),
                   lin(hf[f"{mp}.mtp_block.self_attn.v_proj.weight"])]))
    S.linear("mtp.block.attn.o", lin(hf[f"{mp}.mtp_block.self_attn.o_proj.weight"]))
    S.linear("mtp.block.mlp.gate_up",
             cat1([lin(hf[f"{mp}.mtp_block.mlp.gate_proj.weight"]),
                   lin(hf[f"{mp}.mtp_block.mlp.up_proj.weight"])]))
    S.linear("mtp.block.mlp.down", lin(hf[f"{mp}.mtp_block.mlp.down_proj.weight"]))
    # The three tensors that make the head self-contained. Without them the draft silently
    # borrows the target's embedding, norm and lm_head, which leaves the OUTPUT correct (the
    # target verifies every token) and the acceptance dead - a regression with no visible symptom.
    if not self_contained:
        return
    if f"{mp}.shared_head.norm.weight" in hf:
        S.plain("mtp.norm.weight", hf[f"{mp}.shared_head.norm.weight"])
    if "mtp_embed_tokens.weight" in hf:
        S.plain("mtp.embed_tokens", hf["mtp_embed_tokens.weight"])
    if f"{mp}.shared_head.local_head.weight" in hf:
        S.linear("mtp.head", lin(hf[f"{mp}.shared_head.local_head.weight"]))


def convert(src: Path, sink: Sink, with_mtp: bool = False) -> None:
    hf = load_hf(src)
    S = sink

    # ---------------- language backbone ----------------
    S.plain("embed_tokens", hf["model.embed_tokens.weight"])
    S.plain("norm.weight", hf["model.norm.weight"])
    S.linear("lm_head", lin(hf["lm_head.weight"]))

    n_layers = sum(1 for k in hf if re.fullmatch(r"model\.layers\.\d+\.input_layernorm\.weight", k))
    for i in range(n_layers):
        p = f"model.layers.{i}"
        S.plain(f"layers.{i}.input_layernorm", hf[f"{p}.input_layernorm.weight"])
        S.plain(f"layers.{i}.post_attention_layernorm", hf[f"{p}.post_attention_layernorm.weight"])
        S.linear(f"layers.{i}.attn.wqkv",
                 cat1([lin(hf[f"{p}.self_attn.q_proj.weight"]),
                       lin(hf[f"{p}.self_attn.k_proj.weight"]),
                       lin(hf[f"{p}.self_attn.v_proj.weight"])]))
        S.linear(f"layers.{i}.attn.o", lin(hf[f"{p}.self_attn.o_proj.weight"]))

        if f"{p}.mlp.gate.weight" in hf:                       # sparse MoE layer
            S.plain(f"layers.{i}.mlp.gate", lin(hf[f"{p}.mlp.gate.weight"]))
            experts = sorted({int(m.group(1)) for kk in hf
                              if (m := re.match(rf"{re.escape(p)}\.mlp\.experts\.(\d+)\.gate_proj\.weight", kk))})
            gu = [cat1([lin(hf[f"{p}.mlp.experts.{e}.gate_proj.weight"]),
                        lin(hf[f"{p}.mlp.experts.{e}.up_proj.weight"])]) for e in experts]
            dn = [lin(hf[f"{p}.mlp.experts.{e}.down_proj.weight"]) for e in experts]
            S.linear(f"layers.{i}.mlp.gate_up", stack0(gu))
            S.linear(f"layers.{i}.mlp.down", stack0(dn))
            S.linear(f"layers.{i}.mlp.shared.gate_up",
                     cat1([lin(hf[f"{p}.mlp.shared_experts.gate_proj.weight"]),
                           lin(hf[f"{p}.mlp.shared_experts.up_proj.weight"])]))
            S.linear(f"layers.{i}.mlp.shared.down", lin(hf[f"{p}.mlp.shared_experts.down_proj.weight"]))
        else:                                                   # dense layer (layer 0)
            S.linear(f"layers.{i}.mlp.gate_up",
                     cat1([lin(hf[f"{p}.mlp.gate_proj.weight"]), lin(hf[f"{p}.mlp.up_proj.weight"])]))
            S.linear(f"layers.{i}.mlp.down", lin(hf[f"{p}.mlp.down_proj.weight"]))

    # ---------------- FastMTP draft head (opt-in) ----------------
    if with_mtp:
        _convert_mtp(hf, S, "mtp_module.heads.0", self_contained=sink.mtp_self_contained)

    # ---------------- vision: SAM-ViT-B ----------------
    S.linear("sam.patch_embed.w", patch_flat(hf["model.sam_model.patch_embed.proj.weight"]))
    S.plain("sam.patch_embed.bias", hf["model.sam_model.patch_embed.proj.bias"])
    S.plain("sam.pos_embed", mx.contiguous(hf["model.sam_model.pos_embed"].squeeze(0)))
    for i in range(12):
        sp = f"model.sam_model.blocks.{i}"
        for a, b in ((f"{sp}.norm1.weight", f"sam.blocks.{i}.norm1.weight"),
                     (f"{sp}.norm1.bias", f"sam.blocks.{i}.norm1.bias"),
                     (f"{sp}.norm2.weight", f"sam.blocks.{i}.norm2.weight"),
                     (f"{sp}.norm2.bias", f"sam.blocks.{i}.norm2.bias"),
                     (f"{sp}.attn.qkv.bias", f"sam.blocks.{i}.attn.qkv.bias"),
                     (f"{sp}.attn.proj.bias", f"sam.blocks.{i}.attn.proj.bias"),
                     (f"{sp}.mlp.lin1.bias", f"sam.blocks.{i}.mlp.lin1.bias"),
                     (f"{sp}.mlp.lin2.bias", f"sam.blocks.{i}.mlp.lin2.bias")):
            S.plain(b, hf[a])
        S.linear(f"sam.blocks.{i}.attn.qkv", lin(hf[f"{sp}.attn.qkv.weight"]))
        S.linear(f"sam.blocks.{i}.attn.proj", lin(hf[f"{sp}.attn.proj.weight"]))
        S.plain(f"sam.blocks.{i}.attn.rel_pos_h", hf[f"{sp}.attn.rel_pos_h"])
        S.plain(f"sam.blocks.{i}.attn.rel_pos_w", hf[f"{sp}.attn.rel_pos_w"])
        S.linear(f"sam.blocks.{i}.mlp.lin1", lin(hf[f"{sp}.mlp.lin1.weight"]))
        S.linear(f"sam.blocks.{i}.mlp.lin2", lin(hf[f"{sp}.mlp.lin2.weight"]))
    S.linear("sam.neck.0",
             mx.contiguous(hf["model.sam_model.neck.0.weight"].reshape(256, 768).transpose(1, 0)))
    S.plain("sam.neck.1.weight", hf["model.sam_model.neck.1.weight"])
    S.plain("sam.neck.1.bias", hf["model.sam_model.neck.1.bias"])
    S.plain("sam.neck.2", conv_nhwc(hf["model.sam_model.neck.2.weight"]))
    S.plain("sam.neck.3.weight", hf["model.sam_model.neck.3.weight"])
    S.plain("sam.neck.3.bias", hf["model.sam_model.neck.3.bias"])
    S.plain("sam.net_2", conv_nhwc(hf["model.sam_model.net_2.weight"]))
    S.plain("sam.net_3", conv_nhwc(hf["model.sam_model.net_3.weight"]))

    # ---------------- vision: CLIP-L ----------------
    S.plain("clip.class_embedding", hf["model.vision_model.embeddings.class_embedding"])
    S.plain("clip.position_embedding", hf["model.vision_model.embeddings.position_embedding.weight"])
    S.plain("clip.pre_layrnorm.weight", hf["model.vision_model.pre_layrnorm.weight"])
    S.plain("clip.pre_layrnorm.bias", hf["model.vision_model.pre_layrnorm.bias"])
    for i in range(24):
        cp = f"model.vision_model.transformer.layers.{i}"
        S.plain(f"clip.blocks.{i}.ln1.weight", hf[f"{cp}.layer_norm1.weight"])
        S.plain(f"clip.blocks.{i}.ln1.bias", hf[f"{cp}.layer_norm1.bias"])
        S.plain(f"clip.blocks.{i}.ln2.weight", hf[f"{cp}.layer_norm2.weight"])
        S.plain(f"clip.blocks.{i}.ln2.bias", hf[f"{cp}.layer_norm2.bias"])
        S.linear(f"clip.blocks.{i}.attn.qkv", lin(hf[f"{cp}.self_attn.qkv_proj.weight"]))
        S.plain(f"clip.blocks.{i}.attn.qkv.bias", hf[f"{cp}.self_attn.qkv_proj.bias"])
        S.linear(f"clip.blocks.{i}.attn.out", lin(hf[f"{cp}.self_attn.out_proj.weight"]))
        S.plain(f"clip.blocks.{i}.attn.out.bias", hf[f"{cp}.self_attn.out_proj.bias"])
        S.linear(f"clip.blocks.{i}.mlp.fc1", lin(hf[f"{cp}.mlp.fc1.weight"]))
        S.plain(f"clip.blocks.{i}.mlp.fc1.bias", hf[f"{cp}.mlp.fc1.bias"])
        S.linear(f"clip.blocks.{i}.mlp.fc2", lin(hf[f"{cp}.mlp.fc2.weight"]))
        S.plain(f"clip.blocks.{i}.mlp.fc2.bias", hf[f"{cp}.mlp.fc2.bias"])

    S.linear("projector.weight", lin(hf["model.projector.layers.weight"]))
    S.plain("projector.bias", hf["model.projector.layers.bias"])
    S.plain("image_newline", hf["model.image_newline"])
    S.plain("view_seperator", hf["model.view_seperator"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="model_hf", help="HF snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="dyn-j", choices=sorted(POLICIES))
    ap.add_argument("--early-last", type=int, default=EARLY_LAST_DEFAULT,
                    help="last layer index counted as 'early' for the split policies")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--scale-dtype", default="float32", choices=["float32", "bfloat16"],
                    help="dtype of quant scales/biases. fp32 is the default because bf16 scales "
                         "measurably lost a multi-crop page's exactness at 8 bits while saving "
                         "~7%%; they are a fast-and-approximate option, never an exact one.")
    ap.add_argument("--no-mtp", dest="mtp", action="store_false", default=True,
                    help="omit the FastMTP draft head. ON by default: it costs ~70 MB and buys "
                         "+9%% mean decode (+26%% on dense pages) with output IDENTICAL to greedy, "
                         "because every drafted token is verified by the target.")
    ap.add_argument("--mtp-self-contained", action="store_true",
                    help="also emit mtp_embed_tokens / shared_head.local_head / shared_head.norm. "
                         "They are bit-identical duplicates of the main model's tensors, so this "
                         "adds 662 MB and changes nothing; measured, not assumed.")
    ap.add_argument("--tokenizer-from", default=None,
                    help="directory to copy tokenizer.json / tokenizer_config.json from "
                         "(defaults to --src) so the artifact is self-contained")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = mx.bfloat16 if args.dtype == "bfloat16" else mx.float32
    scale_dt = mx.float32 if args.scale_dtype == "float32" else mx.bfloat16

    t0 = time.time()
    sink = Sink(POLICIES[args.policy], args.early_last, base,
                mtp_self_contained=args.mtp_self_contained)
    convert(src, sink, with_mtp=args.mtp)

    for k, v in list(sink.out.items()):
        if v.dtype in (mx.uint32, mx.uint8):
            continue
        if k.endswith((".scales", ".biases")):
            sink.out[k] = v.astype(scale_dt)
            continue
        sink.out[k] = v.astype(base)

    meta = {
        "model": "jina-ocr-v1-mlx",
        "policy": args.policy,
        "mtp": "1" if args.mtp else "0",
        "early_last": str(args.early_last),
        "dtype": args.dtype,
        "scale_dtype": args.scale_dtype,
        # A global pair exists only as a fallback for a pack the map does not name; the map is
        # the authority and every pack is in it.
        "bits": "0",
        "group_size": "64",
        "quant_map": json.dumps(sink.quant_map, sort_keys=True),
    }

    # Shard under the GitHub release-asset ceiling. Every shard repeats the full metadata so the
    # quant map survives however the set is split or re-ordered.
    keys = sorted(sink.out)
    shards: list[list[str]] = [[]]
    size = 0
    for k in keys:
        n = sink.out[k].nbytes
        if size + n > SHARD_LIMIT and shards[-1]:
            shards.append([])
            size = 0
        shards[-1].append(k)
        size += n

    total = 0
    for i, group in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), {k: sink.out[k] for k in group}, metadata=meta)
        total += (out / name).stat().st_size
        print(f"  {name}  {(out / name).stat().st_size / 1e9:.2f} GB  {len(group)} tensors")

    tok_src = Path(args.tokenizer_from or args.src)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_src / f).exists():
            shutil.copy2(tok_src / f, out / f)

    (out / "omni-ocr.json").write_text(json.dumps({
        "model": "jina-ocr-v1-mlx",
        "policy": args.policy,
        "early_last": args.early_last,
        "dtype": args.dtype,
        "scale_dtype": args.scale_dtype,
        "shards": len(shards),
        "bytes": total,
        "quantized_matrices": len(sink.quant_map),
        "bits_histogram": {str(b): sum(1 for gs, bb in sink.quant_map.values() if bb == b)
                           for b in sorted({bb for gs, bb in sink.quant_map.values()})},
    }, indent=1))
    print(f"policy={args.policy} -> {out}  {total / 1e9:.2f} GB in {len(shards)} shard(s), "
          f"{len(sink.quant_map)} quantized matrices, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
