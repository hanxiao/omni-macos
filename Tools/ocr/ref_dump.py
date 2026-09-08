"""Torch reference dump for the Swift/MLX jina-ocr-v1 port.

The oracle is the ORIGINAL HuggingFace checkpoint at its shipped dtype (bfloat16),
not an fp32 upcast: that is the numeric reference the Swift port is graded against.

Writes one directory per case containing
  stages.safetensors   every intermediate the port can be bisected against
  greedy.json          prompt ids, generated ids, decoded text, timing
so mlx-swift can `loadArrays` the tensors directly (no npz reader in Swift).

Run from the harness work dir (it owns model_hf/ and the ref/ modules):

  cd /Volumes/han2tb/jina-dataroom-harness/runs/fcdd80c5b0b4/work
  .venv-ref/bin/python /Users/hanxiao/Documents/omni-macos/Tools/ocr/ref_dump.py \
      --image bench/doc_small.png --out /Volumes/han2tb/ai-models/jina-ocr-v1-ref/doc_small
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORK = Path.cwd()
sys.path.insert(0, str(WORK / "ref"))

import numpy as np
import torch
from safetensors.numpy import save_file

SNAP = WORK / "model_hf"

OCR_PROMPT = (
    "Transcribe the provided document image into a clean Markdown format, "
    "preserving the natural reading order.\n"
    "Convert all formulas into LaTeX format. Inline formulas should be enclosed in `$ $`. "
    "Display (block) formulas should be enclosed in `$$ $$.\n"
    "Convert tables into HTML format (using <table border='1'><tr><td> tags).\n"
    "Ignore all graphical content in the image document. Do not describe or convert images.\n"
    "Remove the headers and footers, but keep references and footnotes."
)


def np32(t):
    """Every dumped tensor is fp32 on disk regardless of the compute dtype.

    The MODEL runs in bfloat16 (that is the reference being graded against); storing the
    result as fp32 only avoids a lossy second rounding in the comparison itself.
    """
    if isinstance(t, torch.Tensor):
        return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--stages", action="store_true", default=True)
    ap.add_argument("--no-stages", dest="stages", action="store_false",
                    help="greedy reference only (much faster, no per-layer dump)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageOps
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAP), torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    print(f"[ref] loaded {args.dtype} on {dev} in {time.time()-t0:.1f}s", flush=True)

    prompt = args.prompt or OCR_PROMPT
    image = Image.open(args.image).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    p = model.model
    from dynamic_preprocess_local import dynamic_preprocess, BasicImageTransform

    tfm = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    w_crop, h_crop = 1, 1
    crops_raw = []
    if not (image.size[0] <= 640 and image.size[1] <= 640):
        crops_raw, (w_crop, h_crop) = dynamic_preprocess(image, image_size=640)
    global_view = ImageOps.pad(image, (1024, 1024), color=(127, 127, 127))
    images_ori = tfm(global_view).unsqueeze(0)
    images_crop = (torch.stack([tfm(c) for c in crops_raw], 0)
                   if (w_crop > 1 or h_crop > 1) else torch.zeros(0, 3, 640, 640))

    img_tok = tok.convert_tokens_to_ids("<image>")
    n_q_base, n_q_tile = 16, 10
    img_toks = ([img_tok] * (n_q_base + 1)) * n_q_base + [img_tok]
    if w_crop > 1 or h_crop > 1:
        img_toks += ([img_tok] * (n_q_tile * w_crop + 1)) * (n_q_tile * h_crop)
    pre, post = text.split("<image>")
    ids = torch.LongTensor(tok.encode(pre, add_special_tokens=False) + img_toks
                           + tok.encode(post, add_special_tokens=False))
    mask = ids == img_tok
    print(f"[ref] tiles={w_crop}x{h_crop} seq={ids.shape[0]} img_slots={int(mask.sum())}", flush=True)

    tensors: dict[str, np.ndarray] = {
        "input_ids": ids.numpy().astype(np.int32),
        "images_seq_mask": mask.numpy().astype(np.int32),
        "images_ori": np32(images_ori),
        "spatial_crop": np.array([[w_crop, h_crop]], dtype=np.int32),
    }
    if images_crop.numel():
        tensors["images_crop"] = np32(images_crop)

    x_ori = images_ori.to(dev, dtype)
    x_crop = images_crop.to(dev, dtype) if images_crop.numel() else None

    with torch.inference_mode():
        if args.stages:
            inter = {}

            def hook(name):
                def fn(_m, _i, o):
                    inter[name] = o
                return fn

            hs = [p.sam_model.patch_embed.register_forward_hook(hook("sam.patch"))]
            for i, blk in enumerate(p.sam_model.blocks):
                hs.append(blk.register_forward_hook(hook(f"sam.blk{i}")))
            for nm in ("neck", "net_2", "net_3"):
                if hasattr(p.sam_model, nm):
                    hs.append(getattr(p.sam_model, nm).register_forward_hook(hook(f"sam.{nm}")))
            sam_e = p.sam_model(x_ori)
            for h in hs:
                h.remove()
            for k, v in inter.items():
                tensors[k] = np32(v)
            tensors["sam.out"] = np32(sam_e)

            # The SAM absolute-position table AFTER interpolation, for both grids. This is the
            # single primitive that cost the python port the most laps (ATen antialias bicubic,
            # a=-0.5): dumping torch's own answer makes it checkable in one comparison.
            from deepencoder import get_abs_pos_sam
            if getattr(p.sam_model, "pos_embed", None) is not None:
                g_glob = int(p.sam_model.patch_embed(x_ori).size(1))
                tensors["sam.pos@global"] = np32(get_abs_pos_sam(p.sam_model.pos_embed, g_glob))
                if x_crop is not None:
                    g_tile = int(p.sam_model.patch_embed(x_crop[:1]).size(1))
                    tensors["sam.pos@tile"] = np32(get_abs_pos_sam(p.sam_model.pos_embed, g_tile))

            clip_out = p.vision_model(x_ori, sam_e)
            tensors["clip.out"] = np32(clip_out)
            concat = torch.cat((clip_out[:, 1:], sam_e.flatten(2).permute(0, 2, 1)), dim=-1)
            tensors["proj.concat"] = np32(concat)
            tensors["proj.global"] = np32(p.projector(concat))

            if x_crop is not None:
                sam_c = p.sam_model(x_crop)
                clip_c = p.vision_model(x_crop, sam_c)
                concat_c = torch.cat((clip_c[:, 1:], sam_c.flatten(2).permute(0, 2, 1)), dim=-1)
                tensors["sam.local"] = np32(sam_c)
                tensors["proj.local"] = np32(p.projector(concat_c))

            emb = p.compute_inputs_embeds(
                input_ids=ids.unsqueeze(0).to(dev),
                images=[(images_crop.to(dev, dtype), x_ori)],
                images_seq_mask=mask.unsqueeze(0).to(dev),
                images_spatial_crop=torch.tensor([[w_crop, h_crop]]).to(dev))
            tensors["inputs_embeds"] = np32(emb)

            # Routed expert ids per MoE layer. A MoE stage can disagree for two entirely
            # different reasons - rounding, or a different top-k selection - and only the ids
            # separate them. Without this, a single re-routed token looks like a numeric defect.
            router_ids = {}

            def gate_hook(idx):
                def fn(module, _inp, out):
                    router_ids[idx] = out[0].detach().to(torch.int32).cpu()
                return fn

            gate_handles = []
            for li, layer in enumerate(p.layers):
                gate = getattr(getattr(layer, "mlp", None), "gate", None)
                if gate is not None and hasattr(gate, "top_k"):
                    gate_handles.append(gate.register_forward_hook(gate_hook(li)))

            h = emb
            pos_ids = torch.arange(ids.shape[0], device=dev).unsqueeze(0)
            cos, sin = p.rotary_emb(h, pos_ids)
            tensors["rope.cos"] = np32(cos[0])
            tensors["rope.sin"] = np32(sin[0])
            for i, layer in enumerate(p.layers):
                h = layer(h, attention_mask=None,
                          position_ids=torch.arange(1, ids.shape[0] + 1, device=dev).unsqueeze(0),
                          use_cache=False)[0]
                tensors[f"layer{i}.out"] = np32(h)
            for hh in gate_handles:
                hh.remove()
            for li, v in router_ids.items():
                tensors[f"layer{li}.topk"] = v.numpy().astype(np.int32)
            hn = p.norm(h)
            tensors["norm.out"] = np32(hn)
            tensors["prefill.logits"] = np32(model.lm_head(hn[:, -1:, :])[0, -1])
            print("[ref] stages dumped", flush=True)

        t_gen = time.time()
        gen = model.generate(
            input_ids=ids.unsqueeze(0).to(dev),
            images=[(images_crop.to(dev, dtype), x_ori)],
            images_seq_mask=mask.unsqueeze(0).to(dev),
            images_spatial_crop=torch.tensor([[w_crop, h_crop]]).to(dev),
            do_sample=False, max_new_tokens=args.max_new_tokens)
        gen_s = time.time() - t_gen

    new = gen[0, ids.shape[0]:].tolist()
    decoded = tok.decode(new, skip_special_tokens=True)
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else 1
    save_file(tensors, str(out / "stages.safetensors"))
    (out / "greedy.json").write_text(json.dumps({
        "image": str(Path(args.image).resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "prompt_text": text,
        "prompt_ids": ids.tolist(),
        "tiles": [w_crop, h_crop],
        "tokens": new,
        "text": decoded,
        "stopped_by": "eos" if (new and new[-1] == eos) else "cap",
        "gen_seconds": gen_s,
        "tok_per_s": len(new) / max(gen_s, 1e-9),
    }, indent=1))
    print(f"[ref] {len(new)} tokens in {gen_s:.1f}s = {len(new)/max(gen_s,1e-9):.1f} tok/s "
          f"stopped_by={'eos' if (new and new[-1] == eos) else 'cap'} -> {out}", flush=True)
    print("[ref] head:", decoded[:200].replace("\n", " | "), flush=True)


if __name__ == "__main__":
    main()
