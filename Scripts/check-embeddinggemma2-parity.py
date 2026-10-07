#!/usr/bin/env python3
"""Opt-in numeric check: local MLX BF16 against Transformers FP32 on CPU.

Run with the fork's Python interpreter and an optional local checkpoint path.
Small fixtures verify runtime fidelity, not retrieval quality or performance.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import time
from pathlib import Path
import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoProcessor, EmbeddingGemma2Model
from mlx_vlm.embedding_loader import load_embedding_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', nargs='?', type=Path,
                        default=Path.home() / 'Library/Application Support/OmniEmbeddingGemma2/embeddinggemma2')
    args = parser.parse_args()
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    processor.audio_seq_length = 8192
    reference = EmbeddingGemma2Model.from_pretrained(args.checkpoint, local_files_only=True,
                                                    dtype=torch.float32).eval()
    model = load_embedding_model(args.checkpoint)
    fixtures = Path(__file__).resolve().parent.parent / 'Tests/OmniKitTests/Resources'
    image = Image.open(fixtures / 'test_image.png').convert('RGB')
    samples, rate = sf.read(fixtures / 'test_audio.wav', dtype='float32')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
    inputs = [
        ('French text', processor(text=['task: search result | query: Combien coûte la réparation de la pompe ?'],
                                  return_tensors='np')),
        ('image', processor(images=[image], return_tensors='np')),
        ('video', processor(videos=[np.stack([np.asarray(image)] * 2)], return_tensors='np',
                            add_timestamps=False, do_sample_frames=False)),
        ('audio', processor(audio=[samples], sampling_rate=16000, return_tensors='np')),
    ]
    for name, inputs_for_sample in inputs:
        start = time.monotonic()
        arrays = {k: v for k, v in inputs_for_sample.items() if isinstance(v, np.ndarray)}
        values = model(**{k: mx.array(v) for k, v in arrays.items()}).text_embeds
        mx.eval(values)
        actual = np.asarray(values.astype(mx.float32))[0]
        tensors = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
        with torch.inference_mode():
            hidden = reference(**tensors).last_hidden_state
            mask = tensors['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            expected = torch.nn.functional.normalize(pooled, dim=-1)[0].numpy()
        cosine = float(actual @ expected)
        print(json.dumps({'input': name, 'cosine': cosine,
                          'max_abs': float(np.max(np.abs(actual - expected))),
                          'seconds': round(time.monotonic() - start, 2)}), flush=True)
        if not np.isfinite(actual).all() or cosine <= 0.99:
            raise RuntimeError(f'{name}: expected cosine > 0.99, got {cosine}')


if __name__ == '__main__':
    main()
