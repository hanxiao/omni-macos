"""Local, persistent MLX worker. stdout is exclusively newline-delimited JSON.

No listeners, network or model downloads: load only the supplied local directory.
Images/PCM cross the private pipe as base64 bytes; nothing is written to disk.
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def respond(value):
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
    sys.stdout.flush()


def main():
    # Libraries occasionally print loading messages; never let them corrupt the pipe.
    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        from pathlib import Path
        from PIL import Image
        from transformers import AutoProcessor
        from mlx_vlm.embedding_loader import load_embedding_model
        path = Path(sys.argv[1])
        config = json.loads((path / 'config.json').read_text())
        if config.get('model_type') != 'embedding_gemma2':
            raise ValueError('Expected an EmbeddingGemma 2 checkpoint')
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        # The released config still caps audio at 280 tokens; allow bounded 240s
        # index segments to use the shared 8192-token context instead.
        processor.audio_seq_length = 8192
        model = None
        def load(vision, audio):
            cfg = dict(config)
            if not vision:
                cfg['vision_config'] = None
            if not audio:
                cfg['audio_config'] = None
            return load_embedding_model(path, config=cfg)
        vision = sys.argv[2] == '1'
        audio = sys.argv[3] == '1'
        model = load(vision, audio)
        mx.set_cache_limit(256 * 1024 * 1024)
    mx.set_memory_limit(int(os.environ.get('OMNI_GEMMA_MEMORY_LIMIT', '6000000000')))
    respond({'ready': True, 'dimension': 768, 'model': 'google/embeddinggemma-2'})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                op = request['op']
                if op == 'towers':
                    # Keep the current model usable if the replacement fails.
                    replacement = load(request['vision'], request['audio'])
                    model = replacement
                    vision, audio = request['vision'], request['audio']
                    mx.clear_cache()
                    result = {'ok': True}
                elif op == 'memory':
                    mx.set_memory_limit(int(request['bytes']))
                    result = {'ok': True}
                elif op == 'idle':
                    mx.clear_cache()
                    result = {'ok': True}
                elif op in ('text', 'tokenize'):
                    prefix = 'task: search result | query: ' if request.get('query') else 'title: none | text: '
                    texts = [prefix + t for t in request['texts']]
                    inputs = processor.tokenizer(texts, padding=True, truncation=True,
                                                 max_length=8192, return_tensors='np')
                    count = int(inputs['attention_mask'].sum())
                    if op == 'tokenize':
                        result = {'tokens': count}
                    else:
                        values = model(**{k: mx.array(v) for k, v in inputs.items()}).text_embeds
                        mx.eval(values)
                        result = {'vectors': values.astype(mx.float32).tolist(), 'tokens': count}
                elif op in ('images', 'video', 'audio'):
                    if op in ('images', 'video'):
                        if not vision:
                            raise ValueError('Vision encoder disabled')
                        images = [Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
                                  for b in request['images']]
                        if op == 'images':
                            # Each image is an independent corpus item.
                            batches = [processor(images=[im], return_tensors='np') for im in images]
                        else:
                            frames = np.stack([np.asarray(im) for im in images])
                            batches = [processor(videos=[frames], return_tensors='np',
                                                 add_timestamps=False, do_sample_frames=False)]
                    else:
                        if not audio:
                            raise ValueError('Audio encoder disabled')
                        samples = np.frombuffer(base64.b64decode(request['pcm']), dtype='<f4').copy()
                        batches = [processor(audio=[samples], sampling_rate=16000, return_tensors='np')]
                    vectors, count = [], 0
                    for inputs in batches:
                        tensors = {k: mx.array(v) for k, v in inputs.items()
                                   if isinstance(v, np.ndarray)}
                        values = model(**tensors).text_embeds
                        mx.eval(values)
                        vectors.extend(values.astype(mx.float32).tolist())
                        count += int(np.asarray(inputs['attention_mask']).sum())
                    result = {'vectors': vectors, 'tokens': count}
                else:
                    raise ValueError('Unknown operation: ' + op)
            # Validate at the boundary; invalid vectors must never reach the index.
            for vector in result.get('vectors', []):
                a = np.asarray(vector)
                if a.shape != (768,) or not np.isfinite(a).all() or abs(np.linalg.norm(a) - 1) > 0.01:
                    raise ValueError('Invalid embedding: expected finite normalized 768d vector')
            result.update(active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory())
            respond(result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            respond({'error': str(exc)})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        respond({'error': str(exc)})
        sys.exit(1)
