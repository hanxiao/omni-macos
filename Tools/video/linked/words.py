# Word timings for every narration line (mlx-whisper), in film seconds, to work/linked/words.json.
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_whisper, clock as T
out = {}
for line, start in T.VO.items():
    r = mlx_whisper.transcribe(os.path.join(T.VO_DIR, line + ".wav"), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                               word_timestamps=True, language="en")
    out[line] = [(w["word"].strip(), round(start + w["start"], 3), round(start + w["end"], 3))
                 for s in r["segments"] for w in s.get("words", [])]
    print(line, " ".join(f"{w}@{a:.2f}" for w, a, b in out[line]))
json.dump(out, open(os.path.join(T.WORK, "words.json"), "w"), indent=1)
