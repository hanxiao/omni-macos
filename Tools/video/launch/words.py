# Word timings for every narration line (mlx-whisper), in film seconds, to words.json.
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_whisper, timeline as T
out = {}
for line, start in T.VO.items():
    r = mlx_whisper.transcribe(f"vo/fast/{line}.wav", path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                               word_timestamps=True, language="en")
    ws = [(w["word"].strip(), round(start + w["start"], 3), round(start + w["end"], 3))
          for s in r["segments"] for w in s.get("words", [])]
    out[line] = ws
    print(line, " ".join(f"{w}@{a:.2f}" for w, a, b in ws))
json.dump(out, open("words.json", "w"), indent=1)
