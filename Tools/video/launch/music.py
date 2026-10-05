# Score: Lyria from one structured brief. usage: music.py <out-prefix> [n variants] [model]
import sys, os, json, base64, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()

BRIEF = """Instrumental score for a 95-second launch film for a calm, precise Mac app. 104 BPM, 4/4,
steady tempo throughout. Warm, tactile, handcrafted sound: felt piano, plucked muted strings, marimba,
soft round analog synth bass, brushed light percussion, a few subtle paper and wooden foley ticks.
Elegant and confident, never epic, no risers, no dubstep, no vocals.
Structure:
0:00-0:09 a sparse felt-piano motif alone, curious, lots of space.
0:09-0:18 light pizzicato and soft marimba join, airy.
0:18-0:28 a gentle descent in the harmony, low synth bass enters, the beat is still absent.
0:28-0:55 a steady, precise groove: brushed kick and shaker, marimba ostinato, playful and clever.
0:55-1:10 fuller and warmer, soft pads, the groove keeps moving forward.
1:10-1:15 a short breakdown, piano alone.
1:15-1:30 the full groove returns, brighter and confident, crisp percussion.
1:30-1:35 resolve on a warm final piano chord with a short natural tail."""

def one(i, prefix, model):
    body = {"contents": [{"parts": [{"text": BRIEF}]}], "generationConfig": {"responseModalities": ["AUDIO"]}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    for c in d.get("candidates", []):
        for p in c["content"]["parts"]:
            if "inlineData" in p:
                path = f"{prefix}-{i}.mp3"
                open(path, "wb").write(base64.b64decode(p["inlineData"]["data"]))
                return path
    return "none: " + json.dumps(d)[:300]

if __name__ == "__main__":
    prefix = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model = sys.argv[3] if len(sys.argv) > 3 else "lyria-3-pro-preview"
    with ThreadPoolExecutor(n) as ex:
        for r in ex.map(lambda i: one(i, prefix, model), range(n)): print(r)
