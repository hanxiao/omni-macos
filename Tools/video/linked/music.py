# Score for the Linked Macs film: Lyria, the launch film's instrumentation, a 78-second arc.
# usage: music.py <out-prefix> [n variants] [model]
import sys, os, json, base64, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()

BRIEF = """Instrumental score for a 78-second feature film for a calm, precise Mac app. 102 BPM, 4/4,
steady tempo throughout. Warm, tactile, handcrafted sound: felt piano, plucked muted strings, marimba,
soft round analog synth bass, brushed light percussion, a few subtle paper and wooden foley ticks.
Elegant and confident, never epic, no risers, no dubstep, no vocals.
Structure:
0:00-0:08 a sparse felt-piano motif alone, curious, two notes answering each other like a call and reply.
0:08-0:17 light pizzicato and soft marimba join, the call and reply passes between two instruments.
0:17-0:38 a steady, precise groove: brushed kick and shaker, marimba ostinato, soft synth bass, playful and clever.
0:38-0:48 the same groove grows wider and fuller, more voices joining in, many marimbas answering each other.
0:48-0:56 a short breakdown, piano alone, quiet, sleepy.
0:56-1:13 the full groove returns, brighter and confident, crisp percussion.
1:13-1:18 resolve on a warm final piano chord with a long natural tail."""

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
        for r in ex.map(lambda i: one(i, prefix, model), range(n)): print(r, flush=True)
