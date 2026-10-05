# Voiceover: one Gemini TTS clip per line, same voice and direction for all.
# usage: vo.py <outdir> [--voice V] [--model M] [--only id,id]
import sys, os, json, base64, urllib.request, argparse, wave
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()

# The TTS models speak everything after the colon and take what comes before it as direction.
DIRECTION = "Say warmly and clearly, at a natural, slightly brisk pace, like the narrator of a short product film"

LINES = [
    ("old",    "Semantic search has existed for decades."),
    ("cloud",  "Today it is mostly a cloud service, so your files are uploaded and searched on remote servers."),
    ("down",   "Omni keeps all of it on the Mac that holds the files, with no server and no network connection."),
    ("omni",   "This is Omni."),
    ("easy",   "Running the embedding model locally is straightforward. Omni runs it in Swift on Metal, without Python."),
    ("only",   "Inference, however, is only one part of the problem."),
    ("messy",  "The harder part is keeping the index current when folders hold version one, version two, final, and final final."),
    ("once",   "Omni stores each passage once, keyed by its content. A copy adds no vectors, and an edit re-encodes only the passage it changes."),
    ("aside",  "Indexing shares the machine with you. While you type, it gives the GPU smaller units of work, within one memory limit that you set."),
    ("funnel", "There is no vector database. Each query scans a one-bit copy of every vector and rescores the best candidates exactly."),
    ("agents", "Local agents can search the same index over MCP, and their queries never leave the machine."),
    ("paper",  "We describe the design and its measurements in a paper at the NeurIPS twenty twenty-six workshop on on-device intelligence."),
    ("app",    "In the app, you search by meaning in any language, find similar files, and browse folders as you do in Finder."),
    ("end",    "Omni is free and open source."),
]

def say(line_id, text, out, voice, model):
    body = {"contents": [{"parts": [{"text": f"{DIRECTION}: {text}"}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
    pcm = base64.b64decode(part["data"])          # 24 kHz mono s16le
    path = os.path.join(out, f"{line_id}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    return path, len(pcm) / 48000

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("out")
    a.add_argument("--voice", default="Charon"); a.add_argument("--model", default="gemini-3.8-flash-tts")
    a.add_argument("--only", default="")
    o = a.parse_args(); os.makedirs(o.out, exist_ok=True)
    todo = [l for l in LINES if not o.only or l[0] in o.only.split(",")]
    with ThreadPoolExecutor(6) as ex:
        for p, dur in ex.map(lambda l: say(l[0], l[1], o.out, o.voice, o.model), todo):
            print(f"{dur:5.2f}s {p}")
