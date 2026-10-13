# Narration for the Linked Macs film: the launch film's voice (Gemini 2.5 Pro TTS, Sadaltager) and
# direction, two takes per line; the chosen take is trimmed and sped to 1.1x into vo/fast/.
# usage: vo.py <outdir> [--take a] [--only id,id]
import sys, os, json, base64, urllib.request, argparse, wave, time
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()
DIRECTION = "Say warmly and clearly, at a natural, slightly brisk pace, like the narrator of a short product film"

LINES = [
    ("intro",  "Omni already searches every kind of file on your Mac, by meaning. But what if your files are spread across several Macs?"),
    ("link",   "Omni can now link them. Open Link a Mac on both, and type the six digits one shows into the other."),
    ("proof",  "The code proves the two Macs are talking to each other, with nothing in between. No server is involved: they find each other on your network, or over your VPN."),
    ("search", "From then on, one search covers every linked Mac. Each one searches its own index on its own GPU, and its results join yours as they arrive."),
    ("open",   "Open, preview or drag a result, and the file comes across the link. Its folders are in your sidebar."),
    ("swarm",  "Link as many Macs as you like. One search fans out to all of them at once, and every answer comes back to you."),
    ("away",   "If a Mac is asleep, the search simply goes on without it."),
    ("agents", "Local agents get the same reach, through the same tools."),
    ("end",    "Omni Linked Macs: multimodal search, now federated."),
]

def say(line_id, text, out, voice, model):
    body = {"contents": [{"parts": [{"text": f"{DIRECTION}: {text}"}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300)); break
        except Exception as e:
            print("retry", line_id, attempt, str(e)[:120], file=sys.stderr); time.sleep(6 * (attempt + 1))
    else:
        raise SystemExit(f"failed {line_id}")
    pcm = base64.b64decode(d["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    path = os.path.join(out, f"{line_id}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    return path, len(pcm) / 48000

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("out")
    a.add_argument("--voice", default="Sadaltager"); a.add_argument("--model", default="gemini-2.5-pro-preview-tts")
    a.add_argument("--only", default="")
    o = a.parse_args(); os.makedirs(o.out, exist_ok=True)
    todo = [l for l in LINES if not o.only or l[0] in o.only.split(",")]
    with ThreadPoolExecutor(4) as ex:
        for p, dur in ex.map(lambda l: say(l[0], l[1], o.out, o.voice, o.model), todo):
            print(f"{dur:5.2f}s {p}", flush=True)
