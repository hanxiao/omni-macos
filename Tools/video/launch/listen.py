# Ask Gemini about an audio file. usage: listen.py <audio> "<question>" [model]
import sys, os, json, base64, urllib.request
KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()
path, q = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "gemini-3.1-pro-preview"
mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}[path.rsplit(".", 1)[1]]
body = {"contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(open(path, "rb").read()).decode()}},
                                {"text": q}]}]}
req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
d = json.load(urllib.request.urlopen(req, timeout=600))
print("".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]))
