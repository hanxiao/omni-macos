# Ask Gemini to watch a video. usage: watch.py <video.mp4> "<question>" [model]
# Small files go inline; the review copies are kept under 20 MB.
import sys, os, json, base64, urllib.request
KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()
path, q = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "gemini-3.1-pro-preview"
data = open(path, "rb").read()
assert len(data) < 19_500_000, "inline video must stay under 20 MB"
body = {"contents": [{"parts": [{"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(data).decode()}},
                                {"text": q}]}]}
req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
d = json.load(urllib.request.urlopen(req, timeout=900))
print("".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]))
