# Plate generator: Gemini image models with one fixed style.
# usage: gen.py <out.png> "<subject>" [--ref img.png ...] [--aspect 16:9] [--size 2K] [--model m] [--raw]
import sys, os, json, base64, argparse, urllib.request, time

KEY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", ".gemini-key")).read().strip()

STYLE = (
    "Handcrafted cut-paper diorama photographed in a studio. Layered matte cardstock with visible "
    "cut edges, slight paper fiber texture and soft contact shadows between layers. Soft warm key "
    "light from the upper left, gentle falloff, shallow depth of field like a macro lens. Restrained "
    "palette: cream paper #F3EDE2, kraft #C9A27A, soil browns #6B4A33 and #3A2A20, ink #1D1D1F, with "
    "a single accent of soft teal #7FD8D0. Calm, precise, editorial, generous negative space. "
    "No text, no letters, no numbers, no logos, no watermarks, no people's faces."
)

def generate(prompt, refs, aspect, size, model):
    parts = [{"text": prompt}]
    for r in refs:
        mime = "image/png" if r.endswith(".png") else "image/jpeg"
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(open(r, "rb").read()).decode()}})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"],
                                 "imageConfig": {"aspectRatio": aspect, "imageSize": size}}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            for c in d.get("candidates", []):
                for p in c.get("content", {}).get("parts", []):
                    if "inlineData" in p: return base64.b64decode(p["inlineData"]["data"])
            raise RuntimeError(json.dumps(d)[:400])
        except Exception as e:
            print("retry", attempt, str(e)[:200], file=sys.stderr); time.sleep(5 * (attempt + 1))
    raise SystemExit("failed")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("out"); a.add_argument("subject")
    a.add_argument("--ref", action="append", default=[])
    a.add_argument("--aspect", default="16:9"); a.add_argument("--size", default="2K")
    a.add_argument("--model", default="gemini-3-pro-image")
    a.add_argument("--raw", action="store_true", help="subject only, no house style")
    o = a.parse_args()
    prompt = o.subject if o.raw else f"{o.subject}\n\nStyle: {STYLE}"
    data = generate(prompt, o.ref, o.aspect, o.size, o.model)
    open(o.out, "wb").write(data)
    print(o.out, len(data))
