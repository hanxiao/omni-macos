# Launch film

The film on hanxiao.io/omni: a cut-paper world from Gemini image models, narration from Gemini
TTS, a Lyria score, app takes recorded from the real app, all composited in skia-python.

1. `assets.py` generates the plates and sprites (`gen.py`, Gemini 3 Pro Image, one house style)
   and cuts sprites out with Apple Vision (`swiftc -O cutout.swift -o ../work/launch/cutout`).
   `textures.py` derives every paper swatch from one generated sheet, so all colours share a grain.
2. `music.py` asks Lyria for the score; `timeline.py` holds the bar-aligned edit of it.
3. `vo.py` reads the script with Gemini TTS (Sadaltager, `gemini-2.5-pro-preview-tts`); the chosen
   takes are trimmed into `../work/launch/vo/fast/`. `words.py` gives word timings (mlx-whisper),
   which the picture syncs to. `mix.py` joins score and voice, ducking the score under speech.
4. `probe.sh` and `take.sh` record app takes on an APFS clone of the index in a Developer-ID-signed
   dev build, with synthetic sidebar history. Use queries whose visible results are safe to show.
5. `film.py` is the edit, `core.py` the drawing kit, `render.py` the renderer:
   `render.py stills <dir> t1,t2 --fast` for review, `render.py video <out.mp4>` for the master.
   `listen.py` and `watch.py` ask Gemini to review audio and video.

The Gemini key is read from `../work/.gemini-key` (git-ignored).
