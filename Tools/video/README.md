# Intro video

The video on hanxiao.io/omni and in the README, rebuilt from the real app.

1. `swiftc -O recwin.swift -o work/recwin` - records one window with ScreenCaptureKit. The window
   must be on screen and the display awake: a covered window is not redrawn.
2. Index a demo library of public files into `work/db` (the takes show file names and paths).
3. `./take.sh <A|B|C|D> <seconds> "<OMNI_PERF_SCRIPT steps>" [app args]` per scene; the steps are in
   `App/PerfScript.swift` (`type:`, `similar:`, `frame:`, `front`, `appearance:light`, ...).
4. Decode each take's used span to `work/takes/<X>.rgb` at 60 fps with a `<X>.json` of `{"n": frames}`.
5. `uv run --with numpy --with scipy python music.py work/music-raw.wav`, then
   `ffmpeg -i work/music-raw.wav -af loudnorm=I=-14:TP=-1.5 work/music.wav`.
6. `uv run --with pillow --with numpy python render.py out.mp4` (`--preview "t1,t2"` for stills).
