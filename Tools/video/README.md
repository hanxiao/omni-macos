# Intro video

The video on hanxiao.io/omni and in the README, rebuilt from the real app.

1. `swiftc -O recwin.swift -o work/recwin` - records one window with ScreenCaptureKit. The window
   must be on screen and the display awake: a covered window is not redrawn.
2. The takes run on an APFS clone of the real index (`cp -c -R`), in a Developer-ID-signed copy of the build
   so it inherits the installed app's folder grants. Use queries whose results are safe to show.
3. `./take.sh <A|B|C|D> <seconds> "<OMNI_PERF_SCRIPT steps>" [app args]` per scene; the steps are in
   `App/PerfScript.swift` (`type:`, `similar:`, `frame:`, `front`, `appearance:light`, ...).
4. Decode each take's used span to `work/takes/<X>.rgb` at 60 fps with a `<X>.json` of `{"n": frames}`.
5. `uv run --with numpy --with scipy python music.py work/music-raw.wav`, then
   `ffmpeg -i work/music-raw.wav -af loudnorm=I=-14:TP=-1.5 work/music.wav`.
6. `swiftc -O sym.swift -o work/symtool` and render the feature-grid SF Symbols into `work/sym/`.
7. `uv run --with pillow --with numpy python render.py out.mp4` (`--preview "t1,t2"` for stills).
