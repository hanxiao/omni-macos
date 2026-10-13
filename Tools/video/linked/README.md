# Linked Macs film

The one-minute film for Linked Macs: the launch film's world and kit (`../launch`: cut-paper plates,
the mole, `core.py`, the renderer), its voice and the same score instrumentation. `storyboard.md` is
the plan; `clock.py` the measured score marks and where each line starts.

1. `vo.py <dir>` reads the script (Gemini 2.5 Pro TTS, Sadaltager), two takes; the chosen takes are
   trimmed and sped to 1.1x into `../work/linked/vo/fast/`. `words.py` gives word timings.
2. `music.py <prefix> 3` asks Lyria for the score; the chosen one is used whole (`clock.SCORE`).
3. New plates and sprites come from `../launch/gen.py` with the launch assets as references, cut out
   with `../work/launch/cutout`.
4. App takes: two isolated instances of the release build linked over loopback (perf script steps
   `linkwindow`, `linkpair`, `linkcode`, `linkaccept`, `waitfile`), recorded with
   `../work/launch/recwin`; the second Mac's folder is generated photos, the first is the launch
   film's index clone. Only queries whose visible results are landscapes.
5. `mix.py <out.wav>`, then `render.py stills <dir> t1,t2 --fast` to review and
   `render.py video <out.mp4>` for the master.
