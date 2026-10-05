# The film's clock: the score edit (bar-aligned source spans) and where each narration line starts.
# Shared by mix.py and render.py, so picture and sound cannot drift apart.
import os, wave

BAR = 2.3527                       # 102 BPM, measured on the score's downbeats
# (source start, source end) of the Lyria score. Every cut sits 25 ms before a measured kick onset,
# so the crossfade is over before the hit; the ending goes through the score's own last bar into its
# drop rather than cutting into the drop.
PRE = 0.025
MUSIC = [(0.0, 56.572 - PRE),                      # intro, then the groove (first kick 28.338)
         (28.338 - PRE, 56.572 - PRE),             # the groove again, for the mechanisms
         (56.572 - PRE, 103.646 - PRE),            # breakdown, then the lift
         (120.102 - PRE, 133.0)]                   # the last bar, the drop, the final chord

def _starts():
    t, out = 0.0, []
    for a, b in MUSIC:
        out.append(t); t += b - a
    return out, t
SEG_START, LENGTH = _starts()

GROOVE = 28.338                                   # first kick: the title lands here
REPEAT = SEG_START[1] + PRE                       # the groove's first kick, again
BREAK = SEG_START[2] + PRE                        # the breakdown
LIFT = SEG_START[2] + PRE + (75.396 - 56.572)     # the lift
OUTRO = SEG_START[3] + PRE                        # the last bar
DROP = OUTRO + (122.47 - 120.102)                 # the drums stop
CHORD = OUTRO + (123.6 - 120.102)                 # the last chord sounds

# Narration: the chosen takes in vo/fast/<line>.wav, read at a brisk pace; packed from anchors with
# breathing room between them.
VO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "launch", "vo", "fast")
def dur(line):
    with wave.open(os.path.join(VO_DIR, line + ".wav")) as w:
        return w.getnframes() / w.getframerate()

def _place():
    v = {"old": 1.2, "cloud": 7.2, "down": 15.8}
    v["omni"] = GROOVE - 0.45 - dur("omni")             # "...Omni." just before the first kick
    v["easy"] = GROOVE + 1.05
    v["only"] = v["easy"] + dur("easy") + 0.8
    v["messy"] = v["only"] + dur("only") + 1.8          # room for the panel to shrink into the cavity
    v["once"] = v["messy"] + dur("messy") + 0.8
    v["aside"] = v["once"] + dur("once") + 1.3
    v["funnel"] = v["aside"] + dur("aside") + 1.0
    v["agents"] = max(BREAK + 0.5, v["funnel"] + dur("funnel") + 0.9)
    v["paper"] = v["agents"] + dur("agents") + 0.9
    v["app"] = LIFT + 0.9
    v["end"] = CHORD - 0.35 - dur("end")                # "...entirely yours" just before the chord
    return v
VO = _place()
