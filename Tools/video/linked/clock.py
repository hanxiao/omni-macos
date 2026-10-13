# The Linked Macs film's clock: the score (used whole) and where each narration line starts.
# Shared by mix.py and film.py, so picture and sound cannot drift apart.
import os, wave

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..", "work", "linked")
SCORE = os.path.join(WORK, "music", "score-2.mp3")

# Measured on the score (librosa, 2026-10-12): beat 0.418 s; the groove builds from ~10 s;
# the breakdown runs 36.1-46.2; the groove returns at 46.3; the last hit is at 63.4, then silence.
BEAT = 0.418
GROOVE = 10.0
BREAK = 36.1
RETURN = 46.3
LAST_HIT = 63.4
LENGTH = 65.2

VO_DIR = os.path.join(WORK, "vo", "fast")
def dur(line):
    with wave.open(os.path.join(VO_DIR, line + ".wav")) as w:
        return w.getnframes() / w.getframerate()

def _place():
    v = {"many": 1.0}
    v["link"] = 4.6
    v["proof"] = v["link"] + dur("link") + 1.4
    v["search"] = v["proof"] + dur("proof") + 1.2
    v["away"] = BREAK + 0.9                      # the breakdown is the Mac asleep
    v["open"] = v["away"] + dur("away") + 1.1    # it wakes as the groove returns
    v["agents"] = RETURN + 3.3
    v["end"] = LAST_HIT - 0.45 - dur("end")      # "...zero point sixteen" just before the last hit
    return v
VO = _place()
