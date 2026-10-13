# The Linked Macs film's clock: the score edit and where each narration line starts.
# Shared by mix.py and film.py, so picture and sound cannot drift apart.
import os, wave

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..", "work", "linked")
SCORE = os.path.join(WORK, "music", "score78-0.mp3")

# Marks measured on the chosen score (librosa, 2026-10-12): the bass leaves at the breakdown and comes
# back with the groove; the last hit is the strongest onset after 72 s, and the score decays to
# -47 dB by 76 s by itself.
BREAK = 46.115                           # the breakdown: a Mac asleep
RETURN = 57.075                          # the groove returns: it wakes
LAST_HIT = 73.10                         # the final chord
LENGTH = LAST_HIT + 3.4

VO_DIR = os.path.join(WORK, "vo", "fast")
def dur(line):
    with wave.open(os.path.join(VO_DIR, line + ".wav")) as w:
        return w.getnframes() / w.getframerate()

def _place():
    v = {"intro": 0.9}
    v["link"] = v["intro"] + dur("intro") + 1.4
    v["proof"] = v["link"] + dur("link") + 1.4
    v["search"] = v["proof"] + dur("proof") + 1.2
    v["swarm"] = v["search"] + dur("search") + 1.5     # room for the app shot to land
    v["away"] = BREAK + 0.9                           # the breakdown is the Mac asleep
    v["open"] = v["away"] + dur("away") + 1.1         # it wakes as the groove returns
    v["agents"] = RETURN + 3.3
    v["end"] = LAST_HIT - 0.45 - dur("end")           # "...goes federal" just before the last hit
    return v
VO = _place()
