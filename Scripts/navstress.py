#!/usr/bin/env python3
"""Seeded back/forward stress for Omni's folder browser.

Only navigation: Cmd-[ / Cmd-], clicks on the two chevrons, and double-clicks on rows to
descend. No destructive keys, no context menus, no Settings - same blacklist as drive.py.
"""
import Quartz, random, subprocess, sys, time

SEED, N = int(sys.argv[1]), int(sys.argv[2])
X, Y, W, H = (int(v) for v in sys.argv[3:7])
rnd = random.Random(SEED)
CMD = Quartz.kCGEventFlagMaskCommand
LBR, RBR = 33, 30

def key(code, flags=0):
    for down in (True, False):
        e = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        Quartz.CGEventSetFlags(e, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
        time.sleep(0.01)

def click(x, y, count=1):
    for i in range(count):
        for t in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            e = Quartz.CGEventCreateMouseEvent(None, t, (x, y), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventSetIntegerValueField(e, Quartz.kCGMouseEventClickState, i + 1)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
            time.sleep(0.015)

BACK  = (X + 380, Y + 26)     # the two halves of the nav capsule
FWD   = (X + 425, Y + 26)
def row(): return (X + 400, Y + 81 + 20 * rnd.randint(0, 8))

acts = [
    ("cmd_back",  lambda: key(LBR, CMD)),
    ("cmd_fwd",   lambda: key(RBR, CMD)),
    ("click_back", lambda: click(*BACK)),
    ("click_fwd",  lambda: click(*FWD)),
    ("descend",    lambda: click(*row(), count=2)),
    ("burst_back", lambda: [key(LBR, CMD) for _ in range(rnd.randint(2, 6))]),
    ("burst_fwd",  lambda: [key(RBR, CMD) for _ in range(rnd.randint(2, 6))]),
    ("thrash",     lambda: [key(rnd.choice([LBR, RBR]), CMD) for _ in range(8)]),
]

log = open("/tmp/navstress.log", "w")
for i in range(N):
    name, fn = rnd.choice(acts)
    log.write("#%d %s\n" % (i, name)); log.flush()
    try:
        fn()
    except Exception as ex:
        log.write("#%d EXC %s\n" % (i, ex)); log.flush()
    time.sleep(rnd.choice([0.03, 0.06, 0.12, 0.25]))
    if i % 25 == 0 and not subprocess.run(["pgrep", "-x", "Omni"],
                                          capture_output=True).stdout.strip():
        log.write("DEAD at #%d\n" % i); log.flush(); sys.exit(1)
log.write("DONE\n"); log.close()
print("navstress done")
