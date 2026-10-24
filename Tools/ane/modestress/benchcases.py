"""Per case of one bench arm: wall time, the bench process's own GPU time, and the system's energy by unit
(minus an idle baseline per second), from benchgoal.sh's logs. python3 benchcases.py <arm dir> [idle cpu gpu ane dram W]"""
import bisect, re, statistics, sys
O = sys.argv[1]
idle = [float(x) for x in sys.argv[2:6]] if len(sys.argv) >= 6 else [0, 0, 0, 0]
marks = [l.split() for l in open(f"{O}/marks.log") if l.strip()]
t0 = float(marks[0][0])
tend = float(marks[1][0]) if len(marks) > 1 else None
cases = []
for line in open(f"{O}/bench.log", errors="replace"):
    m = re.match(r"\s*\[(\d+):(\d+)\]\s+(\d+)/(\d+) (\S+)", line)
    if m and (not cases or cases[-1][1] != m.group(5)):
        cases.append((t0 + int(m.group(1)) * 60 + int(m.group(2)), m.group(5)))
power = [tuple(map(float, l.split())) for l in open(f"{O}/power.log") if len(l.split()) == 5]
gpu = []
for l in open(f"{O}/gpu.log"):
    p = l.split()
    if len(p) >= 2: gpu.append((float(p[0]), int(p[1])))
def gpu_at(t):
    ts = [g[0] for g in gpu]; i = bisect.bisect_right(ts, t) - 1
    return gpu[max(i, 0)][1] if gpu else 0
bounds = cases + [(tend or (power[-1][0] if power else t0), "end")]
print(f"{'case':22} {'secs':>6} {'own GPU s':>9} {'GPU W':>6} {'ANE W':>6} {'CPU W':>6} {'DRAM W':>6} {'J (net)':>8}")
tot = [0.0] * 4
for (a, name), (b, _) in zip(bounds, bounds[1:]):
    pw = [p for p in power if a <= p[0] <= b]
    if not pw: continue
    avg = [statistics.fmean(p[k] for p in pw) for k in (1, 2, 3, 4)]
    net = sum(max(0.0, avg[k] - idle[k]) for k in range(4)) * (b - a)
    for k in range(4): tot[k] += max(0.0, avg[k] - idle[k]) * (b - a)
    own = (gpu_at(b) - gpu_at(a)) / 1e9
    print(f"{name:22} {b - a:6.0f} {own:9.2f} {avg[1]:6.2f} {avg[2]:6.3f} {avg[0]:6.1f} {avg[3]:6.2f} {net:8.0f}")
print(f"net energy over idle, whole run: CPU {tot[0]:.0f} J, GPU {tot[1]:.0f} J, ANE {tot[2]:.0f} J, DRAM {tot[3]:.0f} J")
