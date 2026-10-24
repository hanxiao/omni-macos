"""Per phase of a stress run: the instance's own GPU time (ns, from its Metal queues), system power by unit,
peak footprint, and every served request (count, failures, empty answers, p50/p95/max ms) by surface.
  python3 analyze.py <run dir>"""
import bisect, collections, csv, statistics, sys, os
W = sys.argv[1]
marks = [(float(t), name) for t, name in (l.split(maxsplit=1) for l in open(f"{W}/marks.log").read().split("\n") if l.strip())]
marks = [(t, n.strip()) for t, n in marks]

def series(path, parse):
    out = []
    if os.path.exists(path):
        for l in open(path):
            p = l.split()
            if len(p) >= 2:
                try: out.append(parse(p))
                except Exception: pass
    return out

gpu = series(f"{W}/gpu.log", lambda p: (float(p[0]), int(p[1]), [int(x) for x in p[2].split(",")] if len(p) > 2 and p[2] else []))
power = series(f"{W}/power.log", lambda p: (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
fp = series(f"{W}/fp.log", lambda p: (float(p[0]), int(p[1])))
load = list(csv.reader(open(f"{W}/load.csv"))) if os.path.exists(f"{W}/load.csv") else []

def at(seq, t):  # last sample at or before t
    i = bisect.bisect_right([s[0] for s in seq], t) - 1
    return seq[max(i, 0)] if seq else None

print(f"{'phase':16} {'secs':>6} {'GPU ms (own)':>12} {'GPU ms/s':>9} {'GPU W':>6} {'ANE W':>6} {'CPU W':>6} {'peak MB':>8}")
for (t0, name), (t1, _) in zip(marks, marks[1:]):
    g0, g1 = at(gpu, t0), at(gpu, t1)
    own = (g1[1] - g0[1]) / 1e6 if g0 and g1 else float("nan")
    pw = [p for p in power if t0 <= p[0] <= t1]
    avg = lambda k: statistics.fmean(p[k] for p in pw) if pw else float("nan")
    pk = max([f[1] for f in fp if t0 <= f[0] <= t1] or [0])
    dur = t1 - t0
    print(f"{name:16} {dur:6.0f} {own:12.1f} {own / dur if dur else 0:9.2f} {avg(2):6.2f} {avg(3):6.3f} {avg(1):6.1f} {pk:8d}")

# Which queues grew in each phase (the MLX queue vs the UI's), for the phases that used any GPU.
print("\nper-queue growth (ms), phases with > 5 ms:")
for (t0, name), (t1, _) in zip(marks, marks[1:]):
    g0, g1 = at(gpu, t0), at(gpu, t1)
    if not g0 or not g1 or (g1[1] - g0[1]) / 1e6 <= 5: continue
    n = min(len(g0[2]), len(g1[2]))
    grew = [(i, (g1[2][i] - g0[2][i]) / 1e6) for i in range(n) if g1[2][i] - g0[2][i] > 1e6]
    print(f"  {name:16} queues {len(g1[2])}: " + ", ".join(f"q{i} {ms:.1f}" for i, ms in grew))

print(f"\n{'phase':14} {'surface':6} {'n':>5} {'fail':>5} {'empty':>6} {'p50':>7} {'p95':>7} {'max':>7}")
by = collections.defaultdict(list)
for r in load:
    by[(r[1], r[2])].append(r)
for (ph, s), rs in sorted(by.items()):
    ms = sorted(float(r[3]) for r in rs)
    fails = [r for r in rs if r[5] != "1"]
    empty = [r for r in rs if r[5] == "1" and int(r[4]) == 0]
    print(f"{ph:14} {s:6} {len(rs):5d} {len(fails):5d} {len(empty):6d} {ms[len(ms)//2]:7.0f} {ms[int(len(ms)*0.95)]:7.0f} {ms[-1]:7.0f}"
          + (f"   e.g. {fails[0][6][:70]}" if fails else ""))
