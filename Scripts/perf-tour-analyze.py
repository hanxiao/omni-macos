# Main-thread stalls and footprint per tour phase: uv run python Scripts/perf-tour-analyze.py <out-dir> [-v]
import os, re, sys, subprocess, collections
R = sys.argv[1]
born = int(subprocess.check_output(["stat", "-f", "%B", f"{R}/hang.log"]).strip())
phases = [(l.split()[1], float(l.split()[2])) for l in open(f"{R}/tour.log")]
def phase_at(t):
    name = "pre"
    for n, e in phases:
        if t >= e: name = n
    return name
rows = collections.defaultdict(list); probes = collections.defaultdict(collections.Counter)
for l in open(f"{R}/hang.log"):
    m = re.search(r"t\+([\d.]+)s blocked (\d+) ms \(cpu (\d+) ms", l)
    if not m: continue
    t = born + float(m.group(1)); p = phase_at(t)
    rows[p].append((int(m.group(2)), int(m.group(3))))
    tail = l.split(")", 3)[-1].strip()
    for name in re.findall(r"([A-Za-z][\w.\-]+) (?:x\d+ )?[\d.]+ ?ms", tail): probes[p][name] += 1
fp = collections.defaultdict(list)
if os.path.exists(f"{R}/footprint.log"):
    for l in open(f"{R}/footprint.log"):
        a = l.split()
        if len(a) == 2 and a[1].endswith(("MB", "GB")):
            v = float(a[1][:-2]) * (1024 if a[1].endswith("GB") else 1)
            fp[phase_at(int(a[0]))].append(v)
print(f"{'phase':9} {'n>50':>5} {'n>100':>5} {'n>250':>5} {'sum ms':>7} {'worst':>6} {'cpu%':>5} {'fp max MB':>9}")
order = ["pre"] + [n for n, _ in phases]
tot = [0, 0, 0, 0]
for p in order:
    r = rows.get(p, []); f = fp.get(p, [])
    s = sum(b for b, _ in r); c = sum(c for _, c in r)
    print(f"{p:9} {len(r):5} {sum(b>100 for b,_ in r):5} {sum(b>250 for b,_ in r):5} {s:7} {max([b for b,_ in r], default=0):6} {100*c/max(s,1):5.0f} {max(f, default=0):9.0f}")
    tot[0]+=len(r); tot[1]+=sum(b>100 for b,_ in r); tot[2]+=sum(b>250 for b,_ in r); tot[3]+=s
print(f"{'TOTAL':9} {tot[0]:5} {tot[1]:5} {tot[2]:5} {tot[3]:7}")
if "-v" in sys.argv:
    for p in order:
        if probes[p]: print(p, probes[p].most_common(8))
