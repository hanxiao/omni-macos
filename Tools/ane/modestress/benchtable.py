"""The paper's table for Nano and Small on the GPU and on the Neural Engine, from benchgoal.sh's arm dirs:
every row of the Settings benchmark (bench.json), the energy each arm drew in its indexing and query cases
(system power by unit over the case's window, less an idle baseline), and the arm's own GPU time outside the
benchmark's GPU instruments (its clock probe after every case, the simulated GPU app, the attention curve,
the canary). python3 benchtable.py <bench dir> > table.md   (arms <dir>/{nano,small}-{gpu,ane}, idle.log)"""
import bisect, json, re, statistics, sys, os
D = sys.argv[1]
ARMS = [("nano", "gpu"), ("nano", "ane"), ("small", "gpu"), ("small", "ane")]
HEAD = ["Nano GPU", "Nano ANE", "Small GPU", "Small ANE"]
INSTRUMENTS = {"gpu_app", "attention", "canary"}

idle_rows = [list(map(float, l.split())) for l in open(f"{D}/idle.log") if len(l.split()) == 5]
IDLE = [statistics.fmean(r[k] for r in idle_rows) for k in (1, 2, 3, 4)]   # cpu gpu ane dram W

def cases(O):
    marks = [l.split() for l in open(f"{O}/marks.log") if l.strip()]
    t0, tend = float(marks[0][0]), float(marks[-1][0])
    out = []
    for line in open(f"{O}/bench.log", errors="replace"):
        m = re.match(r"\s*\[(\d+):(\d+)\]\s+(\d+)/(\d+) (\S+?)\s", line)
        if m and (not out or out[-1][1] != m.group(5)):
            out.append((t0 + int(m.group(1)) * 60 + int(m.group(2)), m.group(5)))
    return [(a, b, n) for (a, n), (b, _) in zip(out, out[1:] + [(tend, "end")])]

def series(O):
    power = [tuple(map(float, l.split())) for l in open(f"{O}/power.log") if len(l.split()) == 5]
    gpu = [(float(p[0]), int(p[1])) for p in (l.split() for l in open(f"{O}/gpu.log")) if len(p) >= 2]
    return power, gpu

def window(power, a, b):
    pw = [p for p in power if a <= p[0] <= b]
    return [statistics.fmean(p[k] for p in pw) for k in (1, 2, 3, 4)] if pw else [0] * 4

def gpu_growth(gpu, a, b):
    """Own GPU seconds in [a, b]: only increases count (a client closing drops the total)."""
    pts = [g for g in gpu if a <= g[0] <= b]
    return sum(max(0, y[1] - x[1]) for x, y in zip(pts, pts[1:])) / 1e9

arms = {}
for m, u in ARMS:
    O = f"{D}/{m}-{u}"
    if not os.path.exists(f"{O}/bench.json"): continue
    bench = json.load(open(f"{O}/bench.json"))
    power, gpu = series(O)
    cs = cases(O)
    # The driver books a client's GPU time seconds late (measured: the attention curve's 3.7 s showed
    # 4-6 s after it ended, in the next case), so an instrument's window runs on LAG seconds.
    LAG = 8
    spans = [(a, b + LAG) for a, b, n in cs if n in INSTRUMENTS]
    own = sum(max(0, y[1] - x[1]) for x, y in zip(gpu, gpu[1:])
              if not any(s0 <= y[0] <= s1 for s0, s1 in spans)) / 1e9
    # The clock probe runs after every case: its GPU time is the instrument's, measured alone (~0.23 s).
    probes = 0.23 * len([c for c in cs if c[2] not in INSTRUMENTS])
    energy = {}
    for a, b, n in cs:
        w = window(power, a, b)
        energy[n] = (b - a, w)
    whole = window(power, cs[0][0], cs[-1][1])
    arms[(m, u)] = dict(table={(r["group"], r["task"]): (r["cells"], r.get("unit", "")) for r in bench["table"]},
                        energy=energy, own=own, probes=probes, wall=cs[-1][1] - cs[0][0], whole=whole)

def fmt(cells, unit):
    if "value" in cells:
        v = cells["value"]
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"
    p50, p95 = cells.get("p50"), cells.get("p95")
    if p50 is None: return "-"
    f = lambda x: f"{x:,.0f}" if x >= 100 else f"{x:.3g}"
    return f(p50) + (f" / {f(p95)}" if p95 is not None else "")

rows = []
for m, u in ARMS:
    if (m, u) in arms:
        for k in arms[(m, u)]["table"]:
            if k not in rows: rows.append(k)
print("| Group | Task | " + " | ".join(HEAD) + " | Unit |")
print("|---|---|" + "---|" * len(HEAD) + "---|")
for k in rows:
    if k[0] == "Conditions": continue
    cells = []
    for m, u in ARMS:
        t = arms.get((m, u), {}).get("table", {}).get(k)
        cells.append(fmt(*t) if t else "-")
    unit = next((arms[a]["table"][k][1] for a in arms if k in arms[a]["table"]), "")
    unit = unit + (" (p50 / p95)" if unit == "ms" else "")
    print(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + f" | {unit} |")

def net(w): return [max(0.0, w[i] - IDLE[i]) for i in range(4)]
print("\n| Energy (system, less idle) | " + " | ".join(HEAD) + " |")
print("|---|" + "---|" * len(HEAD))
for case, label in [("index_text", "Text indexing case"), ("index_image", "Image indexing case"), ("queries", "Query case")]:
    for i, unit in enumerate(["CPU", "GPU", "ANE", "DRAM"]):
        cells = []
        for a in ARMS:
            e = arms.get(a, {}).get("energy", {}).get(case)
            cells.append(f"{net(e[1])[i]:.2f} W" if e else "-")
        print(f"| {label}, {unit} | " + " | ".join(cells) + " |")
cells = []
for a in ARMS:
    e = arms.get(a, {}).get("energy", {}).get("index_text"); t = arms.get(a, {}).get("table", {}).get(("Indexing", "Text indexing"))
    if e and t and t[0].get("value"):
        cells.append(f"{sum(net(e[1])) / t[0]['value'] * 1e6:,.0f} J")
    else: cells.append("-")
print("| Text indexing, energy per million tokens | " + " | ".join(cells) + " |")
cells = []
for a in ARMS:
    x = arms.get(a)
    cells.append(f"{x['wall'] / 60:.0f} min, {sum(net(x['whole'])) * x['wall'] / 1000:.1f} kJ" if x else "-")
print("| Whole run, wall time and energy | " + " | ".join(cells) + " |")
print("| Own GPU time outside the instruments (of it, the clock probes) | "
      + " | ".join(f"{arms[a]['own']:.1f} s ({arms[a]['probes']:.1f} s)" if a in arms else "-" for a in ARMS) + " |")
print(f"\nIdle baseline: CPU {IDLE[0]:.2f} W, GPU {IDLE[1]:.2f} W, ANE {IDLE[2]:.3f} W, DRAM {IDLE[3]:.2f} W.")
