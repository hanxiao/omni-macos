"""ComputeUnitUITests (Scripts/compute-unit-ui.sh): each STEP with its time, and the app's own GPU ms in the segment that step ends, by the
unit the app was on. python3 uisteps.py <out dir> [prefix]"""
import glob, os, re, sys
d = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else ""
for log in sorted(glob.glob(f"{d}/{pre}*.log")):
    if log.endswith(".app.log") or ".stderr-" in log: continue
    base = log[:-4]
    marks = []
    for line in open(log, errors="replace"):
        m = re.match(r"LAUNCH (\S+) ([\d.]+)", line)
        if m: marks.append((float(m.group(2)), "launch " + m.group(1)))
        m = re.match(r"STEP (.+?) ([\d.]+) s ([\d.]+)", line)
        if m: marks.append((float(m.group(3)), f"{m.group(1)} ({float(m.group(2)):.1f} s)"))
    samples = []
    if os.path.exists(base + ".gpu"):
        for line in open(base + ".gpu"):
            p = line.split()
            if len(p) >= 3: samples.append((float(p[1]), p[0], int(p[2])))
    mem = [int(l.split()[2]) for l in open(base + ".mem") if l.split()[2:] and l.split()[2].isdigit()] if os.path.exists(base + ".mem") else []
    def gpu_at(t):
        best = None
        for s in samples:
            if s[0] <= t: best = s
        return best
    print(f"== {os.path.basename(base)}  peak footprint {max(mem) if mem else 0} MB")
    unit, prev = "?", None
    for t, name in marks:
        # the sampler lags ~3 s: read the counter a little after the mark
        s = gpu_at(t + 3)
        if name.startswith("launch"): unit = "GPU" if name.endswith("gpu") else "ANE"; prev = None
        seg = ""
        if prev and s and prev[1] == s[1]:
            # The counter is per GPU client: one closing (the towers let go) takes its time with it.
            seg = f"{(s[2] - prev[2]) / 1e6:8.1f} ms GPU" if s[2] >= prev[2] else "    (a GPU client closed)"
        print(f"   [{unit}] {name:38s} {seg}")
        if name.startswith("to the Neural Engine"): unit = "ANE"
        if name.startswith("to the GPU"): unit = "GPU"
        prev = s
