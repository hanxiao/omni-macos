"""One line of per-process GPU time: '<unix time> <total ns> <q1>,<q2>,...' - every Metal queue of every GPU
driver client the pid opened, in ioreg order (stable while the clients live). python3 gpuq.py <pid>"""
import re, subprocess, sys, time
pid = sys.argv[1]
out = subprocess.run(["ioreg", "-l", "-w0", "-c", "AGXDeviceUserClient"], capture_output=True, text=True).stdout
qs, usage = [], None
for line in out.splitlines():
    if "+-o " in line:   # a new node: a usage record belongs to its own node only
        usage = None
    m = re.search(r'"AppUsage" = \((.*)\)', line)
    if m:
        usage = [int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', m.group(1))]
    m = re.search(r'"IOUserClientCreator" = "pid (\d+),', line)
    if m and m.group(1) == pid and usage is not None:
        qs += usage
        usage = None   # child nodes repeat the creator line: count the record once
print(f"{time.time():.3f} {sum(qs)} {','.join(map(str, qs))}")
