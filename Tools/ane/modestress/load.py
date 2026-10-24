"""Search load on every served surface of an isolated Omni, round-robin: REST /v1/search, MCP tools/call
search, MCP tag_image, and the omni CLI over the private socket. One CSV line per request:
  unix_time,phase,surface,ms,results,ok,detail
  python3 load.py --port 51399 --socket <sock> --cli <omni> --phase <name> --seconds 40 --out load.csv
      [--images <dir>] [--gap 0.2]"""
import argparse, http.client, json, os, random, subprocess, time

QUERIES = ["invoice from the dentist", "photo of a dog on the beach", "meeting notes about the budget",
           "a recording of a lecture", "zoo trip video", "tax documents 2024", "red car", "recipe for pasta",
           "Rechnung Zahnarzt", "会议记录 预算", "contrat de location", "a scanned receipt", "screenshot of an error",
           "where did I write about the launch plan", "a song with a guitar", "chart of quarterly revenue"]

def rest(port, q):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/search", body=json.dumps({"query": q, "top_k": 10}), headers={"Content-Type": "application/json"})
    r = c.getresponse(); d = json.loads(r.read() or b"{}"); c.close()
    res = d.get("results") or d.get("data") or []
    return r.status == 200, len(res), f"http {r.status}"

class MCP:
    def __init__(self, port):
        self.port, self.n = port, 0
    def call(self, name, args):
        self.n += 1
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        body = json.dumps({"jsonrpc": "2.0", "id": self.n, "method": "tools/call", "params": {"name": name, "arguments": args}})
        c.request("POST", "/mcp", body=body, headers={"Content-Type": "application/json",
                  "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"})
        r = c.getresponse(); raw = r.read(); c.close()
        msg = json.loads(raw)
        if "error" in msg: return False, 0, str(msg["error"])[:80]
        res = msg["result"]
        if res.get("isError"): return False, 0, json.dumps(res)[:120]
        sc = res.get("structuredContent") or {}
        files = sc.get("files") or []
        n = len(sc.get("results") or []) or sum(len(f.get("tags") or []) for f in files)
        if n == 0:
            txt = "\n".join(x.get("text", "") for x in res.get("content", []))
            if name == "tag_image":   # "1. <path>  tag, tag, tag" - or a reason in place of the tags
                for line in txt.splitlines():
                    tail = line.split("  ", 1)[1] if "  " in line else ""
                    if tail and not any(w in tail for w in ("outside", "NOT indexed", "no tags", "not an image", "not a decodable")):
                        n += len(tail.split(", "))
            else:
                n = txt.count("\n") + 1 if txt.strip() else 0
        return True, n, ""

def cli(path, sock, q):
    env = dict(os.environ, OMNI_SOCKET=sock)
    p = subprocess.run([path, "search", q, "--top-k", "10", "--json", "--no-launch"], capture_output=True, text=True, env=env, timeout=120)
    if p.returncode != 0: return False, 0, p.stderr.strip()[:120]
    try:
        d = json.loads(p.stdout); res = d.get("results") if isinstance(d, dict) else d
        return True, len(res or []), ""
    except Exception as e:
        return False, 0, f"json {e}"

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--port", type=int, default=51399); a.add_argument("--socket", required=True); a.add_argument("--cli", required=True)
    a.add_argument("--phase", required=True); a.add_argument("--seconds", type=float, default=40); a.add_argument("--out", required=True)
    a.add_argument("--images", default=""); a.add_argument("--gap", type=float, default=0.2)
    o = a.parse_args()
    imgs = [os.path.join(dp, f) for dp, _, fs in os.walk(o.images) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))] if o.images else []
    mcp, end, i = MCP(o.port), time.time() + o.seconds, 0
    with open(o.out, "a") as f:
        while time.time() < end:
            q = QUERIES[i % len(QUERIES)]
            surface = ["rest", "mcp", "cli", "tag"][i % 4] if imgs else ["rest", "mcp", "cli"][i % 3]
            t = time.perf_counter()
            try:
                if surface == "rest": ok, n, d = rest(o.port, q)
                elif surface == "mcp": ok, n, d = mcp.call("search", {"query": q, "top_k": 10})
                elif surface == "cli": ok, n, d = cli(o.cli, o.socket, q)
                else: ok, n, d = mcp.call("tag_image", {"paths": [random.Random(i).choice(imgs)], "recompute": True, "top_k": 5})
            except Exception as e:
                ok, n, d = False, 0, f"{type(e).__name__}: {e}"[:120]
            ms = (time.perf_counter() - t) * 1000
            f.write(f"{time.time():.3f},{o.phase},{surface},{ms:.1f},{n},{int(ok)},{d.replace(',', ';')}\n"); f.flush()
            i += 1
            time.sleep(o.gap)
