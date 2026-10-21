#!/usr/bin/env python3
"""Agent-facing search eval: what an agent gets from Omni's MCP tools, and what it pays.

Runs against an ISOLATED Omni serving the labeled eval testbed (serve-eval.sh starts one on port
51399 with a clone of the eval index). Three modes, stdlib only:

  probe  - every eval query through the real `search` tool exactly as an agent calls it. Scores the
           answer by file AND place (@p page, @L line, @t time), and records what the agent has to
           read: characters returned per call and per query until the answer appears, and latency.
           Deterministic: two runs on one build give the same numbers, so it is the A/B instrument.
  load   - N concurrent clients looping over the queries, plus a 1 Hz `initialize` probe that must
           stay fast throughout (0.17.2 took connections and never answered once its Task pool was
           full). Throughput, p50/p99/max, errors.
  agent  - real agent episodes: the `claude` CLI headless, Omni's tools only, a fixed turn budget.
           Success = the final answer names an acceptable file. Records tool calls, turns, tokens,
           cost and wall time per episode. Sends the SYNTHETIC testbed's text to the model - never
           run it against a real index.

  uv run Tools/agenteval/agenteval.py probe --bed /Volumes/han2tb/omni-eval [--port 51399] [--json out]
  uv run Tools/agenteval/agenteval.py load  --bed ... --clients 16 --seconds 30
  uv run Tools/agenteval/agenteval.py agent --bed ... [--model m] --per-tag 2 [--json out]
  uv run Tools/agenteval/agenteval.py agent --bed ... --qa Tools/agenteval/qa.json   # answer from content
"""
import argparse, concurrent.futures, json, os, re, shutil, statistics, subprocess, sys, tempfile, threading, time, unicodedata
import http.client

# ------------------------------------------------------------------------------------------------ MCP

class MCP:
    """JSON-RPC over HTTP, one KEEP-ALIVE connection per thread, the way MCP clients talk. Not
    urllib: it opened a connection per request and asked macOS SystemConfiguration for proxy
    settings on every one, and under load that stalled the client itself for 10-20 s at a time
    with no socket open to the server - which read exactly like the server pausing."""
    def __init__(self, port, timeout=30):
        self.port = port
        self.timeout = timeout
        self.n = 0
        self.lock = threading.Lock()
        self.local = threading.local()

    def _conn(self, timeout):
        c = getattr(self.local, "conn", None)
        if c is None:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
            self.local.conn = c
        c.timeout = timeout
        return c

    def call(self, method, params=None, timeout=None):
        with self.lock:
            self.n += 1
            rid = self.n
        body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": "2025-06-18"}
        t = time.perf_counter()
        for attempt in (0, 1):
            c = self._conn(timeout or self.timeout)
            try:
                c.request("POST", "/mcp", body=body, headers=headers)
                r = c.getresponse()
                raw = r.read()
                break
            except (http.client.HTTPException, ConnectionError, BrokenPipeError):
                # A kept-alive connection the server closed (idle timeout): reconnect once.
                c.close(); self.local.conn = None
                if attempt: raise
        ms = (time.perf_counter() - t) * 1000
        return json.loads(raw), ms, len(raw)

    def tool(self, name, args, timeout=None):
        msg, ms, nbytes = self.call("tools/call", {"name": name, "arguments": args}, timeout)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        res = msg["result"]
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return res, text, ms

# ------------------------------------------------------------------------------------------------ bed

def load_bed(bed):
    spec = json.load(open(os.path.join(bed, "bed.json")))
    tree = os.path.realpath(os.path.join(bed, "tree"))
    return spec["queries"], tree

def parse_expect(e):
    """'Docs/a.pdf@p2' -> ('Docs/a.pdf', 'p', (2, 2)); '@L100-200' -> ('L', (100, 200)); '@t240' -> ('t', (240, 240))."""
    if "@" not in e:
        return e, None, None
    path, place = e.rsplit("@", 1)
    m = re.fullmatch(r"([pLt])(\d+)(?:-(\d+))?", place)
    if not m:
        return path, None, None
    lo = int(m.group(2)); hi = int(m.group(3) or lo)
    return path, m.group(1), (lo, hi)

def parse_locator(loc):
    """MCP locator text -> (kind, value): 'Page 7' -> ('p', 7); 'Line 240' -> ('L', 240); '4:00' / '1:02:03' -> ('t', seconds)."""
    if not loc:
        return None
    m = re.fullmatch(r"Page (\d+)", loc)
    if m: return ("p", int(m.group(1)))
    m = re.fullmatch(r"Line (\d+)", loc)
    if m: return ("L", int(m.group(1)))
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d\d)", loc)
    if m: return ("t", int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
    return None

HIT = re.compile(r"^(\d+)\. (.+?)  \(([a-z]+), (\d+)%(.*)\)$")

def parse_hits(text):
    """The search tool's text block -> [(rank, path, kind, score, locator, body)] in order."""
    hits, cur = [], None
    for line in text.split("\n"):
        m = HIT.match(line)
        if m:
            meta = [s.strip() for s in m.group(5).split(",") if s.strip()]
            loc = next((s for s in meta if parse_locator(s)), None)
            cur = [int(m.group(1)), m.group(2), m.group(3), int(m.group(4)), loc, ""]
            hits.append(cur)
        elif cur is not None and line.startswith("   "):
            cur[5] += line[3:] + "\n"
    return hits

def nfc(p):
    # The filesystem hands back names decomposed (NFD) where the labels are composed: "Andalucía",
    # "Счёт" and every Hangul name differ byte for byte while naming the same file.
    return unicodedata.normalize("NFC", p)

def matches(hit_path, hit_loc, expect, tree, place_aware=True):
    for e in expect:
        path, kind, rng = parse_expect(e)
        if nfc(os.path.realpath(hit_path)) != nfc(os.path.join(tree, path)):
            continue
        if not place_aware or kind is None:
            return True
        loc = parse_locator(hit_loc)
        if loc and loc[0] == kind and rng[0] <= loc[1] <= rng[1]:
            return True
    return False

# ------------------------------------------------------------------------------------------------ probe

def probe(a):
    queries, tree = load_bed(a.bed)
    mcp = MCP(a.port)
    rows = []
    for q in queries:
        args = {"query": q["q"], "top_k": a.top_k}
        if a.max_snippet is not None:
            args["max_snippet"] = a.max_snippet
        _, text, ms = mcp.tool("search", args)
        hits = parse_hits(text)
        rank_place = next((h[0] for h in hits if matches(h[1], h[4], q["expect"], tree)), None)
        rank_file = next((h[0] for h in hits if matches(h[1], h[4], q["expect"], tree, place_aware=False)), None)
        # What the agent read before the answer: the text of every result up to and including it.
        chars_to_answer = None
        if rank_place:
            blocks = text.split("\n\n")
            chars_to_answer = sum(len(b) for b in blocks[:rank_place])
        rows.append({"q": q["q"], "tag": q["tag"], "rank": rank_place, "rank_file": rank_file,
                     "chars": len(text), "chars_to_answer": chars_to_answer, "ms": ms, "hits": len(hits)})
    report(rows, a)

def report(rows, a):
    def agg(rs):
        n = len(rs)
        r = lambda k: sum(1 for x in rs if x["rank"] and x["rank"] <= k) / n
        mrr = sum(1 / x["rank"] for x in rs if x["rank"]) / n
        rf = sum(1 for x in rs if x["rank_file"] and x["rank_file"] <= 10) / n
        return {"n": n, "R@1": round(r(1), 3), "R@5": round(r(5), 3), "R@10": round(r(10), 3), "MRR": round(mrr, 3),
                "fileR@10": round(rf, 3), "chars": int(statistics.mean(x["chars"] for x in rs)),
                "ms_p50": round(statistics.median(x["ms"] for x in rs), 1)}
    tags = sorted({x["tag"] for x in rows})
    out = {"overall": agg(rows), "by_tag": {t: agg([x for x in rows if x["tag"] == t]) for t in tags}, "rows": rows}
    o = out["overall"]
    lat = sorted(x["ms"] for x in rows)
    print(f"PROBE n={o['n']} R@1={o['R@1']} R@5={o['R@5']} R@10={o['R@10']} MRR={o['MRR']} fileR@10={o['fileR@10']} "
          f"chars/call={o['chars']} p50={o['ms_p50']}ms p90={lat[int(len(lat) * .9)]:.1f}ms max={lat[-1]:.1f}ms")
    for t in tags:
        b = out["by_tag"][t]
        print(f"  {t:13s} n={b['n']:3d} R@1={b['R@1']:.3f} R@10={b['R@10']:.3f} MRR={b['MRR']:.3f} chars={b['chars']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)

# ------------------------------------------------------------------------------------------------ load

def load(a):
    queries, _ = load_bed(a.bed)
    mcp = MCP(a.port, timeout=60)
    stop = time.time() + a.seconds
    lat, errs, init_lat = [], [], []
    lk = threading.Lock()

    def worker(i):
        k = i
        while time.time() < stop:
            q = queries[k % len(queries)]["q"]; k += a.clients
            try:
                _, _, ms = mcp.tool("search", {"query": q, "top_k": 20, "max_snippet": 400})
                with lk: lat.append(ms)
            except Exception as e:
                with lk: errs.append(repr(e)[:120])

    def initprobe():
        while time.time() < stop:
            try:
                _, ms, _ = mcp.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                   "clientInfo": {"name": "agenteval", "version": "1"}}, timeout=20)
                with lk: init_lat.append(ms)
            except Exception as e:
                with lk: errs.append("initialize: " + repr(e)[:100])
            time.sleep(1)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.clients + 1) as ex:
        fs = [ex.submit(worker, i) for i in range(a.clients)] + [ex.submit(initprobe)]
        for f in fs: f.result()
    wall = time.time() - t0
    lat.sort(); init_lat.sort()
    pct = lambda v, p: v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")
    print(f"LOAD clients={a.clients} {len(lat)} searches in {wall:.1f}s = {len(lat) / wall:.1f}/s "
          f"p50={pct(lat, .5):.1f}ms p99={pct(lat, .99):.1f}ms max={lat[-1] if lat else 0:.1f}ms errors={len(errs)} | "
          f"initialize n={len(init_lat)} p50={pct(init_lat, .5):.1f}ms max={init_lat[-1] if init_lat else 0:.1f}ms")
    for e in errs[:5]: print("  error:", e)
    if a.json:
        json.dump({"clients": a.clients, "n": len(lat), "wall": wall, "lat": lat, "init": init_lat, "errors": errs}, open(a.json, "w"))

# ------------------------------------------------------------------------------------------------ agent

OMNI_TOOLS = ["search", "search_inline", "read_file", "file_status", "tag_image", "ocr"]

PROMPT = ("Find the file on this Mac that best matches this request: \"{q}\".\n"
          "Use the omni tools to search. When you are done, reply with ONLY the absolute path of the file "
          "on the first line, and on the second line the page, line or timestamp where it matches if the tools "
          "reported one. If no file matches, reply NONE.")

QA_PROMPT = ("Answer this question about the user's files on this Mac: \"{q}\"\n"
             "Use the omni tools. Reply with the answer on the first line and the absolute path of the file "
             "it came from on the second line. If the files do not say, reply UNKNOWN.")

def norm(s):
    # "4,700" and "4 700" and "4700" are one number; case and spacing do not matter.
    return re.sub(r"(?<=\d)[ ,](?=\d)", "", s.lower())

def episode(q, a, tree, cfg, workdir):
    prompt = QA_PROMPT.format(q=q["q"]) if "keys" in q else PROMPT.format(q=q["q"])
    cmd = [a.claude, "-p", prompt] + (["--model", a.model] if a.model else []) + ["--tools", "",
           "--strict-mcp-config", "--mcp-config", cfg,
           "--allowedTools", ",".join(f"mcp__omni__{t}" for t in OMNI_TOOLS),
           "--permission-mode", "bypassPermissions", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--max-budget-usd", str(a.budget)]
    t = time.time()
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=a.timeout)
    calls, result, tool_chars, trail = {}, None, 0, []
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "user":
            # What the tools handed the model: Omni's share of the context.
            for c in ev.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    body = c.get("content")
                    tool_chars += len(body) if isinstance(body, str) else sum(len(x.get("text", "")) for x in body or [] if isinstance(x, dict))
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    calls[c["name"]] = calls.get(c["name"], 0) + 1
                    trail.append({"tool": c["name"].replace("mcp__omni__", ""), "args": c.get("input", {})})
        elif ev.get("type") == "result":
            result = ev
    answer = (result or {}).get("result") or ""
    first = answer.strip().split("\n")[0].strip().strip("`") if answer.strip() else ""
    if "keys" in q:   # a question: right when the answer carries the key
        ok = any(k in norm(answer.strip().split("\n")[0] if answer.strip() else "") for k in q["keys"])
        usage = (result or {}).get("usage", {})
        return {"q": q["q"], "tag": "qa", "ok": ok, "ok_place": ok, "answer": first[:200], "calls": calls,
                "n_calls": sum(calls.values()), "turns": (result or {}).get("num_turns"), "tool_chars": tool_chars,
                "trail": trail, "in_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                "out_tokens": usage.get("output_tokens", 0), "cost": (result or {}).get("total_cost_usd", 0),
                "s": round(time.time() - t, 1), "model": ",".join((result or {}).get("modelUsage", {}).keys()),
                "error": None if result and not result.get("is_error") else (p.stderr[-300:] or (result or {}).get("result", ""))[:300]}
    ok_file = bool(first) and matches(first, None, q["expect"], tree, place_aware=False)
    second = answer.strip().split("\n")[1].strip() if len(answer.strip().split("\n")) > 1 else ""
    ok_place = ok_file and (all(parse_expect(e)[1] is None for e in q["expect"])
                            or matches(first, normalize_place(second), q["expect"], tree))
    usage = (result or {}).get("usage", {})
    return {"q": q["q"], "tag": q["tag"], "ok": ok_file, "ok_place": ok_place, "answer": first,
            "calls": calls, "n_calls": sum(calls.values()), "turns": (result or {}).get("num_turns"), "tool_chars": tool_chars,
            "trail": trail,
            "in_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
            "out_tokens": usage.get("output_tokens", 0), "cost": (result or {}).get("total_cost_usd", 0),
            "s": round(time.time() - t, 1), "model": ",".join((result or {}).get("modelUsage", {}).keys()),
            "error": None if result and not result.get("is_error") else (p.stderr[-300:] or (result or {}).get("result", ""))[:300]}

def normalize_place(s):
    s = s.strip()
    m = re.search(r"[Pp]age\s+(\d+)", s)
    if m: return f"Page {m.group(1)}"
    m = re.search(r"[Ll]ine\s+(\d+)", s)
    if m: return f"Line {m.group(1)}"
    m = re.search(r"(\d+:)?\d+:\d\d", s)
    if m: return m.group(0)
    return s

def agent(a):
    queries, tree = load_bed(a.bed)
    by_tag = {}
    for q in queries:
        by_tag.setdefault(q["tag"], []).append(q)
    chosen = [q for t in sorted(by_tag) for q in by_tag[t][:a.per_tag]]
    if a.only:
        wanted = set(json.load(open(a.only)))
        chosen = [q for q in queries if q["q"] in wanted]
    if a.qa:
        chosen = json.load(open(a.qa))
    workdir = tempfile.mkdtemp(prefix="agenteval-")   # no CLAUDE.md, no project settings
    cfg = os.path.join(workdir, "mcp.json")
    json.dump({"mcpServers": {"omni": {"type": "http", "url": f"http://127.0.0.1:{a.port}/mcp"}}}, open(cfg, "w"))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(a.parallel) as ex:
        for r in ex.map(lambda q: episode(q, a, tree, cfg, workdir), chosen):
            rows.append(r)
            print(f"  {'OK ' if r['ok'] else 'NO '}{'@' if r['ok_place'] else ' '} {r['tag']:12s} calls={r['n_calls']} "
                  f"turns={r['turns']} in={r['in_tokens']} out={r['out_tokens']} ${r['cost']:.4f} {r['s']}s  {r['q'][:50]}"
                  + (f"  ERR {r['error'][:80]}" if r["error"] else ""), flush=True)
    shutil.rmtree(workdir, ignore_errors=True)
    n = len(rows)
    ok = sum(r["ok"] for r in rows); okp = sum(r["ok_place"] for r in rows)
    print(f"AGENT model={rows[0]['model'] if rows else a.model} n={n} success={ok / n:.3f} place={okp / n:.3f} "
          f"calls/q={statistics.mean(r['n_calls'] for r in rows):.2f} turns/q={statistics.mean((r['turns'] or 0) for r in rows):.2f} "
          f"in_tok/q={int(statistics.mean(r['in_tokens'] for r in rows))} out_tok/q={int(statistics.mean(r['out_tokens'] for r in rows))} "
          f"tool_chars/q={int(statistics.mean(r['tool_chars'] for r in rows))} "
          f"cost/q=${statistics.mean(r['cost'] for r in rows):.4f} s/q={statistics.mean(r['s'] for r in rows):.1f} "
          f"tokens/solved={int(sum(r['in_tokens'] + r['out_tokens'] for r in rows) / max(1, ok))}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)

# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("probe", "load", "agent"):
        s = sub.add_parser(name)
        s.add_argument("--bed", required=True)
        s.add_argument("--port", type=int, default=51399)
        s.add_argument("--json")
    sub.choices["probe"].add_argument("--top-k", type=int, default=10)
    sub.choices["probe"].add_argument("--max-snippet", type=int)
    sub.choices["load"].add_argument("--clients", type=int, default=16)
    sub.choices["load"].add_argument("--seconds", type=float, default=30)
    g = sub.choices["agent"]
    g.add_argument("--model", default="", help="the claude CLI's model; default: whatever the CLI runs")
    g.add_argument("--per-tag", type=int, default=2)
    g.add_argument("--parallel", type=int, default=4)
    g.add_argument("--budget", type=float, default=0.5)
    g.add_argument("--timeout", type=float, default=300)
    g.add_argument("--claude", default=shutil.which("claude") or "claude")
    g.add_argument("--only", help="a JSON list of query texts to run instead of --per-tag")
    g.add_argument("--qa", help="questions with answer keys (Tools/agenteval/qa.json) instead of the find-the-file queries")
    a = ap.parse_args()
    if a.port == 51299:
        sys.exit("51299 is a real install's port; run against an isolated instance (serve-eval.sh, 51399)")
    {"probe": probe, "load": load, "agent": agent}[a.mode](a)

if __name__ == "__main__":
    main()
