/**
 * Omni profiling Worker.
 *
 * POST /omni/profiling  -> ingest one anonymous profiling report (204 / 400 / 429)
 * GET  /omni/profiling  -> aggregated stats for the landing page (JSON)
 * OPTIONS               -> CORS preflight
 *
 * Privacy: never stores raw IP, hostname, or any PII. Client-supplied timestamps
 * are ignored; all times come from the worker runtime clock.
 */

export interface Env {
  DB: D1Database;
  // Optional: a secret used to salt the IP hash. Falls back to a constant if unset.
  // Set with: wrangler secret put RATE_SALT
  RATE_SALT?: string;
}

// Uploads are accepted for every KNOWN dataset and stored with the version they ran on.
// v1 (1000 files) and v2 (300 files, same modality mix) have comparable files/s and tokens/s RATES,
// so the indexing history spans both. bench-v4 is the table benchmark (app 0.15.9+): its indexing
// pass is text only, so its rates are NOT comparable with v1/v2 and stay out of that history; its
// table is aggregated on its own (`bench` in GET).
const DATASET_VERSION = "profiling-v2";
const HISTORY_DATASETS = ["profiling-v1", "profiling-v2"];
const BENCH_DATASET = "bench-v4";
const ACCEPTED_DATASETS = new Set([...HISTORY_DATASETS, BENCH_DATASET]);
const MAX_BODY_BYTES = 8 * 1024; // 8KB
const RATE_LIMIT_PER_HOUR = 20;
const ALLOWED_ORIGIN = "https://hanxiao.io";

// ---- bounds for validation ----
const FILES_MAX = 200_000;
const SECONDS_MAX = 86_400;
const BYTES_MAX = 1e15; // ~1PB, generous upper bound to reject absurd values
const TABLE_ROWS_MAX = 64;
const TABLE_TEXT_MAX = 80;
const TABLE_CELLS = new Set(["p50", "p95", "p99", "max", "value", "op"]);
const CELL_ABS_MAX = 1e9;

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (!url.pathname.endsWith("/omni/profiling")) {
      return new Response("Not found", { status: 404 });
    }

    switch (request.method) {
      case "OPTIONS":
        return handleOptions();
      case "GET":
        return handleGet(env);
      case "POST":
        return handlePost(request, env);
      default:
        return new Response("Method not allowed", {
          status: 405,
          headers: { Allow: "GET, POST, OPTIONS" },
        });
    }
  },
};

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function handleOptions(): Response {
  return new Response(null, { status: 204, headers: corsHeaders() });
}

// ---------------------------------------------------------------------------
// POST: ingest
// ---------------------------------------------------------------------------

/** One row of the benchmark table: group, task, unit, cells (p50/p95/p99/max, value, op). */
interface BenchRow {
  g: string;
  t: string;
  u: string;
  c: Record<string, number>;
}

interface ProfilingReport {
  runId: string;
  appVersion?: string;
  datasetVersion: string;
  model?: string;
  table?: BenchRow[];
  hardware: {
    chip?: string | null;
    hwModel?: string | null;
    releaseYear?: number | null;
    macosVersion?: string | null;
    memoryBytes?: number | null;
    vramBytes?: number | null;
    cpuCores?: number | null;
    diskInternal?: boolean | null;
    diskFileSystem?: string | null;
  };
  metrics: {
    files?: number;
    scanned?: number;
    failed?: number;
    seconds?: number;
    filesPerSec?: number;
    tokens?: number;
    tokensPerSec?: number;
    errorRate?: number;
    peakVramDeltaBytes?: number;
  };
}

async function handlePost(request: Request, env: Env): Promise<Response> {
  // Reject oversized bodies cheaply when possible.
  const declaredLen = request.headers.get("content-length");
  if (declaredLen && Number(declaredLen) > MAX_BODY_BYTES) {
    return bad("payload too large");
  }

  const raw = await request.text();
  if (raw.length > MAX_BODY_BYTES) {
    return bad("payload too large");
  }

  let body: ProfilingReport;
  try {
    body = JSON.parse(raw) as ProfilingReport;
  } catch {
    return bad("invalid json");
  }

  const v = validate(body);
  if (!v.ok) return bad(v.error);

  const now = Date.now(); // worker runtime clock; client time ignored

  // ---- rate limit (per hashed IP, per hour window) ----
  const ip = request.headers.get("CF-Connecting-IP") ?? "";
  if (ip) {
    const ipHash = await sha256Hex(ip + (env.RATE_SALT ?? "omni-profiling-salt"));
    const hourWindow = Math.floor(now / 3_600_000);
    const allowed = await bumpRate(env, ipHash, hourWindow);
    if (!allowed) {
      return new Response("rate limited", {
        status: 429,
        headers: { "Retry-After": "3600" },
      });
    }
  }

  // ---- insert (dedup on runId) ----
  const h = body.hardware;
  const m = body.metrics;
  await env.DB.prepare(
    `INSERT OR IGNORE INTO profiling_runs (
       id, created_at, app_version, dataset_ver,
       chip, hw_model, release_year, macos_version,
       mem_bytes, vram_bytes, cpu_cores, disk_internal, disk_fs,
       files, scanned, failed, seconds, files_per_sec,
       tokens, tokens_per_sec, error_rate, peak_vram_delta,
       model, bench_table
     ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?)`
  )
    .bind(
      body.runId,
      now,
      str(body.appVersion),
      str(body.datasetVersion),
      str(h.chip),
      str(h.hwModel),
      intOrNull(h.releaseYear),
      str(h.macosVersion),
      intOrNull(h.memoryBytes),
      intOrNull(h.vramBytes),
      intOrNull(h.cpuCores),
      boolOrNull(h.diskInternal),
      str(h.diskFileSystem),
      intOrNull(m.files),
      intOrNull(m.scanned),
      intOrNull(m.failed),
      numOrNull(m.seconds),
      numOrNull(m.filesPerSec),
      intOrNull(m.tokens),
      numOrNull(m.tokensPerSec),
      numOrNull(m.errorRate),
      intOrNull(m.peakVramDeltaBytes),
      str(body.model),
      Array.isArray(body.table) ? JSON.stringify(body.table.map(cleanRow)) : null
    )
    .run();

  return new Response(null, { status: 204 });
}

function bad(msg: string): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status: 400,
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

type Validation = { ok: true } | { ok: false; error: string };

function validate(b: ProfilingReport): Validation {
  if (typeof b?.runId !== "string" || b.runId.length < 8 || b.runId.length > 64) {
    return { ok: false, error: "invalid runId" };
  }
  if (!ACCEPTED_DATASETS.has(b.datasetVersion)) {
    return { ok: false, error: "unsupported datasetVersion" };
  }
  if (typeof b.hardware !== "object" || b.hardware === null) {
    return { ok: false, error: "missing hardware" };
  }
  if (typeof b.metrics !== "object" || b.metrics === null) {
    return { ok: false, error: "missing metrics" };
  }

  const m = b.metrics;

  // required numeric metric fields: finite + within bounds
  if (!inRange(m.files, 0, FILES_MAX)) return { ok: false, error: "files out of range" };
  if (!inRange(m.scanned, 0, FILES_MAX)) return { ok: false, error: "scanned out of range" };
  if (!inRange(m.failed, 0, FILES_MAX)) return { ok: false, error: "failed out of range" };
  if (!inRange(m.seconds, 0, SECONDS_MAX)) return { ok: false, error: "seconds out of range" };
  if (!inRange(m.filesPerSec, 0, Infinity)) return { ok: false, error: "filesPerSec out of range" };
  if (!inRange(m.tokensPerSec, 0, Infinity)) return { ok: false, error: "tokensPerSec out of range" };
  if (!inRange(m.errorRate, 0, 1)) return { ok: false, error: "errorRate out of range" };
  if (!inRange(m.tokens, 0, Infinity)) return { ok: false, error: "tokens out of range" };
  if (!inRange(m.peakVramDeltaBytes, 0, BYTES_MAX))
    return { ok: false, error: "peakVramDeltaBytes out of range" };

  // optional byte fields: if present, must be finite and >= 0
  const h = b.hardware;
  if (!nullableInRange(h.memoryBytes, 0, BYTES_MAX)) return { ok: false, error: "memoryBytes invalid" };
  if (!nullableInRange(h.vramBytes, 0, BYTES_MAX)) return { ok: false, error: "vramBytes invalid" };
  if (!nullableInRange(h.cpuCores, 0, 4096)) return { ok: false, error: "cpuCores invalid" };
  if (!nullableInRange(h.releaseYear, 1990, 2100)) return { ok: false, error: "releaseYear invalid" };

  if (b.model !== undefined && b.model !== null && !shortText(b.model)) {
    return { ok: false, error: "model invalid" };
  }
  // The table is required for bench-v4 and refused for the older datasets, which never had one.
  if (b.datasetVersion === BENCH_DATASET) {
    const t = validateTable(b.table);
    if (!t.ok) return t;
  } else if (b.table !== undefined) {
    return { ok: false, error: "table not expected for this dataset" };
  }

  return { ok: true };
}

function validateTable(t: unknown): Validation {
  if (!Array.isArray(t) || t.length === 0 || t.length > TABLE_ROWS_MAX) {
    return { ok: false, error: "table invalid" };
  }
  const seen = new Set<string>();
  for (const r of t as BenchRow[]) {
    if (typeof r !== "object" || r === null) return { ok: false, error: "table row invalid" };
    if (!shortText(r.g) || !shortText(r.t) || !shortText(r.u)) {
      return { ok: false, error: "table row text invalid" };
    }
    const key = r.g + "/" + r.t;
    if (seen.has(key)) return { ok: false, error: "table row duplicated" };
    seen.add(key);
    if (typeof r.c !== "object" || r.c === null || Array.isArray(r.c)) {
      return { ok: false, error: "table cells invalid" };
    }
    const cells = Object.entries(r.c);
    if (cells.length === 0) return { ok: false, error: "table cells empty" };
    for (const [k, v] of cells) {
      if (!TABLE_CELLS.has(k) || !inRange(v, -CELL_ABS_MAX, CELL_ABS_MAX)) {
        return { ok: false, error: "table cell invalid" };
      }
    }
  }
  return { ok: true };
}

/** A non-empty string of at most TABLE_TEXT_MAX characters. */
function shortText(x: unknown): x is string {
  return typeof x === "string" && x.length > 0 && x.length <= TABLE_TEXT_MAX;
}

/** Only the four known fields, cells rounded to 2 decimals (what the app already sends). */
function cleanRow(r: BenchRow): BenchRow {
  const c: Record<string, number> = {};
  for (const [k, v] of Object.entries(r.c)) c[k] = Math.round(v * 100) / 100;
  return { g: r.g, t: r.t, u: r.u, c };
}

/** finite number within [min, max] inclusive. Rejects NaN/Infinity/non-number. */
function inRange(x: unknown, min: number, max: number): boolean {
  return typeof x === "number" && Number.isFinite(x) && x >= min && x <= max;
}

/** null/undefined allowed; otherwise finite number within [min, max]. */
function nullableInRange(x: unknown, min: number, max: number): boolean {
  if (x === null || x === undefined) return true;
  return inRange(x, min, max);
}

// ---------------------------------------------------------------------------
// Coercion helpers for binding
// ---------------------------------------------------------------------------

function str(x: unknown): string | null {
  return typeof x === "string" ? x : null;
}
function numOrNull(x: unknown): number | null {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}
function intOrNull(x: unknown): number | null {
  return typeof x === "number" && Number.isFinite(x) ? Math.trunc(x) : null;
}
function boolOrNull(x: unknown): number | null {
  if (x === null || x === undefined) return null;
  return x ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

/** Returns true if the request is allowed, false if over the limit. */
async function bumpRate(env: Env, ipHash: string, hourWindow: number): Promise<boolean> {
  // Atomic upsert that increments the counter and returns the new value.
  const row = await env.DB.prepare(
    `INSERT INTO rate (ip_hash, hour_window, count)
     VALUES (?, ?, 1)
     ON CONFLICT(ip_hash, hour_window)
     DO UPDATE SET count = count + 1
     RETURNING count`
  )
    .bind(ipHash, hourWindow)
    .first<{ count: number }>();

  const count = row?.count ?? 1;
  return count <= RATE_LIMIT_PER_HOUR;
}

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------------------
// GET: aggregate
// ---------------------------------------------------------------------------

interface RunRow {
  chip: string | null;
  app_version: string | null;
  release_year: number | null;
  macos_version: string | null;
  mem_bytes: number | null;
  vram_bytes: number | null;
  files_per_sec: number | null;
  tokens_per_sec: number | null;
  seconds: number | null;
  peak_vram_delta: number | null;
  created_at: number;
}

async function handleGet(env: Env): Promise<Response> {
  const now = Date.now();

  const { results } = await env.DB.prepare(
    `SELECT chip, app_version, release_year, macos_version, mem_bytes, vram_bytes,
            files_per_sec, tokens_per_sec, seconds, peak_vram_delta, created_at
       FROM profiling_runs
      WHERE dataset_ver IN (${HISTORY_DATASETS.map(() => "?").join(",")})
      ORDER BY created_at DESC`
  )
    .bind(...HISTORY_DATASETS)
    .all<RunRow>();

  const rows = results ?? [];

  // ---- byChip: group + median/std (+ per-version breakdown) ----
  const groups = new Map<string, RunRow[]>();
  for (const r of rows) {
    const key = r.chip ?? "Unknown";
    const arr = groups.get(key);
    if (arr) arr.push(r);
    else groups.set(key, [r]);
  }

  const byChip = [...groups.entries()]
    .map(([chip, list]) => {
      const rep = list[0];
      const fps = nums(list.map((r) => r.files_per_sec));
      const tps = nums(list.map((r) => r.tokens_per_sec));
      // Version breakdown WITHIN this chip - so version-over-version is hardware-controlled and
      // actually comparable (throughput is dominated by the Mac, not the app version).
      const vMap = new Map<string, RunRow[]>();
      for (const r of list) {
        const v = r.app_version ?? "?";
        (vMap.get(v) ?? vMap.set(v, []).get(v)!).push(r);
      }
      const versions = [...vMap.entries()]
        .map(([version, vl]) => ({
          version,
          runs: vl.length,
          medianTokensPerSec: roundInt(median(nums(vl.map((r) => r.tokens_per_sec)))),
          medianFilesPerSec: round1(median(nums(vl.map((r) => r.files_per_sec)))),
        }))
        .sort((a, b) => cmpVersion(b.version, a.version)); // newest first
      return {
        chip,
        releaseYear: firstNonNull(list.map((r) => r.release_year)),
        runs: list.length,
        medianFilesPerSec: round1(median(fps)),
        filesPerSecStd: round1(stddev(fps)),
        medianTokensPerSec: roundInt(median(tps)),
        tokensPerSecStd: roundInt(stddev(tps)),
        medianSeconds: round1(median(nums(list.map((r) => r.seconds)))),
        medianPeakVramDeltaBytes: roundInt(median(nums(list.map((r) => r.peak_vram_delta)))),
        memoryBytes: firstNonNull(list.map((r) => r.mem_bytes)) ?? rep.mem_bytes,
        vramBytes: firstNonNull(list.map((r) => r.vram_bytes)),
        macosVersions: uniq(list.map((r) => r.macos_version)),
        versions,
      };
    })
    .sort((a, b) => b.runs - a.runs);

  // ---- byVersion: overall throughput per app version (across all chips - directional only) ----
  const vGroups = new Map<string, RunRow[]>();
  for (const r of rows) {
    const v = r.app_version ?? "?";
    (vGroups.get(v) ?? vGroups.set(v, []).get(v)!).push(r);
  }
  const byVersion = [...vGroups.entries()]
    .map(([version, list]) => ({
      version,
      runs: list.length,
      medianTokensPerSec: roundInt(median(nums(list.map((r) => r.tokens_per_sec)))),
      medianFilesPerSec: round1(median(nums(list.map((r) => r.files_per_sec)))),
    }))
    .sort((a, b) => cmpVersion(b.version, a.version));

  // ---- recent: last 25 anonymized rows ----
  const recent = rows.slice(0, 25).map((r: RunRow) => ({
    chip: r.chip,
    appVersion: r.app_version,
    macosVersion: r.macos_version,
    memoryBytes: r.mem_bytes,
    filesPerSec: round1(r.files_per_sec),
    tokensPerSec: roundInt(r.tokens_per_sec),
    seconds: round1(r.seconds),
    peakVramDeltaBytes: r.peak_vram_delta,
    createdAt: r.created_at,
  }));

  const payload = {
    datasetVersion: DATASET_VERSION,
    updatedAt: now,
    totalRuns: rows.length,
    latestVersion: byVersion.length ? byVersion[0].version : null,
    byChip,
    byVersion,
    recent,
    bench: await benchAggregate(env),
  };

  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "max-age=120",
      ...corsHeaders(),
    },
  });
}

// ---------------------------------------------------------------------------
// GET: the bench-v4 table
// ---------------------------------------------------------------------------

interface BenchRunRow {
  chip: string | null;
  model: string | null;
  release_year: number | null;
  mem_bytes: number | null;
  vram_bytes: number | null;
  cpu_cores: number | null;
  app_version: string | null;
  macos_version: string | null;
  bench_table: string | null;
}

/**
 * One column per (chip, model): every cell is the median over that machine's runs. Rows keep the
 * order of the newest run's table, then any row only older runs had. Machines are ordered by text
 * indexing throughput, fastest first.
 */
async function benchAggregate(env: Env) {
  const { results } = await env.DB.prepare(
    `SELECT chip, model, release_year, mem_bytes, vram_bytes, cpu_cores, app_version,
            macos_version, bench_table
       FROM profiling_runs
      WHERE dataset_ver = ? AND bench_table IS NOT NULL
      ORDER BY created_at DESC`
  )
    .bind(BENCH_DATASET)
    .all<BenchRunRow>();
  const runs = results ?? [];

  const rows: { g: string; t: string; u: string; kind: "latency" | "value" }[] = [];
  const rowSeen = new Set<string>();
  type Machine = { meta: BenchRunRow[]; cells: Map<string, Map<string, number[]>> };
  const machines = new Map<string, Machine>();

  for (const r of runs) {
    let table: BenchRow[];
    try {
      table = JSON.parse(r.bench_table ?? "[]") as BenchRow[];
    } catch {
      continue;
    }
    const key = (r.chip ?? "Unknown") + "\u0000" + (r.model ?? "?");
    let m = machines.get(key);
    if (!m) machines.set(key, (m = { meta: [], cells: new Map() }));
    m.meta.push(r);
    for (const row of table) {
      const id = row.g + "/" + row.t;
      if (!rowSeen.has(id)) {
        rowSeen.add(id);
        rows.push({ g: row.g, t: row.t, u: row.u, kind: "value" in row.c ? "value" : "latency" });
      }
      let cell = m.cells.get(id);
      if (!cell) m.cells.set(id, (cell = new Map()));
      for (const [k, v] of Object.entries(row.c)) {
        const list = cell.get(k);
        if (list) list.push(v);
        else cell.set(k, [v]);
      }
    }
  }

  const out = [...machines.values()].map((m) => {
    const rep = m.meta[0];
    const cells: Record<string, Record<string, number>> = {};
    for (const [id, cell] of m.cells) {
      const c: Record<string, number> = {};
      for (const [k, list] of cell) {
        const v = median(list);
        if (v !== null) c[k] = Math.round(v * 100) / 100;
      }
      cells[id] = c;
    }
    return {
      chip: rep.chip ?? "Unknown",
      model: rep.model,
      releaseYear: firstNonNull(m.meta.map((r) => r.release_year)),
      memoryBytes: firstNonNull(m.meta.map((r) => r.mem_bytes)),
      vramBytes: firstNonNull(m.meta.map((r) => r.vram_bytes)),
      cpuCores: firstNonNull(m.meta.map((r) => r.cpu_cores)),
      runs: m.meta.length,
      appVersions: uniq(m.meta.map((r) => r.app_version)).sort((a, b) => cmpVersion(b, a)),
      macosVersions: uniq(m.meta.map((r) => r.macos_version)),
      cells,
    };
  });
  const speed = (x: (typeof out)[number]) => x.cells["Indexing/Text indexing"]?.value ?? 0;
  out.sort((a, b) => speed(b) - speed(a));

  const versions = uniq(runs.map((r) => r.app_version)).sort((a, b) => cmpVersion(b, a));
  return {
    datasetVersion: BENCH_DATASET,
    runs: runs.length,
    latestVersion: versions[0] ?? null,
    rows,
    machines: out,
  };
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/** Strip null/undefined/non-finite from a list of numbers. */
function nums(xs: (number | null)[]): number[] {
  return xs.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Sample standard deviation; null for <2 points (no spread to report). */
function stddev(xs: number[]): number | null {
  if (xs.length < 2) return null;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

/** Distinct, non-null values in first-seen order. */
function uniq<T>(xs: (T | null)[]): T[] {
  const out: T[] = [];
  const seen = new Set<T>();
  for (const x of xs) if (x !== null && x !== undefined && !seen.has(x)) { seen.add(x); out.push(x); }
  return out;
}

/** Compare dotted versions numerically ("0.1.29" > "0.1.9"); non-numeric sort last. */
function cmpVersion(a: string, b: string): number {
  const pa = a.split(".").map((n) => parseInt(n, 10));
  const pb = b.split(".").map((n) => parseInt(n, 10));
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i], y = pb[i];
    if (Number.isNaN(x) && Number.isNaN(y)) return 0;
    if (Number.isNaN(x)) return -1;
    if (Number.isNaN(y)) return 1;
    if (x !== y) return x - y;
  }
  return 0;
}

function firstNonNull<T>(xs: (T | null)[]): T | null {
  for (const x of xs) if (x !== null && x !== undefined) return x;
  return null;
}

function round1(x: number | null): number | null {
  return x === null ? null : Math.round(x * 10) / 10;
}
function roundInt(x: number | null): number | null {
  return x === null ? null : Math.round(x);
}
