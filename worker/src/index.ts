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

const DATASET_VERSION = "profiling-v1";
const MAX_BODY_BYTES = 8 * 1024; // 8KB
const RATE_LIMIT_PER_HOUR = 20;
const ALLOWED_ORIGIN = "https://hanxiao.io";

// ---- bounds for validation ----
const FILES_MAX = 200_000;
const SECONDS_MAX = 86_400;
const BYTES_MAX = 1e15; // ~1PB, generous upper bound to reject absurd values

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

interface ProfilingReport {
  runId: string;
  appVersion?: string;
  datasetVersion: string;
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
       tokens, tokens_per_sec, error_rate, peak_vram_delta
     ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)`
  )
    .bind(
      body.runId,
      now,
      str(body.appVersion),
      DATASET_VERSION,
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
      intOrNull(m.peakVramDeltaBytes)
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
  if (b.datasetVersion !== DATASET_VERSION) {
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

  return { ok: true };
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
    `SELECT chip, release_year, macos_version, mem_bytes, vram_bytes,
            files_per_sec, tokens_per_sec, seconds, peak_vram_delta, created_at
       FROM profiling_runs
      ORDER BY created_at DESC`
  ).all<RunRow>();

  const rows = results ?? [];

  // ---- byChip: group + medians ----
  const groups = new Map<string, RunRow[]>();
  for (const r of rows) {
    const key = r.chip ?? "Unknown";
    const arr = groups.get(key);
    if (arr) arr.push(r);
    else groups.set(key, [r]);
  }

  const byChip = [...groups.entries()]
    .map(([chip, list]) => {
      // Representative hardware values: most common non-null (fall back to first).
      const rep = list[0];
      return {
        chip,
        releaseYear: firstNonNull(list.map((r) => r.release_year)),
        runs: list.length,
        medianFilesPerSec: round1(median(nums(list.map((r) => r.files_per_sec)))),
        medianTokensPerSec: roundInt(median(nums(list.map((r) => r.tokens_per_sec)))),
        medianSeconds: round1(median(nums(list.map((r) => r.seconds)))),
        medianPeakVramDeltaBytes: roundInt(median(nums(list.map((r) => r.peak_vram_delta)))),
        memoryBytes: firstNonNull(list.map((r) => r.mem_bytes)) ?? rep.mem_bytes,
        vramBytes: firstNonNull(list.map((r) => r.vram_bytes)),
      };
    })
    .sort((a, b) => b.runs - a.runs);

  // ---- recent: last 25 anonymized rows ----
  const recent = rows.slice(0, 25).map((r: RunRow) => ({
    chip: r.chip,
    macosVersion: r.macos_version,
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
    byChip,
    recent,
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
