# omni-profiling worker

Cloudflare Worker (TypeScript, no framework) backed by D1. Powers `hanxiao.io/omni/profiling`:
the Omni macOS app POSTs anonymous profiling reports, the landing page GETs aggregated stats.

## Endpoints

- `POST /omni/profiling` - ingest one report. Body must be JSON, `<= 8KB`, `datasetVersion` one of
  `profiling-v1`, `profiling-v2` (indexing only) or `bench-v4`/`v5`/`v6` (app 0.15.9+, the table benchmark; every version is stored, the table
  is built from the current one only:
  adds `model` and `table`, up to 64 rows of `{g, t, u, c}` with cells `p50 p95 p99 max value op`).
  Returns `204` on success, `400` on bad input, `429` when over the per-IP rate limit (~20/hour).
  Dedups on `runId` (`INSERT OR IGNORE`). Server sets `created_at` from its own clock; client time is ignored.
- `GET /omni/profiling` - aggregated JSON: `totalRuns`, per-chip medians (`byChip`), last 25 rows
  (`recent`), all over profiling-v1/v2 only (the bench datasets index text only, so its rates are not
  comparable); and `bench`: the table, one column per chip and model, each cell the median of its runs.
  CORS allows `https://hanxiao.io`, `Cache-Control: max-age=120`.
- `OPTIONS` - CORS preflight.

## Privacy

No raw IP, hostname, or PII is ever stored. Rate limiting stores only a salted SHA-256 hash of
`CF-Connecting-IP`, keyed by hour window. Optionally set a salt secret:

```
wrangler secret put RATE_SALT
```

## Prerequisites

A Cloudflare account with Workers and D1 enabled, plus `wrangler` installed (`npm i -g wrangler`)
and authenticated (`wrangler login`).

## Deploy

1. Create the D1 database:

   ```
   wrangler d1 create omni-profiling
   ```

2. Copy the printed `database_id` into `wrangler.toml` (replace `REPLACE_WITH_DATABASE_ID_FROM_WRANGLER_D1_CREATE`).

3. Apply the schema to the remote database:

   ```
   wrangler d1 execute omni-profiling --remote --file schema.sql
   ```

4. Deploy the worker:

   ```
   wrangler deploy
   ```

5. Add the route mapping `hanxiao.io/omni/profiling*` to this worker. Either:
   - Cloudflare dashboard: Workers and Pages -> omni-profiling -> Settings -> Domains and Routes ->
     add route `hanxiao.io/omni/profiling*`; or
   - Uncomment the `[[routes]]` block in `wrangler.toml`, fill in your `zone_id`, and re-run `wrangler deploy`.

## Migrations

An existing database takes the files in `migrations/` in order (`schema.sql` is the full schema for
a new one):

```
wrangler d1 execute omni-profiling --remote --file migrations/0001_bench_v4.sql
```

## Local development

```
wrangler dev
# apply schema to the local D1 instance:
wrangler d1 execute omni-profiling --local --file schema.sql
```
