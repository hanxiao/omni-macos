-- Omni profiling D1 schema.
-- One flattened row per anonymous profiling run. No raw IP / hostname / PII stored.

CREATE TABLE IF NOT EXISTS profiling_runs (
  id              TEXT PRIMARY KEY,      -- runId (uuid v4), used for dedup via INSERT OR IGNORE
  created_at      INTEGER NOT NULL,      -- server-set epoch ms (worker runtime clock; client time ignored)
  app_version     TEXT,
  dataset_ver     TEXT NOT NULL,         -- the dataset the run used (profiling-v1 or profiling-v2)
  -- hardware
  chip            TEXT,
  hw_model        TEXT,
  release_year    INTEGER,               -- nullable
  macos_version   TEXT,
  mem_bytes       INTEGER,
  vram_bytes      INTEGER,               -- nullable
  cpu_cores       INTEGER,
  disk_internal   INTEGER,               -- 0/1 or NULL (stored as bool)
  disk_fs         TEXT,                  -- nullable
  -- metrics
  files           INTEGER,
  scanned         INTEGER,
  failed          INTEGER,
  seconds         REAL,
  files_per_sec   REAL,
  tokens          INTEGER,
  tokens_per_sec  REAL,
  error_rate      REAL,
  peak_vram_delta INTEGER
);

CREATE INDEX IF NOT EXISTS idx_profiling_chip       ON profiling_runs (chip);
CREATE INDEX IF NOT EXISTS idx_profiling_created_at ON profiling_runs (created_at);

-- Per-IP rate limiting. We store only a salted hash of the IP, keyed by hour window.
CREATE TABLE IF NOT EXISTS rate (
  ip_hash     TEXT NOT NULL,             -- SHA-256 hex of (CF-Connecting-IP + salt). Raw IP never stored.
  hour_window INTEGER NOT NULL,          -- floor(epoch_ms / 3600000)
  count       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (ip_hash, hour_window)
);
