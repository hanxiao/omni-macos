-- bench-v4 (app 0.15.9): the table benchmark. Additive; profiling-v1/v2 rows keep NULL here.
ALTER TABLE profiling_runs ADD COLUMN model TEXT;
ALTER TABLE profiling_runs ADD COLUMN bench_table TEXT;
CREATE INDEX IF NOT EXISTS idx_profiling_dataset ON profiling_runs (dataset_ver, created_at);
