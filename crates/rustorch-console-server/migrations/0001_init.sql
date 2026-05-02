-- Initial Console schema. Tables match doc v0.7.2 /Reference/Console API
-- response shapes so the JSON layer is a thin marshalling pass.
--
-- Indexes are tuned for the two hot read paths from the frontend:
--   * GET /runs?status=...              → idx_runs_status
--   * GET /runs/:id/curves              → idx_metrics_run_step

CREATE TABLE IF NOT EXISTS runs (
    -- ULID-as-string for time-sortable IDs without a sequence.
    id          TEXT PRIMARY KEY NOT NULL,
    -- One of: queued, running, paused, completed, failed, cancelled.
    -- Free-form on purpose — the state machine lives in Rust, not SQL.
    status      TEXT NOT NULL,
    -- Optional human label so the UI can show something nicer than the ULID.
    title       TEXT,
    -- TrainCfg JSON blob (hyperparams, model ref, dataset ref, …).
    cfg_json    TEXT NOT NULL,
    -- Sweep grouping; null for single-shot runs.
    sweep_id    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status   ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_sweep    ON runs(sweep_id) WHERE sweep_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at DESC);

-- Scalar metrics — everything graphable on the Run detail charts tab
-- goes here. (step, name) tuple is unique per run.
CREATE TABLE IF NOT EXISTS metrics (
    run_id      TEXT NOT NULL,
    step        INTEGER NOT NULL,
    name        TEXT NOT NULL,
    value       REAL NOT NULL,
    ts          TEXT NOT NULL,
    PRIMARY KEY (run_id, step, name),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_step  ON metrics(run_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name  ON metrics(run_id, name);

-- One row per .safetensors checkpoint persisted by the runner.
CREATE TABLE IF NOT EXISTS checkpoints (
    id          TEXT PRIMARY KEY NOT NULL,
    run_id      TEXT NOT NULL,
    path        TEXT NOT NULL,
    step        INTEGER NOT NULL,
    metrics_json TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ckpts_run ON checkpoints(run_id, step);

-- Activity feed — recent events shown on the Dashboard.
CREATE TABLE IF NOT EXISTS activity (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,        -- run_started, run_completed, checkpoint_saved, …
    run_id      TEXT,
    payload_json TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_created ON activity(created_at DESC);
