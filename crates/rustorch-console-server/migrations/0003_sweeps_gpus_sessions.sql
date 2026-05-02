-- Tables required by P2.3 (sweeps), the cluster registry (gpus),
-- and runner authentication (sessions). Created here so the v1
-- schema is complete; queries land with the respective plans.

CREATE TABLE IF NOT EXISTS sweeps (
    id          TEXT PRIMARY KEY NOT NULL,
    -- one of: grid, random, asha, bayes
    strategy    TEXT NOT NULL,
    -- pending, running, completed, cancelled
    status      TEXT NOT NULL,
    spec_json   TEXT NOT NULL,
    base_cfg_json TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sweeps_status  ON sweeps(status);
CREATE INDEX IF NOT EXISTS idx_sweeps_created ON sweeps(created_at DESC);

-- Persistent record of GPUs the cluster has seen — populated by the
-- gpus polling task, queried by `GET /cluster/gpus` for historical
-- bookkeeping (current snapshot lives in memory).
CREATE TABLE IF NOT EXISTS gpus (
    id          INTEGER PRIMARY KEY,
    -- e.g. "RTX 4090"
    name        TEXT NOT NULL,
    vram_total_mb INTEGER NOT NULL,
    -- 7.x = Volta, 8.x = Ampere, 9.x = Hopper. NULL for non-CUDA.
    compute_capability TEXT,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

-- Runner sessions opened over the gRPC channel (P2.8). Persisted
-- so we can show "runner X disconnected at HH:MM" in the activity
-- feed and survive a Console restart with stale session ids.
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY NOT NULL,
    runner_id   TEXT NOT NULL,
    run_id      TEXT,
    -- connected, disconnected
    state       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    metadata_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_runner ON sessions(runner_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_run    ON sessions(run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_state  ON sessions(state);
