-- Builder graph + small key/value store. One row per logical
-- document keyed by a short string (e.g. "current" for the working
-- builder graph). JSON blob holds the payload; the server doesn't
-- inspect it.

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY NOT NULL,
    json  TEXT NOT NULL
);
