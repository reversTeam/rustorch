//! SQLite persistence layer. We use sqlx for compile-time checked
//! queries plus a thin wrapper that hides the connection-string ←→
//! filesystem-path translation.
//!
//! The schema lives in `migrations/0001_init.sql`. Migrations are
//! applied automatically on startup via `sqlx::migrate!` so the binary
//! is self-bootstrapping (no `sqlx-cli` required at deploy time).

use crate::error::{ApiError, ApiResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Default DB location follows the XDG state dir spec:
/// `~/.local/share/rustorch/console.db`. Tests override via `:memory:`.
pub fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home)
        .join(".local")
        .join("share")
        .join("rustorch")
        .join("console.db")
}

/// Open (or create) the SQLite database at `path`, ensuring the
/// parent directory exists, then run pending migrations. Returns a
/// pool ready to be cloned into handler state.
///
/// Special cases:
/// * `path == ":memory:"` → in-memory DB, no file created. Tests use
///   this to get isolation per spawn.
pub async fn connect(path: &str) -> ApiResult<SqlitePool> {
    let opts = if path == ":memory:" {
        SqliteConnectOptions::from_str("sqlite::memory:")
            .map_err(|e| ApiError::Internal(format!("sqlite memory opts: {e}")))?
    } else {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ApiError::Internal(format!("mkdir {}: {e}", parent.display())))?;
        }
        SqliteConnectOptions::new()
            .filename(p)
            .create_if_missing(true)
            .foreign_keys(true)
            .busy_timeout(std::time::Duration::from_secs(5))
    };

    let pool = SqlitePoolOptions::new()
        .max_connections(8)
        .connect_with(opts)
        .await?;
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .map_err(|e| ApiError::Internal(format!("migrate: {e}")))?;
    Ok(pool)
}

/// Lifecycle states aligned with doc v0.7.2 Run state machine. The
/// transitions live in the handlers; this enum is just the wire
/// representation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, utoipa::ToSchema,
)]
#[sqlx(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl FromStr for RunStatus {
    type Err = ApiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "queued" => Self::Queued,
            "running" => Self::Running,
            "paused" => Self::Paused,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            other => return Err(ApiError::InvalidArg(format!("unknown status: {other}"))),
        })
    }
}

/// Materialized run row — what `GET /runs` returns. The full TrainCfg
/// is kept JSON-encoded in `cfg_json` to avoid coupling the schema to
/// the rapidly evolving training config.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Run {
    pub id: String,
    pub status: RunStatus,
    pub title: Option<String>,
    pub cfg_json: String,
    pub sweep_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Insert payload — separated from `Run` so callers don't have to
/// invent timestamps themselves.
#[derive(Debug, Clone, Deserialize)]
pub struct NewRun {
    pub title: Option<String>,
    pub cfg_json: serde_json::Value,
    pub sweep_id: Option<String>,
}

pub async fn insert_run(pool: &SqlitePool, new: NewRun) -> ApiResult<Run> {
    let id = ulid::Ulid::new().to_string();
    let now = Utc::now();
    let cfg = serde_json::to_string(&new.cfg_json)?;
    let status = RunStatus::Queued;

    sqlx::query(
        "INSERT INTO runs (id, status, title, cfg_json, sweep_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(status.as_str())
    .bind(&new.title)
    .bind(&cfg)
    .bind(&new.sweep_id)
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(pool)
    .await?;

    Ok(Run {
        id,
        status,
        title: new.title,
        cfg_json: cfg,
        sweep_id: new.sweep_id,
        created_at: now,
        updated_at: now,
    })
}

/// Lightweight projection used by list endpoints. Avoids streaming
/// the full cfg_json blob over the wire when the UI just wants a
/// table row.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct RunSummary {
    pub id: String,
    pub status: RunStatus,
    pub title: Option<String>,
    pub sweep_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub async fn list_runs(
    pool: &SqlitePool,
    status: Option<RunStatus>,
    limit: i64,
    offset: i64,
) -> ApiResult<Vec<RunSummary>> {
    let rows = match status {
        Some(s) => {
            sqlx::query_as::<_, RunSummaryRow>(
                "SELECT id, status, title, sweep_id, created_at, updated_at
             FROM runs WHERE status = ?
             ORDER BY created_at DESC LIMIT ? OFFSET ?",
            )
            .bind(s.as_str())
            .bind(limit)
            .bind(offset)
            .fetch_all(pool)
            .await?
        },
        None => {
            sqlx::query_as::<_, RunSummaryRow>(
                "SELECT id, status, title, sweep_id, created_at, updated_at
             FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(pool)
            .await?
        },
    };
    rows.into_iter().map(RunSummary::try_from).collect()
}

pub async fn get_run(pool: &SqlitePool, id: &str) -> ApiResult<Run> {
    let row: Option<RunRow> = sqlx::query_as(
        "SELECT id, status, title, cfg_json, sweep_id, created_at, updated_at
         FROM runs WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    row.ok_or_else(|| ApiError::NotFound(format!("run {id}")))?
        .try_into()
}

pub async fn update_run_status(pool: &SqlitePool, id: &str, status: RunStatus) -> ApiResult<Run> {
    let now = Utc::now();
    let res = sqlx::query("UPDATE runs SET status = ?, updated_at = ? WHERE id = ?")
        .bind(status.as_str())
        .bind(now.to_rfc3339())
        .bind(id)
        .execute(pool)
        .await?;
    if res.rows_affected() == 0 {
        return Err(ApiError::NotFound(format!("run {id}")));
    }
    get_run(pool, id).await
}

pub async fn count_runs(pool: &SqlitePool, status: Option<RunStatus>) -> ApiResult<i64> {
    let n: (i64,) = match status {
        Some(s) => {
            sqlx::query_as("SELECT COUNT(*) FROM runs WHERE status = ?")
                .bind(s.as_str())
                .fetch_one(pool)
                .await?
        },
        None => {
            sqlx::query_as("SELECT COUNT(*) FROM runs")
                .fetch_one(pool)
                .await?
        },
    };
    Ok(n.0)
}

// ---- private row types -------------------------------------------------

#[derive(sqlx::FromRow)]
struct RunSummaryRow {
    id: String,
    status: String,
    title: Option<String>,
    sweep_id: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<RunSummaryRow> for RunSummary {
    type Error = ApiError;
    fn try_from(r: RunSummaryRow) -> Result<Self, Self::Error> {
        Ok(Self {
            id: r.id,
            status: RunStatus::from_str(&r.status)?,
            title: r.title,
            sweep_id: r.sweep_id,
            created_at: parse_ts(&r.created_at)?,
            updated_at: parse_ts(&r.updated_at)?,
        })
    }
}

#[derive(sqlx::FromRow)]
struct RunRow {
    id: String,
    status: String,
    title: Option<String>,
    cfg_json: String,
    sweep_id: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<RunRow> for Run {
    type Error = ApiError;
    fn try_from(r: RunRow) -> Result<Self, Self::Error> {
        Ok(Self {
            id: r.id,
            status: RunStatus::from_str(&r.status)?,
            title: r.title,
            cfg_json: r.cfg_json,
            sweep_id: r.sweep_id,
            created_at: parse_ts(&r.created_at)?,
            updated_at: parse_ts(&r.updated_at)?,
        })
    }
}

fn parse_ts(s: &str) -> ApiResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| ApiError::Internal(format!("ts parse: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn fresh() -> SqlitePool {
        connect(":memory:").await.unwrap()
    }

    #[tokio::test]
    async fn migration_creates_runs_table() {
        let p = fresh().await;
        let n = count_runs(&p, None).await.unwrap();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn insert_then_get_roundtrip() {
        let p = fresh().await;
        let new = NewRun {
            title: Some("hello".into()),
            cfg_json: serde_json::json!({"lr": 1e-3}),
            sweep_id: None,
        };
        let r = insert_run(&p, new).await.unwrap();
        assert_eq!(r.status, RunStatus::Queued);
        let got = get_run(&p, &r.id).await.unwrap();
        assert_eq!(got.id, r.id);
        assert_eq!(got.title.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn list_filters_by_status_and_orders_desc() {
        let p = fresh().await;
        for i in 0..3 {
            insert_run(
                &p,
                NewRun {
                    title: Some(format!("r{i}")),
                    cfg_json: serde_json::json!({}),
                    sweep_id: None,
                },
            )
            .await
            .unwrap();
        }
        let all = list_runs(&p, None, 50, 0).await.unwrap();
        assert_eq!(all.len(), 3);
        assert!(all[0].created_at >= all[1].created_at);

        let queued = list_runs(&p, Some(RunStatus::Queued), 50, 0).await.unwrap();
        assert_eq!(queued.len(), 3);
        let running = list_runs(&p, Some(RunStatus::Running), 50, 0)
            .await
            .unwrap();
        assert!(running.is_empty());
    }

    #[tokio::test]
    async fn update_status_404_on_missing() {
        let p = fresh().await;
        let err = update_run_status(&p, "missing", RunStatus::Running)
            .await
            .unwrap_err();
        matches!(err, ApiError::NotFound(_));
    }

    #[tokio::test]
    async fn update_status_advances_then_persists() {
        let p = fresh().await;
        let r = insert_run(
            &p,
            NewRun {
                title: None,
                cfg_json: serde_json::json!({}),
                sweep_id: None,
            },
        )
        .await
        .unwrap();
        let updated = update_run_status(&p, &r.id, RunStatus::Running)
            .await
            .unwrap();
        assert_eq!(updated.status, RunStatus::Running);
        assert!(updated.updated_at >= r.updated_at);
    }

    #[test]
    fn run_status_round_trips_via_str() {
        for s in [
            RunStatus::Queued,
            RunStatus::Running,
            RunStatus::Paused,
            RunStatus::Completed,
            RunStatus::Failed,
            RunStatus::Cancelled,
        ] {
            assert_eq!(RunStatus::from_str(s.as_str()).unwrap(), s);
        }
        assert!(RunStatus::from_str("nope").is_err());
    }
}
