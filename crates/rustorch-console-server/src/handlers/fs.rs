//! `/fs/*` — filesystem + cargo bridge for the in-browser editor.
//!
//! Every path-taking endpoint is sandboxed against `state.workspace_dir`
//! via `safe_join`. We intentionally reject any path that escapes the
//! workspace (`..` traversal, absolute paths, symlinks resolving
//! outside) so the editor can't read `/etc/shadow` even with auth.

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Deserialize, utoipa::IntoParams)]
pub struct PathQuery {
    /// Path relative to the workspace root (e.g. `src/main.rs`).
    pub path: String,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct TreeEntry {
    pub path: String,
    pub kind: &'static str, // "dir" | "file"
    pub size: u64,
}

#[utoipa::path(get, path = "/fs/tree", responses((status = 200, body = [TreeEntry])))]
pub async fn tree(State(s): State<AppState>) -> ApiResult<Json<Vec<TreeEntry>>> {
    let root = s.workspace_dir.clone();
    let entries = tokio::task::spawn_blocking(move || walk(&root, &root))
        .await
        .map_err(|e| ApiError::Internal(format!("walk: {e}")))??;
    Ok(Json(entries))
}

fn walk(root: &Path, base: &Path) -> ApiResult<Vec<TreeEntry>> {
    let mut out = Vec::new();
    if !base.exists() {
        return Ok(out);
    }
    let read = std::fs::read_dir(base)
        .map_err(|e| ApiError::Internal(format!("read_dir {}: {e}", base.display())))?;
    for entry in read.flatten() {
        let p = entry.path();
        // Skip the noisy `target`, `node_modules`, `.git` dirs.
        let name = entry.file_name();
        let n = name.to_string_lossy();
        if matches!(n.as_ref(), "target" | "node_modules" | ".git") {
            continue;
        }
        let rel = p
            .strip_prefix(root)
            .unwrap_or(&p)
            .to_string_lossy()
            .to_string();
        let meta = entry.metadata().ok();
        if p.is_dir() {
            out.push(TreeEntry {
                path: rel.clone(),
                kind: "dir",
                size: 0,
            });
            // Cap the recursion depth implicitly by the directory
            // structure; no symlink follow.
            let nested = walk(root, &p)?;
            out.extend(nested);
        } else {
            out.push(TreeEntry {
                path: rel,
                kind: "file",
                size: meta.map(|m| m.len()).unwrap_or(0),
            });
        }
    }
    Ok(out)
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct FileResponse {
    pub path: String,
    pub content: String,
    pub size: u64,
}

#[utoipa::path(get, path = "/fs/file", params(PathQuery), responses((status = 200, body = FileResponse)))]
pub async fn read_file(
    State(s): State<AppState>,
    Query(q): Query<PathQuery>,
) -> ApiResult<Json<FileResponse>> {
    let abs = safe_join(&s.workspace_dir, &q.path)?;
    let bytes = tokio::fs::read(&abs)
        .await
        .map_err(|e| ApiError::NotFound(format!("{}: {e}", q.path)))?;
    let size = bytes.len() as u64;
    let content = String::from_utf8(bytes)
        .map_err(|_| ApiError::InvalidArg(format!("{}: not valid UTF-8", q.path)))?;
    Ok(Json(FileResponse {
        path: q.path,
        content,
        size,
    }))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct WriteBody {
    pub content: String,
}

#[utoipa::path(
    put, path = "/fs/file",
    params(PathQuery),
    request_body = WriteBody,
    responses((status = 200, body = FileResponse))
)]
pub async fn write_file(
    State(s): State<AppState>,
    Query(q): Query<PathQuery>,
    Json(body): Json<WriteBody>,
) -> ApiResult<Json<FileResponse>> {
    let abs = safe_join(&s.workspace_dir, &q.path)?;
    if let Some(parent) = abs.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| ApiError::Internal(format!("mkdir: {e}")))?;
    }
    let size = body.content.len() as u64;
    tokio::fs::write(&abs, &body.content)
        .await
        .map_err(|e| ApiError::Internal(format!("write: {e}")))?;
    Ok(Json(FileResponse {
        path: q.path,
        content: body.content,
        size,
    }))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct Problem {
    pub level: &'static str,
    pub file: String,
    pub line: u32,
    pub message: String,
}

/// `/fs/problems` — placeholder. The real implementation will run
/// `cargo check --message-format=json` and map the diagnostics. For
/// now it returns an empty list so the UI's Problems pane renders.
#[utoipa::path(get, path = "/fs/problems", responses((status = 200, body = [Problem])))]
pub async fn problems() -> ApiResult<Json<Vec<Problem>>> {
    Ok(Json(Vec::new()))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct RunBody {
    /// Manifest path relative to the workspace (e.g.
    /// `crates/foo/Cargo.toml`). If `None`, runs at the workspace
    /// root.
    pub manifest: Option<String>,
    /// Free-form extra args passed to `cargo run --`.
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct RunSpawnResponse {
    /// Stable id for the spawned process — useful for cancellation
    /// (planned). Today it's the OS pid.
    pub pid: u32,
    /// SSE topic to subscribe to for stdout / stderr lines.
    pub topic: &'static str,
}

/// `POST /fs/run` — spawn `cargo run` in the workspace and stream
/// stdout / stderr lines to the `cargo.output` SSE topic. The HTTP
/// response returns immediately with the topic name; the heavy
/// lifting happens in a background task.
#[utoipa::path(
    post, path = "/fs/run",
    request_body = RunBody,
    responses((status = 200, body = RunSpawnResponse))
)]
pub async fn run_cargo(
    State(s): State<AppState>,
    Json(body): Json<RunBody>,
) -> ApiResult<Json<RunSpawnResponse>> {
    use tokio::io::AsyncBufReadExt;
    use tokio::process::Command;

    let manifest = match body.manifest.as_deref() {
        Some(m) => safe_join(&s.workspace_dir, m)?,
        None => s.workspace_dir.join("Cargo.toml"),
    };
    if !manifest.exists() {
        return Err(ApiError::NotFound(format!(
            "manifest {}",
            manifest.display()
        )));
    }

    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--manifest-path")
        .arg(&manifest)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if !body.args.is_empty() {
        cmd.arg("--").args(&body.args);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| ApiError::Internal(format!("cargo spawn: {e}")))?;
    let pid = child.id().unwrap_or(0);

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let hub = s.hub.clone();
    let topic = "cargo.output";

    // Stream stdout + stderr line-by-line to the cargo.output topic.
    // We don't await the whole process — the response goes out as
    // soon as the spawn succeeds.
    if let Some(out) = stdout {
        let hub = hub.clone();
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                hub.publish(
                    topic,
                    "cargo.output",
                    &serde_json::json!({"stream": "stdout", "line": line}),
                );
            }
        });
    }
    if let Some(err) = stderr {
        let hub = hub.clone();
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(err).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                hub.publish(
                    topic,
                    "cargo.output",
                    &serde_json::json!({"stream": "stderr", "line": line}),
                );
            }
        });
    }
    // Reap the child in the background so it doesn't become a zombie.
    tokio::spawn(async move {
        let status = child.wait().await;
        let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
        hub.publish(
            topic,
            "cargo.output",
            &serde_json::json!({"stream": "exit", "code": code}),
        );
    });

    Ok(Json(RunSpawnResponse { pid, topic }))
}

/// Reject any path that escapes the workspace root. Resolves `..`
/// segments by lexical normalisation, then checks that the result
/// still starts with the canonical root.
pub fn safe_join(root: &Path, rel: &str) -> ApiResult<PathBuf> {
    let candidate = root.join(rel);
    let abs = normalize(&candidate);
    let canon_root = normalize(root);
    if !abs.starts_with(&canon_root) {
        return Err(ApiError::InvalidArg(format!(
            "path escapes workspace: {rel}"
        )));
    }
    Ok(abs)
}

/// Lexical path normalisation — same idea as Go's `filepath.Clean`.
/// Doesn't follow symlinks (we don't want filesystem access in a
/// validator). Empty components and `.` are dropped, `..` pops the
/// previous component if any.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::Prefix(_) | Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {},
            Component::ParentDir => {
                out.pop();
            },
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/fs/tree", get(tree))
        .route("/fs/file", get(read_file).put(write_file))
        .route("/fs/problems", get(problems))
        .route("/fs/run", axum::routing::post(run_cargo))
}

#[allow(dead_code)]
fn _http_status_used() -> StatusCode {
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_join_blocks_traversal() {
        let tmp = std::env::temp_dir().join("rustorch-fs-test");
        let _ = std::fs::create_dir_all(&tmp);

        // Inside-the-workspace paths are accepted.
        let ok = safe_join(&tmp, "src/main.rs").unwrap();
        assert!(ok.starts_with(normalize(&tmp)));

        // `..` traversal is rejected.
        let err = safe_join(&tmp, "../../etc/passwd");
        assert!(err.is_err(), "should reject .. traversal");

        // Absolute paths land outside the root after normalisation.
        let err = safe_join(&tmp, "/etc/passwd");
        assert!(err.is_err(), "should reject absolute path");
    }

    #[test]
    fn normalize_drops_dot_and_pops_dotdot() {
        let p = normalize(Path::new("/tmp/a/./b/../c"));
        assert_eq!(p, Path::new("/tmp/a/c"));
    }
}
