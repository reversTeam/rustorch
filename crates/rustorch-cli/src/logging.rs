//! Output helpers — pretty (coloured tty) vs JSON (one event per
//! line). Used by every sub-command via [`emit`] so the
//! `--log-format` flag is honoured uniformly.

use crate::cli::LogFormat;
use serde::Serialize;

/// Emit `event` on stdout in the requested format.
///
/// - `Pretty`: human-readable single line `[level] op key=value …`.
/// - `Json`:   `serde_json::to_string` of the event, one per line.
///
/// The event type implements [`Serialize`] for the JSON path and
/// [`std::fmt::Display`] for the pretty path. Callers wrap their
/// payload in an [`Event`] struct that implements both.
pub fn emit<E: Event>(format: LogFormat, event: &E) {
    match format {
        LogFormat::Pretty => println!("{}", event.pretty()),
        LogFormat::Json => match serde_json::to_string(event) {
            Ok(s) => println!("{}", s),
            Err(e) => eprintln!("[rustorch-cli] log_format=json: {e}"),
        },
    }
}

/// Trait every emit-able event implements. See [`Started`],
/// [`Done`] for canonical events.
pub trait Event: Serialize {
    /// Pretty single-line representation for tty.
    fn pretty(&self) -> String;
}

/// `started` event.
#[derive(Serialize)]
pub struct Started<'a> {
    /// Constant `"started"` for filtering.
    pub event: &'static str,
    /// Sub-command name (`"run"`, `"sweep"`, …).
    pub op: &'a str,
    /// Free-form context (script path, run id, …).
    pub context: &'a str,
}

impl Event for Started<'_> {
    fn pretty(&self) -> String {
        format!("[rustorch] {} started → {}", self.op, self.context)
    }
}

/// `done` event.
#[derive(Serialize)]
pub struct Done<'a> {
    /// Constant `"done"`.
    pub event: &'static str,
    /// Sub-command name.
    pub op: &'a str,
    /// Process exit status from the child.
    pub status: i32,
}

impl Event for Done<'_> {
    fn pretty(&self) -> String {
        format!("[rustorch] {} done (exit {})", self.op, self.status)
    }
}

/// `error` event.
#[derive(Serialize)]
pub struct ErrorEvent<'a> {
    /// Constant `"error"`.
    pub event: &'static str,
    /// Sub-command name.
    pub op: &'a str,
    /// Diagnostic message.
    pub message: &'a str,
}

impl Event for ErrorEvent<'_> {
    fn pretty(&self) -> String {
        format!("[rustorch] {} ERROR: {}", self.op, self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_started() {
        let s = Started {
            event: "started",
            op: "run",
            context: "src/train.rs",
        };
        let out = s.pretty();
        assert!(out.contains("run"));
        assert!(out.contains("src/train.rs"));
    }

    #[test]
    fn json_started_serialises_event_field() {
        let s = Started {
            event: "started",
            op: "run",
            context: "src/train.rs",
        };
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains("\"event\":\"started\""));
        assert!(j.contains("\"op\":\"run\""));
    }

    #[test]
    fn done_pretty_includes_exit_code() {
        let d = Done {
            event: "done",
            op: "check",
            status: 0,
        };
        assert!(d.pretty().contains("exit 0"));
    }
}
