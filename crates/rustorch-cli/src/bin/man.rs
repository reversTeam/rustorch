//! Standalone helper — emit man-pages for `rustorch` and every
//! sub-command into a target directory.
//!
//! Invoke: `cargo run -p rustorch-cli --bin rustorch-man -- target/man`.
//! The output directory is created if missing. One file per command:
//! `rustorch.1`, `rustorch-run.1`, `rustorch-sweep.1`, …
//!
//! Used by:
//! * developers who want `man rustorch` after `make install`,
//! * CI to pre-render man-pages for the release tarball.

use clap::CommandFactory;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

// Pull in the same `Cli` derive surface used by the actual binary so
// the man-pages stay in lock-step with the help text.
#[path = "../cli.rs"]
mod cli;

fn main() -> std::io::Result<()> {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "man".into())
        .into();
    fs::create_dir_all(&out_dir)?;

    let mut cmd = cli::Cli::command();
    cmd.build();
    render(&cmd, &out_dir, None)?;

    println!("[rustorch-man] wrote man-pages to {}", out_dir.display());
    Ok(())
}

/// Render `cmd` and recurse into its sub-commands. `prefix` is the
/// chain of parent names so we can name files like `rustorch-run.1`.
fn render(cmd: &clap::Command, out_dir: &PathBuf, prefix: Option<&str>) -> std::io::Result<()> {
    let name = match prefix {
        Some(p) => format!("{p}-{}", cmd.get_name()),
        None => cmd.get_name().to_string(),
    };
    let path = out_dir.join(format!("{name}.1"));
    let mut buf: Vec<u8> = Vec::new();
    clap_mangen::Man::new(cmd.clone()).render(&mut buf)?;
    let mut f = fs::File::create(&path)?;
    f.write_all(&buf)?;

    for sub in cmd.get_subcommands() {
        // Skip the auto-generated `help` sub-command — it has no
        // useful man-page of its own.
        if sub.get_name() == "help" {
            continue;
        }
        render(sub, out_dir, Some(&name))?;
    }
    Ok(())
}
