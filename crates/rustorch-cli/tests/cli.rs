//! End-to-end CLI smoke tests via `assert_cmd`. Exercises the
//! actual `rustorch` binary built by `cargo test`.

use assert_cmd::Command;
use predicates::prelude::*;

fn rustorch() -> Command {
    Command::cargo_bin("rustorch").expect("binary built")
}

#[test]
fn help_lists_every_subcommand() {
    rustorch()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("sweep"))
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("deploy"))
        .stdout(predicate::str::contains("fork"));
}

#[test]
fn version_flag() {
    rustorch().arg("--version").assert().success();
}

#[test]
fn run_help_describes_amp_choices() {
    rustorch()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--amp"))
        .stdout(predicate::str::contains("bf16"));
}

#[test]
fn run_rejects_missing_script() {
    rustorch()
        .args(["run", "/tmp/__nope__.rs"])
        .assert()
        .code(2)
        .stdout(predicate::str::contains("script not found"));
}

#[test]
fn run_rejects_zero_gpus() {
    rustorch()
        .args(["run", "--gpus", "0", "/tmp/whatever.rs"])
        .assert()
        .code(2);
}

#[test]
fn run_rejects_invalid_amp() {
    rustorch()
        .args(["run", "--amp", "fp99", "/tmp/whatever.rs"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("possible values").or(predicate::str::contains("invalid")),
        );
}

#[test]
fn run_rejects_negative_lr() {
    rustorch()
        .args(["run", "--lr", "-1.0", "/tmp/whatever.rs"])
        .assert()
        .code(2);
}

#[test]
fn check_rejects_missing_script() {
    rustorch()
        .args(["check", "/tmp/__nope__.rs"])
        .assert()
        .code(2);
}

#[test]
fn sweep_rejects_no_axes() {
    rustorch().args(["sweep", "/tmp/train.rs"]).assert().code(2);
}

#[test]
fn sweep_grid_lists_combos() {
    rustorch()
        .args([
            "sweep",
            "--lr",
            "1e-4,3e-4,1e-3",
            "--batch",
            "64,128",
            "/tmp/train.rs",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("6 combos"))
        .stdout(predicate::str::contains("sweep_id"));
}

#[test]
fn deploy_rejects_missing_checkpoint() {
    rustorch()
        .args([
            "deploy",
            "--checkpoint",
            "/tmp/__no_ckpt__.safetensors",
            "--to",
            "https://x",
        ])
        .assert()
        .code(2);
}

#[test]
fn deploy_rejects_invalid_url() {
    let f = tempfile::NamedTempFile::new().unwrap();
    rustorch()
        .args([
            "deploy",
            "--checkpoint",
            f.path().to_str().unwrap(),
            "--to",
            "ftp://no",
        ])
        .assert()
        .code(2);
}

#[test]
fn fork_rejects_empty_id() {
    rustorch().args(["fork", ""]).assert().code(2);
}

#[test]
fn fork_happy_path_emits_started_done() {
    rustorch()
        .args(["fork", "abc-123", "--lr", "1e-4"])
        .assert()
        .success()
        .stdout(predicate::str::contains("started"))
        .stdout(predicate::str::contains("done"));
}

#[test]
fn unknown_subcommand_fails() {
    rustorch().args(["fakecmd"]).assert().failure();
}

#[test]
fn run_log_format_json_emits_json() {
    rustorch()
        .args(["run", "--log-format", "json", "/tmp/__nope__.rs"])
        .assert()
        .code(2)
        .stdout(predicate::str::contains("\"event\":\"error\""));
}

#[test]
fn man_page_helper_renders_all_subcommands() {
    // The `rustorch-man` helper binary writes a `.1` file per
    // sub-command. We point it at a temp dir and assert every
    // expected file ended up there.
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().to_str().unwrap();
    Command::cargo_bin("rustorch-man")
        .expect("man helper built")
        .arg(target)
        .assert()
        .success();
    for name in [
        "rustorch.1",
        "rustorch-run.1",
        "rustorch-sweep.1",
        "rustorch-check.1",
        "rustorch-deploy.1",
        "rustorch-fork.1",
    ] {
        let p = dir.path().join(name);
        assert!(p.exists(), "missing man-page: {name}");
    }
}
