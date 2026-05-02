//! rustorch CLI — entry point.
//!
//! `rustorch run / sweep / check / deploy / fork [...]`
//! Parsing lives in [`crate::cli`]; each sub-command's body in
//! [`crate::commands`].

mod cli;
mod commands;
mod config;
mod logging;

use clap::Parser;

fn main() {
    let exit_code = match cli::Cli::parse().command {
        cli::Command::Run(args) => commands::run::execute(&args),
        cli::Command::Sweep(args) => commands::sweep::execute(&args),
        cli::Command::Check(args) => commands::check::execute(&args),
        cli::Command::Deploy(args) => commands::deploy::execute(&args),
        cli::Command::Fork(args) => commands::fork::execute(&args),
    };
    std::process::exit(exit_code);
}
