//! `cpe`: the command-line tools of the Cubeplanet engine (contract 4.5, 5.4).
//!
//! `caps`, `proctex-sheet` and `content-check` (E0); `run`, `shot` and `bench` (E2).
//! `cpe --help` lists them.
//! Results go to stdout, log records and errors to stderr. Exit codes:
//! 0 success, [`cp_app::EXIT_FAILURE`] when a command fails, [`cp_app::EXIT_USAGE`] for a
//! command line that cannot be run.

use std::io::Write;
use std::process::ExitCode;

use cp_app::{cli, logger};

fn main() -> ExitCode {
    logger::install_from_env();
    let command = match cli::parse(std::env::args_os().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            to_stderr(&format!("error: {error}\n{}", error.hint()));
            return ExitCode::from(cp_app::EXIT_USAGE);
        }
    };
    let result = cp_app::run(&command, &mut std::io::stdout().lock());
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            to_stderr(&format!("error: {failure}"));
            ExitCode::from(cp_app::EXIT_FAILURE)
        }
    }
}

/// Writes one message to stderr. When even stderr is gone there is nobody left to tell, so a
/// write error is ignored instead of panicking like `eprintln!`.
fn to_stderr(message: &str) {
    let _ = writeln!(std::io::stderr().lock(), "{message}");
}
