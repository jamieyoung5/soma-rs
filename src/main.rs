use std::env;
use std::process::ExitCode;

use clap::Parser;

use soma::cli::{run, Cli};
use soma::tui::run_tui;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // Launch TUI if no args given, or if explicitly requested.
    let should_launch_tui = args.len() == 1
        || args.iter().any(|a| a == "--tui")
        || (args.len() == 2 && args[1] == "tui");

    if should_launch_tui {
        match run_tui() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        let cli = Cli::parse();
        match run(cli) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    }
}
