mod commands;

pub use commands::run;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::models::Algorithm;

#[derive(Debug, Parser)]
#[command(
    name = "soma",
    version,
    about = "An easy-to-use ML CLI powered by smartcore",
    long_about = "soma is a command-line tool for training, evaluating, and using \
                  machine learning models. It wraps the smartcore library and provides \
                  a simple interface for working with CSV datasets.",
    after_help = "Examples:\n  \
                  soma train --data iris.csv --target species --algorithm knn -o model.soma\n  \
                  soma predict --model model.soma --data new_data.csv\n  \
                  soma evaluate --data iris.csv --target species --algorithm rfc --test-size 0.2\n  \
                  soma inspect --model model.soma\n  \
                  soma algorithms"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Train a model on a CSV dataset and save it to disk.
    #[command(visible_alias = "t")]
    Train(TrainArgs),

    /// Make predictions using a previously saved model.
    #[command(visible_alias = "p")]
    Predict(PredictArgs),

    /// Evaluate a model's performance using train/test split.
    #[command(visible_alias = "e")]
    Evaluate(EvaluateArgs),

    /// Inspect a saved model file and display its metadata.
    #[command(visible_alias = "i")]
    Inspect(InspectArgs),

    /// List all supported machine learning algorithms.
    #[command(visible_alias = "ls")]
    Algorithms,
}

#[derive(Debug, Parser)]
pub struct TrainArgs {
    /// Path to the CSV file containing training data.
    #[arg(short, long)]
    pub data: PathBuf,

    /// Name or index of the target (label) column in the CSV.
    #[arg(short, long)]
    pub target: String,

    /// Algorithm to use (see `soma algorithms` for the full list).
    #[arg(short, long, value_enum)]
    pub algorithm: Algorithm,

    /// Output model file path.
    #[arg(short, long, default_value = "model.soma")]
    pub output: PathBuf,

    /// Fraction of data to hold out for evaluation (0.0–1.0). Set to 0 to skip.
    #[arg(long, default_value = "0.2")]
    pub test_size: f64,
}

#[derive(Debug, Parser)]
pub struct PredictArgs {
    /// Path to a saved .soma model file.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Path to CSV data for prediction (same columns as training data).
    #[arg(short, long)]
    pub data: PathBuf,

    /// Column to exclude from input, if present.
    #[arg(short, long)]
    pub target: Option<String>,

    /// Write predictions to a file instead of stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Parser)]
pub struct EvaluateArgs {
    /// Path to the CSV dataset.
    #[arg(short, long)]
    pub data: PathBuf,

    /// Name or index of the target column.
    #[arg(short, long)]
    pub target: String,

    /// Algorithm to evaluate.
    #[arg(short, long, value_enum)]
    pub algorithm: Algorithm,

    /// Fraction of data to hold out for testing (0.0–1.0).
    #[arg(long, default_value = "0.2")]
    pub test_size: f64,
}

#[derive(Debug, Parser)]
pub struct InspectArgs {
    /// Path to the .soma model file.
    #[arg(short, long)]
    pub model: PathBuf,
}
