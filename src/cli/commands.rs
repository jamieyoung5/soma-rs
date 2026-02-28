use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use comfy_table::{presets, Table};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::cli::{Cli, Command, EvaluateArgs, InspectArgs, PredictArgs, TrainArgs};
use crate::error::{Result, SomaError};
use crate::eval::EvalReport;
use crate::io::{load_csv, split_train_test};
use crate::models::store::ModelMetadata;
use crate::models::{Algorithm, ModelStore};

/// Dispatch the parsed CLI to the right subcommand.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Train(args) => cmd_train(args),
        Command::Predict(args) => cmd_predict(args),
        Command::Evaluate(args) => cmd_evaluate(args),
        Command::Inspect(args) => cmd_inspect(args),
        Command::Algorithms => cmd_algorithms(),
    }
}

fn cmd_train(args: TrainArgs) -> Result<()> {
    println!("Loading data from '{}'...", args.data.display());
    let dataset = load_csv(&args.data, &args.target)?;
    println!("  {dataset}");

    let (train_x, train_y, eval_split) = if args.test_size > 0.0 && args.test_size < 1.0 {
        let split = split_train_test(&dataset, args.test_size)?;
        println!(
            "  Split: {} train / {} test (test_size={:.0}%)",
            split.n_train,
            split.n_test,
            args.test_size * 100.0
        );
        let eval = Some((split.x_test, split.y_test));
        (split.x_train, split.y_train, eval)
    } else {
        println!(
            "  Using all {} samples for training (no test split)",
            dataset.n_samples
        );
        (dataset.x.clone(), dataset.y.clone(), None)
    };

    println!(
        "\nTraining {} on {} samples with {} features...",
        args.algorithm.description(),
        train_y.len(),
        dataset.n_features
    );

    let start = Instant::now();
    let model = args.algorithm.train(&train_x, &train_y)?;
    let elapsed = start.elapsed();
    println!("  Training completed in {elapsed:.2?}");

    if let Some((ref x_test, ref y_test)) = eval_split {
        let preds = model.predict(x_test)?;
        let report = EvalReport::compute(args.algorithm.task(), y_test, &preds);
        println!("\nTest set evaluation:");
        println!("{report}");
    }

    let metadata = ModelMetadata {
        algorithm: args.algorithm,
        feature_names: dataset.feature_names.clone(),
        target_name: dataset.target_name.clone(),
        n_train_samples: train_y.len(),
        n_features: dataset.n_features,
    };

    let store = ModelStore::new(model, metadata);
    store.save(&args.output)?;
    println!("Model saved to '{}'", args.output.display());

    Ok(())
}

fn cmd_predict(args: PredictArgs) -> Result<()> {
    println!("Loading model from '{}'...", args.model.display());
    let store = ModelStore::load(&args.model)?;
    println!("  Algorithm: {}", store.metadata.algorithm.description());
    println!("  Features:  [{}]", store.metadata.feature_names.join(", "));
    println!("  Target:    {}", store.metadata.target_name);

    let target_col = args
        .target
        .as_deref()
        .unwrap_or(&store.metadata.target_name);

    // Try loading with the target column excluded. If that column doesn't
    // exist in the prediction file, just treat every column as a feature.
    let (x, n_features) = load_prediction_data(&args.data, target_col, &store)?;

    store.validate_features(n_features)?;

    let (n_rows, _n_cols) = x.shape();
    println!("\nPredicting on {n_rows} samples...");

    let start = Instant::now();
    let predictions = store.model.predict(&x)?;
    let elapsed = start.elapsed();

    println!("  Prediction completed in {elapsed:.2?}");

    match args.output {
        Some(ref path) => {
            write_predictions_to_file(path, &predictions)?;
            println!("Predictions written to '{}'", path.display());
        }
        None => {
            println!("\nPredictions:");
            let stdout = io::stdout();
            let mut out = stdout.lock();
            for pred in &predictions {
                writeln!(out, "{pred}").map_err(SomaError::from)?;
            }
        }
    }

    Ok(())
}

fn load_prediction_data(
    path: &Path,
    target_col: &str,
    store: &ModelStore,
) -> Result<(DenseMatrix<f64>, usize)> {
    match load_csv(path, target_col) {
        Ok(ds) => Ok((ds.x, ds.n_features)),
        Err(SomaError::ColumnNotFound(_)) => {
            load_all_columns_as_features(path, store.metadata.n_features)
        }
        Err(e) => Err(e),
    }
}

/// Read every column as a feature (no target column to exclude).
fn load_all_columns_as_features(
    path: &Path,
    expected_features: usize,
) -> Result<(DenseMatrix<f64>, usize)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();
    let n_features = headers.len();

    if n_features == 0 {
        return Err(SomaError::data("CSV file has no columns"));
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();

    for (line_no, result) in reader.records().enumerate() {
        let record = result?;
        let mut row = Vec::with_capacity(n_features);
        for (col_idx, col_name) in headers.iter().enumerate() {
            let raw = record.get(col_idx).ok_or_else(|| {
                SomaError::data(format!("row {line_no}: missing column index {col_idx}"))
            })?;
            let val: f64 = raw.trim().parse().map_err(|_| {
                SomaError::data(format!(
                    "row {line_no}, column '{col_name}': cannot parse '{raw}' as a number"
                ))
            })?;
            row.push(val);
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Err(SomaError::data("CSV file contains no data rows"));
    }

    if n_features != expected_features {
        return Err(SomaError::data(format!(
            "model expects {expected_features} features but prediction CSV has {n_features} columns"
        )));
    }

    let n_rows = rows.len();

    // row-major -> column-major for DenseMatrix
    let mut col_major = vec![0.0; n_rows * n_features];
    for (r, row) in rows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            col_major[c * n_rows + r] = val;
        }
    }

    let x = DenseMatrix::new(n_rows, n_features, col_major, true)
        .map_err(|e| SomaError::data(format!("failed to build DenseMatrix: {e}")))?;

    Ok((x, n_features))
}

fn write_predictions_to_file(path: &Path, predictions: &[f64]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = fs::File::create(path)?;
    for pred in predictions {
        writeln!(file, "{pred}").map_err(SomaError::from)?;
    }
    Ok(())
}

fn cmd_evaluate(args: EvaluateArgs) -> Result<()> {
    if args.test_size <= 0.0 || args.test_size >= 1.0 {
        return Err(SomaError::config(format!(
            "test_size must be between 0 and 1 exclusive, got {}",
            args.test_size,
        )));
    }

    println!("Loading data from '{}'...", args.data.display());
    let dataset = load_csv(&args.data, &args.target)?;
    println!("  {dataset}");

    let split = split_train_test(&dataset, args.test_size)?;
    println!(
        "  Split: {} train / {} test (test_size={:.0}%)",
        split.n_train,
        split.n_test,
        args.test_size * 100.0
    );

    println!("\nTraining {}...", args.algorithm.description(),);

    let start = Instant::now();
    let model = args.algorithm.train(&split.x_train, &split.y_train)?;
    let train_elapsed = start.elapsed();
    println!("  Training completed in {train_elapsed:.2?}");

    let start = Instant::now();
    let train_preds = model.predict(&split.x_train)?;
    let test_preds = model.predict(&split.x_test)?;
    let pred_elapsed = start.elapsed();
    println!("  Prediction completed in {pred_elapsed:.2?}");

    let task = args.algorithm.task();

    let train_report = EvalReport::compute(task, &split.y_train, &train_preds);
    let test_report = EvalReport::compute(task, &split.y_test, &test_preds);

    println!("\nTraining set evaluation:");
    println!("{train_report}");

    println!("Test set evaluation:");
    println!("{test_report}");

    Ok(())
}

fn cmd_inspect(args: InspectArgs) -> Result<()> {
    let store = ModelStore::load(&args.model)?;
    let meta = &store.metadata;

    let mut table = Table::new();
    table.load_preset(presets::UTF8_FULL);
    table.set_header(vec!["Property", "Value"]);

    table.add_row(vec![
        "Algorithm".to_string(),
        meta.algorithm.description().to_string(),
    ]);
    table.add_row(vec!["Task".to_string(), meta.algorithm.task().to_string()]);
    table.add_row(vec!["Target column".to_string(), meta.target_name.clone()]);
    table.add_row(vec![
        "Features".to_string(),
        format!("{} columns", meta.n_features),
    ]);
    table.add_row(vec![
        "Feature names".to_string(),
        meta.feature_names.join(", "),
    ]);
    table.add_row(vec![
        "Training samples".to_string(),
        meta.n_train_samples.to_string(),
    ]);
    table.add_row(vec![
        "Model file".to_string(),
        args.model.display().to_string(),
    ]);

    let file_size = fs::metadata(&args.model)
        .map(|m| format_bytes(m.len()))
        .unwrap_or_else(|_| "unknown".to_string());
    table.add_row(vec!["File size".to_string(), file_size]);

    println!("{table}");
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn cmd_algorithms() -> Result<()> {
    let mut table = Table::new();
    table.load_preset(presets::UTF8_FULL);
    table.set_header(vec!["Name", "Alias", "Task", "Description"]);

    for algo in Algorithm::all() {
        let (name, alias) = algorithm_name_and_alias(*algo);
        table.add_row(vec![
            name,
            alias,
            algo.task().to_string(),
            algo.description().to_string(),
        ]);
    }

    println!("Supported algorithms:\n");
    println!("{table}");
    println!("\nUse the Name or Alias with the --algorithm flag.");
    println!("Example: soma train --data data.csv --target label --algorithm knn");

    Ok(())
}

fn algorithm_name_and_alias(algo: Algorithm) -> (String, String) {
    match algo {
        Algorithm::LogisticRegression => ("logistic-regression".into(), "lr".into()),
        Algorithm::KnnClassifier => ("knn-classifier".into(), "knn".into()),
        Algorithm::DecisionTreeClassifier => ("decision-tree-classifier".into(), "dtc".into()),
        Algorithm::RandomForestClassifier => ("random-forest-classifier".into(), "rfc".into()),
        Algorithm::GaussianNb => ("gaussian-nb".into(), "gnb".into()),
        Algorithm::LinearRegression => ("linear-regression".into(), "linreg".into()),
        Algorithm::RidgeRegression => ("ridge-regression".into(), "ridge".into()),
        Algorithm::LassoRegression => ("lasso-regression".into(), "lasso".into()),
        Algorithm::ElasticNet => ("elastic-net".into(), "enet".into()),
        Algorithm::KnnRegressor => ("knn-regressor".into(), "knnr".into()),
        Algorithm::DecisionTreeRegressor => ("decision-tree-regressor".into(), "dtr".into()),
        Algorithm::RandomForestRegressor => ("random-forest-regressor".into(), "rfr".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert!(format_bytes(1536).contains("KiB"));
        assert!(format_bytes(2 * 1024 * 1024).contains("MiB"));
        assert!(format_bytes(3 * 1024 * 1024 * 1024).contains("GiB"));
    }

    #[test]
    fn all_algorithms_have_names() {
        for algo in Algorithm::all() {
            let (name, alias) = algorithm_name_and_alias(*algo);
            assert!(!name.is_empty());
            assert!(!alias.is_empty());
        }
    }
}
