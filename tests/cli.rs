use std::process::Command;

use tempfile::TempDir;

fn soma_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_soma"))
}

fn iris_csv() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/iris_mini.csv")
}

fn housing_csv() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/housing_mini.csv")
}

#[test]
fn algorithms_lists_all() {
    let output = soma_bin().arg("algorithms").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("knn"), "expected knn in output:\n{stdout}");
    assert!(
        stdout.contains("linreg"),
        "expected linreg in output:\n{stdout}"
    );
    assert!(stdout.contains("rfc"), "expected rfc in output:\n{stdout}");
    assert!(
        stdout.contains("ridge"),
        "expected ridge in output:\n{stdout}"
    );
}

#[test]
fn train_classification_knn() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "knn",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists(), "model file should be created");
    assert!(
        model_path.metadata().unwrap().len() > 0,
        "model file should not be empty"
    );
}

#[test]
fn train_regression_linreg() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            housing_csv(),
            "--target",
            "price",
            "--algorithm",
            "linreg",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists());
}

#[test]
fn train_with_evaluation() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "rfc",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0.3")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists());
}

#[test]
fn train_missing_data_file_fails() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            "nonexistent.csv",
            "--target",
            "species",
            "--algorithm",
            "knn",
            "-o",
        ])
        .arg(&model_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn train_bad_target_column_fails() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "nonexistent_column",
            "--algorithm",
            "knn",
            "-o",
        ])
        .arg(&model_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn predict_after_train() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let train_output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "knn",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();
    assert!(
        train_output.status.success(),
        "train failed: {}",
        String::from_utf8_lossy(&train_output.stderr)
    );

    let pred_output = soma_bin()
        .args(["predict", "--data", iris_csv(), "--target", "species"])
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();

    assert!(
        pred_output.status.success(),
        "predict failed: {}",
        String::from_utf8_lossy(&pred_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&pred_output.stdout);

    assert!(!stdout.trim().is_empty(), "predictions should not be empty");
}

#[test]
fn predict_to_output_file() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");
    let output_path = tmp.path().join("predictions.csv");

    let train_output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "dtc",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();
    assert!(train_output.status.success());

    let pred_output = soma_bin()
        .args(["predict", "--data", iris_csv(), "--target", "species"])
        .arg("--model")
        .arg(&model_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(
        pred_output.status.success(),
        "predict failed: {}",
        String::from_utf8_lossy(&pred_output.stderr)
    );
    assert!(output_path.exists(), "output file should be created");
    let contents = std::fs::read_to_string(&output_path).unwrap();
    assert!(
        !contents.trim().is_empty(),
        "output file should not be empty"
    );
}

#[test]
fn predict_with_bad_model_fails() {
    let tmp = TempDir::new().unwrap();
    let bad_model = tmp.path().join("garbage.soma");
    std::fs::write(&bad_model, b"this is not a model").unwrap();

    let output = soma_bin()
        .args(["predict", "--data", iris_csv()])
        .arg("--model")
        .arg(&bad_model)
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn evaluate_classification() {
    let output = soma_bin()
        .args([
            "evaluate",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "rfc",
            "--test-size",
            "0.3",
        ])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("ccuracy") || stdout.contains("Accuracy") || stdout.contains("accuracy"),
        "expected accuracy metric in output:\n{stdout}"
    );
}

#[test]
fn evaluate_regression() {
    let output = soma_bin()
        .args([
            "evaluate",
            "--data",
            housing_csv(),
            "--target",
            "price",
            "--algorithm",
            "linreg",
            "--test-size",
            "0.3",
        ])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        stdout.contains("R²")
            || stdout.contains("MAE")
            || stdout.contains("MSE")
            || stdout.contains("r2"),
        "expected regression metrics in output:\n{stdout}"
    );
}

#[test]
fn evaluate_multiple_algorithms() {
    for algo in &["knn", "dtc", "gnb", "lr"] {
        let output = soma_bin()
            .args([
                "evaluate",
                "--data",
                iris_csv(),
                "--target",
                "species",
                "--algorithm",
                algo,
                "--test-size",
                "0.3",
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "algorithm {algo} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    for algo in &["linreg", "ridge", "lasso", "dtr"] {
        let output = soma_bin()
            .args([
                "evaluate",
                "--data",
                housing_csv(),
                "--target",
                "price",
                "--algorithm",
                algo,
                "--test-size",
                "0.3",
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "algorithm {algo} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn inspect_shows_model_metadata() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let train_output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "rfc",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();
    assert!(train_output.status.success());

    let output = soma_bin()
        .arg("inspect")
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        stdout.contains("random")
            || stdout.contains("rfc")
            || stdout.contains("Random")
            || stdout.contains("algorithm"),
        "expected algorithm info in inspect output:\n{stdout}"
    );
}

#[test]
fn inspect_nonexistent_model_fails() {
    let output = soma_bin()
        .args(["inspect", "--model", "does_not_exist.soma"])
        .output()
        .unwrap();

    assert!(!output.status.success());
}

#[test]
fn full_roundtrip_classification() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("roundtrip.soma");

    let train = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "knn",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "train: {}",
        String::from_utf8_lossy(&train.stderr)
    );

    let inspect = soma_bin()
        .arg("inspect")
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();
    assert!(
        inspect.status.success(),
        "inspect: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );

    let predict = soma_bin()
        .args(["predict", "--data", iris_csv(), "--target", "species"])
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();
    assert!(
        predict.status.success(),
        "predict: {}",
        String::from_utf8_lossy(&predict.stderr)
    );

    let predictions = String::from_utf8_lossy(&predict.stdout);

    let lines: Vec<&str> = predictions.trim().lines().collect();
    assert!(
        lines.len() >= 2,
        "expected at least a header + one prediction row, got {} lines:\n{predictions}",
        lines.len()
    );
}

#[test]
fn full_roundtrip_regression() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("roundtrip_reg.soma");

    let train = soma_bin()
        .args([
            "train",
            "--data",
            housing_csv(),
            "--target",
            "price",
            "--algorithm",
            "ridge",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "train: {}",
        String::from_utf8_lossy(&train.stderr)
    );

    let inspect = soma_bin()
        .arg("inspect")
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();
    assert!(
        inspect.status.success(),
        "inspect: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );

    let predict = soma_bin()
        .args(["predict", "--data", housing_csv(), "--target", "price"])
        .arg("--model")
        .arg(&model_path)
        .output()
        .unwrap();
    assert!(
        predict.status.success(),
        "predict: {}",
        String::from_utf8_lossy(&predict.stderr)
    );

    let predictions = String::from_utf8_lossy(&predict.stdout);
    assert!(
        !predictions.trim().is_empty(),
        "predictions should not be empty"
    );
}

#[test]
fn version_flag() {
    let output = soma_bin().arg("--version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        stdout.contains("soma"),
        "expected 'soma' in version output: {stdout}"
    );
}

#[test]
fn help_flag() {
    let output = soma_bin().arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        stdout.contains("train"),
        "expected 'train' subcommand in help"
    );
    assert!(
        stdout.contains("predict"),
        "expected 'predict' subcommand in help"
    );
    assert!(
        stdout.contains("evaluate"),
        "expected 'evaluate' subcommand in help"
    );
}

#[test]
fn unknown_subcommand_fails() {
    let output = soma_bin().arg("foobar").output().unwrap();
    assert!(!output.status.success());
}

#[test]
fn train_target_by_column_index() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "train",
            "--data",
            iris_csv(),
            "--target",
            "4",
            "--algorithm",
            "knn",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists());
}

#[test]
fn train_alias_t_works() {
    let tmp = TempDir::new().unwrap();
    let model_path = tmp.path().join("model.soma");

    let output = soma_bin()
        .args([
            "t",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "knn",
        ])
        .arg("-o")
        .arg(&model_path)
        .arg("--test-size")
        .arg("0")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn evaluate_alias_e_works() {
    let output = soma_bin()
        .args([
            "e",
            "--data",
            iris_csv(),
            "--target",
            "species",
            "--algorithm",
            "knn",
            "--test-size",
            "0.3",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn algorithms_alias_ls_works() {
    let output = soma_bin().arg("ls").output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
