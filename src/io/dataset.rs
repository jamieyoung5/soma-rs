use std::path::Path;

use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::error::{Result, SomaError};

/// A loaded dataset ready for training or evaluation.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: DenseMatrix<f64>,
    pub y: Vec<f64>,
    pub feature_names: Vec<String>,
    pub target_name: String,
    pub n_samples: usize,
    pub n_features: usize,
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dataset(samples={}, features={}, target='{}')",
            self.n_samples, self.n_features, self.target_name,
        )
    }
}

/// Result of a train/test split.
#[derive(Debug, Clone)]
pub struct SplitData {
    pub x_train: DenseMatrix<f64>,
    pub y_train: Vec<f64>,
    pub x_test: DenseMatrix<f64>,
    pub y_test: Vec<f64>,
    pub n_train: usize,
    pub n_test: usize,
}

/// Load a CSV file into a [`Dataset`].
///
/// `target_col` can be a column name or a zero-based index (as a string).
/// All remaining columns become features.
pub fn load_csv(path: &Path, target_col: &str) -> Result<Dataset> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_path(path)?;

    let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

    if headers.is_empty() {
        return Err(SomaError::data("CSV file has no columns"));
    }

    let target_idx = resolve_column_index(&headers, target_col)?;

    let feature_indices: Vec<usize> = (0..headers.len()).filter(|&i| i != target_idx).collect();

    if feature_indices.is_empty() {
        return Err(SomaError::data(
            "No feature columns remain after selecting the target",
        ));
    }

    let feature_names: Vec<String> = feature_indices
        .iter()
        .map(|&i| headers[i].clone())
        .collect();
    let target_name = headers[target_idx].clone();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for (line_no, result) in reader.records().enumerate() {
        let record = result?;
        let row: Vec<f64> = feature_indices
            .iter()
            .map(|&i| parse_cell(&record, i, line_no, &headers))
            .collect::<Result<Vec<f64>>>()?;

        let target_val = parse_cell(&record, target_idx, line_no, &headers)?;

        rows.push(row);
        targets.push(target_val);
    }

    if rows.is_empty() {
        return Err(SomaError::data("CSV file contains no data rows"));
    }

    let n_samples = rows.len();
    let n_features = feature_indices.len();
    let x = rows_to_dense_matrix(&rows, n_samples, n_features)?;

    Ok(Dataset {
        x,
        y: targets,
        feature_names,
        target_name,
        n_samples,
        n_features,
    })
}

/// Split a dataset into train and test partitions. No shuffling — the split
/// just takes the first N rows for training and the rest for testing.
pub fn split_train_test(dataset: &Dataset, test_fraction: f64) -> Result<SplitData> {
    if test_fraction <= 0.0 || test_fraction >= 1.0 {
        return Err(SomaError::config(format!(
            "test_fraction must be between 0 and 1 exclusive, got {test_fraction}"
        )));
    }

    let n = dataset.n_samples;
    let n_test = (n as f64 * test_fraction).round() as usize;
    let n_test = n_test.clamp(1, n - 1);
    let n_train = n - n_test;

    let split_at = n_train;

    let (train_rows, test_rows) = extract_rows(&dataset.x, dataset.n_features, split_at, n);

    let y_train = dataset.y[..split_at].to_vec();
    let y_test = dataset.y[split_at..].to_vec();

    let x_train = rows_to_dense_matrix_flat(&train_rows, n_train, dataset.n_features)?;
    let x_test = rows_to_dense_matrix_flat(&test_rows, n_test, dataset.n_features)?;

    Ok(SplitData {
        x_train,
        y_train,
        x_test,
        y_test,
        n_train,
        n_test,
    })
}

// --- internals ---

/// Try to resolve a column name (or numeric index string) to the actual index.
fn resolve_column_index(headers: &[String], col: &str) -> Result<usize> {
    // Exact match first.
    if let Some(idx) = headers.iter().position(|h| h == col) {
        return Ok(idx);
    }

    // Case-insensitive fallback.
    if let Some(idx) = headers.iter().position(|h| h.eq_ignore_ascii_case(col)) {
        return Ok(idx);
    }

    // Maybe it's a numeric index.
    if let Ok(idx) = col.parse::<usize>() {
        if idx < headers.len() {
            return Ok(idx);
        }
        return Err(SomaError::ColumnNotFound(format!(
            "column index {idx} is out of range (dataset has {} columns)",
            headers.len()
        )));
    }

    Err(SomaError::ColumnNotFound(format!(
        "'{col}' not found in headers: [{}]",
        headers.join(", ")
    )))
}

fn parse_cell(
    record: &csv::StringRecord,
    col_idx: usize,
    row_idx: usize,
    headers: &[String],
) -> Result<f64> {
    let raw = record
        .get(col_idx)
        .ok_or_else(|| SomaError::data(format!("row {row_idx}: missing column index {col_idx}")))?;

    raw.trim().parse::<f64>().map_err(|_| {
        let col_name = headers.get(col_idx).map(|s| s.as_str()).unwrap_or("?");
        SomaError::data(format!(
            "row {row_idx}, column '{col_name}': cannot parse '{raw}' as a number"
        ))
    })
}

/// Row-major Vec<Vec<f64>> -> column-major DenseMatrix.
fn rows_to_dense_matrix(
    rows: &[Vec<f64>],
    n_samples: usize,
    n_features: usize,
) -> Result<DenseMatrix<f64>> {
    let mut col_major = vec![0.0; n_samples * n_features];
    for (r, row) in rows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            col_major[c * n_samples + r] = val;
        }
    }

    DenseMatrix::new(n_samples, n_features, col_major, true)
        .map_err(|e| SomaError::data(format!("failed to build DenseMatrix: {e}")))
}

/// Flat row-major buffer -> column-major DenseMatrix.
fn rows_to_dense_matrix_flat(
    flat_row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<DenseMatrix<f64>> {
    let mut col_major = vec![0.0; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            col_major[c * n_rows + r] = flat_row_major[r * n_cols + c];
        }
    }

    DenseMatrix::new(n_rows, n_cols, col_major, true)
        .map_err(|e| SomaError::data(format!("failed to build DenseMatrix: {e}")))
}

/// Split matrix rows into two flat row-major buffers at the given index.
fn extract_rows(
    matrix: &DenseMatrix<f64>,
    n_features: usize,
    split_at: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    use smartcore::linalg::basic::arrays::Array;

    let mut train_buf = Vec::with_capacity(split_at * n_features);
    let mut test_buf = Vec::with_capacity((n - split_at) * n_features);

    for row in 0..n {
        for col in 0..n_features {
            let val = *matrix.get((row, col));
            if row < split_at {
                train_buf.push(val);
            } else {
                test_buf.push(val);
            }
        }
    }

    (train_buf, test_buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn load_basic_csv() {
        let csv = "a,b,target\n1,2,0\n3,4,1\n5,6,0\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "target").unwrap();
        assert_eq!(ds.n_samples, 3);
        assert_eq!(ds.n_features, 2);
        assert_eq!(ds.target_name, "target");
        assert_eq!(ds.feature_names, vec!["a", "b"]);
        assert_eq!(ds.y, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn load_by_column_index() {
        let csv = "x,y,z\n1,2,3\n4,5,6\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "2").unwrap();
        assert_eq!(ds.target_name, "z");
        assert_eq!(ds.n_features, 2);
    }

    #[test]
    fn missing_column_errors() {
        let csv = "a,b\n1,2\n";
        let f = write_temp_csv(csv);
        let err = load_csv(f.path(), "nope").unwrap_err();
        assert!(matches!(err, SomaError::ColumnNotFound(_)));
    }

    #[test]
    fn non_numeric_cell_errors() {
        let csv = "a,b\n1,hello\n";
        let f = write_temp_csv(csv);
        let err = load_csv(f.path(), "b").unwrap_err();
        assert!(matches!(err, SomaError::Data(_)));
        assert!(err.to_string().contains("hello"));
    }

    #[test]
    fn empty_data_errors() {
        let csv = "a,b\n";
        let f = write_temp_csv(csv);
        assert!(matches!(
            load_csv(f.path(), "b").unwrap_err(),
            SomaError::Data(_)
        ));
    }

    #[test]
    fn split_roundtrip() {
        let csv = "a,b,t\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "t").unwrap();
        let split = split_train_test(&ds, 0.4).unwrap();
        assert_eq!(split.n_train + split.n_test, 5);
        assert_eq!(split.y_train.len(), split.n_train);
        assert_eq!(split.y_test.len(), split.n_test);
    }

    #[test]
    fn split_rejects_bad_fractions() {
        let csv = "a,t\n1,0\n2,1\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "t").unwrap();
        assert!(split_train_test(&ds, 0.0).is_err());
        assert!(split_train_test(&ds, 1.0).is_err());
        assert!(split_train_test(&ds, -0.1).is_err());
    }

    #[test]
    fn case_insensitive_column_lookup() {
        let headers = vec!["Alpha".into(), "Beta".into()];
        assert_eq!(resolve_column_index(&headers, "alpha").unwrap(), 0);
    }

    #[test]
    fn dataset_display() {
        let csv = "a,t\n1,0\n2,1\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "t").unwrap();
        let s = ds.to_string();
        assert!(s.contains("samples=2"));
        assert!(s.contains("features=1"));
    }

    #[test]
    fn matrix_values_correct() {
        use smartcore::linalg::basic::arrays::Array;

        let csv = "a,b,t\n1,2,10\n3,4,20\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "t").unwrap();

        assert_eq!(*ds.x.get((0, 0)), 1.0);
        assert_eq!(*ds.x.get((0, 1)), 2.0);
        assert_eq!(*ds.x.get((1, 0)), 3.0);
        assert_eq!(*ds.x.get((1, 1)), 4.0);
    }

    #[test]
    fn split_preserves_values() {
        use smartcore::linalg::basic::arrays::Array;

        let csv = "a,b,t\n1,2,0\n3,4,1\n5,6,2\n7,8,3\n";
        let f = write_temp_csv(csv);
        let ds = load_csv(f.path(), "t").unwrap();
        let split = split_train_test(&ds, 0.5).unwrap();

        assert_eq!(*split.x_train.get((0, 0)), 1.0);
        assert_eq!(*split.x_train.get((1, 1)), 4.0);
        assert_eq!(*split.x_test.get((0, 0)), 5.0);
        assert_eq!(*split.x_test.get((1, 1)), 8.0);

        assert_eq!(split.y_train, vec![0.0, 1.0]);
        assert_eq!(split.y_test, vec![2.0, 3.0]);
    }
}
