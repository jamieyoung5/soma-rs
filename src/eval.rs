use comfy_table::{presets, Table};

use crate::models::TaskType;

/// Computed evaluation metrics for a model, either classification or regression.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub task: TaskType,
    pub n_samples: usize,
    pub metrics: Vec<(String, f64)>,
}

impl EvalReport {
    /// Compare predictions against ground truth and compute the relevant metrics.
    ///
    /// Panics if `y_true` and `y_pred` differ in length.
    pub fn compute(task: TaskType, y_true: &[f64], y_pred: &[f64]) -> Self {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have the same length"
        );

        let metrics = match task {
            TaskType::Classification => classification_metrics(y_true, y_pred),
            TaskType::Regression => regression_metrics(y_true, y_pred),
        };

        Self {
            task,
            n_samples: y_true.len(),
            metrics,
        }
    }

    pub fn to_table(&self) -> String {
        let mut table = Table::new();
        table.load_preset(presets::UTF8_FULL);
        table.set_header(vec!["Metric", "Value"]);

        table.add_row(vec!["Task".to_string(), self.task.to_string()]);
        table.add_row(vec!["Samples".to_string(), self.n_samples.to_string()]);

        for (name, value) in &self.metrics {
            table.add_row(vec![name.clone(), format!("{value:.6}")]);
        }

        table.to_string()
    }

    /// Look up a metric by name (case-insensitive). Returns `None` if not found.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| *v)
    }
}

impl std::fmt::Display for EvalReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_table())
    }
}

// Classification

fn classification_metrics(y_true: &[f64], y_pred: &[f64]) -> Vec<(String, f64)> {
    let accuracy = accuracy(y_true, y_pred);

    let mut classes: Vec<f64> = y_true
        .iter()
        .chain(y_pred.iter())
        .copied()
        .collect::<Vec<_>>();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_f1 = 0.0;
    let n_classes = classes.len() as f64;

    for &cls in &classes {
        let tp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| t == cls && p == cls)
            .count() as f64;
        let fp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| t != cls && p == cls)
            .count() as f64;
        let fn_ = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| t == cls && p != cls)
            .count() as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
    }

    let macro_precision = if n_classes > 0.0 {
        total_precision / n_classes
    } else {
        0.0
    };
    let macro_recall = if n_classes > 0.0 {
        total_recall / n_classes
    } else {
        0.0
    };
    let macro_f1 = if n_classes > 0.0 {
        total_f1 / n_classes
    } else {
        0.0
    };

    vec![
        ("Accuracy".into(), accuracy),
        ("Precision (macro)".into(), macro_precision),
        ("Recall (macro)".into(), macro_recall),
        ("F1 Score (macro)".into(), macro_f1),
    ]
}

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < f64::EPSILON)
        .count();
    correct as f64 / y_true.len() as f64
}

// Regression

fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> Vec<(String, f64)> {
    let mae = mean_absolute_error(y_true, y_pred);
    let mse = mean_squared_error(y_true, y_pred);
    let rmse = mse.sqrt();
    let r2 = r_squared(y_true, y_pred);

    vec![
        ("MAE".into(), mae),
        ("MSE".into(), mse),
        ("RMSE".into(), rmse),
        ("R²".into(), r2),
    ]
}

fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum();
    sum / y_true.len() as f64
}

fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    sum / y_true.len() as f64
}

/// R² = 1 - SS_res / SS_tot. Returns 0 when all true values are identical.
fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }

    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;

    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    if ss_tot.abs() < f64::EPSILON {
        return 0.0;
    }

    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_basic() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        assert!((accuracy(&y, &y) - 1.0).abs() < f64::EPSILON);

        let y_pred = vec![0.0, 0.0, 0.0, 0.0];
        assert!((accuracy(&y, &y_pred) - 0.5).abs() < f64::EPSILON);

        assert!((accuracy(&[], &[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mae_and_mse() {
        let y = vec![1.0, 2.0, 3.0];
        assert!(mean_absolute_error(&y, &y).abs() < f64::EPSILON);
        assert!(mean_squared_error(&y, &y).abs() < f64::EPSILON);

        let y_pred = vec![2.0, 3.0, 4.0];
        assert!((mean_absolute_error(&y, &y_pred) - 1.0).abs() < f64::EPSILON);
        // each residual^2 = 1, so MSE = 1
        assert!((mean_squared_error(&y, &y_pred) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn r2_values() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert!((r_squared(&y, &y) - 1.0).abs() < f64::EPSILON);

        // constant target => ss_tot=0, should return 0 gracefully
        let constant = vec![5.0, 5.0, 5.0];
        assert!((r_squared(&constant, &constant) - 0.0).abs() < f64::EPSILON);

        // inversely correlated => negative R²
        let bad_pred = vec![3.0, 2.0, 1.0];
        let y3 = vec![1.0, 2.0, 3.0];
        assert!(r_squared(&y3, &bad_pred) < 0.0);
    }

    #[test]
    fn classification_report_perfect() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let report = EvalReport::compute(TaskType::Classification, &y, &y);
        assert_eq!(report.task, TaskType::Classification);
        assert_eq!(report.n_samples, 5);
        assert!((report.get("accuracy").unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((report.get("f1 score (macro)").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn regression_report_perfect() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let report = EvalReport::compute(TaskType::Regression, &y, &y);
        assert_eq!(report.task, TaskType::Regression);
        assert!(report.get("mae").unwrap().abs() < f64::EPSILON);
        assert!((report.get("r²").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_display_sanity() {
        let y = vec![0.0, 1.0, 1.0];
        let report = EvalReport::compute(TaskType::Classification, &y, &y);
        let output = report.to_string();
        assert!(output.contains("Accuracy"));
        assert!(output.contains("classification"));
    }

    #[test]
    fn single_sample() {
        let report = EvalReport::compute(TaskType::Classification, &[1.0], &[1.0]);
        assert_eq!(report.n_samples, 1);
        assert!((report.get("accuracy").unwrap() - 1.0).abs() < f64::EPSILON);

        let report = EvalReport::compute(TaskType::Regression, &[3.0], &[3.0]);
        assert!(report.get("mae").unwrap().abs() < f64::EPSILON);
    }

    #[test]
    fn get_missing_metric() {
        let report = EvalReport::compute(TaskType::Regression, &[1.0, 2.0], &[1.0, 2.0]);
        assert!(report.get("nonexistent").is_none());
    }
}
