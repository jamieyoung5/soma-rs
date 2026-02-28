use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;

use crate::error::{Result, SomaError};
use crate::models::algorithm::{Algorithm, TaskType};

use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linear::elastic_net::ElasticNet;
use smartcore::linear::lasso::Lasso;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::ridge_regression::RidgeRegression;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;

/// Wraps every supported smartcore model behind a single type so the rest of
/// the codebase doesn't have to be generic over every possible estimator.
///
/// Classifiers predict integer types internally; we convert to/from f64 at
/// the boundary so the public API stays uniform.
#[derive(Debug, Serialize, Deserialize)]
pub enum TrainedModel {
    LogisticRegression(LogisticRegression<f64, i32, DenseMatrix<f64>, Vec<i32>>),
    KnnClassifier(KNNClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>, Euclidian<f64>>),
    DecisionTreeClassifier(DecisionTreeClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>),
    RandomForestClassifier(RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>),
    /// GaussianNB needs u32 targets (Unsigned trait bound).
    GaussianNb(GaussianNB<f64, u32, DenseMatrix<f64>, Vec<u32>>),

    LinearRegression(LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    RidgeRegression(RidgeRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    LassoRegression(Lasso<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    ElasticNet(ElasticNet<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    KnnRegressor(KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>>),
    DecisionTreeRegressor(DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    RandomForestRegressor(RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
}

impl TrainedModel {
    /// Run the model on `x` and return predictions as f64.
    pub fn predict(&self, x: &DenseMatrix<f64>) -> Result<Vec<f64>> {
        let preds: Vec<f64> = match self {
            // classifiers: i32 -> f64
            Self::LogisticRegression(m) => {
                let raw: Vec<i32> = m.predict(x).map_err(SomaError::model)?;
                raw.into_iter().map(|v| v as f64).collect()
            }
            Self::KnnClassifier(m) => {
                let raw: Vec<i32> = m.predict(x).map_err(SomaError::model)?;
                raw.into_iter().map(|v| v as f64).collect()
            }
            Self::DecisionTreeClassifier(m) => {
                let raw: Vec<i32> = m.predict(x).map_err(SomaError::model)?;
                raw.into_iter().map(|v| v as f64).collect()
            }
            Self::RandomForestClassifier(m) => {
                let raw: Vec<i32> = m.predict(x).map_err(SomaError::model)?;
                raw.into_iter().map(|v| v as f64).collect()
            }
            // u32 -> f64
            Self::GaussianNb(m) => {
                let raw: Vec<u32> = m.predict(x).map_err(SomaError::model)?;
                raw.into_iter().map(|v| v as f64).collect()
            }
            // regressors already produce f64
            Self::LinearRegression(m) => m.predict(x).map_err(SomaError::model)?,
            Self::RidgeRegression(m) => m.predict(x).map_err(SomaError::model)?,
            Self::LassoRegression(m) => m.predict(x).map_err(SomaError::model)?,
            Self::ElasticNet(m) => m.predict(x).map_err(SomaError::model)?,
            Self::KnnRegressor(m) => m.predict(x).map_err(SomaError::model)?,
            Self::DecisionTreeRegressor(m) => m.predict(x).map_err(SomaError::model)?,
            Self::RandomForestRegressor(m) => m.predict(x).map_err(SomaError::model)?,
        };
        Ok(preds)
    }

    pub fn algorithm(&self) -> Algorithm {
        match self {
            Self::LogisticRegression(_) => Algorithm::LogisticRegression,
            Self::KnnClassifier(_) => Algorithm::KnnClassifier,
            Self::DecisionTreeClassifier(_) => Algorithm::DecisionTreeClassifier,
            Self::RandomForestClassifier(_) => Algorithm::RandomForestClassifier,
            Self::GaussianNb(_) => Algorithm::GaussianNb,
            Self::LinearRegression(_) => Algorithm::LinearRegression,
            Self::RidgeRegression(_) => Algorithm::RidgeRegression,
            Self::LassoRegression(_) => Algorithm::LassoRegression,
            Self::ElasticNet(_) => Algorithm::ElasticNet,
            Self::KnnRegressor(_) => Algorithm::KnnRegressor,
            Self::DecisionTreeRegressor(_) => Algorithm::DecisionTreeRegressor,
            Self::RandomForestRegressor(_) => Algorithm::RandomForestRegressor,
        }
    }

    pub fn task(&self) -> TaskType {
        self.algorithm().task()
    }

    pub fn description(&self) -> &'static str {
        self.algorithm().description()
    }
}

impl std::fmt::Display for TrainedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TrainedModel({})", self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_classifier_data() -> (DenseMatrix<f64>, Vec<f64>) {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ])
        .unwrap();
        (x, vec![0.0, 0.0, 1.0, 1.0, 1.0])
    }

    #[test]
    fn knn_predict_and_metadata() {
        let (x, y) = toy_classifier_data();
        let model = Algorithm::KnnClassifier.train(&x, &y).unwrap();

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
        assert_eq!(model.algorithm(), Algorithm::KnnClassifier);
        assert_eq!(model.task(), TaskType::Classification);
        assert!(model.to_string().contains("K-Nearest Neighbors Classifier"));
    }

    #[test]
    fn linreg_predict() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]])
            .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let model = Algorithm::LinearRegression.train(&x, &y).unwrap();
        assert_eq!(model.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn serde_roundtrip() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]])
            .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let model = Algorithm::LinearRegression.train(&x, &y).unwrap();

        let encoded = bincode::serialize(&model).unwrap();
        let decoded: TrainedModel = bincode::deserialize(&encoded).unwrap();

        let orig = model.predict(&x).unwrap();
        let after = decoded.predict(&x).unwrap();
        for (a, b) in orig.iter().zip(after.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "diverged after roundtrip: {a} vs {b}"
            );
        }
    }
}
