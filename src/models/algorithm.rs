use std::fmt;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;

use crate::error::{Result, SomaError};
use crate::models::wrapper::TrainedModel;

use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linear::elastic_net::ElasticNet;
use smartcore::linear::lasso::Lasso;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::ridge_regression::RidgeRegression;
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classification => write!(f, "classification"),
            Self::Regression => write!(f, "regression"),
        }
    }
}

/// Every ML algorithm soma knows how to use.
///
/// SVC/SVR are excluded — they borrow the training data, which makes
/// serialization impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum Algorithm {
    #[value(alias = "lr")]
    LogisticRegression,
    #[value(alias = "knn")]
    KnnClassifier,
    #[value(alias = "dtc")]
    DecisionTreeClassifier,
    #[value(alias = "rfc")]
    RandomForestClassifier,
    #[value(alias = "gnb")]
    GaussianNb,
    #[value(alias = "linreg")]
    LinearRegression,
    #[value(alias = "ridge")]
    RidgeRegression,
    #[value(alias = "lasso")]
    LassoRegression,
    #[value(alias = "enet")]
    ElasticNet,
    #[value(alias = "knnr")]
    KnnRegressor,
    #[value(alias = "dtr")]
    DecisionTreeRegressor,
    #[value(alias = "rfr")]
    RandomForestRegressor,
}

impl Algorithm {
    pub fn task(self) -> TaskType {
        match self {
            Self::LogisticRegression
            | Self::KnnClassifier
            | Self::DecisionTreeClassifier
            | Self::RandomForestClassifier
            | Self::GaussianNb => TaskType::Classification,

            Self::LinearRegression
            | Self::RidgeRegression
            | Self::LassoRegression
            | Self::ElasticNet
            | Self::KnnRegressor
            | Self::DecisionTreeRegressor
            | Self::RandomForestRegressor => TaskType::Regression,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::LogisticRegression => "Logistic Regression (classification)",
            Self::KnnClassifier => "K-Nearest Neighbors Classifier",
            Self::DecisionTreeClassifier => "Decision Tree Classifier (CART)",
            Self::RandomForestClassifier => "Random Forest Classifier",
            Self::GaussianNb => "Gaussian Naïve Bayes",
            Self::LinearRegression => "Linear Regression (OLS)",
            Self::RidgeRegression => "Ridge Regression (L2)",
            Self::LassoRegression => "Lasso Regression (L1)",
            Self::ElasticNet => "Elastic Net (L1+L2)",
            Self::KnnRegressor => "K-Nearest Neighbors Regressor",
            Self::DecisionTreeRegressor => "Decision Tree Regressor (CART)",
            Self::RandomForestRegressor => "Random Forest Regressor",
        }
    }

    pub fn all() -> &'static [Algorithm] {
        &[
            Self::LogisticRegression,
            Self::KnnClassifier,
            Self::DecisionTreeClassifier,
            Self::RandomForestClassifier,
            Self::GaussianNb,
            Self::LinearRegression,
            Self::RidgeRegression,
            Self::LassoRegression,
            Self::ElasticNet,
            Self::KnnRegressor,
            Self::DecisionTreeRegressor,
            Self::RandomForestRegressor,
        ]
    }

    /// Train a model with default hyperparameters.
    ///
    /// `y` is always `&[f64]` — for classifiers it gets converted to the
    /// appropriate integer type internally.
    pub fn train(self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<TrainedModel> {
        match self {
            Self::LogisticRegression => {
                let y_i32 = to_i32_vec(y);
                let m = LogisticRegression::fit(x, &y_i32, Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::LogisticRegression(m))
            }
            Self::KnnClassifier => {
                let y_i32 = to_i32_vec(y);
                let params = KNNClassifierParameters::default().with_distance(Euclidian::new());
                let m = KNNClassifier::fit(x, &y_i32, params).map_err(SomaError::model)?;
                Ok(TrainedModel::KnnClassifier(m))
            }
            Self::DecisionTreeClassifier => {
                let y_i32 = to_i32_vec(y);
                let m = DecisionTreeClassifier::fit(x, &y_i32, Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::DecisionTreeClassifier(m))
            }
            Self::RandomForestClassifier => {
                let y_i32 = to_i32_vec(y);
                let m = RandomForestClassifier::fit(x, &y_i32, Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::RandomForestClassifier(m))
            }
            Self::GaussianNb => {
                let y_u32 = to_u32_vec(y)?;
                let m = GaussianNB::fit(x, &y_u32, Default::default()).map_err(SomaError::model)?;
                Ok(TrainedModel::GaussianNb(m))
            }

            Self::LinearRegression => {
                let m = LinearRegression::fit(x, &y.to_vec(), Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::LinearRegression(m))
            }
            Self::RidgeRegression => {
                let m = RidgeRegression::fit(x, &y.to_vec(), Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::RidgeRegression(m))
            }
            Self::LassoRegression => {
                let m = Lasso::fit(x, &y.to_vec(), Default::default()).map_err(SomaError::model)?;
                Ok(TrainedModel::LassoRegression(m))
            }
            Self::ElasticNet => {
                let m = ElasticNet::fit(x, &y.to_vec(), Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::ElasticNet(m))
            }
            Self::KnnRegressor => {
                let params = KNNRegressorParameters::default().with_distance(Euclidian::new());
                let m = KNNRegressor::fit(x, &y.to_vec(), params).map_err(SomaError::model)?;
                Ok(TrainedModel::KnnRegressor(m))
            }
            Self::DecisionTreeRegressor => {
                let m = DecisionTreeRegressor::fit(x, &y.to_vec(), Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::DecisionTreeRegressor(m))
            }
            Self::RandomForestRegressor => {
                let m = RandomForestRegressor::fit(x, &y.to_vec(), Default::default())
                    .map_err(SomaError::model)?;
                Ok(TrainedModel::RandomForestRegressor(m))
            }
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

fn to_i32_vec(y: &[f64]) -> Vec<i32> {
    y.iter().map(|&v| v as i32).collect()
}

/// Fails if any label is negative (GaussianNB needs unsigned targets).
fn to_u32_vec(y: &[f64]) -> Result<Vec<u32>> {
    y.iter()
        .map(|&v| {
            if v < 0.0 {
                Err(SomaError::data(format!(
                    "Gaussian NB requires non-negative class labels, got {v}"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_every_variant() {
        assert_eq!(Algorithm::all().len(), 12);
    }

    #[test]
    fn task_types() {
        let classifiers = [
            Algorithm::LogisticRegression,
            Algorithm::KnnClassifier,
            Algorithm::DecisionTreeClassifier,
            Algorithm::RandomForestClassifier,
            Algorithm::GaussianNb,
        ];
        for algo in &classifiers {
            assert_eq!(algo.task(), TaskType::Classification, "{algo:?}");
        }

        let regressors = [
            Algorithm::LinearRegression,
            Algorithm::RidgeRegression,
            Algorithm::LassoRegression,
            Algorithm::ElasticNet,
            Algorithm::KnnRegressor,
            Algorithm::DecisionTreeRegressor,
            Algorithm::RandomForestRegressor,
        ];
        for algo in &regressors {
            assert_eq!(algo.task(), TaskType::Regression, "{algo:?}");
        }
    }

    #[test]
    fn descriptions_nonempty_and_match_display() {
        for algo in Algorithm::all() {
            assert!(!algo.description().is_empty());
            assert_eq!(algo.to_string(), algo.description());
        }
    }

    #[test]
    fn task_type_display() {
        assert_eq!(TaskType::Classification.to_string(), "classification");
        assert_eq!(TaskType::Regression.to_string(), "regression");
    }

    #[test]
    fn train_linreg() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]])
            .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert!(Algorithm::LinearRegression.train(&x, &y).is_ok());
    }

    #[test]
    fn train_knn() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        assert!(Algorithm::KnnClassifier.train(&x, &y).is_ok());
    }

    #[test]
    fn i32_conversion() {
        assert_eq!(to_i32_vec(&[0.0, 1.0, 2.5, -1.0]), vec![0, 1, 2, -1]);
    }

    #[test]
    fn u32_conversion() {
        assert_eq!(to_u32_vec(&[0.0, 1.0, 2.0]).unwrap(), vec![0, 1, 2]);
        assert!(to_u32_vec(&[0.0, -1.0]).is_err());
    }
}
