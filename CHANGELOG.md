# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added

- Train classification and regression models from CSV datasets.
- Predict with saved `.soma` model files.
- Evaluate model performance with configurable train/test splits.
- Inspect saved model metadata.
- List all supported algorithms via `soma algorithms`.
- Interactive TUI mode when launched with no arguments.
- Classification algorithms: Logistic Regression, KNN, Decision Tree, Random Forest, Gaussian Naïve Bayes.
- Regression algorithms: Linear Regression, Ridge, Lasso, Elastic Net, KNN Regressor, Decision Tree Regressor, Random Forest Regressor.
- Case-insensitive column lookup by name or index.

[Unreleased]: https://github.com/jamieyoung5/soma-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamieyoung5/soma-rs/releases/tag/v0.1.0
