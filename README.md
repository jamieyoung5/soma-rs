# soma

[![CI](https://github.com/jamieyoung5/soma-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jamieyoung5/soma-rs/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A CLI for training and using ML models, powered by [smartcore](https://smartcorelib.org/).

## Install

```sh
cargo install --path .
```

## Usage

Train a model:

```sh
soma train --data iris.csv --target species --algorithm knn -o model.soma
```

Predict with it:

```sh
soma predict --model model.soma --data new_data.csv
```

Evaluate how well an algorithm does:

```sh
soma evaluate --data iris.csv --target species --algorithm rfc --test-size 0.3
```

Inspect a saved model:

```sh
soma inspect --model model.soma
```

List what's available:

```sh
soma algorithms
```

Running `soma` with no arguments launches an interactive TUI.

## Algorithms

### Classification

| Name                       | Alias | Description          |
| -------------------------- | ----- | -------------------- |
| `logistic-regression`      | `lr`  | Logistic Regression  |
| `knn-classifier`           | `knn` | K-Nearest Neighbors  |
| `decision-tree-classifier` | `dtc` | Decision Tree (CART) |
| `random-forest-classifier` | `rfc` | Random Forest        |
| `gaussian-nb`              | `gnb` | Gaussian Naïve Bayes |

### Regression

| Name                      | Alias    | Description          |
| ------------------------- | -------- | -------------------- |
| `linear-regression`       | `linreg` | OLS                  |
| `ridge-regression`        | `ridge`  | L2 regularized       |
| `lasso-regression`        | `lasso`  | L1 regularized       |
| `elastic-net`             | `enet`   | L1+L2                |
| `knn-regressor`           | `knnr`   | K-Nearest Neighbors  |
| `decision-tree-regressor` | `dtr`    | Decision Tree (CART) |
| `random-forest-regressor` | `rfr`    | Random Forest        |

Use either the full name or the alias with `--algorithm`.

## Data format

- CSV with a header row
- All values must be numeric — encode categoricals as integers beforehand
- Target column can be specified by name (`--target species`) or index (`--target 4`)
- Column lookup is case-insensitive

## Limitations

- **No SVM** — smartcore's SVC/SVR borrow the training data, which makes serialization impossible
- **Default hyperparameters only** for now
- **No categorical encoding** — your data needs to be numeric already
- **No shuffling** — train/test splits just take the first N rows for training

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
