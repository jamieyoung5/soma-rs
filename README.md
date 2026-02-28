# soma

[![CI](https://github.com/jamieyoung5/soma-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jamieyoung5/soma-rs/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A command-line tool for training, evaluating, and using machine learning models. Works as a traditional CLI or as an interactive TUI (just run `soma` with no arguments). Built on top of [smartcore](https://smartcorelib.org/), so everything runs locally. No Python, no cloud APIs, no faffing about.

## Install

From source (you'll need [Rust](https://rustup.rs/)):

```sh
cargo install --path .
```

Or grab a binary from the [releases page](https://github.com/jamieyoung5/soma-rs/releases).

## Quickstart

Train a model:

```sh
soma train --data iris.csv --target species --algorithm knn -o model.soma
```

See how well it did:

```sh
soma evaluate --data iris.csv --target species --algorithm knn --test-size 0.3
```

Use it to make predictions on new data:

```sh
soma predict --model model.soma --data new_data.csv
```

Check what's inside a saved model:

```sh
soma inspect --model model.soma
```

See all available algorithms:

```sh
soma algorithms
```

Every subcommand has a short alias too: `t`, `e`, `p`, `i`, `ls` if you don't feel like typing the whole thing.

## Algorithms

### Classification

| Name                       | Alias | What it is           |
| -------------------------- | ----- | -------------------- |
| `logistic-regression`      | `lr`  | Logistic Regression  |
| `knn-classifier`           | `knn` | K-Nearest Neighbors  |
| `decision-tree-classifier` | `dtc` | Decision Tree (CART) |
| `random-forest-classifier` | `rfc` | Random Forest        |
| `gaussian-nb`              | `gnb` | Gaussian Naïve Bayes |

### Regression

| Name                      | Alias    | What it is           |
| ------------------------- | -------- | -------------------- |
| `linear-regression`       | `linreg` | OLS                  |
| `ridge-regression`        | `ridge`  | L2 regularised       |
| `lasso-regression`        | `lasso`  | L1 regularised       |
| `elastic-net`             | `enet`   | L1+L2                |
| `knn-regressor`           | `knnr`   | K-Nearest Neighbors  |
| `decision-tree-regressor` | `dtr`    | Decision Tree (CART) |
| `random-forest-regressor` | `rfr`    | Random Forest        |

Use the full name or the alias with `--algorithm`, whichever you prefer.

## What your data needs to look like

- CSV with a header row
- Every value needs to be numeric, encode your categoricals as integers before you feed them in
- Target column can be a name (`--target species`) or an index (`--target 4`)
- Column lookup is case-insensitive, so `Species`, `species`, and `SPECIES` all work

## Known limitations

These are things I'm aware of and might address later:

- **No SVM**: smartcore's SVC/SVR borrow the training data, which makes serialisation impossible with the current architecture
- **Default hyperparameters only**: no way to tune them yet
- **No categorical encoding**: you need to handle that yourself before passing data in
- **No shuffling**: train/test splits take the first N rows for training and the rest for testing, so if your data is sorted by class you'll have a bad time

## Contributing

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up.

## License

GPL-3.0, see [LICENSE](LICENSE) for details.
