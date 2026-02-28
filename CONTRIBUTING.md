# Contributing to soma

Thanks for your interest in contributing! Here's how to get started.

## Getting started

1. **Fork** the repo and clone your fork.
2. Make sure you have Rust installed ([rustup.rs](https://rustup.rs/)).
3. Build and run the tests:

```sh
cargo build
cargo test
```

## Development workflow

### Code style

This project uses standard `rustfmt` formatting and `clippy` linting. Before submitting a PR, make sure both pass cleanly:

```sh
cargo fmt -- --check
cargo clippy -- -D warnings
```

### Running tests

```sh
cargo test
```

All existing tests must pass. If you're adding new functionality, please add tests to cover it.

### Project structure

```
src/
├── cli/          # CLI argument parsing and command handlers
├── io/           # CSV loading and dataset handling
├── models/       # Algorithm definitions, model wrapper, serialization
├── tui/          # Interactive terminal UI
├── error.rs      # Error types
├── eval.rs       # Evaluation metrics and reporting
├── lib.rs        # Public module exports
└── main.rs       # Entry point
tests/
└── data/         # Sample CSV files for testing
```

## Submitting a pull request

1. Create a feature branch from `main`:
   ```sh
   git checkout -b my-feature
   ```
2. Make your changes in small, focused commits.
3. Make sure `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` all pass.
4. Push your branch and open a pull request against `main`.
5. Describe what your PR does and why in the PR description.

## Adding a new algorithm

If you're adding a new algorithm from smartcore:

1. Add the variant to the `Algorithm` enum in `src/models/algorithm.rs`.
2. Add training and prediction logic in `src/models/wrapper.rs`.
3. Add tests covering the new algorithm.
4. Update the algorithm table in `README.md`.

## Reporting bugs

Open an issue with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS and Rust version (`rustc --version`)

## License

By contributing, you agree that your contributions will be licensed under the [GPL-3.0 License](LICENSE).
