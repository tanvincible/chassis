# Contributing to Chassis

Thank you for looking at this project. Chassis is a small embedded vector index for local semantic search: one file, in your process, no server. It stays small, correct and fast only through careful work.

Whether you are fixing a bug in the Rust core, improving the Python bindings, or helping someone else with a question, your help is appreciated.

If you like what we are doing but do not have the time to write code or documentation, you can still help by starring the repository or mentioning the project to others who might find a local vector index useful.

## I Have a Question

We maintain two primary sources of documentation:
1. **[The Chassis book](docs/src/SUMMARY.md)**: guides, the API and error references, architecture, the file format, and the decision records.
2. **[Python docs](chassisdb/docs/index.md)**: usage guides and API reference for the Python package (`chassisdb`, imported as `chassis`).

Before asking a question, please check these resources. If the answer isn't there, search the existing [issues](https://github.com/tanvincible/chassis/issues) on GitHub.

If you still need help, feel free to [open an issue](https://github.com/tanvincible/chassis/issues/new). Please be specific about which part of the stack you are using (Rust Core, C FFI, or Python).

## How to Contribute

We value stability and correctness above all else. 

By contributing, you agree that your work is your own and that it may be distributed under the project’s MIT and Apache 2.0 licenses.

### Reporting Bugs

A good bug report helps us fix things quickly.

Before reporting a bug, please check the [issue tracker](https://github.com/tanvincible/chassis/issues?q=label%3Abug). When opening a new report, please include:

1. **Component**: Is this a bug in the Python client, the C FFI, or the Rust core?
2. **Reproduction**: A minimal code example (Python script or Rust test case) that triggers the issue.
3. **Environment**: OS (Linux/Mac/Windows), Architecture (x86/ARM), and Python/Rust versions.
4. **Logs/Errors**: Full stack traces or panic messages.

### Security Issues

If you discover a security vulnerability (e.g., memory safety issue in the FFI boundary), please do not report it publicly. Instead, send an email to **[tanvipm19@gmail.com](mailto:tanvipm19@gmail.com)** so it can be handled responsibly.

### Suggesting Enhancements

Chassis is intentionally small and focused. If you have an idea for a new feature, please consider whether it fits the project’s goal: local semantic search in one file, in your process, with no server.

We prefer features that benefit most users rather than specialized functionality that can be built on top of Chassis by the host application.

### Development Setup

Chassis is a monorepo containing three distinct components:

* `chassis-core`: The index itself: storage, graph and search (Rust).
* `chassis-ffi`: The C API (Rust).
* `chassisdb`: The Python package, imported as `chassis` (Python, over the C API with ctypes).

#### 1. Rust Development (Core & FFI)

You will need Rust 1.88 or later.

```bash
# Build the entire workspace
cargo build

# Run all tests (Core + FFI)
cargo test --workspace

# Check for linting errors, as CI does on Linux x86
cargo clippy --all-features --all-targets -- -D warnings
```

On ARM, clippy currently stops at known warnings in the NEON kernels in `chassis-core/src/distance.rs`.

#### 2. Python Development (`chassisdb`)

To work on the Python bindings, you need the C library built first. The package finds it in `target/release`, or wherever `CHASSIS_LIB_PATH` points.

```bash
# 1. Build the shared library
cargo build --release -p chassis-ffi

# 2. Setup Python environment
cd chassisdb
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install in editable mode with dev dependencies (add `docs` to build the docs)
pip install -e ".[dev]"

# 4. Run Python tests
pytest
```

### Style Guidelines

#### Code Formatting

- **Rust**: We use `rustfmt` and `clippy`. Run `cargo fmt` and `cargo clippy` before committing.
- **Python**: We use `black` and `ruff`.

```bash
# Inside chassisdb/
black .
ruff check . --fix
mypy .
```

#### Commit Messages

We use the Conventional Commits specification to keep history readable.

* `feat`: A new feature
* `fix`: A bug fix
* `docs`: Documentation changes
* `build`: Build system or dependency changes
* `refactor`: Code changes that do not alter behavior
* `perf`: Performance improvements
* `test`: Adding or fixing tests
* `chore`: Maintenance tasks

**Example:**
`feat: add buffer flushing support to VectorIndex`

For larger changes, you may include a message body after a blank line with additional context. A git commit template is provided in the [repository](https://github.com/tanvincible/chassis/blob/main/.gitmessage).

### Improving Documentation

Documentation is as important as code.

* **Architecture Docs**: Located in [`docs`](https://github.com/tanvincible/chassis/tree/main/docs) (built with `mdbook`).
* **Python Docs**: Located in [`chassisdb/docs/`](https://github.com/tanvincible/chassis/tree/main/chassisdb/docs) (built with `mkdocs`).

If you find a typo or unclear explanation, please feel free to submit a pull request.

## Final Note

This project exists because we believe in local, reliable software. We are glad you are here to help build it.
