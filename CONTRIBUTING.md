# Contributing

Thank you for looking at Chassis. It is an embedded vector index for local semantic search: one
file, in your process, no server. It stays small, correct and fast only through careful work, so
this page is specific about what a change needs before it can be merged.

You don't have to write code to help. A clear bug report helps. So does answering someone's
question, fixing a sentence that confused you, or mentioning Chassis to someone who needs a local
vector index.

Everyone here follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

Start with the docs:

- [The Chassis book](docs/src/SUMMARY.md): the guides, the API and error references, how Chassis
  works, and the file format.
- [The Python docs](chassisdb/docs/index.md): the `chassisdb` package.
- [The decision records](docs/adr): why Chassis works the way it does, with the measurements
  behind each choice.

If the answer isn't there, search the [issues](https://github.com/tanvincible/chassis/issues). If
nobody has asked yet, [open one](https://github.com/tanvincible/chassis/issues/new), and say
whether you are using the Rust crate, the C API or the Python package.

## Bugs

A bug we can run gets fixed first, so the most useful thing you can send is a short script or test
that shows it. Include the vectors it needs, or the code that makes them. Please also tell us:

- Which part you are using, and its version.
- What you expected, and what happened instead, with the full error or stack trace.
- Your operating system, your CPU (x86-64 or ARM64), and your Rust or Python version.
- The file system the index is on. Some bugs happen on only one, such as a network share.

If you think the index file is damaged, keep a copy. It is often the fastest way to the cause. An
index holds your vectors, so think about that before you attach it to a public issue.

## Security

Don't report a vulnerability in a public issue. Email
[tanvipm19@gmail.com](mailto:tanvipm19@gmail.com) instead. That includes a memory-safety bug at the
C boundary, and a crafted file that makes Chassis read or write memory it shouldn't.

## Features

Chassis is small on purpose. It does local semantic search in one file, in your process. It is not
a database server, a distributed system or a metadata store, and the
[non-goals](docs/src/introduction.md#non-goals) say why. [Limits](docs/src/guide/limits.md#poor-fits)
says where it is a poor fit, and [ROADMAP.md](ROADMAP.md) lists what is planned.

Before you build a feature, open an issue that says what you need it for. That way you find out
whether it fits before you spend your time on it. If your application could build it on top of
Chassis, it most likely belongs there.

## Setup

The repository has five parts:

- `chassis-core/`: the `chassisdb` crate, which is the index itself (Rust)
- `chassis-ffi/`: the C API (Rust)
- `chassisdb/`: the Python package, over the C API with ctypes
- `chassisdb-mcp/`: the MCP memory server (Python)
- `integrations/langchain-chassisdb/`: the LangChain vector store (Python)

### Rust

`rust-toolchain.toml` pins stable Rust, and rustup installs it on the first build. The oldest Rust
that builds Chassis is 1.89.

```bash
cargo build
cargo test --workspace --all-features
cargo fmt --all
cargo clippy --all-features --all-targets -- -D warnings
```

Run the last two before you push: CI fails on an unformatted file or a single clippy warning.

### Python

The Python package loads the C library, so build that first. From a checkout, the package finds it
in `target/release` on its own. If yours is somewhere else, point `CHASSIS_LIB_PATH` at it.

```bash
cargo build --release -p chassis-ffi
cd chassisdb
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

CI also runs these from `chassisdb/`, so run them before you push:

```bash
ruff check chassisdb tests examples
black --check chassisdb tests examples
mypy chassisdb
```

[Testing](docs/src/dev/testing.md) has the commands for the MCP server and the LangChain store, and
what each suite checks. [Building](docs/src/dev/building.md) covers each platform and
cross-compiling.

## Pull requests

A typo or a broken link needs no ceremony. Open the pull request.

Anything larger has to meet these before it can be merged:

- A fix comes with a test that fails without it. [Testing](docs/src/dev/testing.md#suites) says
  which suite it belongs in.
- A speed-up holds at equal recall, because a faster search that finds less hasn't won anything.
  CI benchmarks each pull request that touches the core against main, on x86-64 and ARM64, and
  fails it if anything got markedly slower or lost recall
  ([Benchmarking](docs/src/dev/benchmarking.md)).
- Files written by earlier releases still open. Formats 3 and 4 are frozen, so a format change
  either adds something older releases can ignore, or raises the version so they refuse the file
  instead of misreading it.
- A design choice has a decision record in `docs/adr`, with the measurements behind it.
- A new error says what was wrong, with the values, then a `help:` line saying what to do instead.
  It is listed in [Errors](docs/src/guide/errors.md).
- Docs follow [STYLE.md](STYLE.md): every number names its machine, data and date, and every
  example runs as written.

## Commits

Titles follow [Conventional Commits](https://www.conventionalcommits.org): a type, a colon, then
what changed. The types are `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `build`, `ci` and
`chore`. A `!` after the type marks a breaking change.

Write the title as what changed for someone using Chassis, not what you did to the code:

```text
fix: a full disk is an error with what to do, not a crash, on Linux too
fix: on a Windows network share, adds and backups no longer fail with "Access is denied"
```

Name the decision record when there is one, as in `(ADR-0026)`. The why goes in the body.

## License

By contributing, you agree that your work is your own and that it may be distributed under the
project's MIT and Apache 2.0 licenses.

## Thanks

This project exists because we believe in local, reliable software. We are glad you are here to
help build it.
