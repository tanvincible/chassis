# Testing

This page shows how to run Chassis's tests, what each suite checks, and what CI runs, so you know
which tests cover a change and where a new one belongs.

## Running

```bash
# Every Rust test in the workspace
cargo test --workspace --all-features

# One suite
cargo test -p chassisdb --test crash_tests

# The random-history test with many more histories (12 by default)
CHASSIS_MODEL_SEEDS=2000 cargo test -p chassisdb --release --test model_tests
```

Run the slower suites (`crash_tests`, `model_tests`, `batch_tests`) in release when you run them
often: a debug build computes distances many times slower.

The Python package, the MCP server and the LangChain integration test against the C library in
`target/release`. The Python package's tests need its dev extras, which include `pytest-cov`:

```bash
cargo build --release -p chassis-ffi
pip install -e "./chassisdb[dev]"

(cd chassisdb && CHASSIS_LIB_PATH=../target/release PYTHONPATH=. pytest)
(cd chassisdb-mcp && pip install -e ../chassisdb -e . && python -m pytest tests)
(cd integrations/langchain-chassisdb && pip install -e ../../chassisdb -e . && python -m pytest tests)
```

## Suites

The Rust suites are in `chassis-core/tests`; each file says at its top what it checks.

| Suite | What it checks |
| --- | --- |
| `index_integration` | `VectorIndex` end to end: open, add, delete, search, flush, reopen |
| `storage_tests` | The file: the lock, growth, commits and reopening |
| `graph_io_tests`, `link_tests` | Neighbor lists on disk, and linking with the diversity heuristic |
| `search_tests` | Search: results, ordering, edge cases |
| `batch_tests` | `add_batch` on every core builds a graph as good as one built a vector at a time |
| `filter_tests` | Filtered search returns only allowed ids, and the nearest of them |
| `metric_tests` | Cosine ranking and distances match brute force; the metric is the file's |
| `half_precision_tests` | Half precision finds what brute force over the rounded vectors finds |
| `compaction_tests` | `compact()` keeps every vector under its id and drops the deleted ones |
| `migration_tests` | Opening a format 1 or 2 file migrates it to format 3 |
| `reader_tests` | Readers in other processes, while a writer changes the file |
| `crash_tests` | A writer killed at random points: every reopen shows exactly the last flush |
| `model_tests` | Random sequences of every write, checked against a model of what the index should hold |
| `error_tests` | Each mistake gets its error kind, its values and a `help:` line |
| `warm_tests`, `huge_pages_tests`, `fork_tests` | `warm`, huge pages, and a forked child dropping an index its parent was reading in |
| `compile_fail` | Code that would let a vector outlive its borrow doesn't compile (`tests/ui`) |

Unit tests sit next to the code they test. Among them, `power_loss.rs` stands in for a power loss
at every fsync of every operation. It builds the files a power loss could leave, with each 512-byte
sector as last synced, as last written, or torn between the two, and checks that each one reopens
to the last flush or the one in progress.

The Python tests are in `chassisdb/tests`: the bindings, every error, and the code agents copy,
which they run: the agent skill's Python blocks and the examples.

## CI

| Workflow | What it runs |
| --- | --- |
| CI | `cargo test --all-features` on Linux, macOS and Windows, with stable Rust and 1.88; the Python tests on Python 3.11 with numpy 1.26 and on 3.14; the MCP and LangChain tests; `cargo fmt`, `cargo clippy -D warnings`, `cargo doc`; ruff, black and mypy |
| MSRV | `cargo check` and `cargo test` on Rust 1.88 |
| Coverage | `cargo llvm-cov` on Linux |
| Build | The C library and the Python wheels for Linux x86-64 and ARM64, macOS and Windows |
| Benchmark | Each pull request against main, on Linux x86-64 and ARM64 ([Benchmarking](./benchmarking.md)) |

`compile_fail` runs only on stable: it compares rustc's error text, which changes between releases.

## Writing tests

- Put each index in a temporary directory (`tempfile::tempdir()`), never a fixed path.
- Make random data deterministic: seed it, as `model_tests.rs` does with its own xorshift.
- To test a crash, run the writer in a child process and kill it, as `crash_tests.rs` does.
- When a test checks recall, which varies with thread timing, check it over many queries with a
  margin, not one query exactly.
