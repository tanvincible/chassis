# Benchmarking

This page shows how to measure Chassis: against other engines on standard datasets, one feature at
a time, on every pull request, and down to single functions. The numbers these produce, with the
machines and dates, are on [Performance](../architecture/performance.md).

## Datasets

`bench/ann` measures Chassis against hnswlib and usearch on the ANN benchmark datasets, with the
same settings for every engine: `M` 16, `ef_construction` 200, recall@10 against queries a second
on one thread.

```bash
pip install numpy h5py pyarrow hnswlib usearch
python bench/ann/prepare.py bench/ann/data sift-128     # also glove-100, dbpedia-openai3-1536

cargo run --release --example ann -- bench/ann/data sift-128          # an add per vector
cargo run --release --example ann -- bench/ann/data sift-128 batch    # add_batch on every core
python bench/ann/reference_bench.py hnswlib bench/ann/data sift-128
python bench/ann/reference_bench.py usearch bench/ann/data sift-128
```

## Features

Each of these examples measures one thing, and names the decision record it was written for:

| Example | What it measures |
| --- | --- |
| `recall` | Build time, recall@10 and search latency on random vectors: `-- [count] [dims]` |
| `filtered` | Filtered search latency and recall on a dataset ([ADR-0009](https://github.com/tanvincible/chassis/blob/main/docs/adr/009-filtered-search.md)) |
| `churn` | What deletes cost, and what `compact()` gives back ([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)) |
| `lost_adds` | What a crash that loses adds does to the vectors committed before it ([ADR-0012](https://github.com/tanvincible/chassis/blob/main/docs/adr/012-rolling-back-lost-adds.md)) |
| `undo_cost` | What saving committed vectors' lists costs ([ADR-0012](https://github.com/tanvincible/chassis/blob/main/docs/adr/012-rolling-back-lost-adds.md)) |

Run one with `cargo run --release --example <name> -- <arguments>`; the top of each file says
what it takes.

## Pull requests

Every pull request is measured against main on Linux x86-64 and ARM64, on SIFT (128 dimensions,
100,000 vectors) and dbpedia (1,536 dimensions, 20,000 vectors, in full and half precision).
`bench/compare.py` builds both versions of `chassis-core/examples/bench.rs` and runs them by turns
on one machine, so the machine's speed cancels out. It posts the result as a comment, and the check
fails if anything got markedly slower or lost recall.

## Functions

The criterion benchmarks in `chassis-core/benches` time single functions on small, hand-built
inputs: distances, storage writes and reads, linking, graph reads and search. They are for
comparing two versions of one function, not for numbers to publish.

```bash
cargo bench -p chassisdb --bench distance_bench                 # also storage_bench, link_bench,
                                                               # search_bench, graph_io_bench,
                                                               # hnsw_node_bench
cargo bench -p chassisdb --bench search_bench -- high_ef        # only the groups matching high_ef
cargo bench -p chassisdb --bench search_bench -- --save-baseline before
cargo bench -p chassisdb --bench search_bench -- --baseline before
```

The last two compare a change with the code before it. Criterion writes HTML reports to
`target/criterion`.

## Measuring

- Compare at equal recall. A faster search that finds less hasn't won anything
  ([ADR-0017](https://github.com/tanvincible/chassis/blob/main/docs/adr/017-equal-recall-and-long-vectors.md)).
- Use release builds, and a quiet machine: close what you can, and run each side more than once.
- Say whether the index was in memory. On macOS a file you just copied isn't, so a first run reads
  it from disk.
- With every number, give the machine, the data and the date, as [STYLE.md](https://github.com/tanvincible/chassis/blob/main/STYLE.md)
  asks.
