# Performance Guide

Measured on 2026-10-03 to 2026-10-05 on an Apple M5 (10 cores, 24 GiB) running macOS 26.6 on APFS,
with rustc 1.99.0 and release builds, on a machine that was also running other applications. Numbers
on other machines will differ, so the commands to reproduce them are at the end. Search and build
numbers below are for the code after the 33-candidate cap was removed (ADR-0004 amendment).

## Against hnswlib and usearch

From `bench/ann`: the same settings for every engine (`M` 16, `ef_construction` 200), one thread for
building and searching, 1,000 queries per dataset, recall@10 against the datasets' ground truth.
Chassis runs from Rust; hnswlib 0.8.0 and usearch 2.26.3 run through their Python batch APIs with
one thread. usearch built exactly the same graph as hnswlib at these settings (identical results for
every query), so it differs only in speed.

| Dataset | Engine | Build | ef 64: recall / QPS | ef 256: recall / QPS | ef 512: recall / QPS | File |
|---------|--------|-------|---------------------|----------------------|----------------------|------|
| SIFT-1M, 128 dims | Chassis | 567 s | 0.969 / 3,741 | 0.998 / 1,433 | 0.999 / 964 | 3,379 MB |
| | hnswlib | 769 s | 0.960 / 3,904 | 0.997 / 1,098 | 0.999 / 633 | 661 MB |
| | usearch | 938 s | 0.959 / 2,143 | 0.997 / 642 | 0.999 / 361 | 661 MB |
| GloVe, 1.18M × 100 | Chassis | 1,430 s | 0.788 / 3,419 | 0.897 / 1,328 | 0.934 / 678 | 3,289 MB |
| | hnswlib | 988 s | 0.763 / 2,780 | 0.887 / 1,009 | 0.930 / 567 | 649 MB |
| | usearch | 1,536 s | 0.763 / 2,027 | 0.887 / 588 | 0.930 / 264 | 649 MB |
| dbpedia, 99k × 1536 (OpenAI) | Chassis | 420 s | 0.978 / 918 | 0.998 / 288 | 1.000 / 158 | 982 MB |
| | hnswlib | 673 s | 0.970 / 478 | 0.996 / 154 | 0.999 / 81 | 623 MB |
| | usearch | 607 s | 0.970 / 518 | 0.996 / 140 | 0.999 / 70 | 623 MB |

What these numbers do and don't show:

* **One thread is not how the others are used.** hnswlib and usearch build on every core; on this
  machine that would make them roughly 8× faster to build. Chassis builds on one thread.
* **This is an ARM machine.** hnswlib's SIMD distance code only targets x86, so here it runs
  compiler-vectorized loops (about 264 ns per 1,536-dim distance, against 149 ns for Chassis's NEON
  code). On x86 the gap at 1,536 dims may close. usearch has ARM SIMD and was still slower here.
* **Chassis searches warm memory-mapped pages.** The harness runs an untimed pass after each open; a
  cold first pass after opening was about half as fast on SIFT at ef 16.
* **Chassis's file and memory are much larger** (peak memory about 4.2 GB against 0.8–0.9 GB on SIFT
  and GloVe), from its fixed 2,192-byte node records. ADR-0008 proposes the fix.
* **GloVe's build spent 568 of its 1,430 s in the kernel,** moving the graph within the file and
  remapping it as it grows, which ADR-0008 also removes. Chassis's user-space CPU time there, 797 s,
  was below hnswlib's 978 s.
* **Removing the 33-candidate cap made dbpedia's build 19% slower** (420 s against 352 s), in
  exchange for higher recall at every `ef`; on SIFT and the 20k example, build time didn't increase.

## End to End

From
[`examples/recall.rs`](https://github.com/tanvincible/chassis/blob/main/chassis-core/examples/recall.rs):
uniform random vectors in [-1, 1], default `IndexOptions` (`max_connections` 16, `ef_construction`
200), build time including the final `flush()`, 200 queries, recall@10 against brute force.

| Vectors | Dims | Build | `ef_search` | Recall@10 | Mean search |
|---------|------|-------|-------------|-----------|-------------|
| 20,000 | 128 | 7.5–7.6 s (2 runs) | 50 | 0.58 | 112–132 µs |
| | | | 200 | 0.90 | 325–341 µs |
| 20,000 | 1536 | 92.1 s | 50 | 0.37 | 1.2 ms |
| | | | 200 | 0.69 | 3.6 ms |

Uniform random vectors are close to the worst case for HNSW. Real embeddings usually reach higher
recall at the same settings, so measure on your own data before choosing `ef_search`.

## Deletes

Same 20,000 × 128 index with part of it deleted, from the last lines of `examples/recall.rs`
(`ef_search` 50, two runs on 2026-10-05 while the machine was busy, so latency is relative to the
same run's undeleted search, and noisy):

| Deleted | Recall@10 (live vectors) | Search time vs. none deleted | Flush with these deletes |
|---------|--------------------------|------------------------------|--------------------------|
| 0% | 0.58 | 1× | 3.9 ms |
| 10% | 0.59–0.60 | 0.9–1.2× | 17 ms (2,000 deletes) |
| 50% | 0.72–0.74 | 1.1–1.8× | 30–42 ms (8,000 deletes) |

Search still walks through deleted vectors to keep the graph connected, so it slows down as deletes
accumulate; recall rises because it explores further to collect enough live results. Deleted vectors
keep their space until the index is rebuilt.

## Storage

From `storage_bench`, 768-dimensional vectors, in a temp dir on APFS.

| Operation | Time |
|-----------|------|
| Insert, no flush | 17 µs (12–22 µs) |
| Insert, then flush | 3.9 ms |
| 10 inserts, then one flush | 4.1 ms |
| 100 inserts, then one flush | 4.3 ms |
| 1,000 inserts, then one flush | 7.9 ms |
| Read a cached vector | 85 ns |
| Read a vector after reopening | 79 µs |
| Read 1,000 vectors in order | 118 µs |
| Grow an empty file to 1,000 vectors | 6.8 ms |

`flush()` costs milliseconds because it waits for the disk (`F_FULLFSYNC` on macOS, `fsync`
elsewhere). Insert in batches and flush once per batch. A temp dir on tmpfs makes fsync free, so
benchmarks run there understate this cost.

## Distance Kernels

From `distance_bench`, NEON on this machine (AVX2 on x86_64).

| Dims | SIMD | Scalar | Speedup |
|------|------|--------|---------|
| 64 | 8.1 ns | 21.5 ns | 2.7x |
| 128 | 15.3 ns | 59.1 ns | 3.9x |
| 384 | 35.5 ns | 239 ns | 6.7x |
| 768 | 76.4 ns | 613 ns | 8.0x |
| 1536 | 149 ns | 1.39 µs | 9.4x |
| 3072 | 315 ns | 2.88 µs | 9.1x |

SIMD throughput is about 10 billion elements per second at 384 dimensions and up.

## Other Benchmarks

`search_bench`, `link_bench`, `graph_io_bench` and `hnsw_node_bench` are microbenchmarks.
`search_bench` searches hand-wired graphs (single-layer, except the greedy-descent bench) over
vectors with at most 16 non-zero dimensions, not graphs built by `VectorIndex`, so its latencies are
far below real search. Use `examples/recall.rs` for end-to-end numbers.

## Index Size

Every node gets a fixed-size record: `16 + M0 * 8 + (max_layers - 1) * M * 8` bytes, which is 2,192
bytes with the defaults (`M` 16, `M0` 32, `max_layers` 16). About 94% of nodes only use layer 0 (272
bytes of it).

For 1M vectors with 768 dimensions:

- Vectors: 1M × 768 × 4 = 3.07 GB
- Graph: 1M × 2,192 = 2.19 GB
- Slack before the graph: 25% of the vector zone, at least 8 MiB
- Growth headroom: up to 25% of the file

Measured through `VectorIndex::add`: 10,000 × 768 dims is a 64.4 MB file for 52.6 MB of vectors and
graph; 100,000 × 128 dims is 290 MB for 270 MB. Slack that vectors have not reached yet usually
holds an earlier copy of the graph, so it takes disk space. While the graph moves, the file briefly
needs room for one more copy of the graph: about 1.9× the live data at 128 dimensions, 1.5× at 768
and up.

## Reproduce

```bash
cargo run --release --example recall -- 20000 128
cargo bench --bench storage_bench
cargo bench --bench distance_bench

# Against hnswlib and usearch (downloads about 1.9 GB of datasets)
pip install numpy h5py pyarrow hnswlib usearch
python bench/ann/prepare.py
cargo run --release --example ann -- bench/ann/data sift-128
python bench/ann/reference_bench.py hnswlib bench/ann/data sift-128
python bench/ann/reference_bench.py usearch bench/ann/data sift-128
```

Criterion reports go to `target/criterion/`.
