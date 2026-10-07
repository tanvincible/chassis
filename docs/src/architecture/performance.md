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

* **One thread is not how builds usually run.** On every core (see [Parallel builds](#parallel-builds)),
  Chassis and hnswlib build 2.4 to 5.2 times faster, and on this machine Chassis still leads on SIFT.
* **This is an ARM Mac.** hnswlib's SIMD distance code only targets x86, so here it runs
  compiler-vectorized loops (about 264 ns per 1,536-dim distance, against 149 ns for Chassis's NEON
  code). usearch has ARM SIMD and was still slower here. **On an x86 Linux runner, hnswlib was
  faster than Chassis,** and on an ARM Linux runner the two were even: see
  [On GitHub's runners](#on-githubs-runners).
* **Chassis searches warm memory-mapped pages.** The harness runs an untimed pass after each open; a
  cold first pass after opening was about half as fast on SIFT at ef 16.
* **These Chassis rows are file format 2.** Its files and memory were much larger (peak memory about
  4.2 GB against 0.8–0.9 GB on SIFT and GloVe), from fixed 2,192-byte node records, and GloVe's build
  spent 568 of its 1,430 s in the kernel, moving the graph within the file and remapping it as it
  grew. Format 3 removes both; see [Format 3](#format-3).
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

Each slot takes a 24-byte slot header, its vector, and a level-0 record of `8 + 4 × M0` bytes (136
with the defaults). The few nodes above layer 0 add `4 × M` bytes per extra layer in the upper heap,
about 4 bytes per vector on average. For 1M vectors with 768 dimensions that is 3.07 GB of vectors
and 0.16 GB of everything else.

Segments double in size up to 256 MiB, so the file can be up to one segment larger than the data
written. That unused end is never written; whether it stays sparse depends on the file system (on
APFS, ranges above about 16 MiB did and smaller ones sometimes didn't; ext4 is untested). Nothing is
ever copied to grow the file.

## Format 3

[ADR-0008](../adr/008-format-v3-and-multi-process-readers.md) "Implementation Status" has the full
results. In short, on the same graphs converted from format 2:

* **Search returns identical results,** ids and distances, for every query at every ef on SIFT-1M
  and dbpedia, and is 1.5–12% faster (median paired ratio per ef), from smaller records and a
  smaller working set. Re-measured after the reader changes to the search path: no slower.
* **Files are 4.7× smaller on SIFT-1M** (713 MB against 3,379 MB) and 1.2× on dbpedia (821 MB
  against 982 MB). Converting SIFT-1M took 9.4 s.
* **Opening a 10M-slot file takes under 1 ms;** it only maps the file, so pages load on first
  touch. An index with its own ids builds its id table on the first add or delete in each process:
  at 10M ids, a median of 0.47 s with the file in the page cache and 0.91 s right after copying it,
  on a machine with other load. Filling the hash table is over 90% of that.

Built from scratch on format 3 (one thread, same settings as the table above, with other
applications running, so build times are indicative):

| Dataset | Build | In the kernel | Peak memory | File | Recall at ef 64 / 256 / 512 |
|---------|-------|---------------|-------------|------|-----------------------------|
| SIFT-1M | 527 s | 10.6 s | 1.09 GB | 713 MB | 0.968 / 0.998 / 0.999 |
| dbpedia | 431 s | 6.4 s | 1.22 GB | 821 MB | 0.978 / 0.998 / 1.000 |

Peak memory includes the training vectors the harness holds (0.51 GB for SIFT, 0.61 GB for
dbpedia). Format 2 peaked at about 4.2 GB on SIFT, and GloVe's format 2 build spent 568 s in the
kernel.

## Parallel builds

Measured on 2026-10-07 on the same M5, which has 4 performance and 6 efficiency cores and was in
Low Power Mode, with other applications running. One build at a time, `M` 16, `ef_construction`
200; "every core" is Chassis's `add_batch` and hnswlib's `num_threads=-1`. Search is one thread.

| Dataset | Engine | One thread | Every core | Speedup | ef 64, every core: recall / QPS |
|---------|--------|------------|------------|---------|---------------------------------|
| SIFT-1M | Chassis | 545 s | 110 s | 5.0× | 0.968 / 4,111 |
| | hnswlib | 682 s | 131 s | 5.2× | 0.959 / 2,878 |
| dbpedia, 99k × 1536 | Chassis | 417 s | 176 s | 2.4× | 0.978 / 841 |
| | hnswlib | 596 s | 163 s | 3.7× | 0.970 / 481 |

Graphs built on every core reach the same recall as one-thread builds of either engine (within
0.002 from `ef` 32 up, and 0.007 below). Their QPS varied by up to 40% either way between single
passes, so these runs don't compare search speed. An earlier parallel dbpedia run the same day took 159 s for Chassis and 158 s
for hnswlib, so builds vary by about 10% between runs. The 1,536-dimension build scales less for
both engines; [ADR-0010](../adr/010-parallel-batch-builds.md) has the details.

## On GitHub's runners

From the Benchmark workflow on 2026-10-07: SIFT-1M on GitHub's `ubuntu-latest` (x86) and
`ubuntu-24.04-arm` runners, 4 vCPUs each, one thread, run in the order Chassis, hnswlib, hnswlib,
Chassis. hnswlib was compiled on each runner with `-march=native`. QPS is the median of five
passes; ranges are the two runs of each engine.

| Runner | Engine | Build | ef 64: recall / QPS | ef 128: recall / QPS |
|--------|--------|-------|---------------------|----------------------|
| x86 | Chassis | 614–622 s | 0.968 / 4,250–4,347 | 0.992 / 2,328–2,590 |
| | hnswlib | 386–396 s | 0.960 / 6,751–6,871 | 0.989 / 3,763–3,771 |
| ARM | Chassis | 714–790 s | 0.968 / 3,165–3,406 | 0.992 / 1,811–1,915 |
| | hnswlib | 717–720 s | 0.960 / 3,247–3,322 | 0.989 / 1,827–1,870 |

* **On x86, hnswlib builds about 1.6× faster and searches about 1.4× faster at equal recall.**
  Chassis's lead on the M5 above comes largely from hnswlib's SIMD not targeting ARM. Not yet
  measured: how much of the x86 gap is hnswlib's software prefetching (x86 only), its compiling
  for the exact CPU, or Chassis's per-access address arithmetic.
* **On ARM Linux, the two are about even:** builds within the runs' spread, and Chassis slightly
  ahead in search at equal recall.

## Reproduce

```bash
cargo run --release --example recall -- 20000 128
cargo bench --bench storage_bench
cargo bench --bench distance_bench

# Against hnswlib and usearch (downloads about 1.9 GB of datasets)
pip install numpy h5py pyarrow hnswlib usearch
python bench/ann/prepare.py
cargo run --release --example ann -- bench/ann/data sift-128          # add `batch` for every core
python bench/ann/reference_bench.py hnswlib bench/ann/data sift-128   # add `0` for every core
python bench/ann/reference_bench.py usearch bench/ann/data sift-128
```

Criterion reports go to `target/criterion/`.

The **Benchmark** workflow (`.github/workflows/bench.yml`) runs the Chassis and hnswlib comparison on
GitHub's x86 and ARM Linux runners: start it from the Actions tab, optionally naming Chassis versions
to compare, or add the `benchmark` label to a pull request, which compares its base and head too.
Each job runs every engine forwards then backwards and writes the tables to the run's summary.
Shared runners are noisy, so compare engines within one job rather than across runs.
