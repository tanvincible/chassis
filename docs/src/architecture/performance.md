# Performance

How fast Chassis searches and builds, against hnswlib and FAISS, with the machines, the data and
the commands to measure it yourself.

## Summary

Measured from 2026-10-08 to 2026-10-10, one thread searching, on GitHub's server runners (AMD EPYC
Zen 3, 4 and 5, Intel Xeon, Arm Neoverse-N2) and an Apple M5. The decision records hold the
details and the commands.

**Against hnswlib, at the same recall**: queries a second, Chassis over hnswlib
([ADR-0017](https://github.com/tanvincible/chassis/blob/main/docs/adr/017-equal-recall-and-long-vectors.md), [ADR-0018](https://github.com/tanvincible/chassis/blob/main/docs/adr/018-half-precision.md)).

| Vectors | x86 | Neoverse-N2 |
| --- | --- | --- |
| 128 dims × 1M | 1.2–2.6× | 3.1–4.4× |
| 960 dims × 200k | 0.86–1.05×; in half precision 1.0–2.1× | 1.65–1.77×; half 2.3–2.5× |
| 1,536 dims × 99k | 0.82–1.13×; in half precision 1.0–1.7× | 1.85–2.33×; half 2.45–2.93× |

These came before distances were computed in groups ([ADR-0021](https://github.com/tanvincible/chassis/blob/main/docs/adr/021-grouped-distances.md)),
which made searches over vectors of a kilobyte or more another 11 to 60% faster out of cache.

**Through Python**: each engine through its Python package on one thread, every query in one call
(Chassis's `search_batch`), queries a second for 0.95 recall, Chassis over the other engine
(2026-10-10, one run, one machine per CPU). With one call per query instead, Chassis lost a fifth to
a third of its speed from Rust, 15 to 38 µs a query; in one call it is within 5% of it. Since #109
`search` spends half as long outside the library: on 2026-10-17, against one call for all, it took
13 to 26 µs a query where the package before took 30 to 61.

| Vectors | Over hnswlib | Over FAISS's HNSW |
| --- | --- | --- |
| 128 dims × 1M | Zen 3 1.13×, Xeon 1.33×, N2 3.46× | Zen 3 0.98×, Xeon 1.22×, N2 1.69× |
| 128 dims × 1M, half | Zen 3 1.61×, Xeon 2.01×, N2 3.63× | Zen 3 1.40×, Xeon 1.85×, N2 1.77× |
| 1,536 dims × 99k | Zen 4 1.26×, Xeon 1.05×, N2 2.92× | Zen 4 0.78×, Xeon 0.95×, N2 1.09× |
| 1,536 dims × 99k, half | Zen 4 1.91×, Xeon 1.55×, N2 2.92× | Zen 4 1.19×, Xeon 1.40×, N2 1.09× |

The same run on the Xeon, across search settings:

![Recall@10 against queries a second on one thread, for Chassis in full and half precision, FAISS's HNSW, hnswlib and usearch, on SIFT and dbpedia](recall-qps.svg)

[Comparison](./comparison.md) has Annoy, sqlite-vec and LanceDB too, with build times, file sizes
and cold starts.

**From a cold start**
([ADR-0019](https://github.com/tanvincible/chassis/blob/main/docs/adr/019-warm.md), Apple M5, 99,000
× 1,536 dims, the file not in memory): the first result came 0.11 s after the process started.
hnswlib, which loads its file first, gave its first at 0.5 to 0.8 s. With `warm()`, the hundredth
result came at 0.37 s instead of 1.9 s.

## Builds

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
passes, so these runs don't compare search speed. An earlier parallel dbpedia run the same day
took 159 s for Chassis and 158 s for hnswlib, so builds vary by about 10% between runs. The
1,536-dimension build scales less for both engines;
[ADR-0010](https://github.com/tanvincible/chassis/blob/main/docs/adr/010-parallel-batch-builds.md)
has the details.

A one-thread build of SIFT-1M peaked at 1.09 GB of memory, 0.51 GB of it the vectors the harness
holds, and one of dbpedia at 1.22 GB with 0.61 GB.

## Size

Each slot takes a 24-byte slot header, its vector, and a level-0 record of `8 + 4 × M0` bytes (136
with the defaults). The few nodes above layer 0 add `4 × M` bytes per extra layer in the upper heap,
about 4 bytes per vector on average. For 1M vectors with 768 dimensions that is 3.07 GB of vectors
and 0.16 GB of everything else; half precision halves the vectors. Built as above, SIFT-1M's file is
713 MB and dbpedia's 821 MB.

Segments double in size up to 256 MiB, so the file can be up to one segment longer than the data
written. That unused end is never written, and takes no disk where the file system keeps holes:
ext4, XFS and btrfs keep all of them, and APFS those of 16 MiB or more. On APFS, files as long as
SIFT-1M's and dbpedia's took 688 and 630 MB of disk. A backup keeps the holes too.

## Deletes

A search still walks through deleted vectors, to keep the graph connected, so deletes cost speed
and space but not recall. With half the vectors deleted, searches ran at about 70% of their speed,
and with 90% deleted at a fifth. `compact()` gives the speed back
([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)).

## History

Measurements of file format 2, before v0.7.0, are in the decision records that each change came
with, and in [this page's history](https://github.com/tanvincible/chassis/commits/main/docs/src/architecture/performance.md).

## Reproduce

```bash
# Against hnswlib and usearch (downloads about 1.9 GB of datasets)
pip install numpy h5py pyarrow hnswlib usearch
python bench/ann/prepare.py
cargo run --release --example ann -- bench/ann/data sift-128          # add `batch` for every core
python bench/ann/reference_bench.py hnswlib bench/ann/data sift-128   # add `0` for every core
python bench/ann/reference_bench.py usearch bench/ann/data sift-128
```

[Benchmarking](../dev/benchmarking.md) has the other harnesses, and how to measure fairly.
