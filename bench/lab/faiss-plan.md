# FAISS at full precision, and the Python call

Written and committed before any run.

## The questions

1. In `incumbents.yml` RUN 7, FAISS's HNSW searched 1,536-dimension vectors in full precision at
   1.28 times Chassis's speed on Zen 4 and 1.05 times on a Xeon 6973P-C, and at 0.92 times on
   Neoverse-N2. FAISS picks a build for the CPU, AVX-512 where the CPU has it; Chassis's kernels stop
   at AVX2 and FMA. Is AVX-512 FAISS's lead?
2. How long does `search` spend outside the library on server CPUs, before and after
   `perf/python-call` (4fcddd5), which halved it on an Apple M5 (10.8 µs to 4.9)?

## What is measured

RUN 8 of `incumbents.yml`, Chassis from `perf/python-call`. dbpedia, 99,000 vectors of 1,536
dimensions, on four x86 jobs and one ARM; SIFT, 1M of 128, on two x86 and one ARM. The x86 jobs
land on whatever CPUs GitHub gives them. Engines, one after another in this order:

- `chassis-py-old`: main's Python package (0.9.1) on the branch's C library, through `PYTHONPATH`.
  The C library is the same code as main's.
- `chassis-py`, `chassis-py-batch`, `chassis-py-half`, `chassis-py-half-batch`, `hnswlib`.
- `faiss`; `faiss-avx2`, with `FAISS_SIMD_LEVEL=AVX2` (and `FAISS_OPT_LEVEL=avx2` for a build
  without dynamic dispatch); `faiss-again`, the first one again, to see the runner drift.

**Speed** is queries a second at 0.95 recall, interpolated linearly between the two `ef` settings
around it, as in RUNs 6 and 7. **The time outside the library** is, at each `ef`,
`1/QPS(chassis-py) - 1/QPS(chassis-py-batch)`: the same file, the same queries and the same
results, one call a query against one call for all. Its median over the four `ef` settings is
reported; the same for `chassis-py-old`. It includes the harness's own handling of each
query's results, the same for both.

## Checks on the harness

- Each FAISS engine logs the SIMD level it runs at. `faiss` has to name AVX512 on a CPU whose flags
  have `avx512f`, and `faiss-avx2` has to name AVX2. A job where either fails is left out of
  question 1.
- `faiss-again` has to be within 5% of `faiss` at every `ef`; a job where it isn't is noisy and
  left out of question 1.
- `chassis-py-old` has to log main's package and `chassis-py` the branch's, and they have to find
  the same recall at every `ef`.

## Predictions

1. On every x86 job with AVX-512, at 1,536 dimensions, `faiss` makes at least 1.10 times
   `faiss-avx2`'s queries a second.
2. On those jobs, `chassis-py-batch` makes at least 0.95 times `faiss-avx2`'s.
3. On x86 jobs without AVX-512, `faiss` and `faiss-avx2` are within 5%: the same level.
4. On every job, `chassis-py` spends at most 0.6 times what `chassis-py-old` spends outside the
   library.

## The decision

- **1 and 2 hold**: AVX-512 is FAISS's lead. Next: AVX-512 kernels for full and half precision,
  with an ADR, measured against this run. The intrinsics need Rust 1.89, one past the MSRV.
- **1 holds, 2 doesn't**: AVX-512 is part of it. The kernels still go ahead; the rest is in the
  search loop, and the next lab profiles it.
- **1 fails**: AVX-512 isn't it, and there is no kernel work. The next lab compares the two search
  loops with performance counters.
- **4** gives the numbers the PR and the docs quote. It doesn't hold the PR back: its results are
  identical to main's, and it was faster on the M5.
- With no job on an AVX-512 CPU, question 1 has no answer, and the run is repeated.

## Results

RUN 8, [run 39062936133](https://github.com/tanvincible/chassis/actions/runs/39062936133), Chassis
at `perf/python-call` 4fcddd5, FAISS 1.15.1, which picks its SIMD level as it starts. The x86 jobs
drew five AMD EPYC 7763 (Zen 3, no AVX-512) and one EPYC 9V74 (Zen 4, AVX-512): dbpedia shard c.
No Xeon. `run8_report.py` (scratch) interpolates queries a second at 0.95 recall.

**Checks.** On the Zen 4 job `faiss` ran at AVX512_VPOPCNT and `faiss-avx2` at AVX2; on Zen 3 both
ran at AVX2. `faiss-again` was within 5% of `faiss` at every `ef` on every x86 job but dbpedia b
(8.7% at `ef` 256), which is noisy; it has no AVX-512 and only counts toward prediction 3.
`chassis-py-old` loaded main's package and `chassis-py` the branch's, with the same recall at every
`ef` on every job. On Neoverse-N2, `faiss-avx2` ran at NONE: there is no AVX2 there.

| Job | CPU | faiss / faiss-avx2 | batch / faiss | batch / faiss-avx2 | half batch / faiss |
| --- | --- | --- | --- | --- | --- |
| dbpedia c | Zen 4 | 1.026 | 0.741 | 0.761 | 1.138 |
| dbpedia a, b, d | Zen 3 | 0.996, 1.006, 1.016 | 0.853, 0.793, 0.839 | 0.849, 0.798, 0.853 | 1.127, 1.006, 1.089 |
| SIFT a, b | Zen 3 | 1.009, 1.002 | 1.048, 1.007 | 1.057, 1.009 | 1.458, 1.461 |
| dbpedia | N2 | (NONE) | 1.062 | | 1.242 |
| SIFT | N2 | (NONE) | 1.552 | | 1.660 |

Time outside the library, median over `ef`, in µs a query (`1/QPS(per query) - 1/QPS(batch)`,
harness included), and one call a query as a share of one call for all:

| Job | CPU | main's package | this one | share, main's | share, this one |
| --- | --- | --- | --- | --- | --- |
| SIFT a | Zen 3 | 40.8 | 19.4 | 0.72 | 0.86 |
| SIFT b | Zen 3 | 29.8 | 13.3 | 0.76 | 0.89 |
| SIFT | N2 | 60.9 | -6.9 | 0.57 | 0.89 |
| dbpedia c | Zen 4 | 46.1 | 18.9 | 0.92 | 0.95 |
| dbpedia | N2 | 49.0 | 26.3 | 0.90 | 0.94 |
| dbpedia a, b, d | Zen 3 | 13.7, 7.8, -21.9 | 12.9, 30.9, -18.5 | 0.91, 1.00, 0.98 | 0.95, 0.92, 1.00 |

**Predictions.**

1. **Failed.** On the one AVX-512 job, `faiss` made 1.026 times `faiss-avx2`'s queries a second.
2. **Failed.** `chassis-py-batch` made 0.761 times `faiss-avx2`'s: FAISS at AVX2 is 1.31 times
   Chassis there.
3. **Held.** Without AVX-512, `faiss` and `faiss-avx2` were within 1.6% on all five jobs.
4. **Held where the run can tell, 0.41 to 0.54** on SIFT Zen 3 (twice), dbpedia Zen 4 and dbpedia
   N2. Not on dbpedia Zen 3: a search there takes about 350 µs, and the per-`ef` estimates swing
   from -119 to +105 µs, so runner noise swamps a 20 µs difference. On SIFT N2 this package's
   estimate is below zero.

## What it means

- **AVX-512 isn't FAISS's lead.** It gives FAISS 2.6% on Zen 4, and FAISS leads at 1,536
  dimensions in full precision on Zen 3 too, 1.17 to 1.26 times, with no AVX-512 there. At 128
  dimensions Chassis and FAISS are even on Zen 3 (1.01 to 1.05), and on N2 Chassis leads. By the
  decision rule there is no kernel work. The lead is in what each search does per query at long
  vectors.
- **Next**, before any performance counters, which GitHub's VMs may not expose: distances computed
  per query at 0.95 recall, Chassis's from the lab's counters and FAISS's from `hnsw_stats.ndis`.
  More distances means the graph or the search order; the same number means each distance costs
  Chassis more at 6 KB a vector: memory, prefetch or the kernel.
- **Half precision** beats FAISS's full precision on every x86 job, 1.01 to 1.46 times.
- **The Python package**: one call a query at 128 dimensions kept 0.86 to 0.89 of a batch's speed,
  against 0.57 to 0.76 before; the time outside the library about halved, as on the M5. #109
  quotes 13 to 26 µs.
