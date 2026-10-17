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
