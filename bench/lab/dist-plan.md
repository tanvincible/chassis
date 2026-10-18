# What a distance costs, Chassis and FAISS

Written and committed before any run.

## The question

RUN 9 (`ndis-plan.md`) found that FAISS's HNSW computes no fewer distances than Chassis for 0.95
recall on dbpedia, yet each costs Chassis 1.21 to 1.22 times as long on Zen 3 at 1,536 dimensions
in full precision. What makes a distance cost more: the kernel, the prefetch, or the memory under
the vectors?

The memory became a suspect while this plan was written. GitHub's x86 runners have transparent
huge pages set to `always` (RUN 9's logs), so FAISS's vectors, in anonymous memory, sit on 2 MB
pages. Chassis's sit in a mapped file on 4 KB pages unless `huge_pages` is on, which it wasn't in
RUN 9. A 6 KB vector spans two or three 4 KB pages, and 608 MB of them is far past what the TLB
covers. The ARM runners have THP at `madvise`, so FAISS gets 4 KB pages there; on Neoverse-N2,
Chassis was ahead.

## What is measured

**Part 1: one distance alone.** `distbench.yml`, RUN 1, Chassis at `lab/exp-dist` (RUN 9's
code, `lab/exp-ndis`, plus this lab's harness and `LAB_PF`). Eight jobs on `ubuntu-latest`. On
99,000 random vectors of 1,536 dimensions, nanoseconds a distance, as a search computes them:

* 32 vectors picked at random, a node's neighbors.
* **Chassis's kernel**, as its search runs it (four vectors at a time, inlined into the loop),
  with each vector's first `N` cache lines asked for into L2 first: `pf0`, `pf8`, `pf16`,
  `pf32`, `pf96` (the whole vector). `now` is this CPU's policy: 8 lines on Zen 3, 32 on Zen 4.
  `now-again` is the same arm again, timed last.
* **FAISS's kernel**, `faiss`: its own `fvec_L2sqr_batch_4` at AVX2, called from its wheel's
  `libfaiss.so` (faiss-cpu 1.15.1), four vectors at a time, with nothing asked for ahead, as
  its search does. Its search calls it through a virtual call and a dispatcher; called here
  directly, it pays less per group of four than in a search, which favours it.

Each arm is timed over the vectors in five kinds of memory, each read into memory before any
timing:

| memory | what it stands for |
| --- | --- |
| `cache` | 40 vectors (240 KB), in L2 |
| `anon` | anonymous memory, no advice: FAISS's vectors |
| `anon-4k` | anonymous memory with `MADV_NOHUGEPAGE` |
| `file` | a file mapped shared and writable, no advice: Chassis's default |
| `file-huge` | the same with `MADV_HUGEPAGE` and `MADV_RANDOM`: Chassis's `huge_pages` |

The best of five rounds, the arms in turn within each round. Each memory's share on huge pages,
from `/proc/self/smaps`, is printed beside it.

**Part 2: whole searches.** `incumbents.yml`, RUN 10, Chassis at `lab/exp-dist`. dbpedia, 99,000
vectors of 1,536 dimensions, six jobs on `ubuntu-latest`. One Chassis index, built as in RUN 9,
searched by the lab's Rust harness six ways: `LAB_PF` at 0, 8 and 32 lines, each without and
with `huge_pages` (`pfN`, `pfN-huge`). Before each, the file is put out of memory, so each reads
it back under its own advice. Three rounds; queries a second the median. FAISS through
`incumbents.py`: `faiss`, `faiss-4k` (the same, with `PR_SET_THP_DISABLE`, so its memory gets no
huge pages) and `faiss-again`. Queries a second at 0.95 recall are interpolated linearly between
the two `ef` settings around it, as in RUNs 6 to 9.

Decided on Zen 3, and on Zen 4 if a job draws one, each CPU on its own, the median over its
jobs. Other CPUs are reported, not decided on.

## Checks on the harness

* Both kernels within 1e-4 of an f64 reference on 3,200 distances in each memory; the run
  stops otherwise.
* Pages, part 1: `anon` at least 0.9 on huge pages and `anon-4k` at most 0.05. `file` and
  `file-huge` count as 4 KB or huge only at most 0.05 or at least 0.9; otherwise they are
  reported as they are and not decided on.
* Pages, part 2: from `/proc/self/smaps_rollup` after each arm's searches, `pfN-huge` with at least
  500 MB of file on huge pages (`FilePmdMapped`) and `pfN` with at most 10 MB; `faiss` with at
  least 500 MB of anonymous huge pages and `faiss-4k` with at most 10 MB. An arm that fails is
  reported and not decided on.
* Noise: `now-again` within 5% of `now` in each memory; `faiss-again` within 5% of `faiss` at
  every `ef`. A job that fails one is not decided on.
* Every Chassis arm in part 2 finds the same recall at every `ef`: prefetch and pages don't
  change results.

## Predictions

On Zen 3:

1. Part 1: `now` takes at least 1.15 times as long on `anon-4k` as on `anon`.
2. Part 1: `now` on `file` within 5% of `now` on `anon-4k`.
3. Part 1: on `anon`, `pf0` within 10% of `faiss`.
4. Part 1: in `cache`, `pf0` takes at most 0.95 times `faiss`'s time.
5. Part 2: `pf8-huge` makes at least 1.10 times `pf8`'s queries a second.
6. Part 2: `faiss` makes at least 1.10 times `faiss-4k`'s.
7. Part 2: `pf8-huge` makes at least 0.95 times `faiss`'s.
8. Part 2: no `pfN` makes more than 1.05 times `pf8`'s queries a second, with the same pages.

## The decision

For each CPU decided on, `own` is its own policy (8 lines on Zen 3, 32 on Zen 4). These can hold
together; each gives its own next step, in this order:

* **Pages.** `own-huge` makes at least 0.95 times `faiss`'s queries a second, and `faiss` at
  least 1.05 times `faiss-4k`'s: the lead is the pages. Next: a plan for `huge_pages` on by
  default on Linux, which ADR-0016 kept opt-in, with a run on the cases where it cost (Zen 5 at
  128 dimensions, down to 0.93) before any change.
* **Prefetch.** Some `pfN` at least 1.05 times `own`'s queries a second with the same pages, and
  faster than `now` in part 1 on the matching memory: next, a PR moving that CPU to `N` lines,
  with ADR-0015 amended.
* **Kernel.** In part 1, `faiss` at most 0.90 times `pf0`'s time on `anon` or in `cache`: next,
  a lab of FAISS's loop shape (one sum a vector) against Chassis's (four).
* **None of these**, with `own-huge` still under 0.95 times `faiss`: the cost is outside the
  distance. Next: the search loop's other work (the heap, the visited list), counted.
