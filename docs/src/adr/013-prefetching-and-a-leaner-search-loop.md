# ADR-0013: Prefetching into L2, a Leaner Search Loop, and a Smaller Fill

**Date:** 2026-10-08
**Status:** Proposed

## Context

On GitHub's x86 runners hnswlib searched faster than Chassis, by a factor that changed with the
CPU: 1.4 times on an EPYC 9V74 (Zen 4), about even on a Xeon 8573C, while on an EPYC 9V45 (Zen 5)
and on ARM Chassis led. Its one-thread builds were 1.35 to 1.6 times slower on every x86 machine.
The speedups of 2026-10-07 (prefetching neighbors' vectors, one visited filter per thread) had
helped a great deal on ARM, some on the Xeon, and not at all on Zen 4.

To find out why, an experiment branch ran both engines in one job per machine, with hnswlib
compiled there with `-march=native`: the distance kernels alone, an index small enough to stay in
cache, and SIFT-1M, counting distance computations in both. Several x86 jobs and one ARM job ran
per push, on whatever CPUs GitHub handed out: EPYC 7763 (Zen 3), 9V74 (Zen 4) and 9V45 (Zen 5),
Xeon 8573C (Emerald Rapids) and 8370C (Ice Lake), and Neoverse-N2.

What it showed:

* **Not the kernel.** At 128 dimensions Chassis's AVX2 kernel takes 6.2 ns on Zen 5 and 12.8 ns
  on Zen 3; hnswlib's, with AVX-512 where the machine has it, takes 7.3 and 13.4. At 960 and
  1,536 dimensions they are level. An AVX-512 kernel would gain nothing here.
* **The prefetch.** A node has up to 32 neighbors, and Chassis asked for the first eight cache
  lines of each vector into L1: up to 256 hints in a burst. AMD cores before Zen 5 track 24 misses
  to L1 at a time and drop hints past that; Zen 5 tracks 124. On Zen 4, on Neoverse-N2 and on the
  Xeons, asking for the same lines into L2 instead is better, on the first two by far.
* **Work per neighbor.** Where the index fits in cache, Chassis ran at 0.67 to 0.80 of hnswlib's
  speed. For every neighbor the loop found the vector's slot twice (once to prefetch, once for
  the distance), tested the CPU's features, called a kernel that can't be inlined, and read the
  worst result off a heap.
* **A denser graph.** With the same `M`, Chassis computed 11 to 13% more distances per query than
  hnswlib and got more recall for them (0.968 against 0.960 at `ef = 64`). A selection that kept
  fewer than half a list was filled to the whole list with the nearest of the rest, which on real
  data is nearly every selection. So every list was full, every backlink had to prune one, and
  builds did about twice the pruning.
* **Huge pages.** The x86 runners give anonymous memory transparent huge pages, so hnswlib's index
  sat on 2 MB pages and Chassis's file mapping on 4 KB ones. Turning them off for hnswlib cost it
  6 to 13%.

## Decision

### 1. Vectors are asked for into L2, except on Zen 5

Before computing distances to a node's unvisited neighbors, a search asks for the first eight
cache lines of each one's vector. It asks for them into L2. On AMD processors of CPUID family
0x1A and up, Zen 5 and its successors, it asks into L1. The choice is made once per process.

Eight lines into L2 was the best of what was tried on Zen 4, both Xeons and Neoverse-N2, and on
Zen 3 as good as anything. Only Zen 5 did better with L1, by a tenth or more.

### 2. A candidate's neighbor list is asked for when it is queued

A node's level-0 list is in another array than its vector, so expanding a node began with a miss
that nothing else could overlap. hnswlib asks for the list of the best candidate after each push.
Chassis asks for the list of every candidate it queues.

### 3. Less work per neighbor

The loop locates each unvisited neighbor's vector once and keeps the slice for the prefetch and
the distance; it looks the kernel up once per search and calls it through a pointer; and it keeps
the worst result's distance in a local. Selecting a new node's neighbors takes the distances its
search has just computed, where it recomputed and re-sorted them.

None of this changes a result: on 1.9 million results over five indexes, with deletes, filters and
a reader, ids and distances are the same bit for bit as before.

### 4. A short selection is filled to a quarter of the list, not all of it

Selection keeps a candidate only if it is closer to the node than to every neighbor already kept.
Filling the list with the rest undid that on nearly every node. The HNSW paper and hnswlib have no
fill at all. Chassis keeps a small one: if fewer than a quarter of the list was kept, the nearest
of the rest make up the quarter. Vectors that lie along a line leave only two diverse neighbors
each, a chain that one lost link cuts, and the kill tests, whose vectors do lie on a line, fail
without any fill.

This changes the graphs that new builds produce, not the file format.

## Measurements

All on 2026-10-08, on GitHub's runners (4 vCPUs), SIFT-1M, `M = 16`, `ef_construction = 200`,
search on one thread as the median of five passes, each engine run two or three times per job.
A figure "against hnswlib" is Chassis's queries per second over hnswlib's in the same job, at the
same `ef`, and a range covers `ef` 32 to 256 and every machine of that kind. Runners of the same
model differ, so nothing here compares one job with another.

### Where the hints go

The experiment branch's own loop (decision 3 as a prototype, with decision 2), asking for eight
lines of each vector, against hnswlib:

| CPU | Into L1 | Into L2 |
| --- | --- | --- |
| EPYC 7763 (Zen 3) | 0.87 to 1.01 | 0.85 to 1.02 |
| EPYC 9V74 (Zen 4) | 0.79 to 0.87 | 1.34 to 1.60 |
| EPYC 9V45 (Zen 5) | 1.73 to 1.90 | 1.56 to 1.75 |
| Xeon 8573C | 1.01 to 1.21 | 1.07 to 1.37 |
| Xeon 8370C | 0.87 to 0.89 | 0.86 to 0.93 |
| Neoverse-N2 | 1.48 to 1.66 | 2.30 to 2.78 |

### Before and after

The code before this ADR and after it, in the same job. "Same index" has both search the index
the old code built, so only the search differs; "its own index" has the new code search the index
it built, with the smaller fill. Zen 4 and the Xeon come from earlier runs of the same loop, the
Xeon's with hints into L1; the final policy was not run on either.

Search at a million vectors, against hnswlib:

| CPU (machines) | Before | After, same index | After, its own index |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (4) | 0.73 to 0.79 | 0.88 to 0.93 | 0.94 to 0.99 |
| EPYC 9V74, Zen 4 (2) | 0.66 to 0.75 | 1.49 to 1.63 | 1.53 to 1.68 |
| EPYC 9V45, Zen 5 (1) | 1.40 to 1.43 | 1.61 to 1.68 | 1.66 to 1.74 |
| Xeon 8573C (1) | 0.89 to 0.92 | 0.97 to 1.00 | 1.03 to 1.04 |
| Neoverse-N2 (2) | 1.46 to 1.57 | 2.48 to 2.81 | 2.52 to 2.89 |

Search at 20,000 vectors, which stay in cache, each version on its own index:

| CPU | Before | After |
| --- | --- | --- |
| EPYC 7763 | 0.67 to 0.79 | 0.78 to 0.86 |
| EPYC 9V74 | 0.65 to 0.84 | 0.84 to 0.94 |
| EPYC 9V45 | 0.70 to 0.82 | 0.79 to 0.85 |
| Xeon 8573C | 0.76 to 0.92 | 0.92 to 1.00 |
| Neoverse-N2 | 1.34 to 1.54 | 1.59 to 1.75 |

Builds, in seconds: 100,000 vectors on one thread, then the million on all four vCPUs.

| CPU | hnswlib | Before | After |
| --- | --- | --- | --- |
| EPYC 7763 | 18 to 22, 138 to 149 | 30 to 34, 187 to 198 | 18 to 19, 133 to 139 |
| EPYC 9V74 | 16 to 25, 130 to 158 | 26 to 38, 186 to 251 | 15 to 20, 96 to 123 |
| EPYC 9V45 | 13, 110 | 18 to 19, 93 | 12, 72 |
| Xeon 8573C | 20, 141 | 31, 181 | 19, 138 |
| Neoverse-N2 | 35 to 40, 187 to 192 | 32 to 36, 141 to 149 | 19 to 20, 92 |

Recall@10 at `ef` 32, 64, 128 and 256, over all the machines:

| | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- |
| Before | 0.907 to 0.912 | 0.967 to 0.970 | 0.992 to 0.993 | 0.998 |
| After | 0.905 to 0.909 | 0.966 to 0.967 | 0.991 to 0.992 | 0.997 |
| hnswlib | 0.890 to 0.895 | 0.957 to 0.960 | 0.988 to 0.989 | 0.997 |

* **Search at a million vectors is 1.2 times faster than before on Zen 3 and Zen 5, 2.2 times on
  Zen 4, 1.7 to 1.8 times on Neoverse-N2**, on the same index. It leads hnswlib everywhere
  measured but Zen 3, where it is within 6% on its own index, and Ice Lake, which the final code
  was not run on.
* **In cache Chassis still trails hnswlib on x86**, by up to a fifth. What is left of the work
  per neighbor is the next thing to look at there (ADR-0014).
* **One-thread builds are 1.6 to 1.9 times faster** and level with hnswlib's on x86, where they
  took half as long again. Nearly all of that is the smaller fill.
* **The smaller fill costs about 0.003 of recall at `ef = 32`, 0.002 at 64 and under 0.001
  above**, and on the graphs it builds a search is up to 7% faster.

## What Was Tried and Left Out

* **A rolling window**: asking for the vector a few neighbors ahead while computing a distance.
  Better than L1 hints all at once on Zen 3 and Neoverse-N2, but worse on Zen 5 (1.0 to 1.3
  times hnswlib's speed against 1.7) and far behind L2 hints on Neoverse-N2 (1.7 against 2.5).
* **Reads in place of hints**, which can't be dropped: they stall the loop instead. 0.5 to 0.7 of
  hnswlib's speed on x86.
* **Fewer lines per vector.** In the experiment branch's loop, three lines into L1 was the best
  setting on Zen 3, about level with hnswlib. The shipped loop reaches 0.76 to 0.94 with it, no
  better than with eight lines, in three further runs that changed one thing at a time. The
  cause was not found, so nothing is built on it.
* **An AVX-512 kernel.** The AVX2 kernel is already as fast as hnswlib's AVX-512 one.
* **Link-time optimization** and one codegen unit: nothing on the search loop.
* **hnswlib's rule for a new node**, which selects `M` neighbors on layer 0 where Chassis selects
  up to `M0`. Builds were 5% faster again, but recall per distance computed was worse than with
  `M0` and no fill.

Two things helped and are left for later. Compiling the whole crate for x86-64-v3 gave 3 to 9%,
which a search loop compiled for AVX2 with the kernel inlined should also give. And on Linux 6.17
with ext4, asking for huge pages on the mapping gave 6 to 8% on Zen 3 and Zen 4 once the index
had been read back from disk; without them hnswlib's own index, which gets them on the x86
runners, was 4 to 13% slower.

## Consequences

### Positive

* Search is faster on every CPU measured, and no longer behind hnswlib on Zen 4 or Emerald
  Rapids at a million vectors.
* Builds on x86 are level with hnswlib's on one thread and ahead of it on four.
* No change to results from the search loop, and none to the file format.

### Negative

* The policy is one rule with one exception, fitted to six CPUs. A CPU that wants something else
  gets hints into L2, which was never much worse than the best measured except on Zen 5.
* New builds produce sparser graphs: recall at a given `ef` is lower by up to 0.003, and a caller
  that tuned `ef` to a recall target may have to raise it slightly.
* A vector's lines past the first eight are left to the hardware prefetcher, as before. Nothing
  here was measured above 128 dimensions (ADR-0015 does).
* The measurements are one dataset on shared runners.
