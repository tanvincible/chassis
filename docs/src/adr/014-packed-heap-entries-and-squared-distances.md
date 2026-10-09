# ADR-0014: Packed Heap Entries, Squared Distances, and a Search Loop That Inlines

**Date:** 2026-10-09
**Status:** Proposed

## Context

ADR-0013 left one case behind: where the index fits in cache, Chassis searched slower than
hnswlib on x86. Measured again on 2,000 and 20,000 SIFT vectors it ran at 0.79 to 0.87 of
hnswlib's speed on Zen 3, Zen 4 and Zen 5, and led only from 100,000 vectors up.

The experiment branch of ADR-0013 ruled out what it could with both engines in one job:

* **Not more work.** Chassis computed up to 3% more distances per query than hnswlib, expanded
  the same number of nodes, and looked at 5 to 12% more neighbors.
* **Not huge pages.** Without them hnswlib was as fast at 2,000 vectors and 2 to 6% slower at
  20,000.
* **Not the prefetch policy.** One line or eight, into L1 or L2, or no hints at all: 0.78 to
  0.92 of hnswlib's speed on the three AMD processors.
* **Not the kernel.** It took 13 to 17% of Chassis's search time.

So both engines were sampled with `perf` on the CPU clock, about a thousand samples a second, and
the samples were laid over the instructions. Shares of search time at 20,000 vectors and
`ef = 256`, with decision 1 already in place:

| | EPYC 9V45 (Zen 5) | Xeon 6973P-C |
| --- | --- | --- |
| The two heaps | 28% | 21% |
| Marking neighbors visited | 13% | 6% |
| The distance kernel | 13% | 17% |
| Reading a node's neighbor ids | 9% | 5% |
| Hints for vectors | 9% | 23% |
| Locating a vector | 8% | 7% |
| The square root and the comparison after it | 5% | 5% |
| Sorting the results | 4% | 4% |
| Hints for neighbor lists | 3% | 5% |
| Everything else | 8% | 7% |

What stood behind the rows:

* **The heaps.** An entry was a slot and a distance, 16 bytes, compared with `f32::total_cmp`,
  which rewrites both floats' bits before every comparison. `BinaryHeap::pop` was not inlined. A
  candidate nearer than the worst result was pushed and the worst then popped: one walk up the
  heap, one to the bottom and one back up.
* **Visited.** The filter lists the words it sets a bit in, so that the next search clears only
  those. Whether a word is still empty was a branch, and on an index of thousands to hundreds of
  thousands of vectors it goes either way about as often.
* **Locating a vector.** The lookup formats an error message when a slot is out of bounds. That
  made it too large to inline, so every neighbor cost a call that returned through memory.
* **The square root** sat between each distance and the branch that depends on it.
* **The hints.** The loop issuing them worked out its bounds again for every neighbor and chose
  L1 or L2 per line. On the Xeon the hints themselves were slow to retire.
* **The kernel** was the smaller part in Chassis and the larger in hnswlib: 9 ns per distance
  against 21 ns on the Xeon at 2,000 vectors and `ef = 32`. hnswlib's kernel waits on the
  vector's cache misses; Chassis has asked for the vectors by then and pays for the hints
  instead.

A profile of the loop after decisions 1 to 6 showed one more thing, at the other end of the
scale. At a million vectors and `ef = 32` the descent through the upper layers took a tenth to a
fifth of the search. It asked for nothing ahead, so every distance waited on its own cache
misses, and its kernel sat behind a closure that was not compiled for AVX2 and so was called, not
inlined.

## Decision

### 1. The search loops are compiled once per kernel

`Kernel` is a trait with two implementations: `Avx2`, and `Portable` (NEON on aarch64, plain
arithmetic elsewhere). The layer-0 loop and the greedy descent are generic over it and inlined
into a wrapper compiled for AVX2 and FMA, which a search enters after one feature test. The
kernel inlines into the loop and the loop itself may use AVX2. ADR-0013 expected 3 to 9% of this
from what compiling the whole crate for x86-64-v3 gave. It gave about 1%.

### 2. Lookup errors are built out of line

The out-of-bounds and bad-record errors are built in functions marked cold. The lookups shrink to
a few instructions and inline into the loop.

### 3. A heap entry is one integer

A node and its distance are packed into a `u64`: the distance's bits above the slot. A distance
is never negative, so its bits order as its value does, and the integers order by distance, then
slot. An entry is half the size, a comparison is one instruction, and nothing is left to call.

A candidate nearer than the worst of `ef` results replaces it at the top of the heap and sinks
from there, one walk in place of three. The results are sorted as integers before they are
unpacked.

### 4. Distances stay squared inside a search

The kernels return the sum of squares, which orders vectors as the distance does. The root is
taken of the results a layer search returns, `ef` of them, not of every distance it computes.

### 5. The visited filter lists a word without a branch

It writes the word's index at the end of its list every time and advances the end only if the
word was empty.

### 6. A search works out its hints once

How many lines of a vector to ask for, and how many of them into L1, is settled before the loop.
Asking is two counted loops with nothing to decide inside.

### 7. The upper layers ask for vectors too

The greedy descent asks for every unvisited neighbor's vector before it computes a distance, as
layer 0 does, and its kernel is inlined.

### What changes in results

* Results at the same distance come back in slot order. They came back in whatever order the
  heap left them.
* A NaN distance is farther than every other, whatever its sign. `total_cmp` put a negative NaN
  first.
* A candidate nearer than the worst result is now taken even when the two distances' roots round
  to the same float. On five indexes of 5,000 to 60,000 vectors, with deletes, filters and a
  reader, 96,000 result lists came out the same as before but for seven, which differ in the
  order of two results at the same distance. At a million vectors recall moved by under 0.001,
  upward in all but one case.
* Builds run the same search, so the graphs they produce can differ where candidates tie. The
  file format does not change.

## Measurements

On 2026-10-08, on GitHub's runners (4 vCPUs), SIFT, `M = 16`, `ef_construction = 200`, search on
one thread as the median of three passes, each engine run three times per job, every version
searching the same index. A figure "against hnswlib" is Chassis's queries per second over
hnswlib's in the same job at the same `ef`, with hnswlib compiled there with `-march=native`; a
range covers `ef` 32 to 256 and every machine of that kind. Nothing here compares one job with
another.

### Before and after

The code of ADR-0013 and the code of this one, against hnswlib, and the second over the first.

2,000 vectors:

| CPU (machines) | Before | After | Speedup |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (6) | 0.80 to 0.88 | 1.10 to 1.17 | 1.31 to 1.38 |
| EPYC 9V74, Zen 4 (3) | 0.82 to 0.89 | 1.12 to 1.20 | 1.34 to 1.37 |
| EPYC 9V45, Zen 5 (2) | 0.81 to 0.88 | 1.08 to 1.15 | 1.32 to 1.40 |
| Xeon 6973P-C (1) | 0.90 to 1.02 | 1.13 to 1.30 | 1.25 to 1.29 |
| Neoverse-N2 (2) | 1.23 to 1.46 | 1.45 to 1.74 | 1.16 to 1.21 |

20,000 vectors:

| CPU (machines) | Before | After | Speedup |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (6) | 0.79 to 1.00 | 1.15 to 1.43 | 1.38 to 1.46 |
| EPYC 9V74, Zen 4 (3) | 0.84 to 1.11 | 1.19 to 1.58 | 1.36 to 1.42 |
| EPYC 9V45, Zen 5 (2) | 0.77 to 0.87 | 1.08 to 1.20 | 1.36 to 1.53 |
| Xeon 6973P-C (1) | 0.92 to 1.03 | 1.11 to 1.27 | 1.20 to 1.23 |
| Neoverse-N2 (2) | 1.57 to 1.86 | 1.85 to 2.28 | 1.17 to 1.23 |

100,000 vectors:

| CPU (machines) | Before | After | Speedup |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (6) | 1.08 to 1.25 | 1.29 to 1.58 | 1.15 to 1.30 |
| EPYC 9V74, Zen 4 (3) | 1.41 to 1.76 | 1.92 to 2.34 | 1.33 to 1.42 |
| EPYC 9V45, Zen 5 (2) | 1.45 to 1.60 | 1.91 to 2.10 | 1.30 to 1.35 |
| Xeon 6973P-C (1) | 0.85 to 0.97 | 0.98 to 1.19 | 1.15 to 1.24 |
| Neoverse-N2 (2) | 2.43 to 2.96 | 2.94 to 3.55 | 1.16 to 1.29 |

A million vectors:

| CPU (machines) | Before | After | Speedup |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (6) | 0.80 to 1.01 | 0.96 to 1.12 | 1.06 to 1.20 |
| EPYC 9V74, Zen 4 (3) | 1.41 to 1.65 | 1.84 to 2.02 | 1.21 to 1.36 |
| EPYC 9V45, Zen 5 (2) | 1.58 to 1.71 | 2.15 to 2.34 | 1.35 to 1.40 |
| Xeon 6973P-C (1) | 1.14 to 1.15 | 1.30 to 1.33 | 1.13 to 1.15 |
| Neoverse-N2 (2) | 2.48 to 3.00 | 3.19 to 3.59 | 1.13 to 1.30 |

Decisions 1 to 6 alone, in earlier runs of the same kind, on the two Xeons the final code did not
meet:

| CPU | Vectors | Before | After |
| --- | --- | --- | --- |
| Xeon 8573C (2) | 2,000 and 20,000 | 0.85 to 0.99 | 1.09 to 1.25 |
| | 100,000 | 0.89 to 1.14 | 1.07 to 1.42 |
| | a million | 1.12 to 1.27 | 1.31 to 1.44 |
| Xeon 8370C (1) | 2,000 and 20,000 | 0.85 to 0.98 | 1.07 to 1.29 |
| | 100,000 | 1.11 to 1.15 | 1.28 to 1.37 |
| | a million | 1.05 to 1.08 | 1.13 to 1.19 |

### What each decision gave

Each commit against the one before, as the median over `ef` and six x86 machines (decision 7
over twelve):

| Decision | 2,000 | 20,000 | 100,000 |
| --- | --- | --- | --- |
| 1. Loops compiled per kernel | +1% | +1% | 0% |
| 2. Errors out of line | +3% | +3% | +3% |
| 3. One integer per heap entry | +20% | +15% | +11% |
| 4. Squared distances | +3% | +4% | +3% |
| 5. Visited without a branch | 0% | +5% | +2% |
| 6. Hints worked out once | +6% | +7% | +7% |
| 7. Hints in the upper layers | −2% | −1% | +1% |

Decision 3 gave 12 to 27% across machines and `ef` at 2,000 vectors. Decision 5 was worth 2 to
10% at 20,000 and cost up to 3% at 2,000, the more the larger `ef`: few words are still empty
there, and the branch it removes was easy to predict. Decision 7 is for the large index: 3% at a
million vectors on x86 in the median, up to 11% at `ef = 32`, and 1 to 27% on Neoverse-N2. On an
index that sits in cache its hints buy nothing and cost 1 to 3% at `ef = 32`, 5% on one machine.

On Neoverse-N2 decision 3 gave 7 to 12% and no other more than 5%.

### Builds

One thread, 100,000 vectors, in seconds: the best of two runs on each machine.

| CPU | hnswlib | Before | After |
| --- | --- | --- | --- |
| EPYC 7763 (6) | 19.7 to 22.5 | 18.4 to 19.9 | 14.7 to 17.2 |
| EPYC 9V74 (3) | 19.0 to 22.8 | 15.9 to 19.5 | 12.6 to 15.0 |
| EPYC 9V45 (2) | 13.0 to 14.8 | 12.1 to 12.2 | 9.6 to 9.9 |
| Xeon 6973P-C (1) | 14.2 | 14.5 | 12.0 |
| Neoverse-N2 (2) | 36.9 to 42.4 | 18.8 to 19.7 | 16.9 |

* **In cache Chassis now leads hnswlib on every x86 processor measured**, by 7% or more, where
  it trailed by up to 23%. Search there is 1.2 to 1.5 times faster than before.
* **At a million vectors search is 1.06 to 1.4 times faster.** Zen 3, the one processor where
  Chassis trailed at that size, is level: 0.96 to 1.12.
* **One-thread builds are 1.1 to 1.3 times faster**, and ahead of hnswlib's on every machine.

## What Was Tried and Left Out

* **Leaving a visited node's word unwritten.** Decision 5 writes the word back even when the bit
  was already set. Returning early instead gave 2 to 3% at 2,000 vectors with a large `ef` and
  cost about 3% at 100,000 on Zen 3 and Zen 4, up to 6, most likely because the compiler then
  folds the two tests back into the branch.
* **Forcing the neighbor iterator inline.** Within two percent either way on x86, and 1 to 4%
  slower on Neoverse-N2 below a million vectors.
* **A prefetch policy for small indexes.** In the new loop eight lines per vector is the best
  setting or level with it at every size. No hints at all are level to 5% slower at 2,000
  vectors, 7 to 10% slower at 20,000 and 11 to 44% slower at 100,000 on x86; on Neoverse-N2 10,
  29 and 46%.
* **Hints into L1 on Granite Rapids.** On the one Xeon 6973P-C that ran it, L1 was 2% faster
  than L2 at 2,000 vectors, 3% at 20,000 and about 10% at 100,000. Ice Lake preferred L2
  slightly and Emerald Rapids did in ADR-0013. One machine, and nothing at a million vectors:
  not enough to add a second exception to the policy.
* **Skipping the upper layers' hints on a small index.** It would win back what decision 7
  costs there, with a threshold that depends on the machine's caches.

## Consequences

### Positive

* Chassis is no longer behind hnswlib where the index fits in cache, on any CPU measured.
* Search and one-thread builds are faster at every size, on x86 and on ARM.
* Results at equal distances now come in a defined order, and a NaN distance can't come first.

### Negative

* Results are not the same bit for bit as before: the order of equal distances, and at a million
  vectors a few candidates taken or left that were not.
* The search loop is compiled twice on x86, once per kernel, and the binary grows by that.
* A packed entry holds a 32-bit slot. The format already limits an index to 2³² − 1 slots.
* The upper layers' hints cost a small index 1 to 3% at a small `ef`.
* The measurements are one dataset at 128 dimensions on shared runners.
