# ADR-0017: Compare at Equal Recall; Long Vectors Out of Cache Are Bound by Memory

**Date:** 2026-10-09
**Status:** Proposed

## Context

Chassis is for local semantic search in one file. Someone searching picks the recall they need,
not an `ef`.

ADR-0013 to ADR-0016 compared Chassis with hnswlib at the same `ef`. At the same `ef` Chassis
returns more of the true neighbors, and computes more distances to do it: on 99,000 OpenAI
embeddings of 1,536 dimensions, recall of 0.938, 0.979, 0.993 and 0.998 at `ef` 32 to 256 where
hnswlib has 0.925, 0.969, 0.987 and 0.996. So those figures set a faster, less exact search
beside a slower, more exact one, and at 960 and 1,536 dimensions they showed Chassis behind on
x86.

## Decision

### 1. Engines are compared at equal recall

Chassis's speed is taken at the other engine's recall, between the two `ef` values whose recall
lies on either side of it, on a straight line through the logarithms of the miss rate and of the
queries per second.

### 2. Nothing more is done to the search over long vectors that keeps them as they are

Two changes were tried for it and are left out. With full-precision vectors, an index of long
vectors that doesn't fit in cache is bound by how fast memory delivers them.

## Measurements

From the runs of ADR-0015 and ADR-0016 and one more of the same kind, on 2026-10-09: GitHub's
runners, one thread, hnswlib compiled with `-march=native` in the same job. Chassis against
hnswlib, out of cache, with `huge_pages` off:

| CPU | 128 × 1,000,000, same `ef` | Same recall | 960 × 200,000, same `ef` | Same recall | 1,536 × 99,000, same `ef` | Same recall |
| --- | --- | --- | --- | --- | --- | --- |
| EPYC 7763 (Zen 3) | 1.06 to 1.11 | 1.15 to 1.31 | 0.83 to 0.90 | 0.91 to 1.01 | 0.75 to 0.83 | 0.92 to 1.15 |
| EPYC 9V74 (Zen 4) | 1.85 to 2.04 | 2.01 to 2.43 | 0.92 to 0.94 | 0.99 to 1.01 | 0.65 to 0.68 | 0.79 to 0.93 |
| EPYC 9V45 (Zen 5) | 2.17 to 2.33 | 2.37 to 2.72 | 0.86 to 0.92 | 0.92 to 1.03 | 0.81 to 0.83 | 0.99 to 1.10 |
| Xeon 8370C | 1.15 to 1.21 | 1.32 to 1.38 | | | 0.84 to 0.88 | 1.05 to 1.15 |
| Xeon 8573C | 1.32 to 1.48 | 1.48 to 1.72 | | | | |
| Neoverse-N2 | 2.87 to 3.56 | 3.12 to 4.06 | 1.45 to 1.68 | 1.55 to 1.86 | 1.54 to 1.73 | 1.89 to 2.32 |

With `huge_pages` on, at the same recall: 1.05 to 1.20 at 960 dimensions and 1.01 to 1.25 at
1,536 on the three AMD processors.

* **At equal recall Chassis is level with hnswlib on long vectors on x86**, from 9% behind to
  15% ahead, **and ahead with huge pages.** Zen 4 at 1,536 dimensions without them is the
  exception: 7 to 21% behind.
* **At 128 dimensions its lead is a tenth to a fifth larger than the same-`ef` figures said.**

### Why there is little left

On one Zen 3 machine a query at `ef = 64` over the 1,536-dimension index took 643 µs. A search
like it computes about 1,500 distances, counted on another build of the same index: some 430 ns
each, for 6 KB of vector, or 14 GB a second. The kernel alone needs 157 ns when the vectors are
in cache. The rest is waiting for memory.

## What Was Tried and Left Out

* **Stopping a distance once it is past the worst result.** A sum of squares only grows, so the
  result is exact. Of the distances a search computes over these embeddings, 85% are for nodes
  it then turns away, but they pass the limit late: 87 to 90% of the floats are still read. The
  kernel looked at its running sum every 256 floats. At 1,536 dimensions out of cache that was
  2% faster on Zen 3, 5% on Zen 4 and the Xeon 8370C, 9% on Neoverse-N2; in cache 3 to 6% slower
  on three of the four; and at 960 dimensions 8 to 9% slower on Zen 5. The looking costs about
  what the stopping saves.
* **A kernel that asks for the vector ahead of where it is reading**, 4, 8 or 16 lines. Against
  the same build without it: 3 to 6% faster on Zen 4 and Zen 5, within 5% either way on Zen 3,
  from 4% slower to 8% faster on the Xeon 8370C, and 4 to 9% slower on Neoverse-N2.

What would move a search over long vectors by more than a few percent is fewer bytes per vector:
half-precision or 8-bit storage. That changes the file format and, for 8 bits, the results. It is
not decided here.

## Consequences

### Positive

* The figures now answer the question a user has: how fast at the recall I need.
* Two changes that cost code and paid little, or cost speed elsewhere, are not carried.

### Negative

* The same-`ef` tables of ADR-0013 to ADR-0016 understate Chassis by a tenth to a quarter and
  stay as they were measured.
* Equal-recall figures are interpolated between four `ef` values, and cover only the recalls
  both engines reach within them.
* Zen 4 without huge pages stays behind at 1,536 dimensions.
