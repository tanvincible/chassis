# ADR-0004: Diversity Heuristics & Lazy Distance Caching

**Date:** 2026-01-24  
**Status:** Accepted, cache withdrawn 2026-10-04 (see [Amendment](#amendment-2026-10-04))

## Context

In HNSW graphs, the quality of the small-world navigation property depends heavily on how neighbors are selected during construction.

1. **Naïve Selection (Heuristic 1):** Simply selecting the closest neighbors creates highly clustered graphs. This forces search algorithms to crawl through dense local regions, significantly increasing hop count and degrading navigability.
2. **Diverse Selection (Heuristic 2):** The HNSW paper proposes selecting a neighbor only if it is closer to the base node than to any already-selected neighbor. This enforces long-range connections and creates graph-level “expressways”.

The problem is computational cost.
Heuristic 2 is expensive: for a candidate set of size N and target size M, it requires O(N·M) distance checks. With high-dimensional vectors (e.g., 1536 dimensions), redundant distance computation becomes the dominant cost during graph construction, accounting for the majority of CPU time.

This ADR assumes a symmetric metric distance function (e.g., Euclidean distance), which guarantees distance symmetry and non-negativity.

## Decision

We adopt **Heuristic 2** for all neighbor selection and augment it with a **Lazy Symmetric Distance Cache** to eliminate redundant computation.

### 1. Heuristic Logic

We implement Heuristic 2 with a starvation fallback:

* **Pruning Rule:** A candidate is retained only if its distance to the base node is strictly smaller than its distance to every already-selected neighbor.
* **Fallback:** If pruning yields fewer than a policy-defined minimum number of neighbors, the remaining slots are filled using closest-first selection, ignoring diversity, to preserve graph connectivity.

The minimum degree is treated as a construction policy parameter rather than a strict mathematical invariant.

### 2. Stack-Allocated Lazy Cache

To eliminate the O(N·M) recomputation bottleneck, we introduce a specialized distance cache:

* **Stack Allocation:** A fixed-size `[f32; 1089]` array is allocated on the stack, representing a 33×33 symmetric distance matrix. This avoids heap allocation in the construction hot path and is safe given the shallow call depth and typical multi-megabyte thread stacks.
* **Symmetry:** Because the distance function is symmetric, computed values are stored in both `[i][j]` and `[j][i]`, halving the number of required computations.
* **Lazy Evaluation:** Distances are computed only when requested by the heuristic. If a candidate pair is never compared, its distance is never computed.

## Consequences

### Positive

#### Significant Construction Speedup

Benchmarks show that the lazy cache reduces pruning time per node from roughly 200 microseconds to approximately 50 microseconds for typical parameters. Most diversity checks resolve via cache hits rather than repeated vector arithmetic.

#### High Graph Quality

The ability to apply Heuristic 2 without prohibitive cost results in graphs with stronger navigability properties, improving downstream search performance compared to naïve neighbor selection.

#### Zero-Allocation Hot Path

The pruning logic performs no heap allocation. This reduces allocator pressure, memory fragmentation, and latency variance during large batch insertions.

> Amended 2026-10-03: only the distance cache is on the stack. The candidate and selection lists are
> `Vec`s, so pruning does allocate.

### Negative

#### Compile-Time Limits on M

The stack-allocated cache requires a compile-time maximum size. We currently enforce `MAX_M = 32`, supporting candidate sets up to 33 nodes. Increasing this limit would require a larger stack allocation or a slower heap-backed alternative.

#### Increased Complexity

The lazy cache introduces substantially more code and complexity compared to a simple distance loop. This is an explicit tradeoff in favor of predictable performance and graph quality.

## Compliance

* **Fixed Size Enforcement:** The distance cache enforces a compile-time capacity limit.
* **Lazy Access:** The pruning logic mediates all distance access through a cache-aware helper, ensuring each pairwise distance is computed at most once.
* **Symmetry Guarantee:** All cache writes update both symmetric entries to maintain consistency.

## Amendment (2026-10-04)

The cache never helped. Heuristic 2 takes each candidate once and compares it with each neighbor
already kept, so no pair is ever looked up twice: an instrumented build counted 0 hits in 157
million lookups (20,000 × 128 and 3,000 × 1,536 dimensions). The pruning speedup quoted under
Consequences can't have come from it.

Its 33-entry capacity also cost graph quality. New nodes chose neighbors from only the 33 nearest
of their `ef_construction` candidates. With every candidate considered, as hnswlib does, search was
7–18% faster at equal recall on a 200k SIFT prefix, and SIFT-1M's recall at `ef_search` 512 rose
from 0.998 to 0.999.

The cache and both 33-candidate caps (`MAX_M` in `link.rs`, `MAX_CANDIDATES_FOR_HEURISTIC` in
`lib.rs`) were removed on 2026-10-04, and the heuristic now computes each distance directly. Build
time on 20,000 × 128 random vectors didn't change (7.7–8.2 s before, 8.1–8.2 s after), and recall@10
at `ef_search` 200 rose from 0.880–0.885 to 0.902–0.908.
