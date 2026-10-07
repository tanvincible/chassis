# ADR-0010: Parallel Batch Builds

**Date:** 2026-10-07
**Status:** Proposed

## Context

Chassis links one vector at a time (ADR-0002), so a build uses one core: SIFT-1M took 805 s on
the measured machine. hnswlib links on every core, and anyone comparing the two meets the
difference in their first five minutes.

ADR-0002 chose sequential construction to keep one invariant: every edge points at a node whose
data is already written. Two things since then make a parallel build fit that invariant:

* **Lists are written for concurrent readers.** Readers in other processes traverse neighbor
  lists while the writer changes them (ADR-0008): every entry is an atomic word, and `store_list`
  rewrites a list so a surviving entry keeps its place. Threads of the writer can traverse the same
  way.
* **A record can exist before it is linked.** A slot's vector, id and graph record are written
  before anything links to it, and readers skip ids past the routing count they were given.

## Decision

### 1. `add_batch` and `add_batch_with_ids`

`VectorIndex::add_batch(vectors)` takes vectors back to back and returns their ids, assigned as
`add` assigns them; `add_batch_with_ids(ids, vectors)` takes the caller's. C's `chassis_add_batch`
now uses it, `chassis_add_batch_with_ids` is new, and Python gains `add_batch(vectors, ids=None)`.
A batch is added whole or not at all: ids and dimensions are checked first, and a failure while
writing or linking rolls the slots back.

### 2. Write the batch, then link it on every core

The writer first appends every vector of the batch, draws its layer, and writes an empty graph
record for it, one at a time. Then a thread per core takes the batch's nodes in order and links
each one: it searches the graph from the current entry point, selects neighbors with the usual
heuristic, writes its own lists, and adds a backlink to each neighbor, pruning as before. Every
edge points at a written vector and record throughout, so ADR-0002's invariant holds.

### 3. One lock per list change, none for reading

Every change to a node's lists, its own and each backlink with its pruning, holds that node's
lock, one of 4,096 striped by slot. A thread holds one lock at a time, so threads can't deadlock.
A node is unreachable until its own lists and backlinks are written, since searches only follow
lists, so its own write rarely waits. Searches read lists without locks, as readers in other
processes do, and may see a list mid-change; they tolerate that the same way.

A node whose layer is above the current top holds the entry-point lock while it links, as in
hnswlib, so the new top connects to the old one. That is rare (the top layer grows with the log of
the index size), and other threads wait for it.

### 4. Readers see the batch when it is linked

The routing count readers follow is published once the whole batch is linked, so readers in
other processes skip the batch's ids until then, and see it in their results after the next
`flush()`, as for single adds. A crash during a batch leaves the file as the last flush left it,
with backlinks to rolled-back slots handled as after any crash (ADR-0005 amendment).

### 5. Thread timing shapes the graph

Which nodes a thread sees depends on what other threads have linked, so two builds of the same
batch differ, as two sequential builds already do through their random layers.

## Measurements

On 2026-10-07, on an Apple M5 with 4 performance and 6 efficiency cores, in Low Power Mode, with
other applications running. Builds ran one at a time, with `M = 16` and `ef_construction = 200`;
"every core" is `add_batch` for Chassis and `num_threads=-1` for hnswlib 0.8.0. Searches ran on one
thread over 1,000 queries. `cargo run --release --example ann -- <data> <dataset> [batch]` and
`python bench/ann/reference_bench.py hnswlib <data> <dataset> [threads]` reproduce them.

| Dataset | Engine | One thread | Every core | Speedup |
| --- | --- | --- | --- | --- |
| SIFT-1M (128 dims) | Chassis | 545 s | 110 s | 5.0× |
| | hnswlib | 682 s | 131 s | 5.2× |
| dbpedia (99k × 1,536 dims) | Chassis | 417 s | 176 s | 2.4× |
| | hnswlib | 596 s | 163 s | 3.7× |

Recall@10 / queries per second of each graph, built on one thread → built on every core:

| Dataset | Engine | `ef = 64` | `ef = 256` |
| --- | --- | --- | --- |
| SIFT-1M | Chassis | 0.970 / 4,338 → 0.968 / 4,111 | 0.998 / 1,681 → 0.998 / 1,484 |
| | hnswlib | 0.960 / 2,934 → 0.959 / 2,878 | 0.997 / 1,041 → 0.997 / 883 |
| dbpedia | Chassis | 0.978 / 938 → 0.978 / 841 | 0.998 / 303 → 0.998 / 271 |
| | hnswlib | 0.970 / 484 → 0.970 / 481 | 0.996 / 163 → 0.996 / 161 |

* **Parallel builds match on recall.** Within 0.002 from `ef = 32` up, for both engines; at 10 and
  16, Chassis's SIFT graph was up to 0.007 lower.
* **Search speed can't be compared from these runs.** One pass per `ef` varied by up to 40% between
  graphs, in both directions and for both engines (Chassis's parallel SIFT graph was 44% faster at
  `ef = 10` and 35% slower at 32). Paired, repeated searches would be needed to see a difference.
* **Chassis builds at about hnswlib's speed on every core:** faster on SIFT, within run-to-run
  variation on dbpedia (an earlier pair of parallel builds took 159 s and 158 s).
* **dbpedia scales less for both engines.** Its vectors are 6 KB, twelve times SIFT's, so memory
  bandwidth may limit it; this wasn't measured. Low Power Mode and the efficiency cores cap every
  number here; a desktop with only performance cores should scale further.
* **Threads don't wait on each other.** Sampling a dbpedia build, every worker spent about 80% of
  its time searching and 18% pruning backlinks, and under 1% waiting on a lock.


## Consequences

### Positive

* Builds use every core, through the same API in Rust, C and Python.
* No format change, and crash and reader guarantees are unchanged.

### Negative

* A batch uses every core, with no way yet to limit it; an application serving searches from the
  same process competes with its own build.
* A batch's graph depends on thread timing as well as on the random layer draws every build makes.
* A batch holds the write lock for its whole duration, so searches through a shared C or Python
  handle wait for it.
* Two parts aren't pinned by a test. Without the backlink lock, two threads can each add their
  node to a neighbor's list and one addition is lost, which lowers quality too little for a test to
  see, as in hnswlib. Nothing in a test can make linking fail, so the rollback of a failed batch
  runs only in review.
