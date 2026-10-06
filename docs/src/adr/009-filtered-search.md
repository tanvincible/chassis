# ADR-0009: Filtered Search

**Date:** 2026-10-07
**Status:** Proposed

## Context

Applications rarely want the nearest vectors in the whole index. They want the nearest among one
user's documents, one language, the rows a permission check allows. Chassis stores vectors and
caller ids only, and its docs tell applications to keep everything else in their own store, such
as SQLite.

Filtering after the search doesn't work. Asking for `k / s` results to keep `k` that match a filter
passing a fraction `s` of the index returns too few whenever the matches aren't spread evenly, and
costs `1 / s` times a normal search when they are.

Two things already in the code shape the design:

* **Deletes are a filter.** Layer-0 search walks through deleted nodes but leaves them out of the
  results (ADR-0007), and readers apply their snapshot the same way (ADR-0008). A caller's filter
  can use the same path.
* **The format reserves room for stored metadata.** Every slot header has a metadata reference,
  and ADR-0008 plans a metadata heap, added later with only a write-version bump. Until this ADR's
  review that promise didn't hold: a newer write version stopped readers too, and so did a longer
  header. Both are fixed, while format 3 is still unreleased.

## Decision

### 1. The caller supplies the filter

`search_filtered(query, k, filter)` takes a function from a caller id to `bool`, in Rust on both
`VectorIndex` and `IndexReader`. C (`chassis_search_filtered`) and Python (`allowed=`) take a list
of allowed ids instead, since a Python callback per visited node would cost more than the search;
the list becomes a hash set for the search. Chassis stores no metadata. An application keeps it
where it already does and passes the ids its query selects.

### 2. Walk the graph, and give up when a scan is cheaper

Layer-0 search treats rejected nodes like deleted ones: they are traversed, so the graph stays
connected, but never returned. Measured below, that keeps mean recall@10 at about 0.98 or better
at every selectivity. Its cost is the problem: to find `ef` matches when a fraction `s` of the index
matches, it visits about `1 / s` times more nodes. At 1% it visits half of dbpedia, and at 0.1%
all of it.

So a filtered search first runs the filter on about 1,024 slots, one per equal stretch of the
index at a jittered position so a filter periodic in the id can't line up with them, to estimate
how many vectors match, `m`, and gives the graph search a budget of the smaller of `m / 2` and
`32 × ef / s` distance computations. If it finishes within the budget, its results stand.
Otherwise it stops, and the search checks every slot and returns the exact nearest matches.

* **`m / 2`.** Checking every slot computes `m` distances, but in order. A graph visit reads a
  vector and a neighbor list from random places and keeps two heaps: from the visit counts of the
  graph-only runs below, it costs 0.5 to 2 times a scanned distance for random filters on both
  datasets, and up to 6 times for filters far from the query on SIFT, whose heaps grow large. Half
  the matches is about the scan's cost in the usual case.
* **`32 × ef / s`.** An unfiltered search visits about `16 × ef` nodes at `ef = 64` (10 to 18,
  inferred from random filters passing 3% to 30%), so a
  filter whose matches are spread at random needs about `16 × ef / s` visits. A filter whose
  matches lie far from the query needs many more, because the search crosses everything nearer
  first: on SIFT-1M, 310,000 visits for a filter passing 30% of the index. Twice the random need
  cuts those off early.

The budget counts distance computations, not time, so the same query on the same index always
takes the same path.

### 3. Stored metadata stays out, and can come later

Keeping metadata in the caller's store covers filters of any kind with no format change, so
ADR-0008 can freeze without it. If Chassis later stores metadata, the slot headers' metadata
references and a metadata heap carry it, behind a write-version bump that older releases read
through.

## Measurements

On 2026-10-07, on a shared Apple Silicon workstation (4 performance and 6 efficiency cores, load
average 5 to 7), one search
thread, `M = 16`, `ef_construction = 200`, `ef_search = 64`, 200 queries per row, recall@10 against
the exact nearest matches. Four filters per fraction: random ids, looked up in a `Vec<bool>` (r) and
in a `HashSet` (s), as an allow-list is; the Voronoi cell of `1 / fraction` random pivots that holds
the query (n, matches clustered near it); another cell (f, matches far from it). Matches are
nominal; a near cell can hold twice as many. "Exact" computes distances to the matches only;
`cargo run --release --example filtered` reproduces it and the shipped columns. "Graph only" is a
patched build without the budget, from an earlier run on the same machine, with other random ids
and pivots of the same shapes. r runs first at each fraction, on a cache the cell assignment just
evicted, so at 30% it reads slower than s, which filters the same ids. Times in ms.

**dbpedia-openai3-1536** (99,000 OpenAI embeddings, 1,536 dimensions; unfiltered 1.2 ms):

| Fraction | Matches | Exact r / s / n / f | Graph only r / n / f | Shipped r / s / n / f |
| --- | --- | --- | --- | --- |
| 30% | 30,000 | 15 / 16 / 17 / 15 | 4.0 / 2.9 / 5.4 | 7.7 / 3.4 / 2.6 / 10 |
| 10% | 10,000 | 6.6 / 7.7 / 7.0 / 6.7 | 8.0 / 3.9 / 18 | 11 / 12 / 5.1 / 10 |
| 3% | 3,000 | 2.4 / 4.3 / 3.0 / 2.6 | 20 / 6.5 / 38 | 3.5 / 5.5 / 4.4 / 4.1 |
| 1% | 1,000 | 0.5 / 1.5 / 1.1 / 1.0 | 37 / 8.5 / 58 | 0.9 / 1.9 / 1.8 / 1.6 |
| 0.3% | 300 | 0.3 / 1.2 / 0.5 / 0.5 | 87 / 15 / 99 | 0.6 / 1.5 / 0.9 / 0.8 |
| 0.1% | 100 | 0.2 / 1.7 / 0.3 / 0.3 | 106 / 39 / 102 | 0.5 / 2.0 / 0.6 / 0.5 |

**SIFT-1M** (1,000,000 vectors, 128 dimensions; unfiltered 0.36 ms; graph only not run below 3%):

| Fraction | Matches | Exact r / s / n / f | Graph only r / n / f | Shipped r / s / n / f |
| --- | --- | --- | --- | --- |
| 30% | 300,000 | 42 / 51 / 31 / 32 | 1.1 / 0.6 / 263 | 3.3 / 0.7 / 0.4 / 25 |
| 10% | 100,000 | 17 / 41 / 17 / 15 | 3.0 / 0.5 / 212 | 1.3 / 1.5 / 0.4 / 17 |
| 3% | 30,000 | 7.9 / 17 / 11 / 7.3 | 3.4 / 0.5 / 132 | 9.2 / 15 / 0.7 / 11 |
| 1% | 10,000 | 4.7 / 16 / 5.2 / 4.0 | | 7.8 / 18 / 2.1 / 6.9 |
| 0.3% | 3,000 | 2.7 / 21 / 3.3 / 2.7 | | 4.7 / 23 / 3.4 / 5.1 |
| 0.1% | 1,000 | 2.6 / 11 / 2.6 / 2.7 | | 5.0 / 12 / 4.7 / 4.9 |

* **The shipped search takes at most 2.1 times the exact scan** in every row of both tables, and at
  `ef_search = 256` (not shown). Its recall is 0.98 or better at 64 and 0.998 or better at 256, and
  exact wherever it scans, up to ties between equal distances (SIFT has duplicate vectors).
* **The graph alone keeps its recall but not its speed.** Random filters below 1% visit nearly the
  whole index: at 0.1% on dbpedia, 620 times a scan of every slot in the same run. Filters far
  from the query are slow at every fraction: on SIFT, 6 to 26 times the scan, and 310,000 visits
  at 30%.
* **The cost of the budget.** Where the graph alone is fast, the budget can give up on it too early:
  a random 3% filter on SIFT takes 9.2 ms against 3.4 for the graph alone in the earlier run. It
  spends about 15,000 visits before giving up, and scanning a million slots costs about 5 ms
  however few match. Counting that cost into the budget keeps such searches on the graph, but lets
  far filters, whose visits cost up to 6 times a scanned distance, run longer.
  Measuring both costs at run time would fit each machine, but makes the path, and so the results,
  vary between runs of the same search.
* **Allow-lists pay for hash lookups.** Scanning a million slots with a `HashSet` filter costs 10 to
  23 ms. A scan driven by the list itself would cost only its length, but needs an id-to-slot map,
  which readers don't keep.

## Options Considered

* **Filter after searching.** Fails as described in the context.
* **Filter inside HNSW only,** as hnswlib does. Measured above: up to 620 times slower than
  scanning every slot for selective filters, and 6 times for a filter passing 30% whose matches lie
  far from the query.
* **A fixed selectivity threshold** for switching to a scan. Where the two paths cross depends on
  the index size (the graph visits about `V / s` nodes, a scan computes `s × N` distances, so they
  cross near `m = √(V × N)` matches) and on the filter: matches clustered near the query cost the
  graph search a sixth to three quarters of what random ones do, and matches far from it up to
  twice as much on dbpedia and 40 to 240 times on SIFT. A budget adapts to all three.
* **Storing metadata with a filter language,** as Qdrant and Weaviate do. Much larger, and a
  metadata store is what the introduction says Chassis isn't. Decision 3 keeps the option.
* **Graphs built for filters,** like ACORN or Filtered-DiskANN. They need the filters at build
  time, or more edges per node, to make very selective filters fast. At the sizes Chassis targets,
  a scan already answers those exactly in milliseconds.
* **A C callback instead of a list.** A list serves C and Python alike; a callback can be added if
  a C caller needs a computed filter.

## Consequences

### Positive

* Filtered search in Rust, C and Python, exact whenever the filter is selective, at most about
  twice an exact scan of the matches in every measured case.
* No format change, so ADR-0008 can freeze without metadata, and format 3 can now gain metadata
  later without locking out older readers.

### Negative

* The filter runs once per visited node, and once per vector when scanning: an expensive filter
  slows every filtered search. Allow-lists from C and Python are hashed for every search.
* A scan checks every slot, so a selective filter costs time in proportion to the index: about
  5 ms per million vectors on the measured machine, before the matches.
* The budget's two constants are fitted to two datasets on one machine, and favor bounding the
  worst case over keeping every search that the graph alone would finish fast.
