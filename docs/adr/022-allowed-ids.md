# ADR-0022: Allowed Ids That Cost Little to Pass

**Date:** 2026-10-15
**Status:** Accepted

## Context

A filtered search through C or Python takes its ids as an array. `chassis_search_filtered` and
`chassis_search_excluding` copied every id into a `std::collections::HashSet`, hashed with
SipHash, on every call. The Rust search itself costs little more with a filter: it estimates how
many vectors match from about 1,024 of them (ADR-0009), whatever the length of the list.

The database guide made every search pay for the list: to keep out a vector a crash left without
its row, it selected every row's id and passed them all as `allowed`. The MCP server's `recall`
without a tag, and its `remember`, did the same.

Measured on 2026-10-15 on an Apple M5 in Low Power Mode, release builds, through Python, on an
index of 20,000 random vectors, ids 0 to 19,999, `k` 10, the median of 20 to 50 searches:

| Allowed ids | None | 20,000 | 100,000 | 1,000,000 |
| --- | --- | --- | --- | --- |
| As a numpy array, 64 dimensions | 0.068 ms | 0.49 ms | 1.35 ms | 13.7 ms |
| As a list, 64 dimensions | | 0.94 ms | 4.5 ms | 46.8 ms |
| As a numpy array, 384 dimensions | 0.18 ms | 0.45 ms | 1.46 ms | 13.7 ms |

The cost grows with the list, not with the index or the dimensions. A list adds Python's
conversion of each element to the array.

## Options

Each way of holding the ids, built from 20,000 to 1,000,000 of them and asked about 3,000 ids,
timed in Rust on the same machine (median of 9 to 31 runs):

| Holding the ids | Build, 20k | Build, 100k | Build, 1M | 3,000 lookups |
| --- | --- | --- | --- | --- |
| SipHash `HashSet` (before) | 0.2–0.5 ms | 1.2 ms | 12.0 ms | 0.03 ms |
| `HashSet` keyed by one multiply, as the id table is | 0.07–0.13 ms | 0.58 ms | 5.8 ms | 0.007 ms |
| The caller's array, if sorted, searched by bisection | 0.007 ms | 0.018 ms | 0.20 ms | 0.04–0.06 ms |
| The same, sorted first when it isn't | 0.21 ms | 1.3 ms | 17.3 ms | 0.04–0.06 ms |
| A bitmap over slots, when ids equal slots | 0.01–0.05 ms | 0.1–0.24 ms | 0.5–0.8 ms | |

Building dominates: a search asks about the ids a few thousand times at most. A bitmap over slots
is the fastest for ids in any order, but only the core can build one, since only it knows each
id's slot, and that needs a new Rust API for slices of ids beside the closure. A filter handle
that callers build once and pass to many searches would also need new API in all three languages.

## Decision

1. **The guide and the MCP server stop passing every id.** A search joins its hits to their rows
   by id and drops a hit without a row; `mend()` at start removes such vectors. `allowed` is for a
   real filter, in id order (`ORDER BY id`). The MCP server's untagged `recall` searches
   everything; `remember` leaves out only the id it is about to reuse; tagged ids come sorted.
2. **The C API reads sorted ids where they lie.** `chassis_search_filtered` and
   `chassis_search_excluding` check whether the ids are in ascending order, which is one pass
   with no copy, and if so look ids up by bisection in the caller's array. Otherwise they build a
   hash set keyed by one multiply, with a random key per call.
3. **The Rust API stays as it is.** `search_filtered` takes a closure; a Rust caller holds its ids
   however suits it.

## Results

The same measurement after the change:

| Allowed ids, 64 dimensions | 20,000 | 100,000 | 1,000,000 |
| --- | --- | --- | --- |
| Sorted numpy array | 0.093 ms | 0.12 ms | 0.58 ms |
| Shuffled numpy array | 0.17 ms | 0.74 ms | 6.9 ms |
| List | 0.70 ms | 3.0 ms | 31.8 ms |

384 dimensions: 0.14, 0.19 and 0.66 ms sorted; 0.21, 0.81 and 6.9 ms shuffled.

The guide's own search, with notes in SQLite and 384-dimensional vectors, `k` 5:

| Rows | Every id as `allowed` (before) | Search, then join (after) |
| --- | --- | --- |
| 20,000 | 4.2 ms | 0.13 ms |
| 100,000 | 20.1 ms | 0.16 ms |

## Consequences

### Positive

* Sorted ids, as `np.arange`, `np.unique` and `ORDER BY id` give them, cost a pass without a copy:
  24 times less than before at a million.
* Ids in any other order cost half what they did.
* The guide's search no longer grows with the number of rows.

### Negative

* A list still pays Python's conversion, about 31 ms for a million ids.
* A vector a crash left without its row now takes a place among a search's `k` results until
  `mend()` removes it, so such a search returns one fewer. The MCP server doesn't mend; its next
  memory reuses the id and replaces the vector.
* Unsorted ids still cost a pass that builds a set: 6.9 ms at a million.

### Not done

* A bitmap over slots, and a filter handle reused across searches. Measure again if callers pass
  large unsorted filters in a loop.
