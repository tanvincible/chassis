# ADR-0011: Compaction

**Date:** 2026-10-08
**Status:** Proposed

## Context

A deleted vector keeps its slot: its vector, id and graph record stay in the file, and searches
keep walking through it (ADR-0007). An application that replaces documents grows its file without
bound, and its searches slow down as they step over more deleted nodes: measured below, five times
slower with 90% deleted, though recall doesn't fall. Nothing repairs either. A flush interrupted
by a crash can also leave committed vectors with fewer edges than they should have (ADR-0005
amendment).

ADR-0008 settled how any wholesale change must happen: never in place, because readers in other
processes are traversing the file, but by writing a new file and renaming it over the old one
(decision 8), after committing a "superseded" flag so readers move over. It left the operation
itself, `rebuild_graph()`, unbuilt, and its third acceptance experiment waits on it.

## Decision

### 1. One operation: `compact()`

`VectorIndex::compact()` rewrites the index without its deleted vectors and with a newly built
graph, and replaces the file. It is ADR-0008's `rebuild_graph()` and its vacuum in one, since both
are the same copy; C gets `chassis_compact` and Python `compact()`. Like a flush, it makes every
add and delete so far durable, flushed before or not.

### 2. Build a new file beside the old one

`compact()` creates `<name>.compacting`, replacing one a crash left, with the index's dimensions,
metric and graph parameters. It copies each live slot's id and stored vector, in slot order, and
links them with the parallel batch build (ADR-0010). Ids don't change, so callers' references and
a reader's results stay valid; slots do.

Two things the old file showed only through its deleted slots are carried in the new header:

* **The id high-water mark.** `add` never reuses an id (ADR-0007), which it knew from the largest
  id in any slot, deleted or not. The header gains a field, after its tables, holding one past
  the largest id ever used; `add` starts from it.
* **The delete epoch.** A reader's snapshot is named by its committed count and delete epoch. The
  new file's epoch is one more than the old one's, so no snapshot of the new file shares a name
  with an earlier one.

### 3. Swap it in

Once the new file is fsynced, `compact()`:

1. commits the superseded flag in the old file's header;
2. renames the new file over the old path and fsyncs the directory, as migration does;
3. continues on the new file, whose writer lock it has held since creating it.

A failure before step 1 removes the new file and leaves the index as it was. If the rename fails,
the flag is cleared again. A crash between 1 and 2 leaves the old file in place with the flag set:
the next writer to open it clears it. After a power loss either file may be at the path; both are
complete, and they hold the same vectors under the same ids.

### 4. Readers follow

A reader that finds the superseded flag in a new snapshot opens the path again. If the file there
is a different one, it continues on that; if it is still the same file, because the rename hasn't
happened yet or never will, it keeps its snapshot and looks again on its next search. Its mapping
of the old file stays valid throughout, since an open file outlives its name.

On Windows a file can't be renamed over while any process has it open. `compact()` there closes
its own handles first, and fails, leaving the index unchanged, if another process has the index
open.

## Measurements

On 2026-10-08, SIFT-1M on the M5 used for ADR-0010 (in Low Power Mode, under heavy load from other
work, so queries per second are rough), `M = 16`, `ef_construction = 200`, 1,000 queries, the
median of three passes, recall@10 against the exact nearest live vectors.
`cargo run --release --example churn -- <data> sift-128` reproduces it.

| State | Live vectors | File | `ef = 64`: recall / QPS | `ef = 256`: recall / QPS |
| --- | --- | --- | --- | --- |
| Built | 1,000,000 | 713 MB | 0.968 / 5,140 | 0.998 / 2,035 |
| 50% deleted | 499,522 | 713 MB | 0.987 / 3,568 | 0.999 / 1,079 |
| 90% deleted | 99,511 | 713 MB | 0.999 / 1,038 | 1.000 / 355 |
| Compacted, in 11 s | 99,511 | 88 MB | 0.988 / 8,986 | 1.000 / 3,424 |

* **Deletes cost speed and space, not recall.** A search keeps going until it has `ef` live
  results, so with more deleted nodes it visits more and finds a little more: recall rises while
  queries per second fall to a fifth.
* **Compaction gives both back.** The file shrinks 8 times, to the size of an index built from the
  live vectors, and search at `ef = 64` is 8.7 times faster. At recall no lower than before
  (`ef = 256` after against `ef = 64` before), it is 3.3 times faster.

## Consequences

### Positive

* Deleted vectors' space comes back, and the graph is rebuilt from the vectors alone, which also
  repairs edges lost to an interrupted flush.
* The last unbuilt piece of ADR-0008 exists, so its third experiment can run.

### Negative

* It needs free disk for a second copy of the live data while it runs, and takes as long as
  building the index, on every core.
* It holds the writer for its whole run: adds, deletes and, through a shared C or Python handle,
  searches wait. Readers in other processes keep searching.
* On Windows it can't run while another process has the index open.
* ADR-0008 also described a reader reopening when its file has no name left (`st_nlink == 0`),
  for replacements made without the flag. That check isn't built: every replacement goes through
  the flag.
* ADR-0008's third experiment asks for paired recall measurements across processes while the
  writer compacts. What is tested here is narrower: in one process, two readers search through
  ten compactions with no failed search and no wrong result.
