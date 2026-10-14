# ADR-0012: Rolling Back Lost Adds

**Date:** 2026-10-08
**Status:** Proposed

## Context

An add changes the neighbor lists of nodes that are already committed, in place, before it is
committed itself: each neighbor gets a link back to the new node, and a full list drops an older
link to make room. A crash rolls the new nodes back, since the header still counts only the
committed ones, but not the lists (ADR-0005 amendment). The committed nodes are left with links
to slots that are gone, in place of the links they had.

ADR-0011 measured what that costs. With 1,000 vectors committed and 20,000 lost, a search for a
committed vector found it for 178 to 429 of the 1,000. Adding the 20,000 again brought that to
about 720: a node that has lost every link pointing at it is never reached, so nothing links to
it again. With 100 committed, 3 were found, before and after. Only `compact()` restored them,
and nothing ran it.

The same leftover links were behind two bugs. A batch reusing the lost slots could strand its
own nodes (ADR-0010, fixed with a flag per node), and a filtered search could return fewer ids
than its filter accepts (ADR-0009, fixed with a fallback to the scan).

ADR-0005 rejected a write-ahead log for every write as too costly. What is missing is much
narrower: the committed lists, as they were, for the nodes an uncommitted add has touched.

## Decision

### 1. Save a committed node's lists before they first change

Before any list of a committed node changes for the first time since the last commit, the node's
lists on every layer are appended to an undo file, `<name>.undo`, next to the index. A bit per
committed slot records that it is saved, so a node is saved once per commit however often it
changes after that. Nodes added since the commit are not saved: a crash takes them whole.

An add saves all the neighbors it is about to link back from in one write, since a write per
node costs many times more. Every place that changes a list also checks the bit itself, so no
caller can change a committed list unsaved.

The file starts with a magic number. Each entry is the slot, its layer count, each layer's
neighbors that are themselves committed, and an xxh3-64 checksum seeded with a hash of the
committed graph: its vector count, entry point, top layer, delete epoch, deleted count and id
mark. Entries saved from any other commit fail the checksum.

The file is created with the first entry, so an index built in one go, or a batch's own nodes
linking to each other, never creates one.

### 2. A writer writes them back when it opens the index

Opening an index for writing reads the undo file, writes each saved node's lists back, syncs the
index, and removes the file. The adds made since the commit are gone, and now so are their links
in committed nodes: the graph is the one the commit left.

If two threads of a batch save the same node, the first entry in the file is the one read before
any change (a list changes only after an entry for it is written), so that one is used. Parsing
stops at the first entry that is torn or fails its checksum. Writing back is idempotent, so a
crash during it changes nothing: the next open does it again.

Readers don't write, so a reader that opens the file after a crash sees the damaged graph until a
writer has opened it, as before.

### 3. A commit starts over

A commit that changes the graph removes the undo file and clears the saved bits, after its header
is durable. Before that commit forces the changed lists to disk, it fsyncs the undo file, so a
power loss during a flush can't keep the lists and lose what was saved for them.

A header written only for a flag, or for a flush's intent to delete, commits the same graph, so
the undo file stays valid across it.

### 4. The file follows the index

Compaction and migration build a new file and rename it over the path; the undo file's path
follows. Neither leaves one behind: compaction flushes first, and a new file has nothing
committed until its first flush.

## What holds

* **After the process dies, or an index is dropped without a flush:** opening it for writing
  leaves every committed node with the neighbors the last flush gave it, exactly.
* **After a power loss:** the same for every list whose saved entry reached disk. Between
  flushes the OS may write a changed list before the entry saved for it; that list then keeps
  what ADR-0005's amendment described. It is never worse than before this ADR.
* **An index copied or opened without its undo file,** or by a release from before this ADR,
  behaves as before: the lost adds' links stay, and compaction removes them.

## Measurements

On 2026-10-08, on the M5 used for ADR-0010 and ADR-0011 (in Low Power Mode, under load from other
work).

**The graph after lost adds.** ADR-0011's table again, with the undo file: recall@10 over 200
queries, then how many of the committed vectors a search for them finds.
`cargo run --release --example lost_adds -- <committed> <lost>` reproduces it.

| Committed | Lost | Before | After the loss | Lost ones added again |
| --- | --- | --- | --- | --- |
| 20,000 | 2,000 | 0.988, 20,000 | 0.988, 20,000 | 0.986, 20,000 |
| 20,000 | 20,000 | 0.987, 20,000 | 0.987, 20,000 | 0.979, 20,000 |
| 500 | 5,000 | 1.000, 500 | 1.000, 500 | 0.996, 500 |
| 1,000 | 20,000 | 1.000, 1,000 | 1.000, 1,000 | 0.985, 1,000 |
| 100 | 20,000 | 1.000, 100 | 1.000, 100 | 0.988, 100 |

**What it costs.** 128-dimensional random vectors, `M = 16`, `ef_construction = 100`: onto
200,000 committed vectors, a batch of 50,000, a flush, 5,000 single adds, a flush, then 200
times one add and a flush. Three runs with and three without, alternating. CPU time, since the
machine was too loaded for wall time to mean much.
`cargo run --release --example undo_cost -- 200000 50000 5000` reproduces it.

| | Without | With |
| --- | --- | --- |
| Batch of 50,000, CPU on all cores | 37.1, 37.8, 39.1 s | 37.9, 39.0, 39.0 s |
| Single add, CPU | 513, 535, 627 µs | 473, 523, 544 µs |
| One add and a flush | 12.3, 12.6, 12.6 ms | 18.3, 18.3, 18.4 ms |
| Undo file after the 5,000 adds | none | 8.5 MB |

* **Adds cost the same**, within what these runs can show.
* **A flush after adds onto committed vectors costs one more fsync**: 6 ms here, on a Mac, where
  an fsync flushes the drive's cache. A flush of a new index, or one with no adds since the last,
  writes no undo file and pays nothing.
* **The file is bounded by the committed graph**, at about 150 bytes for a node whose lists are
  full, because a node is saved once per commit. Here it grew by 1.7 KB per add.

## Consequences

### Positive

* A crash, a kill, or a drop without a flush leaves the committed graph as the last flush left
  it. Searches after one find what they found before it.
* The random-history test (`model_tests.rs`) now demands a connected graph after every reopen
  that loses adds, where before it had to excuse them until the next compaction.
* The links that stranded batch nodes and cut filtered searches short no longer survive a
  process crash. Their fixes stay, for power losses and for files older releases wrote.

### Negative

* A second file can exist beside the index: after a crash, and between a flush and the adds
  that follow it. An index moved or copied in that state has to take it along.
* One more fsync per flush that follows adds onto committed vectors.
* After a crash, the next open reads the whole undo file and syncs the index before returning.
* A power loss between flushes can still leave some lists as ADR-0005's amendment described.
* A batch that fails while linking is not rolled back this way: the undo file holds the lists
  as committed, and writing those back would also drop links to vectors added before the batch
  and not yet flushed. Linking can't fail short of a corrupt file, so this is left as it was.

## Alternatives Considered

* **Save only the links each add displaces.** Smaller entries, but one for every link an add
  makes, with no bound until the next flush, and a write for each.
* **Rebuild damaged nodes when a writer opens.** No extra file, but finding them means reading
  every committed list on every open, and relinking a node needs a graph that can still reach
  its neighbors, which is what is missing in the worst case.
* **Compact when a writer opens after a crash.** Exact, but as slow as building the index, for
  damage that is usually small.
* **Hold back the links to new nodes until the flush.** Committed lists would never change
  early, but neither readers in other processes nor later adds would reach the new nodes through
  them, and the flush would still change them in place.
* **A write-ahead log.** ADR-0005 rejected it: every write twice, and replay on open.
