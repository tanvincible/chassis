# Overview

This page is a map of how Chassis keeps and searches your vectors: what is in the file, what a
write does, what a crash leaves behind, and what each choice costs. Each section links to the page
or decision record with the details.

## One file

An index is one file. It holds every vector, each in a slot numbered in the order it was added,
and an HNSW graph over them: the lists of near neighbors that a search walks. Chassis maps the
file into memory and reads vectors and lists where they lie, so opening an index loads nothing,
and the operating system's page cache decides what stays in memory.

The file only grows. Slots fill fixed-size segments; when one is full, the next region is appended
and mapped. Nothing already mapped ever moves, so a reference into the file stays valid while the
index grows, and a reader in another process never finds part of its mapping gone. A deleted
vector keeps its space until `compact()` writes a new file without it.
[Storage](./storage.md) and [File Format](./file-format.md) have the details.

## Graph

Every vector has a list of neighbors on layer 0, which sits in its slot. A few vectors also have
lists on sparser layers above, kept in a heap of their own; a search crosses the index on those in
a few steps, then narrows in on layer 0.

To link a new vector, Chassis searches the graph for its nearest vectors, keeps a set of them that
point in different directions (heuristic 2 of the HNSW paper, over all `ef_construction`
candidates), and adds the new vector to each one's list, pruning a list that is full. Every edge
points at a vector whose data is already written. `add` links one vector at a time; `add_batch`
writes the whole batch first, then links it on every core, so its graph depends on thread timing.
See [Graph](./graph.md),
[ADR-0004](https://github.com/tanvincible/chassis/blob/main/docs/adr/004-diversity-heuristic-with-lazy-cache.md)
and [ADR-0010](https://github.com/tanvincible/chassis/blob/main/docs/adr/010-parallel-batch-builds.md).

## Ids

Your id and the slot are separate. Each slot keeps the id the vector was added under, and search
returns ids. While every id equals its slot, a lookup needs no table; once one differs, the first
lookup in a process scans the slots once to build an id-to-slot table in memory
([ADR-0007](https://github.com/tanvincible/chassis/blob/main/docs/adr/007-ids-and-deletes.md)).

## Search

A search descends the upper layers greedily, then explores layer 0, keeping the `ef_search` nearest
candidates it has seen. In that loop:

- the set of visited vectors is a bitmap that each thread reuses, clearing only the words it set;
- a candidate is its slot and squared distance packed into one integer;
- distances are computed several neighbors at a time, with the next neighbors' vectors fetched
  ahead;
- the distance code is chosen when the program runs: AVX2 and FMA on x86-64 (with F16C for half
  precision), NEON on ARM, and portable code elsewhere.

A search with `allowed` ids walks the graph past the others, or, when that would cost more, as when
few vectors match, checks every allowed vector instead and returns exact results
([ADR-0009](https://github.com/tanvincible/chassis/blob/main/docs/adr/009-filtered-search.md)).
[Performance](./performance.md) has the measured numbers and how to reproduce them.

## Writes

One process writes a file at a time. The writer holds an exclusive lock (`flock` on Unix,
`LockFileEx` on Windows), and a second writer's open fails at once instead of waiting. Within the
process, `search` takes `&self` while `add`, `delete` and `flush` take `&mut self`, so many threads
can search one index while writes wait their turn.

An add or delete goes straight into the mapped file, and searches see it at once. It is durable
after `flush()`, which syncs the file (`msync`, then `fsync`; `F_FULLFSYNC` on macOS,
`FlushFileBuffers` on Windows), writes whichever of the two header copies doesn't hold the newest
header, and syncs again. The header's
counts are what is committed. A flush with deletes writes an intent header first, so a crash
partway rolls its deletes back together. If a sync fails, the index refuses every later flush
until it is reopened: after a failed fsync the OS may have dropped the pages, and a flush that then
succeeded would claim data that is gone.

## Readers

An `IndexReader` searches a file that another process writes, and takes no lock. Before each
search it reads the newest valid header and maps any region added since, and it returns only
vectors the last flush committed. The writer stores lists, delete marks and headers as atomic
words, and rewrites a list in place so that a neighbor in both the old and the new list keeps its
position: a reader partway through a list never misses an edge that was there before and after
([ADR-0008](https://github.com/tanvincible/chassis/blob/main/docs/adr/008-format-v3-and-multi-process-readers.md)).

## Crashes

After a crash, reopening keeps every add and delete up to the last `flush()`, drops the rest, and
finds the graph as that flush left it:

- Each header copy has a checksum, so a torn header write is skipped and the other copy is used.
- Slots past the committed count are ignored, and the next adds reuse them.
- Adds after a flush change the lists of vectors that flush committed. Before a committed vector's
  lists first change, they are saved to an undo file, `<name>.undo`, next to the index, and
  reopening puts them back
  ([ADR-0012](https://github.com/tanvincible/chassis/blob/main/docs/adr/012-rolling-back-lost-adds.md)).
- A crash during a flush with deletes rolls those deletes back.

A power loss between flushes can still leave some lists changed, which can lower recall. Process
kills at random points are tested (`chassis-core/tests/crash_tests.rs`), and a simulator drops,
reorders and tears unsynced writes to stand in for power loss (`chassis-core/src/power_loss.rs`);
neither has run on real hardware.

## Compaction

`compact()` builds a new file beside the old one, `<name>.compacting`, copying each live vector
under its id and linking them as a batch. Then it marks the old file superseded, renames the new
one over it, and continues on the new file. Ids stay the same; slots change. A crash at any point
leaves one complete file at the path, and readers move to the new file on their next search
([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)).

## Trade-offs

| Choice | What it gives | What it costs |
| --- | --- | --- |
| A mapped file | Nothing to load at open; vectors read where they lie | `unsafe` code. A disk error while reading a mapped page, or a full disk on Unix, can crash the process (SIGBUS) instead of returning an error |
| A file that only grows | Mappings never move, so references and readers stay valid | Deleted vectors keep their space until `compact()` |
| One writer | No coordination between writers; readers take no lock | Writes from several processes have to go through one |
| Commits at `flush()` | An add costs no sync | A crash loses what came after the last flush; a flush waits for the disk |
| An undo file | A crash leaves the committed graph whole | A second file beside the index from the first add after a flush until the next one, and one more fsync in that flush |
| Batches on every core | `add_batch` links with every core | The graph depends on thread timing |
