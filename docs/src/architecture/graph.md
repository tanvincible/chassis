# Graph

Chassis searches a persistent HNSW graph (Hierarchical Navigable Small World): lists of near
neighbors, kept in the index file next to the vectors. This page explains how a vector is linked
into the graph, how batches link in parallel, and what a crash can leave in the lists. The
[Overview](./overview.md) shows where the graph sits among the other parts.

## Layers

Every vector is on layer 0, where its list holds up to `2 × max_connections` neighbors. When a
vector is added, it draws a top layer at random, as the HNSW paper describes. Most vectors stay on
layer 0, and each layer up holds about one in `max_connections` of the layer below, up to 16 layers.
On each upper layer a vector's list holds up to `max_connections` neighbors. The graph's entry point
is a vector on its highest layer.

A search starts at the entry point, takes the nearest neighbor on each upper layer until it
reaches layer 0, and then explores layer 0 keeping the `ef_search` nearest vectors it has seen.

## Linking

Adding a vector links it in three steps:

1. **Search.** From the entry point, Chassis searches each layer the new vector is on, keeping the
   `ef_construction` nearest vectors it finds.
2. **Select.** When there are more candidates than the list holds, it keeps a set that point in
   different directions: heuristic 2 of the HNSW paper. Taking candidates nearest first, it keeps
   one only if it is no nearer to a neighbor already kept than to the new vector. If that keeps
   fewer than a quarter of the list, the nearest of the rest fill it up to a quarter, so vectors
   along a line stay connected
   ([ADR-0013](https://github.com/tanvincible/chassis/blob/main/docs/adr/013-prefetching-and-a-leaner-search-loop.md)).
3. **Link back.** It adds the new vector to each neighbor's list. When a list is full, it selects
   again among the old neighbors and the new vector the same way, and keeps the new vector if it
   is among the nearest. Adding a link a list already has changes nothing.

Every edge points at a vector whose data is already written: a vector's own lists can only name
vectors that exist, and a backlink is added only after the new vector is written.

## Batches

`add_batch` writes every vector of the batch and an empty list for each first. Then a thread per
core takes the batch's vectors in order and links each one as above
([ADR-0010](https://github.com/tanvincible/chassis/blob/main/docs/adr/010-parallel-batch-builds.md)).

- A thread holds a vector's lock while it changes that vector's lists. There are 4,096 locks,
  shared by slot number, so threads seldom wait for each other.
- No search reaches a batch vector before that vector has written its own lists.
- A vector that raises the graph's top layer holds the entry point until it is linked, so the new
  top connects to the old one. This is rare, and other threads wait meanwhile.

The graph then depends on the order in which threads finish, so two builds of the same batch
differ, as two sequential builds already do through their random layers.

## Deletes

A deleted vector stays in the graph. Searches still walk through it, so the graph stays connected,
but never return it. It keeps its space until `compact()` builds a new graph without it.

## Crashes

The write order of a link, the new vector's lists before any backlink and the header last, holds
only against a process crash: the operating system writes changed pages back to disk in any
order. Against power loss, the syncs in `flush()` are what order the writes
([ADR-0005 amendment](https://github.com/tanvincible/chassis/blob/main/docs/adr/005-crash-consistent-linking.md#amendment-2026-10-03)).

After a process crash, reopening finds the graph as the last flush left it. Before a committed
vector's lists first change after a flush, they are saved to an undo file, and reopening puts them
back
([ADR-0012](https://github.com/tanvincible/chassis/blob/main/docs/adr/012-rolling-back-lost-adds.md)).

A power loss between flushes can leave lists that name a vector the crash rolled back. Search
skips such a neighbor, and linking drops it, until the next add reuses its slot; after that it is an
ordinary edge to the new vector. A link on a layer the new vector isn't on finds no neighbors
there. Edges that were pruned to make room for a lost vector stay lost, which can lower recall
until the next `compact()`.

## Edge cases

| Case | What happens |
| --- | --- |
| A vector as its own neighbor | Never kept: the search for its neighbors leaves it out. |
| A link a list already has | Skipped, so retrying a link writes nothing. |
| Exact copies of a vector | A list keeps one, plus any the quarter floor adds: copies kept among themselves would fill every list and leave other vectors out of reach. |
| A neighbor on fewer layers than the link | Skipped: the link names a slot that, after a power loss, a vector on fewer layers reused. |
