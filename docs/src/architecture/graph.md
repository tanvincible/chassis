# Graph Topology & Construction

Chassis implements a persistent, single-file HNSW (Hierarchical Navigable Small World) graph. Unlike in-memory implementations, the Chassis graph is designed for **crash consistency**, **zero-copy access**, and **deterministic behavior**.

This document outlines the construction policies, persistence model, and neighbor selection algorithms.

## 1. Construction Policy: Sequential Existence

Chassis enforces a **Strict Sequential Linking** policy to ensure the graph remains navigable and corruption-free without complex runtime checks.

**Invariant:** A node with ID `N` may only link to neighbors with IDs `M < N`.

### Implications
* **Safety**: Forward links in a `NodeRecord` only reference existing nodes. We never store a pointer to uninitialized memory.
* **Search Efficiency**: Search traversal does not need to check if a neighbor exists. If the ID is in the neighbor list, the data is guaranteed to be on disk.
* **Crash Recovery**: There are no "orphaned" forward edges pointing to the future.

### Usage Pattern
The graph must be built by inserting nodes in dense, monotonic order (0, 1, 2...).

```rust
// WRONG: Cannot link to non-existent nodes
// Error: nodes 1, 2, 3 don't exist yet
graph.link_node_bidirectional(0, 1, &[vec![1, 2, 3]])?;

// CORRECT: Sequential construction
graph.link_node_bidirectional(0, 1, &[vec![]])?;        // Node 0: no neighbors yet
graph.link_node_bidirectional(1, 1, &[vec![0]])?;       // Node 1: links to 0
graph.link_node_bidirectional(2, 1, &[vec![0, 1]])?;    // Node 2: links to 0, 1

```

## 2. Crash Consistency Model

After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. Process kills are tested by
`chassis-core/tests/crash_tests.rs`, and power loss is simulated by `chassis-core/src/power_loss.rs`
(not on real hardware). The write order below protects against a process crash only, because the OS
writes mmap pages back in any order; under power loss the fsyncs in `flush()` are what order writes. See the
[ADR-0005 amendment](../adr/005-crash-consistent-linking.md#amendment-2026-10-03).

### The Atomic Write Sequence

1. **Step A (Persist Node):** The new node `A` is written to disk with its forward links (`A` → `Neighbors`).
2. **Step B (Update Neighbors):** We iterate through the neighbors (`B`, `C`...) and update their backward links (`B` → `A`).
3. **Step C (Update Header):** The global node count and entry point are updated.

### Why This Ordering Matters

* **After Step A:** Node `A` exists on disk but has no incoming edges. It is technically "invisible" to search from the entry point, but the file is valid.
* **During Step B:** Some neighbors point to `A`, others don't. This creates "one-way edges," which are valid in HNSW and do not break search.
* **After a crash before `flush()`:** `node_count` rolls back but backlinks to `A` may survive. Search
  skips them until the next insert reuses `A`'s slot; after that they are ordinary edges to the new
  node. If the new node is on fewer layers than `A` was, a surviving link on a higher layer points at
  a node not on that layer: search finds no neighbors there, and linking adds no backlink to it.
  Edges pruned to make room for `A`'s backlinks stay lost.

## 3. Neighbor Selection (Diversity Heuristic)

To prevent "cluster collapse" and ensure the graph maintains its Small World properties, Chassis implements a variant of **Heuristic 2** from the HNSW paper.

### The Selection Algorithm

When a neighbor list exceeds its capacity (`M` or `M0`), we prune it using **Diversity Selection**:

1. **Candidate Set**: We gather the current neighbors plus the new node.
2. **Diversity Check**: We keep a candidate `C` only if it is closer to the base node `B` than it is to any *already-selected* neighbor.
* *Goal*: Prioritize neighbors in different directions rather than just the closest ones.


3. **Starvation Fallback**: If diversity pruning is too aggressive and yields fewer than `M/2` neighbors, we fill the remaining slots with the nearest remaining candidates. This ensures robust connectivity even in highly clustered data.

A new node chooses from all of its `ef_construction` search candidates. Each candidate is compared
with each neighbor already kept at most once, so there is nothing to cache. An earlier lazy distance
cache never got a hit and limited selection to 33 candidates; it was removed (see the
[ADR-0004 amendment](../adr/004-diversity-heuristic-with-lazy-cache.md#amendment-2026-10-04)).

## 4. Idempotency & Retry Safety

The `link_node_bidirectional` operation is **idempotent**. It is safe to retry the operation if a crash occurs mid-update.

* **Duplicate Detection**: Before adding a backward link (`B` → `A`), the engine checks if `B` already links to `A`.
* **No Side Effects**: Retrying a completed operation results in zero disk writes.

## 5. Edge Case Policies

| Case | Policy | Rationale |
| --- | --- | --- |
| **Self-Links** | **Filtered** | A node linking to itself wastes capacity and offers no search value. |
| **Invalid IDs** | **Error** | Linking to an ID `>= node_count` returns an explicit error. |
| **Layer Independence** | **Enforced** | Neighbor lists are processed independently per layer; candidates are not shared across layers. |
| **Identical Vectors** | **Fallback** | If all vectors are identical, the diversity heuristic fails; the *Starvation Fallback* ensures connectivity. |
