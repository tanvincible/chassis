# ADR-0002: Sequential Graph Construction Policy

**Date:** 2026-01-24  
**Status:** Accepted

## Context

Constructing an HNSW graph requires establishing edges between nodes. A fundamental architectural choice arises during index construction: whether a node is permitted to reference neighbors that have not yet been persisted.

Two competing models exist:

### Model A: Strict Sequential Construction

A node may link only to neighbors that have already been persisted.

**Pros**

* Guarantees that every edge resolves to valid, initialized data.
* Enables a fully unchecked, branch-free traversal path.
* Simplifies persistence and crash recovery semantics.

**Cons**

* Enforces a strict monotonic insertion order (0, 1, 2, …).
* Limits opportunities for parallel graph construction without a merge phase.

### Model B: Future References (Non-Sequential)

A node may link to neighbor IDs that do not yet exist on disk.

**Pros**

* Enables non-sequential or partially parallel construction strategies.
* Allows speculative or distributed graph planning.

**Cons**

* Creates dangling forward edges pointing to uninitialized or invalid file offsets.
* Requires runtime existence checks during search.
* Complicates crash recovery and persistence guarantees.

---

This decision directly impacts three critical system dimensions:

* **Search latency**
* **Crash consistency**
* **Parallelization potential**

## Decision

Chassis enforces a **Strict Sequential Construction Policy (Model A)**.

### Invariant

> A node with ID **N** may only link to neighbors with IDs **M < N**.

Any attempt to violate this invariant results in filtering or rejection before persistence.

## Enforcement Mechanisms

This policy is enforced at multiple layers:

1. **Filter-on-Write**
   `link_node_bidirectional` automatically filters out any neighbor IDs
   `>= self.node_count` before writing node records to disk.

2. **Monotonic Insertion**
   `insert` requires that the new node ID is **exactly equal** to the current `node_count`.
   Any attempt to skip or reorder IDs results in an error.

## Consequences

### Positive

#### Zero-Check Search Path

The hot traversal path (`search_layer`, `neighbors_iter_from_mmap`) performs **no existence checks**.

If a neighbor ID appears in an adjacency list, it is *guaranteed* to correspond to valid, initialized data on disk. This removes a conditional branch from every edge traversal and improves branch predictor stability.

#### Crash Safety by Design

The system never persists pointers to future file offsets.

In the event of a crash, the graph remains consistent up to the last successfully written node. There are no orphaned edges or forward references requiring repair or validation during recovery.

#### Deterministic Persistence

Graph construction reduces to a linear, append-only write pattern.

There is no need for a secondary “patching” or resolution phase to fix up forward references, greatly simplifying durability and recovery logic.

### Negative

#### Serialized Construction Bottleneck

Graph construction must proceed in strict ID order (0, 1, 2, …).

This limits parallelization across threads or machines without introducing a separate planning or merge phase, which may be costly.

#### Testing Rigidity

Unit tests cannot create sparse or non-monotonic graphs (e.g., nodes 0, 10, 100).

Tests must construct intermediate nodes, resulting in slightly more boilerplate and setup code.

## Compliance

This decision is enforced throughout the implementation:

* **Insertion Invariant**
  `HnswGraph::insert` asserts `vector_id == self.node_count` in both debug and release builds, returning an error if violated.

* **Neighbor Filtering**
  `HnswGraph::link_node_bidirectional` applies
  `.filter(|&id| id < self.node_count)`
  to all neighbor candidates prior to persistence.

* **Safe Traversal**
  `NeighborIterator` assumes all returned IDs are valid and performs **no bounds checks** against the global node count.
