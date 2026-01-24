# ADR-0005: Crash-Consistent Linking Protocol

**Date:** 2026-01-24  
**Status:** Accepted

## Context

Chassis persists graph mutations directly into a memory-mapped file. A process crash or power loss during a write operation can leave persistent state in a partially updated form.

Common corruption scenarios in graph databases include:

1. **Dangling Pointers:** A node refers to a neighbor ID that has not yet been allocated or written.
2. **Torn Logical Updates:** A partial mutation leaves a node record in a structurally inconsistent state.
3. **Topology Partitioning:** A crash leaves the graph disconnected due to lost or partially applied edges.

Traditional systems address these issues using a Write-Ahead Log (WAL). However, WALs introduce additional I/O (double writes), recovery complexity (log replay, checkpoints), and operational overhead. For Chassis, which prioritizes low-latency access and simplicity, a WAL is undesirable.

We require a lighter-weight mechanism that guarantees structural integrity at all times without a recovery phase.

This ADR assumes the guarantees provided by modern operating systems and filesystems for memory-mapped writes (page-level coherence and ordering). It addresses logical and structural consistency, not arbitrary bit-level corruption due to faulty hardware.

## Decision

We implement a **Strict Ordering Protocol** for all graph mutations.

This protocol ensures that the persistent file is always in a structurally valid state, regardless of when a crash occurs.

### The Atomic Write Sequence

All graph updates must follow this exact order:

1. **Persist Node Record (Forward Links)**
   The new node `A` is written to disk first, including all of its outgoing edges (`A → neighbors`). The global `node_count` is not incremented at this stage.

2. **Update Neighbors (Backward Links)**
   Each neighbor referenced by `A` is updated individually to include a backlink to `A`. These updates may complete partially if a crash occurs.

3. **Update Header**
   The global `node_count`, `entry_point`, and related metadata are updated in the file header. Only after this step is `A` considered visible to the system.

This protocol relies on sequential node IDs (ADR-0002) and a single-writer model (ADR-0003).

### Ghost Node Acceptance

We explicitly accept one benign inconsistency class: **ghost nodes**.

If a crash occurs after Step 1 but before Step 3, node `A` exists physically on disk but is not reachable:

* Its ID is greater than the header’s `node_count`
* It is unreachable from the graph entry point
* Readers ignore it entirely

Because file offsets are derived from the header’s `node_count`, future insertions will overwrite this region by construction. No garbage collection or recovery pass is required.

## Consequences

### Positive

#### No Write-Ahead Log

We eliminate journaling entirely. Persistence is handled with a single file and a single write path, halving write amplification and significantly simplifying the storage engine.

#### Structural Corruption Immunity

The ordering protocol guarantees that we never persist references to uninitialized or invalid data:

* Crash before Step 1: No observable change.
* Crash during Step 2: One-way edges may exist. These are legal in HNSW and do not break search.
* Crash before Step 3: The node is a ghost and safely ignored.

At all times, the on-disk graph remains structurally valid.

#### Zero-Recovery Startup

Opening the database requires no log replay, scanning, or consistency verification. Startup time is constant, regardless of index size.

### Negative

#### Serialized Mutation Path

The protocol enforces strictly ordered writes, preventing parallel mutation of the graph. Write throughput is therefore bounded by a single thread. This is an intentional tradeoff in favor of correctness and simplicity.

#### Temporary Space Loss on Crash

A crash after Step 1 may leave unused space corresponding to a ghost node. In the current append-only, monotonic-ID design, this space is deterministically reclaimed on the next insertion. Long-term reclamation or compaction is deferred as future work.

## Compliance

* **Code Structure:** `link_node_bidirectional` is explicitly structured to follow the Step 1 → Step 2 → Step 3 sequence.
* **Invariant Enforcement:** Any reordering of these steps is treated as a correctness bug.
* **Header Authority:** The storage layer treats the header’s `node_count` as the sole source of truth, ignoring any data beyond it during initialization and traversal.
