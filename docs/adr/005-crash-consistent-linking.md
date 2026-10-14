# ADR-0005: Crash-Consistent Linking Protocol

**Date:** 2026-01-24  
**Status:** Accepted, amended 2026-10-03 (see [Amendment](#amendment-2026-10-03))

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

## Amendment (2026-10-03)

The Context assumed memory-mapped writes reach disk in program order. They don't: the OS writes
dirty pages back in any order, so the step ordering only holds against a process crash. Only
`flush()` (msync + fsync) orders anything on disk.

Two gaps in the protocol, both fixed on 2026-10-03:

* **Backlinks outlive their node.** Step 2 writes backlinks into existing records, but `node_count`
  is only persisted by `flush()`. After a crash, open rolls the node back while the backlinks
  survive, pointing at an ID the next insert reuses. Pruning such a list read the missing vector and
  failed every later insert. `add_backward_link_with_pruning` now drops IDs at or past `node_count`,
  and search already skipped them. Once the next insert reuses the ID, the surviving backlinks are
  ordinary edges to the new node.
* **Moving the graph zone was not crash-safe.** It copied the zone in place over an overlapping
  range, so a crash mid-copy destroyed the only valid graph. A copy now never overwrites the one the
  header points at (an overlapping move goes past both ranges first), and is fsynced before the
  header switches to it.

What holds now: after a process kill, reopening keeps every vector up to the last `flush()` and the
index takes new inserts, tested by `chassis-core/tests/crash_tests.rs`. Graph edges are not rolled
back: edges pruned after the last flush to make room for backlinks stay lost, which can lower recall.
The same is expected after power loss, because `flush()` is the fsync barrier.

Since 2026-10-08 the edges are rolled back too after a process crash: ADR-0012 saves a committed
node's lists before they change and writes them back on the next open. ADR-0011 measured what the
loss had cost.

Since 2026-10-05, `chassis-core/src/power_loss.rs` simulates power loss. At every fsync and between
operations, each 512-byte sector of a crash image keeps its durable or its current contents, and
every image must reopen to exactly the last completed flush or the one in progress, then keep
accepting writes. Against format 2 it passed 248 images and caught each of five protocol mutants: a
missing fsync before the graph header, a missing fsync before a graph move switches, no dirty flag
before delete marks, in-place overlapping graph moves, and recovery disabled. Format 3 replaced
that code; the simulator's results for it, which also tear sectors and crash inside recovery, are in
the [ADR-0008](008-format-v3-and-multi-process-readers.md) implementation status. It does not
model a sector persisted at an intermediate version, and real hardware is not tested.
