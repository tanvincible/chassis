# ADR-0003: Single-Writer / Multi-Reader (SWMR) Concurrency

**Date:** 2026-01-24  
**Status:** Accepted

## Context

Chassis is designed as an embedded, in-process storage engine backed by memory-mapped files. It must safely support concurrent access from multiple threads (and potentially multiple processes) without data corruption, while prioritizing low-latency search.

HNSW graph mutation is particularly sensitive to concurrency:

* **Race Conditions:** Concurrent writers updating adjacency lists can overwrite each other, corrupting graph topology.
* **Reader Safety:** Readers traversing memory-mapped structures must never observe half-written records, dangling pointers, or invalid node IDs.
* **Lock Contention:** Fine-grained locking (e.g., per-node mutexes) introduces memory overhead, implementation complexity, and search-path latency due to lock acquisition and branch misprediction.

The system must therefore prioritize **search latency and correctness** over write parallelism.

## Decision

Chassis adopts a **Single-Writer / Multi-Reader (SWMR)** concurrency model.

### Enforcement Mechanisms

1. **Exclusive Write Access (Inter-process):**
   When the index is opened in write mode, the underlying file is protected by an exclusive OS-level lock (`flock` / `LockFile`). This prevents multiple processes from mutating the same index concurrently.

2. **Exclusive Write Access (In-process):**
   All mutating APIs (`insert`, `link_node_bidirectional`, `ensure_capacity`) require `&mut self`, ensuring via Rust’s borrow checker that only one writer exists within a process at any time.

3. **Lock-Free Readers:**
   Readers operate on immutable memory-mapped data structures and never acquire node-level locks. Search APIs accept `&self` and may execute concurrently across threads without synchronization.

4. **Crash-Safe Write Ordering:**
   Writers follow a strict persistence order (node record → backward links → header update), ensuring readers only observe valid, prefix-consistent graph states (as defined in ADR-0002).

## Consequences

### Positive

#### Lock-Free Search Path

Search operations never acquire mutexes or perform existence checks. If a neighbor ID is present in an adjacency list, it is guaranteed to resolve to valid data. This enables consistently low-latency queries (P99).

#### Strong Corruption Guarantees

By serializing all mutations through a single writer, we eliminate entire classes of concurrency bugs, including race conditions, torn writes, and deadlocks.

#### Implementation Simplicity

The design avoids complex lock hierarchies, reference counting, or hazard pointer schemes. The OS kernel handles inter-process exclusion, while Rust enforces in-process safety.

### Negative

#### Limited Write Throughput

All insertions are serialized through a single writer thread. Parallel graph construction is not supported.
*Mitigation:* Batch insertions to amortize I/O and pruning costs.

#### Reader Constraints During Remap

When the file grows (`ensure_capacity`), the writer must remap the file, invalidating existing memory mappings.

**Invariant:**
Readers must not retain mmap-backed references across writer-induced remap boundaries. Search operations must be scoped to a stable mapping.

## Compliance

* **Initialization Lock:** `Storage::open` attempts to acquire an exclusive file lock in write mode and returns an error if the lock cannot be obtained.
* **Borrowing Discipline:** Search APIs require `&self`; mutation APIs require `&mut self`, enforcing SWMR at compile time.
* **Safe Traversal:** Neighbor iteration assumes all IDs are valid and performs no runtime existence checks.
