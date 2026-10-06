# Architecture Overview

Chassis is a high-performance, embedded vector database built on three foundational architectural principles:

## Core Design Pillars

### 1. Memory-Mapped Storage (ADR-0001)

Chassis uses memory-mapped I/O (`mmap`) as its exclusive persistence mechanism, providing:

- **Zero-copy access**: Direct pointer arithmetic to vectors without heap allocation
- **Instant startup**: Opening 100GB+ indices in milliseconds (only virtual address space is mapped)
- **OS-managed caching**: Delegates page management to the kernel's VMM
- **Burst-friendly durability**: Memory-speed writes with async kernel flushes

**Trade-off**: Requires `unsafe` Rust and careful lifetime management. Since file format 3 the file
grows by appending regions that are each mapped once, so growth never invalidates a mapping
([ADR-0008](../adr/008-format-v3-and-multi-process-readers.md)).

### 2. Sequential Construction (ADR-0002)

The graph enforces **strict monotonic node insertion** (0, 1, 2, ...):

```text
Invariant: A node with ID N may only link to neighbors M where M < N
```

**Benefits**:
- Zero-check search path (no existence validation)
- Crash-safe by design (no dangling forward pointers)
- Deterministic O(1) addressing: a slot's segment and index follow from its number

**Trade-off**: Limits parallel construction without a merge phase.

### 3. Single-Writer Concurrency (ADR-0003)

Chassis implements a **SWMR (Single-Writer, Multi-Reader)** model:

- **Writers**: `&mut self` borrow, one writer at a time
- **Readers**: Concurrent `&self` searches from many threads of the same process
- **Other processes**: Writers are kept out by an exclusive file lock; any number of `IndexReader`s
  search without a lock, each search on the writer's newest flush (ADR-0008, decision 5)

**Benefits**:
- Lock-free search (no mutex acquisition overhead)
- No data races between threads

**Trade-off**: Serialized mutation path.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Chassis Application Layer                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  HNSW Graph      │         │  Storage Layer   │          │
│  │  - Search        │◄────────┤  - Vectors       │          │
│  │  - Linking       │         │  - Mmap Manager  │          │
│  │  - Pruning       │         │  - Zero-copy     │          │
│  └──────────────────┘         └──────────────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              Memory-Mapped File (Single File)               │
│  ┌────────────┬──────────────────────────┬────────────────┐ │
│  │  Header    │  Segments: slot headers, │  Upper heap,   │ │
│  │  A and B   │  vectors, level-0 lists  │  table pages   │ │
│  └────────────┴──────────────────────────┴────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
        Kernel Page Cache (LRU eviction)
              │
              ▼
        Physical Storage (SSD/NVMe)
```

## Performance Characteristics

Measured numbers, the machine they came from and how to reproduce them are in
[Performance](./performance.md).

## Component Responsibilities

### Orchestration Layer (`lib.rs`)
- **`VectorIndex`**: The public facade. It manages the `Storage` and `HnswGraph` instances, ensuring that all operations follow the **Crash Consistency Protocol** (e.g., correct write ordering). Slots written after the last flush are reused by the next add.

### Storage Layer (`storage.rs`)
- **File lifecycle**: Open, migration from formats 1 and 2, exclusive locking
- **Slots**: Append-only; the file grows by appending segments and heap chunks, and committed
  regions never move
- **Zero-copy reads**: `get_vector_slice()` returns `&[f32]` backed by mmap
- **Durability**: Two checksummed header copies; a commit fsyncs data, then writes the other copy

### HNSW Graph (`hnsw/graph.rs`)
- **Topology management**: Node records and adjacency lists, read and written through `Storage`
- **O(1) addressing**: A slot's segment and index are computed, not looked up
- **Persistence**: Write-ahead ordering (node → neighbors → header)
- **Traversal**: Neighbor iteration reads the mmap directly via `neighbors_iter_from_mmap()`

### Distance Metrics (`distance.rs`)
- **SIMD acceleration**: AVX2 (x86_64) and NEON (ARM) intrinsics
- **Fallback**: Portable scalar implementation
- **Optimization**: 4-way accumulator unrolling for pipeline saturation

### Node Layout (`hnsw/node.rs`)
- **In-memory records**: A node's id, layers and neighbor lists while it is linked
- **Format 2 records**: The fixed-size byte layout of format 2, read only to migrate old files

### Linking (`hnsw/link.rs`)
- **Bidirectional edges**: Forward (A→B) and backward (B→A) link maintenance
- **Diversity heuristic**: Heuristic 2 over all `ef_construction` candidates (ADR-0004)
- **Crash consistency**: Ordered write sequence (ADR-0005, see its amendment)

### Search (`hnsw/search.rs`)
- **Dense visited filter**: O(1) array access instead of HashSet hashing
- **Allocations**: Every layer search allocates a `node_count`-bit visited set; layer 0 also allocates two heaps and the result `Vec`
- **NaN-safe ordering**: `f32::total_cmp` for deterministic behavior

## Key Invariants

1. **Node ID density**: IDs must be 0, 1, 2, ..., N without gaps
2. **Forward links validity**: All neighbor IDs < current `node_count`
3. **Mmap stability**: Mappings never move; a reference into one lives as long as its borrow
4. **Write ordering**: Node → backward links → header (crash safety)
5. **Alignment**: All offsets and sizes are 8-byte aligned

## File Format

```text
Offset          Content
─────────────────────────────────────────────────────────
0               Header copy A
64 KiB          Header copy B
128 KiB         Live page, for readers in other processes
192 KiB         Regions: segments, heap chunks, table pages (64 KiB aligned)
```

See [File Format](./file-format.md) for detailed layout specifications.

## Crash Consistency Model

Chassis guarantees structural integrity without Write-Ahead Logging (ADR-0005):

1. **Ghost nodes**: Slots written but not committed (slot >= header count) are ignored and reused
2. **One-way edges**: Incomplete backward links are legal in HNSW and don't break search
3. **Header authority**: the header's slot count is what is committed; readers in other processes
   route past it but never return a slot at or past it
4. **Stale backlinks**: Links to rolled-back nodes are skipped by search until their slot is reused,
   and linking skips a stale link on a layer the reusing node isn't on

After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. Process kills are tested (`chassis-core/tests/crash_tests.rs`), and power loss is simulated (`chassis-core/src/power_loss.rs`), though not on real hardware.

**Recovery**: Opening the index after a crash reads the newest valid header copy. Only a crash during a flush with deletes leaves work: clearing that flush's delete marks, one pass over the slot headers.

## Design Trade-offs Summary

| Decision | Benefit | Cost |
|----------|---------|------|
| Memory-mapping | ns-latency reads, instant startup | `unsafe` Rust, SIGBUS risk |
| Sequential construction | Zero-check search, crash safety | No parallel building |
| SWMR concurrency | Lock-free reads, no races | Serialized writes |
| Level-0 records in slots, upper layers in a heap | Records 13× smaller than format 2's | One heap lookup for the few nodes above layer 0 |
| Diversity heuristic | Better graph quality | O(N·M) pruning complexity |
