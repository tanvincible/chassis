# Architecture Overview

Chassis is a high-performance, embedded vector database built on three foundational architectural principles:

## Core Design Pillars

### 1. Memory-Mapped Storage (ADR-0001)

Chassis uses memory-mapped I/O (`mmap`) as its exclusive persistence mechanism, providing:

- **Zero-copy access**: Direct pointer arithmetic to vectors without heap allocation
- **Instant startup**: Opening 100GB+ indices in milliseconds (only virtual address space is mapped)
- **OS-managed caching**: Delegates page management to the kernel's VMM
- **Burst-friendly durability**: Memory-speed writes with async kernel flushes

**Trade-off**: Requires `unsafe` Rust and careful lifetime management. File growth invalidates existing pointers.

### 2. Sequential Construction (ADR-0002)

The graph enforces **strict monotonic node insertion** (0, 1, 2, ...):

```text
Invariant: A node with ID N may only link to neighbors M where M < N
```

**Benefits**:
- Zero-check search path (no existence validation)
- Crash-safe by design (no dangling forward pointers)
- Deterministic O(1) addressing: `offset = base + (node_id × record_size)`

**Trade-off**: Limits parallel construction without a merge phase.

### 3. Single-Writer Concurrency (ADR-0003)

Chassis implements a **SWMR (Single-Writer, Multi-Reader)** model:

- **Writers**: `&mut self` borrow, one writer at a time
- **Readers**: Concurrent `&self` searches from many threads of the same process
- **Other processes**: Kept out entirely by an exclusive file lock, readers included

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
│  ┌────────────┬────────────┬──────────────────────────────┐ │
│  │  Header    │  Vectors   │  Graph (Nodes + Adjacency)   │ │
│  │  4KB       │  Dense     │  Fixed-size records          │ │
│  └────────────┴────────────┴──────────────────────────────┘ │
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
- **`VectorIndex`**: The public facade. It manages the `Storage` and `HnswGraph` instances, ensuring that all operations follow the **Crash Consistency Protocol** (e.g., correct write ordering) and handling **Ghost Node Recovery** automatically on startup.

### Storage Layer (`storage.rs`)
- **File lifecycle**: Open, growth, exclusive locking
- **Vector persistence**: Append-only insertion; the file grows 25% at a time
- **Zero-copy reads**: `get_vector_slice()` returns `&[f32]` backed by mmap
- **Durability**: `commit()` performs msync + `fsync` (`F_FULLFSYNC` on macOS)

### HNSW Graph (`hnsw/graph.rs`)
- **Topology management**: Node records, adjacency lists, graph header
- **O(1) addressing**: Direct offset computation without hash maps
- **Persistence**: Write-ahead ordering (node → neighbors → header)
- **Traversal**: Neighbor iteration reads the mmap directly via `neighbors_iter_from_mmap()`

### Distance Metrics (`distance.rs`)
- **SIMD acceleration**: AVX2 (x86_64) and NEON (ARM) intrinsics
- **Fallback**: Portable scalar implementation
- **Optimization**: 4-way accumulator unrolling for pipeline saturation

### Node Layout (`hnsw/node.rs`)
- **Fixed-size records**: Determined by `(M, M0, max_layers)` at index creation
- **Direct addressing**: `offset = graph_start + (node_id × record_size)`
- **Compact representation**: Unused neighbor slots filled with `INVALID_NODE_ID`

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
3. **Mmap stability**: No references held across `ensure_capacity()` calls
4. **Write ordering**: Node → backward links → header (crash safety)
5. **Alignment**: All offsets and sizes are 8-byte aligned

## File Format

```text
Offset          Content                 Size
─────────────────────────────────────────────────────────
0               Storage Header          4KB (page-aligned)
4096            Vector Data             count × dims × 4 bytes
<vector_end>    [padding to 4KB]
<graph_start>   Graph Header            64 bytes
<graph_start>+64 Node Records           node_count × record_size
```

See [File Format](./file-format.md) for detailed layout specifications.

## Crash Consistency Model

Chassis guarantees structural integrity without Write-Ahead Logging (ADR-0005):

1. **Ghost nodes**: Nodes written but not reachable (ID >= header.node_count) are safely ignored
2. **One-way edges**: Incomplete backward links are legal in HNSW and don't break search
3. **Header authority**: `node_count` is the sole source of truth for valid data range
4. **Stale backlinks**: Links to rolled-back nodes are skipped by search until their ID is reused

After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. Process kills are tested (`chassis-core/tests/crash_tests.rs`), and power loss is simulated (`chassis-core/src/power_loss.rs`), though not on real hardware.

**Recovery**: Zero-cost. Opening the index after a crash requires no log replay or validation—just read the header and resume operations.

## Design Trade-offs Summary

| Decision | Benefit | Cost |
|----------|---------|------|
| Memory-mapping | ns-latency reads, instant startup | `unsafe` Rust, SIGBUS risk |
| Sequential construction | Zero-check search, crash safety | No parallel building |
| SWMR concurrency | Lock-free reads, no races | Serialized writes |
| Fixed-size records | O(1) addressing, no pointers | Wasted space for low-layer nodes |
| Diversity heuristic | Better graph quality | O(N·M) pruning complexity |
