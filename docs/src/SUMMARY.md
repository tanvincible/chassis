# Summary

[Introduction](./introduction.md)

# User Guide

- [Getting Started](./guide/getting-started.md)
- [Basic Usage](./guide/basic-usage.md)
- [API Reference](./guide/api-reference.md)
- [C API Reference](./guide/c-api.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Storage Layer](./architecture/storage.md)
- [Graph Topology & Construction](./architecture/graph.md)
- [File Format](./architecture/file-format.md)
- [Performance Guide](./architecture/performance.md)

# Architectural Decision Records

- [ADR-0001: Memory-Mapped Storage](./adr/001-memory-mapped-storage.md)
- [ADR-0002: Sequential Graph Construction](./adr/002-sequential-graph-construction.md)
- [ADR-0003: Single-Writer Concurrency](./adr/003-single-writer-concurrency.md)
- [ADR-0004: Diversity Heuristics & Caching](./adr/004-diversity-heuristic-with-lazy-cache.md)
- [ADR-0005: Crash-Consistent Linking](./adr/005-crash-consistent-linking.md)
- [ADR-0006: SIMD Acceleration](./adr/006-simd-acceleration.md)
- [ADR-0007: Caller Ids and Deletes](./adr/007-ids-and-deletes.md)
- [ADR-0008: File Format v3 and Multi-Process Readers (Proposed)](./adr/008-format-v3-and-multi-process-readers.md)
- [ADR-0009: Filtered Search (Proposed)](./adr/009-filtered-search.md)
- [ADR-0010: Parallel Batch Builds (Proposed)](./adr/010-parallel-batch-builds.md)
- [ADR-0011: Compaction (Proposed)](./adr/011-compaction.md)
- [ADR-0012: Rolling Back Lost Adds (Proposed)](./adr/012-rolling-back-lost-adds.md)
- [ADR-0013: Prefetching into L2, a Leaner Search Loop, and a Smaller Fill (Proposed)](./adr/013-prefetching-and-a-leaner-search-loop.md)
- [ADR-0014: Packed Heap Entries, Squared Distances, and a Search Loop That Inlines (Proposed)](./adr/014-packed-heap-entries-and-squared-distances.md)
- [ADR-0015: Prefetch Depth for Long Vectors (Proposed)](./adr/015-prefetch-depth-for-long-vectors.md)
- [ADR-0016: Huge Pages for the Vectors, on Request (Proposed)](./adr/016-huge-pages-on-request.md)
- [ADR-0017: Compare at Equal Recall; Long Vectors Out of Cache Are Bound by Memory (Proposed)](./adr/017-equal-recall-and-long-vectors.md)
- [ADR-0018: Half Precision for the Vectors, on Request (Proposed)](./adr/018-half-precision.md)
- [ADR-0019: Reading an Index into Memory in the Background, on Request (Proposed)](./adr/019-warm.md)
- [ADR-0021: Distances Computed in Groups (Proposed)](./adr/021-grouped-distances.md)

# Development

- [Building from Source](./dev/building.md)
- [Running Tests](./dev/testing.md)
- [Benchmarking](./dev/benchmarking.md)
- [Contributing Guidelines](./dev/contributing.md)
