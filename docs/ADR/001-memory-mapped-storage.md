# ADR-0001: Memory-Mapped Storage Strategy

**Date:** 2026-01-18  
**Status:** Accepted

## Context

Chassis is designed to operate across a wide range of environments, from resource-constrained edge devices (e.g., Raspberry Pi) to high-memory servers. A core requirement is the ability to store and search vector datasets that may exceed available physical RAM.

Traditional approaches present fundamental limitations:

* Standard file I/O (`read` / `write`) incurs kernel-to-user copying overhead and deserialization costs, resulting in microsecond-scale access latency.
* Loading the full dataset into heap-allocated structures (`Vec<f32>`) causes long startup times and introduces hard memory ceilings, often leading to OOM failures.
* Copy-based access patterns prevent the system from fully exploiting CPU cache locality during distance computation.

The system therefore requires a persistence mechanism that:

* supports datasets larger than RAM,
* provides pointer-dereference-level access latency (~ns),
* minimizes startup time,
* and integrates cleanly with zero-copy search and linking logic.

## Decision

Chassis will use **memory-mapped I/O (`mmap`) as the exclusive mechanism for data access and persistence**.

The design has the following properties:

1. **Single File Backing**
   The entire index—vectors, graph topology, and metadata—is stored in a single contiguous file on disk.

2. **OS-Managed Caching**
   Page caching, eviction (LRU), and write-back behavior are delegated entirely to the operating system’s virtual memory manager (VMM).

3. **Zero-Copy Read Path**
   All reads are performed by casting raw pointers (`*const u8`) into typed views (`&[f32]`, `&NodeRecord`) directly within the mapped address space. No intermediate buffers or heap allocations are permitted on the read path.

4. **Cross-Platform Abstraction**
   The implementation relies on the `memmap2` crate, providing a safe and portable abstraction over `mmap` (POSIX) and `CreateFileMapping` (Windows).

## Consequences

### Positive

* **Zero-Copy Access**
  Retrieving a vector becomes pure pointer arithmetic. On a cache hit, access latency is on the order of a few nanoseconds, enabling high arithmetic density during distance computation.

* **Instant Startup**
  Opening a large index (e.g., 100GB) completes in milliseconds, as only the virtual address space is mapped; no data is eagerly loaded into RAM.

* **Burst-Friendly Durability**
  Writes occur at memory speed, with dirty pages flushed asynchronously by the OS. Explicit durability boundaries are provided via `commit()` (fsync), allowing callers to trade latency for persistence guarantees.

### Negative

* **Use of `unsafe` Rust**
  Memory-mapped access requires `unsafe` blocks. Chassis assumes responsibility for validating alignment, bounds, and structural integrity before casting pointers to typed references.

* **Non-Standard I/O Failure Modes**
  Hardware-level I/O failures (e.g., disk full, corrupted blocks) may manifest as `SIGBUS` rather than recoverable `Result::Err` values. While rare in typical deployments, this requires operational awareness.

* **Complex File Growth Semantics**
  The backing file cannot be grown transparently. Expanding capacity requires explicit truncation and re-mapping (`ensure_capacity`), which invalidates all previously derived pointers. As a result, references into the mmap must never be held across insertion or growth boundaries.

## Compliance

This decision is enforced throughout the codebase:

* `Storage::get_vector_slice` returns `&[f32]` backed directly by the memory map, strictly enforcing zero-copy access.
* `Storage::ensure_capacity` manages file growth and re-mapping, ensuring page-aligned (4KB) expansion.
* All file access in `chassis-core` is routed exclusively through the `Storage` abstraction; direct use of `std::fs::File` for reads or writes is prohibited.
