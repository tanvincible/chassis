# Architecture Overview

Chassis is designed around three principles: simplicity, correctness, and respect for hardware constraints.

## Memory-Mapped I/O

The entire Chassis file is mapped into the process address space using `mmap`. This allows the operating system to manage paging and caching. When you read a vector, the OS loads only the necessary pages from disk. When the file is idle, the OS can evict pages to reclaim memory.

This approach minimizes resident memory usage and eliminates the need for explicit buffer management.

## Page Alignment

All file growth is aligned to 4KB boundaries. This matches the page size of most operating systems and the block size of modern storage devices. Aligned writes reduce write amplification on SSDs and avoid partial page updates.

The header itself is exactly 4096 bytes, ensuring that vector data begins on a page boundary.

## Single-Writer, Multi-Reader

The current implementation supports a single writer at a time. This is enforced through exclusive file locking at the OS level. Multiple processes cannot open the same Chassis file simultaneously.

This design eliminates the need for row-level locking or multi-version concurrency control. It keeps the implementation small and the failure modes obvious.

Multi-reader support is planned but not yet implemented.

## Data-First Writes

When inserting a vector, Chassis writes the vector data to disk before updating the header count. If a crash occurs between these two steps, the file contains orphaned data but the header remains valid.

This ordering prevents the header from claiming vectors that were never fully written.
