# Chassis

Chassis is an embeddable, on-disk vector storage engine written in Rust.

It is designed to be used as a local storage component for vector similarity search. Chassis runs in-process, stores data on disk, and does not require a server or external dependencies.

The project is early-stage and focused on establishing a correct, stable storage core.

## Current Capabilities

Chassis provides a high-performance vector storage, graph construction, and search core:

### Universal Interface (New in v0.5.0)
* **Stable C ABI**: A fully compliant C-compatible FFI layer enables Chassis to be embedded in C, C++, Python, Node.js, and Go.
* **Safety Fortress**: The `ffi_guard` architecture guarantees that Rust panics never crash the host process.
* **Opaque Handle Design**: Uses the "Pimpl" pattern to ensure ABI stability while hiding internal Rust implementation details.

### High-Level API
* **`VectorIndex` Facade**: A clean, unified entry point that orchestrates storage, compute, and graph operations.
* **Consistency Orchestration**: Automates the "Register Last" insertion protocol to guarantee readers never see uninitialized data.
* **Ghost Node Recovery**: Automatically detects and recovers from partial writes during power loss ([ADR-005](https://github.com/tanvincible/chassis/blob/main/docs/src/adr/005-crash-consistent-linking.md)).

### Storage Layer
* **Zero-Copy I/O**: Memory-mapped vectors allow accessing 1536d embeddings in nanoseconds.
* **ACID Persistence**: Explicit `fsync`-backed commit strategy.
* **Fixed-Width Geometry**: O(1) deterministic addressing for all on-disk lookups.

### Graph Layer
* **HNSW Construction**: Fully persistent, crash-safe graph topology builder.
* **Bidirectional Linking**: Maintains graph navigability with "Small World" guarantees.
* **Diversity Heuristics**: Implements robust neighbor pruning (Heuristic 2) to prevent clustering.
* **Crash Consistency**: Atomic write ordering ensures the graph structure is never corrupted, even on power loss.

### Search Layer
* **SIMD Acceleration**: Hardware-accelerated distance kernels (AVX2 for x86, NEON for ARM) provide ~23 Gelem/s throughput.
* **Zero-Allocation Traversal**: The hot search path allocates no heap memory, ensuring consistent P99 latency.
* **High Performance**: Achieves sub-50µs latency for 1536d vectors (OpenAI embeddings) on commodity hardware.

## Design Principles

Chassis prioritizes:

* Correctness over feature breadth
* Explicit invariants over implicit behavior
* Local-first operation with predictable performance
* Simple, inspectable file formats

The storage layer is intentionally conservative. Durability, growth strategy, and concurrency semantics are defined explicitly and documented.

## Non-Goals

Chassis does not aim to be:

* A database server
* A cloud service
* A distributed system
* A query engine

These concerns are intentionally left to the embedding application.

## Status

**Alpha (v0.5.0)**

The core storage engine and C FFI layer are feature-complete.

* **Core Engine**: Stable Rust API (`VectorIndex`) with crash consistency.
* **Universal Interface**: Stable C ABI (`chassis.h`) for embedding.

Work is currently underway on **native language bindings** (`chassis-py` and `chassis-node`) to make the engine installable via `pip` and `npm`.

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome.

The project currently prioritizes correctness, simplicity, and clear invariants over feature breadth. See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for details.
