# Introduction

Chassis is an embeddable, on-disk vector storage engine written in Rust. It is designed to be the local storage primitive for embedding-based search applications, running directly within your process without external dependencies.

## What Chassis Does

Chassis provides a complete engine for storing and searching high-dimensional vectors. It manages a single memory-mapped file that contains both the raw vector data and a persistent HNSW graph index.

### Key Capabilities

* **Vector Similarity Search**: Performs approximate nearest neighbor (ANN) search using a fully persistent HNSW graph.
* **High-Level Orchestration**: The `VectorIndex` facade manages the complexity of coordinate storage, graph topology, and search logic.
* **Crash Consistency**: Implements a "Register Last" write protocol ([ADR-005](https://github.com/tanvincible/chassis/blob/main/docs/src/adr/005-crash-consistent-linking.md)). Readers are guaranteed to never see uninitialized data, even if the process crashes mid-write.
* **Zero-Copy Access**: Vectors are accessed directly from the OS page cache via memory mapping, providing nanosecond-level read latency.

## What Chassis Does Not Do

Chassis is intentionally limited in scope to ensure correctness and performance. It is **not**:

* **A Database Server**: There is no network listener, SQL interface, or daemon.
* **A Distributed System**: Replication and sharding are left to the application layer.
* **A Metadata Store**: Chassis stores vectors and IDs only. You should map these IDs to your application data (JSON, text, etc.) using a separate store like SQLite.

## Current Status

**Alpha (v0.4.0)**

The core storage engine and graph algorithms are feature-complete. The API is exposed through the stable `VectorIndex` abstraction.

* **Storage**: Stable.
* **Graph**: Stable (HNSW Heuristic 2).
* **Search**: High performance (~15µs latency).
* **Bindings**: FFI layer is currently in development (Step 9).

## Requirements

* Rust 1.85 or later
* A filesystem that supports memory mapping (Linux, macOS, Windows) and `fsync` for durability.

## License

Chassis is dual-licensed under MIT and Apache 2.0. You may use either license at your option.
