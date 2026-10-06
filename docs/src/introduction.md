# Introduction

Chassis is an embeddable, on-disk vector storage engine written in Rust. It is designed to be the local storage primitive for embedding-based search applications, running directly within your process without external dependencies.

## What Chassis Does

Chassis provides a complete engine for storing and searching high-dimensional vectors. It manages a single memory-mapped file that contains both the raw vector data and a persistent HNSW graph index.

### Key Capabilities

* **Vector Similarity Search**: Performs approximate nearest neighbor (ANN) search using a fully persistent HNSW graph.
* **High-Level Orchestration**: The `VectorIndex` facade manages the complexity of coordinate storage, graph topology, and search logic.
* **Crash Consistency**: After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. ([ADR-005](https://github.com/tanvincible/chassis/blob/main/docs/src/adr/005-crash-consistent-linking.md)).
* **Zero-Copy Access**: Vectors are accessed directly from the OS page cache via memory mapping, providing nanosecond-level read latency.

## What Chassis Does Not Do

Chassis is intentionally limited in scope to ensure correctness and performance. It is **not**:

* **A Database Server**: There is no network listener, SQL interface, or daemon.
* **A Distributed System**: Replication and sharding are left to the application layer.
* **A Metadata Store**: Chassis stores vectors and IDs only. You should map these IDs to your application data (JSON, text, etc.) using a separate store like SQLite, and pass the ids it selects to `search_filtered` to search among them.

## Current Status

**v0.6.3.** Storage, HNSW search, the C ABI and the Python bindings work end to end through
`VectorIndex`. Deleting or updating vectors is not supported yet. Measured numbers are on the
[Performance](./architecture/performance.md) page.

## Requirements

* Rust 1.88 or later
* A filesystem that supports memory mapping (Linux, macOS, Windows) and `fsync` for durability.

## License

Chassis is dual-licensed under MIT and Apache 2.0. You may use either license at your option.
