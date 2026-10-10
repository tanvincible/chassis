# Introduction

Chassis is an embedded vector index for local semantic search, written in Rust. It keeps an index in one file and runs inside your process, with no server to run or connect to.

## What Chassis Does

Chassis stores vectors under your ids and searches them. It manages a single memory-mapped file that holds both the vectors and a persistent HNSW graph over them.

### Key Capabilities

* **Vector Similarity Search**: Performs approximate nearest neighbor (ANN) search using a fully persistent HNSW graph.
* **High-Level Orchestration**: The `VectorIndex` facade manages the complexity of coordinate storage, graph topology, and search logic.
* **Crash Consistency**: After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. ([ADR-005](https://github.com/tanvincible/chassis/blob/main/docs/src/adr/005-crash-consistent-linking.md)).
* **Memory-Mapped Access**: Searches read vectors straight from the operating system's page cache, so opening an index loads nothing and takes about a millisecond.

## What Chassis Does Not Do

Chassis is intentionally limited in scope to ensure correctness and performance. It is **not**:

* **A Database Server**: There is no network listener, SQL interface, or daemon.
* **A Distributed System**: Replication and sharding are left to the application layer.
* **A Metadata Store**: Chassis stores vectors and IDs only. You should map these IDs to your application data (JSON, text, etc.) using a separate store like SQLite, and pass the ids it selects to `search_filtered` to search among them.

## Current Status

The latest release is **v0.7.1**: deletes and your own ids, filtered search, compaction, readers
in other processes, half precision and more, in a new file format that v0.6.3 can't open. Opening
an older file for writing converts it. Search, the C API and the Python bindings work end to end.
Measured numbers are in the [decision records](./adr) and on the
[Performance](./architecture/performance.md) page. To replace a vector, delete its id and add it
again.

## Requirements

* Rust 1.88 or later
* A filesystem that supports memory mapping (Linux, macOS, Windows) and `fsync` for durability.

## License

Chassis is dual-licensed under MIT and Apache 2.0. You may use either license at your option.
