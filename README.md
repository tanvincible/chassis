# Chassis

Chassis is an embeddable, on-disk vector storage engine written in Rust.

It is designed to be used as a local storage component for vector similarity search. Chassis runs in-process, stores data on disk, and does not require a server or external dependencies.

The project is early-stage and focused on establishing a correct, stable storage core.

## Current Capabilities

* One memory-mapped file, used in-process, with no server
* Approximate nearest-neighbor search (HNSW) by Euclidean or cosine distance
* Filtered search, over the ids your application allows
* Your own ids, and deletes
* Batch builds on every core
* Compaction, which drops deleted vectors and rebuilds the graph
* Half-precision vectors, for half the file and half the memory
* Durable `flush()`: after a crash, everything up to the last one is kept
* One writer and any number of readers, in other processes too
* AVX2 and NEON distance kernels, with a scalar fallback
* Rust, C and Python APIs

Not supported yet: storing metadata in the index.

How to use these is in the [guide](docs/src/guide/getting-started.md) and, for Python, the [bindings' README](pychassis/README.md). How they work and what was measured is in the [architecture notes](docs/src/architecture/overview.md) and the [decision records](docs/src/adr). Performance numbers, the machine they were measured on and how to reproduce them are in [Performance](docs/src/architecture/performance.md).

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

**v0.6.3 (Stable)** — May 2026

Patch release: SPDX workspace license, `deny.toml` for `cargo deny`, and `rand` bump (RUSTSEC-2026-0097). See [CHANGELOG.md](CHANGELOG.md).

The storage engine, C FFI layer and Python bindings work end to end. Release history and per-version notes live in [CHANGELOG.md](CHANGELOG.md).

What is planned next is in [ROADMAP.md](ROADMAP.md).

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome.

The project currently prioritizes correctness, simplicity, and clear invariants over feature breadth. See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for details.
