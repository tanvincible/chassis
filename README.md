# Chassis

Chassis is an embeddable, on-disk vector storage engine written in Rust.

It is designed to be used as a local storage component for vector similarity search. Chassis runs in-process, stores data on disk, and does not require a server or external dependencies.

The project is early-stage and focused on establishing a correct, stable storage core.

## Current Capabilities

* **One file, in-process**: Vectors and an HNSW graph live in a single memory-mapped file. There is no server.
* **Search**: Approximate nearest neighbor search with HNSW, Euclidean (L2) distance only. For cosine similarity, normalize vectors first.
* **Your ids and deletes**: `add_with_id(id, vector)` stores your own `u64` id, which search returns; `delete(id)` removes a vector. A delete and an add in the same `flush()` are all-or-nothing, so replacing a vector is safe ([ADR-0007](docs/src/adr/007-ids-and-deletes.md)).
* **SIMD distance kernels**: AVX2 on x86_64 and NEON on aarch64, with a scalar fallback.
* **Durability**: `flush()` calls msync and fsync. After a crash, reopening keeps every add and delete up to the last `flush()` and drops later ones; graph edges changed after that flush may be partly lost, which can lower recall. Process kills are tested by [`crash_tests.rs`](chassis-core/tests/crash_tests.rs), and power loss is simulated by [`power_loss.rs`](chassis-core/src/power_loss.rs), where each disk sector keeps either its last synced or its latest contents; real hardware is not tested ([ADR-005](https://github.com/tanvincible/chassis/blob/main/docs/src/adr/005-crash-consistent-linking.md)).
* **Concurrency**: One writer and any number of concurrent searches within a process. An exclusive file lock keeps other processes out.
* **Bindings**: A C ABI (`chassis-ffi`) that catches Rust panics at the boundary, and Python bindings (`pychassis/`, version tracks the Rust release, currently **v0.6.3**) with NumPy support. Build `chassis-ffi` (`cargo build --release -p chassis-ffi`), then run `pip install -e .` from `pychassis/`.

Not supported yet: metadata, filtering, and reclaiming the space of deleted vectors.

Performance numbers, the machine they were measured on and how to reproduce them are in [Performance](docs/src/architecture/performance.md).

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

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome.

The project currently prioritizes correctness, simplicity, and clear invariants over feature breadth. See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for details.
