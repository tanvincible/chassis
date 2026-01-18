# Chassis

Chassis is an embeddable, on-disk vector storage engine written in Rust.

It is designed to be used as a local storage component for vector similarity search. Chassis runs in-process, stores data on disk, and does not require a server or external dependencies.

The project is early-stage and focused on establishing a correct, stable storage core.

## Current Capabilities

At its current stage, Chassis provides a minimal vector storage layer with explicit invariants:

* Create or open a `.chassis` file with a validated header
* Append fixed-dimension vectors to disk
* Retrieve vectors by index (returns owned copies)
* Memory-mapped read and write paths
* Explicit durability via commit (`fsync`-backed)
* Exclusive file locking to prevent concurrent corruption

Chassis currently implements **storage only**. Indexing and search are not yet implemented.

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

**Alpha (v0.1.0)**

The API and file format are not yet stable and may change. Performance characteristics and supported features will evolve as the core design solidifies.

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome.

The project currently prioritizes correctness, simplicity, and clear invariants over feature breadth. See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for details.
