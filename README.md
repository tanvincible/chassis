# Chassis

Chassis is an embedded vector index for local semantic search, written in Rust.

Your index is one file, and Chassis runs inside your process: no server to run, nothing to connect to. Store embeddings under your own ids, search them, and keep the rest of your data where it already lives.

```python
from chassis import VectorIndex

index = VectorIndex("notes.chassis", dimensions=768)
index.add_batch(embeddings, ids=note_ids)
index.flush()

for hit in index.search(query, k=5):
    print(hit.id, hit.distance)
```

An index opens in about a millisecond: only its headers are read until a search needs more. At the same recall, it searches a million 128-dimension vectors faster than hnswlib on every server CPU it was measured on, x86 and ARM.

## Current Capabilities

* One file, used in-process, with no server
* Approximate nearest-neighbor search (HNSW) by Euclidean or cosine distance
* Filtered search, over the ids your application allows
* Your own ids, and deletes
* Batch builds on every core
* Half-precision vectors, for half the file and half the memory
* Durable `flush()`: after a crash, everything up to the last one is kept
* One writer and any number of readers, in other processes too
* Errors that say what was wrong and what to do instead
* Rust, C and Python APIs

It is not a database server or a distributed system, and it stores vectors and ids, not metadata.

## Learn More

* [Getting started](docs/src/guide/getting-started.md), and the [Python bindings](pychassis/README.md)
* [How it works](docs/src/architecture/overview.md), and why, in the [decision records](docs/src/adr)
* [Benchmarks](docs/src/architecture/performance.md), and what is [planned](ROADMAP.md)

## Status

Early, and moving fast. The last release is v0.6.3; `main` is well ahead of it, in a new file format, and not released yet. See [CHANGELOG.md](CHANGELOG.md).

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
