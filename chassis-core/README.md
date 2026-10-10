# chassis-core

The core of [Chassis](https://github.com/tanvincible/chassis), an embedded vector index for local semantic search.

Your index is one file, and Chassis runs inside your process: no server to run, nothing to connect to. Store embeddings under your own ids, search them, and keep the rest of your data where it already lives.

```toml
[dependencies]
chassis-core = "0.7"
```

```rust,no_run
use chassis_core::{IndexOptions, VectorIndex};

// `embeddings` holds one 768-dimension vector after another, one per id.
fn index_notes(note_ids: &[u64], embeddings: &[f32], query: &[f32]) -> chassis_core::Result<()> {
    let mut index = VectorIndex::open("notes.chassis", 768, IndexOptions::default())?;
    index.add_batch_with_ids(note_ids, embeddings)?;
    index.flush()?;

    for hit in index.search(query, 5)? {
        println!("{} {}", hit.id, hit.distance);
    }
    Ok(())
}
```

An index opens in about a millisecond: only its headers are read until a search needs more. In half precision, it searches a million 128-dimension vectors faster than hnswlib at the same recall, on every server CPU measured, x86 and ARM.

## Capabilities

* One file, used in-process, with no server
* Approximate nearest-neighbor search (HNSW) by Euclidean or cosine distance
* Filtered search, over the ids your application allows
* Your own ids, and deletes
* Batch builds on every core
* Half-precision vectors, for half the file and half the memory
* Durable `flush()`: after a crash, everything up to the last one is kept
* One writer and any number of readers, in other processes too
* Errors that say what was wrong and what to do instead
* A C API too, and Python as [`chassisdb`](https://pypi.org/project/chassisdb/)

It is not a database server or a distributed system, and it stores vectors and ids, not metadata.

## Learn More

* [Getting started](https://tanvincible.github.io/chassis/guide/getting-started.html), and the [API reference](https://docs.rs/chassis-core)
* [How it works](https://tanvincible.github.io/chassis/architecture/overview.html), and why, in the decision records
* [Benchmarks](https://tanvincible.github.io/chassis/architecture/performance.html)

## License

Dual licensed under the Apache License 2.0 and the MIT License, at your option.
