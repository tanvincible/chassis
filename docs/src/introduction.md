# Chassis

Chassis is an embedded vector index for local semantic search. Your index is one file, and Chassis
runs inside your process: no server to run, nothing to connect to. Store embeddings under your own
ids, search them, and keep the rest of your data where it already lives.

```sh
pip install chassisdb   # Python 3.11+: import chassisdb
cargo add chassisdb     # Rust: use chassisdb
```

The C library and its header come with each
[release](https://github.com/tanvincible/chassis/releases).

## Example

```python
import numpy as np
from chassisdb import VectorIndex

embeddings = np.random.rand(1000, 768).astype(np.float32)  # stands for your model's output
note_ids = np.arange(1000, 2000)  # the ids your notes already have

index = VectorIndex("notes.chassis", dimensions=768)
index.add_batch(embeddings, ids=note_ids)
index.flush()

query = embeddings[0]
for hit in index.search(query, k=5):
    print(hit.id, hit.distance)  # 1000 first, at distance 0
```

## Start

- **Python**: the [Python docs](https://tanvincible.github.io/chassis/python/), from the quickstart
  to every method.
- **Rust**: [Getting started](./guide/getting-started.md), then the [Rust API](./guide/api-reference.md).
- **C**: the [C API](./guide/c-api.md).
- **Your text and metadata**: [With a database](./guide/with-a-database.md) keeps them in SQLite
  under the same ids as the vectors.
- **Coding agents**: [Coding agents](./guide/agents.md), and [llms.txt](./llms.md) for the docs as
  Markdown.

## Features

- Approximate nearest-neighbor search (HNSW) by Euclidean or cosine distance.
- Filtered search, over the ids your application allows.
- Your own ids, deletes, and `compact()` to reclaim their space.
- Batch builds on every core.
- Half precision, for half the file and half the memory.
- Durable `flush()`: after a crash, reopening keeps every add and delete up to the last flush and
  drops the rest.
- One writer and any number of readers, in other processes too.
- Errors that say what was wrong and what to do instead.

Opening an index reads only its headers. On an Apple M5 (2026-10-15), opening one of 50,000 vectors
of 384 dimensions took 0.08 ms through Python, with the file in memory. At the same recall, Chassis
searches a million 128-dimension vectors faster than hnswlib on every server CPU measured, x86 and
ARM ([Performance](./architecture/performance.md)).

## Non-goals

Chassis is kept small on purpose, so that it stays correct and fast. It is not:

- **A database server**: there is no network listener, query language or daemon.
- **A distributed system**: replication and sharding are left to the application.
- **A metadata store**: Chassis keeps vectors and ids only. Keep your text and metadata in a store
  of your own, such as SQLite, under the same ids, and pass the ids it selects to a filtered search.

## Status

The latest release is **v0.8.0**; the
[changelog](https://github.com/tanvincible/chassis/blob/main/CHANGELOG.md) lists what each release
changed. Files from older releases open, and opening one for writing converts it.

## Requirements

- Python 3.11 or later with numpy 1.26 or later, or Rust 1.88 or later.
- Linux, macOS or Windows, on a file system that supports memory mapping and `fsync`. Wheels and
  the C library are built for Linux (x86-64 and ARM64), macOS (ARM64) and Windows (x86-64).

## License

Chassis is dual-licensed under MIT and Apache 2.0. You may use either license at your option.
