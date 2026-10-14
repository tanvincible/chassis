# Chassis

Chassis is an embedded vector index for local semantic search, written in Rust.

Your index is one file, and Chassis runs inside your process: no server to run, nothing to connect to. Store embeddings under your own ids, search them, and keep the rest of your data where it already lives.

```sh
pip install chassisdb   # Python 3.11+: import chassisdb
cargo add chassisdb     # Rust: use chassisdb
```

The C library and its header come with each [release](https://github.com/tanvincible/chassis/releases).

```python
from chassisdb import VectorIndex

index = VectorIndex("notes.chassis", dimensions=768)
index.add_batch(embeddings, ids=note_ids)
index.flush()

for hit in index.search(query, k=5):
    print(hit.id, hit.distance)
```

An index opens in about a millisecond: only its headers are read until a search needs more. At the same recall, it searches a million 128-dimension vectors faster than hnswlib on every server CPU measured, x86 and ARM.

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

## When to use it

Chassis fits an app that embeds its own data and searches it on the same machine: a desktop or command-line app, a local RAG or agent memory, a cache one process writes while others search it. Other tools fit other shapes:

* [sqlite-vec](https://github.com/asg017/sqlite-vec), when the data is in SQLite already and the search should be part of SQL queries.
* [LanceDB](https://github.com/lancedb/lancedb), when you want one store for vectors and their columns, with filters in the query and versions of the data.
* A server such as pgvector or Qdrant, when many machines write to one index, or other services query it over the network.

## Learn More

* [Getting started](docs/src/guide/getting-started.md), and the [Python bindings](chassisdb/README.md)
* [How it works](docs/src/architecture/overview.md), and why, in the [decision records](docs/src/adr)
* [Benchmarks](docs/src/architecture/performance.md), and what is [planned](ROADMAP.md)
* LangChain: [langchain-chassisdb](integrations/langchain-chassisdb/README.md), a vector store that passes LangChain's standard tests
* Long-term memory for agents, on your machine: [chassisdb-mcp](chassisdb-mcp/README.md), an MCP server built on Chassis
* Coding with an agent? In Claude Code, `/plugin install chassis --marketplace tanvincible/chassis`; for Cursor, Codex and the rest, see [Chassis for coding agents](docs/src/guide/agents.md), and [llms.txt](llms.txt)

## Status

Early, and moving fast. The latest release is v0.8.0. See [CHANGELOG.md](CHANGELOG.md).

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
