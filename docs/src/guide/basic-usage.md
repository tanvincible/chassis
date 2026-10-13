# Basic Usage

## Workflow Overview

Using Chassis involves three steps:
1. **Open**: Initialize the `VectorIndex`.
2. **Write**: Add vectors and flush to disk.
3. **Read**: Search for nearest neighbors.

## 1. Initialization

```rust
use chassisdb::{VectorIndex, IndexOptions};

let options = IndexOptions {
    max_connections: 16,
    ef_construction: 200,
    ef_search: 50,
    ..IndexOptions::default()
};

let mut index = VectorIndex::open("./data.chassis", 1536, options)?;
```

## 2. Insertion

Chassis assigns a sequential ID (`u64`) to every vector you add. You should store this ID in your external database (e.g., SQLite/Postgres) to map it back to your application data.

```rust
let embedding = compute_embedding("Example text"); // [f32; 1536]

let id = index.add(&embedding)?;
println!("Stored vector at ID: {}", id);

// Important: Commit to disk!
index.flush()?;
```

Dropping the index without `flush()` discards what was added since the last one, as a crash
would. To store vectors under your application's own ids instead, use `add_with_id(id, &embedding)`.

## 3. Search

`search` takes `&self`, so many threads can search one `VectorIndex` at once; `add`, `delete` and
`flush` take `&mut self`, so they wait for those searches. To search while it writes, open an
`IndexReader` on the same file, in another thread or another process: it sees what the writer's
last `flush()` committed.

```rust
let query = compute_embedding("Search query");
let k = 5;

let results = index.search(&query, k)?;

println!("Found {} matches:", results.len());
for res in results {
    println!("- ID: {}, Score: {}", res.id, res.distance);
}
```
