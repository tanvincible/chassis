# Basic Usage

## Workflow Overview

Using Chassis involves three steps:
1. **Open**: Initialize the `VectorIndex`.
2. **Write**: Add vectors and flush to disk.
3. **Read**: Search for nearest neighbors.

## 1. Initialization

```rust
use chassis_core::{VectorIndex, IndexOptions};

let options = IndexOptions {
    max_connections: 16,
    ef_construction: 200,
    ef_search: 50,
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

## 3. Search

Search is thread-safe and can run concurrently with other readers (but not writers).

```rust
let query = compute_embedding("Search query");
let k = 5;

let results = index.search(&query, k)?;

println!("Found {} matches:", results.len());
for res in results {
    println!("- ID: {}, Score: {}", res.id, res.distance);
}
```
