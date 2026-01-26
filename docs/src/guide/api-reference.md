# API Reference

## Core Facade

### `VectorIndex`

The primary entry point for Chassis. `VectorIndex` orchestrates the storage engine, graph topology, and search index into a single, crash-consistent unit.

#### Opening an Index

```rust
use chassis_core::{VectorIndex, IndexOptions};

// Open or create a persistent index
let mut index = VectorIndex::open(
    "embeddings.chassis", 
    768, 
    IndexOptions::default()
)?;
```

**Parameters**:

* `path`: Path to the backing file (created if missing).
* `dims`: Vector dimensionality (must be static for the file's lifetime).
* `options`: Configuration for graph construction and search.

**Returns**:

* `Ok(VectorIndex)`: Handle to the index.
* `Err`: If file is locked, corrupted, or dimensions mismatch.

#### Adding Vectors

```rust
let vector = vec![0.5; 768];
let id = index.add(&vector)?;
```

**Behavior**:

* **Atomic**: Follows the "Register Last" protocol. Readers never see uninitialized data.
* **Sequential**: Returns a monotonically increasing `u64` ID (0, 1, 2...).
* **Durability**: Data is written to memory-mapped pages immediately but requires `flush()` for persistence guarantees.

**Errors**:

* Dimension mismatch.
* Storage write failure (e.g., disk full).

#### Searching

```rust
let query = vec![0.5; 768];
let k = 10; // Neighbors to retrieve

let results = index.search(&query, k)?;

for match in results {
    println!("ID: {}, Distance: {}", match.id, match.distance);
}
```

**Returns**: `Vec<SearchResult>`, sorted by distance (nearest first).

#### Persistence

```rust
// Flush all pending writes to physical disk (fsync)
index.flush()?;
```

**Recommendation**: `flush()` is an expensive syscall. Call it after a batch of insertions (e.g., every 1,000 vectors) or before shutting down.

#### Metadata

```rust
let len = index.len();           // Total vectors
let dim = index.dimensions();    // Vector size
let empty = index.is_empty();    // True if count == 0
```

## Configuration

### `IndexOptions`

Parameters tuning the HNSW graph trade-offs between recall, speed, and memory.

```rust
pub struct IndexOptions {
    /// Max connections per node (M). Default: 16
    pub max_connections: u16,
    
    /// Size of the dynamic candidate list during construction. Default: 200
    /// Higher = Better graph quality, slower inserts.
    pub ef_construction: usize,
    
    /// Size of the dynamic candidate list during search. Default: 50
    /// Higher = Better recall, slower search.
    pub ef_search: usize,
}
```

**Tuning Guide**:

* **High Recall**: Increase `ef_construction` to 400 and `max_connections` to 32.
* **Fast Search**: Decrease `ef_search` to 20-30.
* **Low Memory**: Decrease `max_connections` to 8-12.

## Data Types

### `SearchResult`

```rust
pub struct SearchResult {
    /// The internal sequential ID of the vector
    pub id: u64,
    
    /// Euclidean distance from the query vector
    pub distance: f32,
}

```
