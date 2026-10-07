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

* **Crash behavior**: If the process dies before the next `flush()`, this vector is gone on reopen. The index stays usable.
* **Ids**: `add` returns one past the largest id used so far (0, 1, 2... unless you chose ids yourself). It never reuses an id, even after a delete.
* **Durability**: Data is written to memory-mapped pages immediately but requires `flush()` for persistence guarantees.

**Errors**:

* Dimension mismatch.
* Storage write failure (e.g., disk full).

#### Your Own Ids

```rust
index.add_with_id(42, &vector)?;
```

Search results report `42`. Fails if a live vector already has that id; delete it first to replace
it. `u64::MAX` is reserved. Indexes that use their own ids build an id table in memory on the first
lookup in a process, which scans the index once ([ADR-0007](../adr/007-ids-and-deletes.md)).

#### Adding Many Vectors

```rust
let vectors: Vec<f32> = rows.concat(); // count × dims floats, back to back
let ids = index.add_batch(&vectors)?;  // or index.add_batch_with_ids(&my_ids, &vectors)?
```

Links the batch on every core, so a large batch builds many times faster than `add` in a loop
([ADR-0010](../adr/010-parallel-batch-builds.md)). Ids are assigned as `add` assigns them. A batch is
added whole or not at all. The graph depends on thread timing, so two builds of the same data
differ slightly, as sequential builds already do through their random layers.

#### Deleting

```rust
let deleted = index.delete(42)?; // false if no live vector has id 42
```

Search stops returning the vector immediately, and `len()` drops by one. The delete is durable
after the next `flush()`; a crash before then rolls it back. A delete and an add in the same flush
are all-or-nothing, so `delete(id)` followed by `add_with_id(id, new_vector)` replaces a vector
safely. Deleted vectors keep their disk space until `compact()`.

#### Compacting

```rust
index.compact()?;
```

Rewrites the index without its deleted vectors and with a newly built graph, then replaces the
file with the copy ([ADR-0011](../adr/011-compaction.md)). Ids don't change, and `add` still never
reuses one. Like `flush()`, it makes every add and delete so far durable. It takes as long as
building the index, on every core, and needs free disk for a second copy of the live vectors.
Readers in other processes keep searching and move to the new file by themselves. On Windows it
fails, leaving the index as it was, while another process has the index open.

#### Searching

```rust
let query = vec![0.5; 768];
let k = 10; // Neighbors to retrieve

let results = index.search(&query, k)?;

for match in results {
    println!("ID: {}, Distance: {}", match.id, match.distance);
}
```

**Returns**: `Vec<SearchResult>`, sorted by distance (nearest first), identified by the ids from `add`/`add_with_id`. Deleted vectors are never returned.

#### Filtered search

```rust
// The ids your application allows, e.g. from `SELECT id FROM docs WHERE owner = ?`
let allowed: HashSet<u64> = owner_doc_ids();
let results = index.search_filtered(&query, 10, |id| allowed.contains(&id))?;
```

Only vectors whose id the filter accepts are returned. The filter is called with ids from
`add`/`add_with_id`, many times per search, so it should be cheap. When the filter passes many
vectors, the search walks the graph as usual; when it passes few, or its matches lie far from the
query, walking the graph would visit most of the index to find them, so the search checks every
vector instead and returns the exact nearest. Measured, a filtered search takes at most about twice an exact scan of the matching
vectors ([ADR-0009](../adr/009-filtered-search.md)). `IndexReader` has the same method.

#### Persistence

```rust
// Flush all pending writes to physical disk (fsync)
index.flush()?;
```

**Recommendation**: `flush()` is an expensive syscall. Call it after a batch of insertions (e.g., every 1,000 vectors) or before shutting down.

#### Metadata

```rust
let len = index.len();           // Live vectors (deleted ones excluded)
let dim = index.dimensions();    // Vector size
let metric = index.metric();     // DistanceMetric::Euclidean or ::Cosine
let empty = index.is_empty();    // True if count == 0
```

### `IndexReader`

Searches an index while a `VectorIndex` in another process writes it. It takes no lock, so any
number of processes can open readers on one file next to its single writer
([ADR-0008](../adr/008-format-v3-and-multi-process-readers.md), decision 5).

```rust
use chassis_core::{IndexOptions, IndexReader};

let mut reader = IndexReader::open("embeddings.chassis", 768, IndexOptions::default())?;
let results = reader.search(&query, 10)?;
```

Every search first takes a new snapshot: it returns what the writer's newest `flush()` committed,
and nothing that flush deleted, while routing through vectors added since, so a long unflushed batch
costs readers little recall. A flush becomes visible just before its final fsync, so one that then
fails, or is lost to power loss, may already have been seen. `len()` reports the last snapshot;
`refresh()` takes a new one without searching, and `snapshot()` identifies it, for keying a cache.
`search` takes `&mut self`: open one reader per thread. The file must exist in the current format;
open it once with `VectorIndex::open` to create or migrate it.

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

    /// `DistanceMetric::Euclidean` (default) or `DistanceMetric::Cosine`.
    pub metric: DistanceMetric,
}
```

The metric is fixed when the index is created: reopening with another one is an error, an
`IndexReader` uses the file's, and `metric()` on either reports it. A cosine index stores vectors
scaled to unit length, rejects zero vectors and vectors with NaN or infinite components, and
reports `1 - cosine similarity`, from 0 to 2. Search speed is the same for both.

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
    
    /// Distance from the query by the index's metric
    pub distance: f32,
}

```
