# API Reference

## Core Facade

### `VectorIndex`

The primary entry point for Chassis. `VectorIndex` orchestrates the storage layer, graph topology, and search index into a single, crash-consistent unit.

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
let precision = index.precision(); // Precision::Full or ::Half
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

## Errors

`VectorIndex` and `IndexReader` return `chassis_core::Result<T>`, whose error says what was wrong,
with the value, and on a line starting `help:` what to do instead. `Error::kind()` is an
`ErrorKind` to act on ([Errors](./errors.md) lists them, with causes and fixes):

```rust
use chassis_core::{ErrorKind, IndexOptions, IndexReader, VectorIndex};

match VectorIndex::open("embeddings.chassis", 768, IndexOptions::default()) {
    Ok(index) => { /* write */ }
    // One writer at a time; a reader searches while it writes.
    Err(e) if e.kind() == ErrorKind::Locked => {
        let reader = IndexReader::open("embeddings.chassis", 768, IndexOptions::default())?;
    }
    Err(e) => return Err(e.into()),
}
```

`chassis_core::Error` converts into `anyhow::Error` and `Box<dyn std::error::Error>`, so `?` works
in functions that return those.

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

    /// `Precision::Full` (default) or `Precision::Half`.
    pub precision: Precision,

    /// Ask the operating system to keep the vectors on huge pages. Default: false
    pub huge_pages: bool,

    /// Read the index into memory on another thread from the moment it is opened. Default: false
    pub warm: bool,
}
```

The metric is fixed when the index is created: reopening with another one is an error, an
`IndexReader` uses the file's, and `metric()` on either reports it. A cosine index stores vectors
scaled to unit length, rejects zero vectors and vectors with NaN or infinite components, and
reports `1 - cosine similarity`, from 0 to 2. Search speed is the same for both.

`precision` is fixed when the index is created, as the metric is: reopening with another one is
an error, an `IndexReader` uses the file's, and `precision()` on either reports it.
`Precision::Half` keeps each component of a vector as a 16-bit float
([ADR-0018](../adr/018-half-precision.md)), so the vectors take half the space in the file and in
memory. A search over an index too large for the CPU's caches is faster for it, by a tenth to
two thirds on most of the machines measured, and builds on x86 take 11 to 48% less time. An index
that fits in cache searches no faster, and on Neoverse-N2 a fifth more slowly when its vectors are
long. Each component is rounded to 11 significant bits, about three decimal digits, and reported
distances are to the vectors as kept; on the embeddings measured, recall was the same. A vector
with a component of 65,520 or more in magnitude is refused. A file in half precision is format
version 4, which releases from before this option refuse to open.

Besides half precision, the way to shrink an index is fewer dimensions. Some embedding models are
trained so that the first part of a vector works on its own ("Matryoshka" embeddings, such as
OpenAI's text-embedding-3 with its `dimensions` argument): 512 of 1,536 dimensions make the index
about a third of the size. Measure recall on your own queries first, and cut vectors yourself only
with `DistanceMetric::Cosine`, or normalize them again.

`huge_pages` is for an index too large for the CPU's caches: with the vectors on 2 MB pages,
searches are up to a quarter faster and batch builds a little faster
([ADR-0016](../adr/016-huge-pages-on-request.md)). It works on Linux, where the kernel and
filesystem keep files on huge pages (ext4 on Linux 6.17 does), and does nothing elsewhere. A writer
and its readers each ask for themselves. It is off by default because a page not yet in memory is
then read 2 MB at a time, which an index much larger than memory pays for on every miss.
`VectorIndex::use_huge_pages` and `IndexReader::use_huge_pages` turn it on after opening.

`warm` is for an index whose file is not in memory yet, as after a reboot. Until it is, each page
a search touches is read from the disk, one read at a time. With `warm`, another thread reads in
what the index holds, at the disk's sequential speed, while searches go on
([ADR-0019](../adr/019-warm.md)). On an Apple M5, from the start of a process, the hundredth search over 99,000
vectors of 1,536 dimensions returned after 0.37 s with it and 1.9 s without; on Linux servers
with slower disks, after 1.3 to 1.7 s where it took 1.5 to 2.9 s. It reads only what has been written, and changes
nothing in the file. It is off by default because an index much larger than memory would push
everything else out, itself included, and on a slow disk reading the whole file can take longer
than the searches would. It takes Linux 5.14 or later, or macOS; an older Linux reads
in only some of the index, and Windows none. `VectorIndex::warm` and `IndexReader::warm` ask for
it after opening and return at once; from then on the option is on, so a reader that opens the
file again after a compaction reads the new one in too.

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
