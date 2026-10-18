# Rust API

Every type and method of the `chassisdb` crate, with what each one does to the file and when it
fails. [docs.rs](https://docs.rs/chassisdb) has the same API generated from the code.

## Indexes

### `VectorIndex`

`VectorIndex` opens an index file to add, delete and search vectors. One process writes a file at
a time; to search a file that another process writes, use [`IndexReader`](#indexreader). The
examples below continue from one another, starting with the index opened here.

#### Open

```rust
use chassisdb::{VectorIndex, IndexOptions};

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
* `options`: Configuration for graph construction and search. To reopen an existing index, pass the
  metric, precision and `max_connections` it was created with; others are an `OptionsMismatch`
  error that names them.

**Returns**:

* `Ok(VectorIndex)`: Handle to the index.
* `Err`: If file is locked, corrupted, or dimensions mismatch.

#### Add

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

#### Your ids

```rust
index.add_with_id(42, &vector)?;
```

Search results report `42`. Fails if a live vector already has that id; delete it first to replace
it. `u64::MAX` is reserved. Indexes that use their own ids build an id table in memory on the first
lookup in a process, which scans the index once ([ADR-0007](https://github.com/tanvincible/chassis/blob/main/docs/adr/007-ids-and-deletes.md)).

#### Batches

```rust
let rows: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0; 768]).collect(); // your vectors
let vectors: Vec<f32> = rows.concat(); // count × dims floats, back to back
let ids = index.add_batch(&vectors)?;  // or index.add_batch_with_ids(&my_ids, &vectors)?
```

Links the batch on every core, so a large batch builds many times faster than `add` in a loop
([ADR-0010](https://github.com/tanvincible/chassis/blob/main/docs/adr/010-parallel-batch-builds.md)). Ids are assigned as `add` assigns them. A batch is
added whole or not at all. The graph depends on thread timing, so two builds of the same data
differ slightly, as sequential builds already do through their random layers.

#### Delete

```rust
let deleted = index.delete(id)?; // false if no live vector has that id
```

Search stops returning the vector immediately, and `len()` drops by one. The delete is durable
after the next `flush()`; a crash before then rolls it back. A delete and an add in the same flush
are all-or-nothing, so `delete(id)` followed by `add_with_id(id, new_vector)` replaces a vector
safely. Deleted vectors keep their disk space until `compact()`; `deleted_count()` says how many
the file holds.

#### Compact

```rust
index.compact()?;
```

Rewrites the index without its deleted vectors and with a newly built graph, then replaces the
file with the copy ([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)). Ids don't change, and `add` still never
reuses one. Like `flush()`, it makes every add and delete so far durable. It takes as long as
building the index, on every core, and needs free disk for a second copy of the live vectors.
Readers in other processes keep searching and move to the new file by themselves. On Windows it
fails, leaving the index as it was, while another process has the index open.

When to compact: deletes cost speed and space, not recall. With half the vectors deleted, searches
ran at about 70% of their speed, and with 90% deleted at a fifth
([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)).
Compacting once `deleted_count()` nears `len()` keeps most of the speed; compact sooner when disk
space matters. The file shrinks only by whole segments, which double in size as an index grows, so
it gets smaller only when the live vectors fit in fewer of them.

#### Search

```rust
let query = vec![0.5; 768];
let k = 10; // Neighbors to retrieve

let results = index.search(&query, k)?;

for hit in results {
    println!("id {}, distance {}", hit.id, hit.distance);
}
```

**Returns**: `Vec<SearchResult>`, sorted by distance (nearest first), identified by the ids from `add`/`add_with_id`. Deleted vectors are never returned.

`k` is at least 1. The search keeps the best `max(k, ef_search)` candidates, so a `k` above
`ef_search` is fine; it returns at most `len()` results.

#### Filtered search

```rust
use std::collections::HashSet;

// Stands for the ids your application allows, e.g. from `SELECT id FROM docs WHERE owner = ?`
let allowed: HashSet<u64> = HashSet::from([42, 101, 102]);
let results = index.search_filtered(&query, 10, |id| allowed.contains(&id))?;
```

Only vectors whose id the filter accepts are returned. The filter is called with ids from
`add`/`add_with_id`, many times per search, so it should be cheap. When the filter passes many
vectors, the search walks the graph as usual. When it passes few, or its matches lie far from the
query, walking the graph would visit most of the index to find them. Then the search checks every
vector instead and returns the exact nearest. Measured, a filtered search takes at most about twice
an exact scan of the matching vectors
([ADR-0009](https://github.com/tanvincible/chassis/blob/main/docs/adr/009-filtered-search.md)).
`IndexReader` has the same method.

To leave a few ids out, as "more like this" leaves out the vector it starts from, pass a filter
that rejects them:

```rust
let vector = index.get(42)?.expect("42 is in the index");
let results = index.search_filtered(&vector, 10, |id| id != 42)?;
```

#### Flush

```rust
// Every add and delete so far is durable once this returns.
index.flush()?;
```

A flush waits for the disk, so it takes milliseconds. Call it after a batch of changes, every
1,000 vectors say, or before shutting down. To bound what a crash can lose by time instead, flush
from a timer.

Dropping a `VectorIndex` doesn't flush: what was added or deleted since the last `flush()` is
discarded, as after a crash.

#### Rollback

```rust
// Every add, delete and user version since the last flush is gone.
index.rollback()?;
```

As closing the index and opening it again would, without letting go of the writer's lock. Call it
where a write's transaction in your own store rolls back, or the next flush commits what that write
did to the index ([With a database](./with-a-database.md#writing)). If it fails, later flushes fail
too: close the index and open it again.

At rest the index is one file. A `<name>.undo` file sits beside it from the first add after a flush
until the next flush, and after a crash until a writer reopens the index ([file
format](../architecture/file-format.md#undo-file)). Move, copy or delete the two together, or copy
the index with `backup` while it is open.

#### Backup

```rust
// A copy of the index, every add and delete so far: it flushes first.
index.backup("backups/embeddings.chassis")?;
```

The backup opens like any index. It is written beside its path and renamed into place once whole,
so a crash never leaves half a backup there, and it replaces a file already at that path. An
`IndexReader` makes one too, of the writer's last flush, while the writer keeps writing; see
[`IndexReader`](#indexreader). To restore a backup, copy it into place while nothing has the index
open ([ADR-0024](https://github.com/tanvincible/chassis/blob/main/docs/adr/024-backups.md)).

#### Lookups

```rust
let vector = index.get(42)?;     // Some(vector), or None if no live vector has id 42
let known = index.contains(42)?; // Whether a live vector has id 42
let ids = index.ids()?;          // Every live id, in the order they were added
```

`get` returns the vector as the index keeps it: scaled to length 1 in a cosine index, and rounded
in half precision. With ids you chose, the first lookup in a process scans the index once to map
ids to vectors, as `add_with_id` does.

#### Metadata

```rust
let len = index.len();           // Live vectors (deleted ones excluded)
let deleted = index.deleted_count(); // Deleted vectors the file holds until compact()
let dim = index.dimensions();    // Vector size
let metric = index.metric();     // DistanceMetric::Euclidean or ::Cosine
let precision = index.precision(); // Precision::Full or ::Half
let empty = index.is_empty();    // True if count == 0
```

#### User version

```rust
index.set_user_version(42); // Committed by the next flush, with the data
index.flush()?;
let version = index.user_version(); // 42; 0 until one is set
```

Like SQLite's `user_version`, a number for your application. The next `flush` commits it with the
adds and deletes, so after a crash it says which flush the index holds. Keep the same number in your
own store, in the transaction the flush sits in, and compare the two on open and as each write
begins: they differ only when a flush's commit never came, after a crash or a failed commit
([With a database](./with-a-database.md#opening),
[ADR-0023](https://github.com/tanvincible/chassis/blob/main/docs/adr/023-user-version.md)).
The writer returns the number last set, flushed or not, and `rollback` sets it back to the flushed
one. `IndexReader::user_version` returns the number of the flush its last snapshot shows.

#### Check

```rust
let check = index.check()?; // Reads the whole graph and every vector
assert!(check.is_ok());     // Nothing out of reach, nothing that isn't a finite number
```

Checks what opening can't; opening checks the header and that the file holds every region it
names. A check of 200,000 vectors of 384 dimensions takes about 0.1 s.

- `check.unreachable`: live vectors that no path through the graph reaches, so no search returns
  them. Damage leaves them, and `compact()` builds the graph again. A sound graph can have a few
  too. On uniform random numbers in 384 dimensions, 0.3% to 1.4% of them, rebuilt or not; on
  clustered data, as embeddings are, none. Among many exact copies of one vector, some, and a
  search for that vector finds the others.
- `check.not_finite`: vectors with a component that isn't a finite number, which only damage to
  the file leaves, since `add` refuses them. Delete them and add them again from your own data.

### `IndexReader`

Searches an index while a `VectorIndex` writes it, in another process or this one. It takes no
lock, so any number of readers can open one file next to its single writer
([ADR-0008](https://github.com/tanvincible/chassis/blob/main/docs/adr/008-format-v3-and-multi-process-readers.md), decision 5). It takes the metric
and precision from the file; `max_connections` has to be the one the index was created with.

```rust
use chassisdb::{IndexOptions, IndexReader};

let mut reader = IndexReader::open("embeddings.chassis", 768, IndexOptions::default())?;
let results = reader.search(&query, 10)?;
```

Every search first takes a new snapshot: it returns what the writer's newest `flush()` committed,
and nothing that flush deleted, while routing through vectors added since, so a long unflushed batch
costs readers little recall. A flush becomes visible just before its final fsync, so one that then
fails, or is lost to power loss, may already have been seen. `get`, `contains`, `ids` and `check`
take a new snapshot too; while the writer is linking a vector, `check` can find it out of reach for
a moment. `len()` and `deleted_count()` report the last snapshot; `refresh()` takes a new one
without searching, and `snapshot()` identifies it, for keying a cache.
`search` takes `&mut self`: open one reader per thread. The file must exist in the current format;
open it once with `VectorIndex::open` to create or migrate it.

`reader.backup(path)` copies the writer's last flush while the writer keeps writing. It copies the
file, then the undo file beside it, and starts over if a flush lands meanwhile; after five tries it
gives up. The copy needs a pause in the writer's flushes as long as itself, about a third of a
second for 200 MB on a laptop: a writer that flushes more often than that backs up itself.

## Errors

`VectorIndex` and `IndexReader` return `chassisdb::Result<T>`, whose error says what was wrong,
with the value, and on a line starting `help:` what to do instead. `Error::kind()` is an
`ErrorKind` to act on ([Errors](./errors.md) lists them, with causes and fixes):

```rust
use chassisdb::{ErrorKind, IndexOptions, IndexReader, VectorIndex};

match VectorIndex::open("embeddings.chassis", 768, IndexOptions::default()) {
    Ok(index) => { /* write */ }
    // One writer at a time; a reader searches while it writes.
    Err(e) if e.kind() == ErrorKind::Locked => {
        let reader = IndexReader::open("embeddings.chassis", 768, IndexOptions::default())?;
    }
    Err(e) => return Err(e.into()),
}
```

`chassisdb::Error` converts into `anyhow::Error` and `Box<dyn std::error::Error>`, so `?` works
in functions that return those.

## Options

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

    /// Ask the operating system to keep the vectors on huge pages. Default: None, left to `warm`
    pub huge_pages: Option<bool>,

    /// Read the index into memory on another thread from the moment it is opened. Default: false
    pub warm: bool,
}
```

The metric is fixed when the index is created: reopening with another one is an error, an
`IndexReader` uses the file's, and `metric()` on either reports it. A cosine index stores vectors
scaled to unit length, rejects zero vectors and vectors with NaN or infinite components, and
reports `1 - cosine similarity`, from 0 to 2. Search speed is the same for both.

`precision` is fixed when the index is created, as the metric is: reopening with another one is an
error, an `IndexReader` uses the file's, and `precision()` on either reports it. `Precision::Half`
keeps each component of a vector as a 16-bit float
([ADR-0018](https://github.com/tanvincible/chassis/blob/main/docs/adr/018-half-precision.md)), so
the vectors take half the space in the file and in memory. A search over an index too large for the
CPU's caches is a tenth to two thirds faster for it on most of the machines measured. Builds on x86
take 11 to 48% less time. An index that fits in cache searches no faster, and on Neoverse-N2 a fifth
more slowly when its vectors are long. Each component is rounded to 11 significant bits, about three
decimal digits, and reported distances are to the vectors as kept; on the embeddings measured,
recall was the same. A vector with a component of 65,520 or more in magnitude is refused. A file in
half precision is format version 4, which releases from before this option refuse to open.

Besides half precision, the way to shrink an index is fewer dimensions. Matryoshka embeddings, such
as OpenAI's text-embedding-3 with its `dimensions` argument, are trained so that the first part of a
vector works on its own. 512 of 1,536 dimensions make the index about a third of the size. Measure
recall on your own queries first, and cut vectors yourself only with `DistanceMetric::Cosine`, or
normalize them again. `max_connections: 12` instead of 16 saves less. A 128-dimension file in half
precision is 7% smaller, and one of 1,536 dimensions 1%. Builds are 7 to 18% faster, but searches
for 0.99 recall at 128 and 384 dimensions take up to 14% longer.

`huge_pages` is for an index too large for the CPU's caches: with the vectors on 2 MB pages,
searches are up to a quarter faster and batch builds a little faster
([ADR-0016](https://github.com/tanvincible/chassis/blob/main/docs/adr/016-huge-pages-on-request.md)). It works on Linux, where the kernel and
filesystem keep files on huge pages (ext4 on Linux 6.17 does), and does nothing elsewhere. A writer
and its readers each ask for themselves. On most CPUs measured, searches were faster on huge pages,
by up to 27%. With vectors of 128 dimensions they were 2 to 5% slower on AMD's Zen 5 and level on a
Xeon 6973P-C, and at 960 dimensions level on a Xeon 8370C. In half precision, under long vectors,
they ranged from 3% slower to 8% faster.

`Some(true)` asks from the moment the index is opened. A page not yet in memory is then read 2 MB at
a time, which an index much larger than memory pays for on every miss. `Some(false)` never asks.
`None`, the default, leaves it to `warm`, which asks under vectors of 1 KB or more: 256 dimensions
at full precision, 512 at half. `use_huge_pages` and `keep_small_pages`, on `VectorIndex` and
`IndexReader`, do the same after opening.

Only pages read from the disk after asking go on huge pages. Vectors a writer added one at a time,
and pages that a process which didn't ask read first, stay on 4 KB pages until they leave memory,
and searches over them are no faster. Copying the file may do the same.

`warm` is for an index whose file is not in memory yet, as after a reboot. Until it is, each page a
search touches is read from the disk, one read at a time. With `warm`, another thread reads in what
the index holds, at the disk's sequential speed, while searches go on
([ADR-0019](https://github.com/tanvincible/chassis/blob/main/docs/adr/019-warm.md)). On an Apple M5,
from the start of a process, the hundredth search over 99,000 vectors of 1,536 dimensions returned
after 0.37 s with it and 1.9 s without. On Linux servers with slower disks, it returned after 1.3 to
1.7 s where it took 1.5 to 2.9 s. It reads only what has been written, and changes nothing in the
file. It is off by default because an index much larger than memory would push everything else out,
itself included, and on a slow disk reading the whole file can take longer than the searches would.
It takes Linux 5.14 or later, or macOS; an older Linux reads in only some of the index, and Windows
none. On Linux, unless `huge_pages` says otherwise, it asks for huge pages under vectors of 1 KB or
more: read in whole, the index costs no more on them. `VectorIndex::warm` and `IndexReader::warm` ask for it after opening and return at once. From
then on the option is on, so a reader that opens the file again after a compaction reads the new one
in too.

**Tuning Guide**:

* **High Recall**: Increase `ef_construction` to 400 and `max_connections` to 32.
* **Fast Search**: Decrease `ef_search` to 20-30.
* **Low Memory**: `Precision::Half`, or vectors with fewer dimensions.

## Results

### `SearchResult`

```rust
pub struct SearchResult {
    /// The id the vector was added under
    pub id: u64,
    
    /// Distance from the query by the index's metric
    pub distance: f32,
}
```

Results order by distance, then by id, and two are equal only when both their ids and their
distances are. Check a search's results with `==`: a result with the right distance but another
id doesn't match.
