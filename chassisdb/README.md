# chassisdb

Python bindings for Chassis, an embedded vector index for local semantic search: one file, in your process, no server.

## Features

- Vectors go in as NumPy arrays or lists, and come out as NumPy arrays.
- Many threads search one index at once, while writes run one at a time.
- Other processes search the file while one writes it, with `read_only=True`.
- Every error says what was wrong and what to do, and is also the matching built-in exception.
- Type hints throughout.

## Install

```bash
pip install chassisdb
```

Import it as `chassisdb`. Wheels are built for Python 3.11 and later on Linux (x86-64 and ARM),
macOS on Apple silicon and Windows (x86-64). Elsewhere, or from a
checkout of the repository, `pip install .` in this directory builds it, which needs Rust.

### Development

```bash
# Build the library, then install with dev dependencies
cargo build --release -p chassis-ffi
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy chassisdb

# Formatting
black chassisdb tests
ruff check chassisdb tests
```

## Quickstart

The vectors here are random; in your app they come from your embedding model.

```python
import numpy as np
from chassisdb import VectorIndex

index = VectorIndex("embeddings.chassis", dimensions=768)  # created if it doesn't exist

vectors = np.random.rand(1000, 768).astype(np.float32)
ids = index.add_batch(vectors)  # linked on every core; ids 0 to 999
index.flush()  # on disk once this returns

query = np.random.rand(768).astype(np.float32)
for result in index.search(query, k=10):  # nearest first
    print(result.id, result.distance)

print(len(index), index.dimensions)  # 1000 768
index.close()
```

## Usage

### Options

```python
from chassisdb import VectorIndex, IndexOptions

# Configure HNSW parameters
options = IndexOptions(
    max_connections=32,      # Higher = better recall, more memory
    ef_construction=400,     # Higher = better index, slower build
    ef_search=100,           # Higher = better search, slower queries
    metric="cosine",         # "euclidean" (default) or "cosine"; fixed at creation
    precision="half",        # "full" (default) or "half": 16-bit floats, half the file; fixed at creation
)

index = VectorIndex("tuned.chassis", dimensions=768, options=options)
```

### Closing

```python
with VectorIndex("vectors.chassis", dimensions=128) as index:
    index.add([0.1] * 128)
# Flushed and closed here
```

Leaving the block flushes, unless an exception is leaving it, then closes the index. Closing
without a flush, with `close()` or by garbage collection, discards what was added or deleted since
the last `flush()`, as a crash would.

### Batches

```python
import numpy as np

vectors = np.random.rand(10000, 768).astype(np.float32)
queries = np.random.rand(100, 768).astype(np.float32)

with VectorIndex("large.chassis", dimensions=768) as index:
    # Added and linked on every core
    index.add_batch(vectors)
    index.flush()

    # One call for every query: ids and distances are (100, 10) arrays
    ids, distances = index.search_batch(queries, k=10)
```

### Inputs

A vector can be a NumPy array, a list or a tuple. Each is converted to 32-bit floats.

```python
import numpy as np
from chassisdb import VectorIndex

with VectorIndex("small.chassis", dimensions=3) as index:
    index.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    index.add([0.1, 0.2, 0.3])
    index.add((0.1, 0.2, 0.3))
```

### Errors

```python
from chassisdb import VectorIndex, DimensionMismatchError, IndexLockedError

index = VectorIndex("vectors.chassis", dimensions=128)
try:
    index.add([0.1] * 64)
except DimensionMismatchError as e:
    print(e)
    # The vector has 64 components, but this index holds vectors of 128
    # help: make the vector with the same model as the index's vectors; this index was created for 128 dimensions

try:
    VectorIndex("vectors.chassis", dimensions=128)
except IndexLockedError:
    # One writer at a time; a reader searches while it writes.
    reader = VectorIndex("vectors.chassis", dimensions=128, read_only=True)
```

## Reference

### `VectorIndex`

Opens an index file to add, delete and search vectors.

#### Constructor

```python
VectorIndex(path: str | Path, dimensions: int, options: IndexOptions | None = None, read_only: bool = False)
```

To reopen an existing index, pass the metric, precision and `max_connections` it was created
with; others raise `OptionsMismatchError`, which names them. With `read_only=True` it only
searches, takes no lock, and takes the metric and precision from the file (`max_connections` has
to match); any number can search while one writer writes, in this process or others.

#### Methods

- **`add(vector: Sequence[float] | ndarray, id: int | None = None) -> int`**  
  Add a vector under `id` (0 to 2**64 - 2; `IdInUseError` if a live vector has it), or else under
  one past the largest id so far. Returns the id.

- **`add_batch(vectors: ndarray, ids: Iterable[int] | None = None) -> ndarray`**  
  Add a `(count, dimensions)` array at once, linking it on every core: much faster than `add` in a
  loop. Returns the ids. All or nothing: if any vector or id is rejected, none is added.

- **`delete(id: int) -> bool`**  
  Delete the vector with this id. Returns whether there was one. Search stops returning it at
  once; the delete is durable after the next `flush()`. To replace a vector, delete it and add
  the new one under the same id: the next flush saves both or neither.

- **`get(id: int) -> ndarray | None`**  
  The vector with this id, or `None` if no live vector has it. It comes back as the index keeps
  it: scaled to length 1 in a cosine index, and rounded in half precision.

- **`id in index`**  
  Whether a live vector has this id.

- **`ids() -> ndarray`**  
  The ids of the live vectors, in the order they were added.

- **`search(query: Sequence[float] | ndarray, k: int = 10, allowed: Iterable[int] | ndarray | None = None, excluded: Iterable[int] | ndarray | None = None) -> List[SearchResult]`**  
  Search for k nearest neighbors. Returns sorted list of results. With `allowed`, only those ids
  can be returned, e.g. `allowed=[row[0] for row in db.execute("SELECT id FROM docs WHERE owner = ? ORDER BY id", (user,))]`;
  ids in ascending order cost the least. When walking the graph would cost more, as when few
  vectors match, every vector is checked instead and the results are exact. With `excluded`, those ids are never returned, as for "more
  like this": `index.search(index.get(42), k=10, excluded=[42])`.

- **`search_batch(queries: ndarray, k: int = 10) -> tuple[ndarray, ndarray]`**  
  Search for the k nearest neighbors of each row of a `(count, dimensions)` array, in one call:
  faster than `search` in a loop. Returns `(ids, distances)`, two `(count, k)` arrays, nearest
  first; a row with fewer than k results is filled out with id `2**64 - 1` and distance `inf`.

- **`flush() -> None`**  
  Make every add and delete so far durable. It waits for the disk, so call it after a batch of
  changes, not after each one.

- **`rollback() -> None`**  
  Discard every add, delete and `user_version` since the last `flush()`, as closing the index and
  opening it again would, without letting go of the writer's lock. Call it where a write's
  transaction in your own store rolls back, or the next flush commits what that write did to the
  index. If it raises, later flushes fail too: close the index and open it again.

- **`backup(path) -> None`**  
  Write a copy of the index to `path`, which opens like any index. A writer flushes first, so the
  backup holds every add and delete so far; a `read_only=True` index copies the writer's last
  flush while the writer keeps writing, and gives up if a flush lands during each of five tries.
  The copy is renamed into place once whole, replacing a file already there.

- **`warm() -> None`**  
  Read the index into memory on another thread, so that searches stop waiting for the disk while
  its file is not in memory yet, as after a reboot. Returns at once; searches go on meanwhile. On
  Linux it asks for huge pages too. `IndexOptions(warm=True)` does the same from the moment the
  index is opened.

- **`compact() -> None`**  
  Rewrite the index without its deleted vectors and with a rebuilt graph, reclaiming their space.
  Ids don't change. Takes as long as building the index and needs disk for a second copy.
  Deletes cost speed, not recall: with half the vectors deleted, searches ran at about 70% of
  their speed, so compacting once `deleted_count()` nears `len()` is a good rule. The file
  shrinks by whole segments, which double as an index grows, so only when the live vectors fit in
  fewer of them.

- **`close() -> None`**  
  Close the index and free resources. Adds and deletes since the last `flush()` are discarded, as
  after a crash; leaving a `with` block flushes first, unless an exception is leaving it.

- **`__len__() -> int`**  
  The number of live vectors.

- **`deleted_count() -> int`**  
  Deleted vectors the file still holds, until `compact()` removes them.
- **`user_version: int`**  
  Your application's number for the index, 0 until you set one, like SQLite's `user_version`.
  The next `flush()` commits it with the data, so after a crash it says which flush the index
  holds: keep the same number in your own store, in the transaction the flush sits in, and
  compare the two on open and as each write begins. The writer reports the number last set,
  flushed or not, and `rollback()` sets it back to the flushed one. A `read_only=True` index
  reports its newest snapshot's; setting it there raises `ReadOnlyError`.

- **`check() -> Check`**  
  Checks what opening can't, reading the whole graph and every vector. `check().unreachable`
  lists the ids of live vectors that no path through the graph reaches, so no search returns
  them. Damage leaves them, and `compact()` builds the graph again; a sound graph can have a few
  too, on data with no structure or among many exact copies of one vector. `check().not_finite`
  lists vectors with a component that isn't a finite number, which only damage to the file
  leaves: delete them and add them again from your own data. `check().ok` is true when both are
  empty. A `read_only=True` index checks the writer's last flush.

- **`is_empty() -> bool`**  
  Whether the index has no live vectors.

#### Properties

- **`dimensions: int`** - Number of dimensions per vector
- **`metric: str`** - `"euclidean"` or `"cosine"`, as the index was created; opening with
  `read_only=True` and explicit `options` naming another raises `OptionsMismatchError`
- **`precision: str`** - `"full"` or `"half"`, as the index was created
- **`path: Path`** - Path to the index file
- **`options: IndexOptions`** - HNSW configuration

### `IndexOptions`

The index's options. `metric`, `precision` and `max_connections` are fixed when the index is
created.

```python
@dataclass
class IndexOptions:
    max_connections: int = 16      # M parameter
    ef_construction: int = 200     # Build-time search quality
    ef_search: int = 50            # Query-time search quality
    metric: str = "euclidean"      # or "cosine": 1 - cosine similarity, unit-length storage
    precision: str = "full"        # or "half": vectors kept as 16-bit floats, half the file and memory
    huge_pages: bool = False       # Linux: keep the vectors on 2 MB pages (faster search if it fits in memory)
    warm: bool = False             # read the index in from the start, in the background, on huge pages
```

### `SearchResult`

One result: the vector's id, and its distance from the query.

```python
@dataclass
class SearchResult:
    id: int          # the vector's id
    distance: float  # its distance from the query; smaller is nearer
```

### Exceptions

Each says what was wrong, with the value, and on a line starting `help:` what to do instead. Each
is a `ChassisError` and also the built-in exception it is a kind of, so `except ValueError`
catches a bad argument. The [errors reference](https://github.com/tanvincible/chassis/blob/main/docs/src/guide/errors.md)
lists their causes and fixes.

- **`InvalidArgumentError`** (`ValueError`) - an argument out of range: dimensions, an option, an
  id, a vector with NaN or infinity. A wrong type is a plain `TypeError`.
- **`DimensionMismatchError`** (`ValueError`) - a vector or query of other dimensions than the index's
- **`OptionsMismatchError`** (`ValueError`) - reopened with another metric, precision or
  `max_connections`
- **`IndexNotFoundError`** (`FileNotFoundError`) - no index at the path, for `read_only=True`
- **`NotAnIndexError`** (`ValueError`) - the file isn't a Chassis index
- **`IdInUseError`** (`ValueError`) - the id is taken; delete it first to replace it
- **`IndexLockedError`** - another writer has the index open
- **`ReadOnlyError`** - a `read_only=True` index asked to write
- **`CorruptIndexError`**, **`IndexFullError`**, **`InvalidPathError`** (`OSError`) - a damaged
  file, a full index, an operating-system error

## Threads

Every method is safe to call from any thread. Searches run concurrently (ctypes releases the GIL
during each call); `add()`, `delete()` and `flush()` take the index's write lock, so they run one
at a time and searches wait for them. Don't call `close()` while other threads still use the index.

Other processes can search the file while one process writes it: open it with
`VectorIndex(path, dimensions, read_only=True)`. Such an index takes no lock, sees the writer's
newest `flush()` on every search, runs one search at a time, and raises `ReadOnlyError` on writes.
A flush becomes visible just before its final fsync, so one that then fails, or is lost to power
loss, may already have been seen.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from chassisdb import VectorIndex

index = VectorIndex("large.chassis", dimensions=768)
queries = np.random.rand(100, 768).astype(np.float32)
with ThreadPoolExecutor() as pool:
    results = list(pool.map(lambda q: index.search(q, k=10), queries))
```

## Performance

1. Add many vectors with `add_batch`, which links them on every core, and flush once after a batch:
   a flush waits for the disk.
2. Search many queries with `search_batch`. Measured on 2026-10-10, one call for every query was
   within 5% of Rust's speed; on 2026-10-17, a call per query cost 13 to 26 µs more each
   ([Performance](https://tanvincible.github.io/chassis/architecture/performance.html)).
3. Pass 32-bit float NumPy arrays: lists and other types are converted first.
4. Tune `ef_search`, `ef_construction` and `max_connections` as
   [Performance & tuning](https://tanvincible.github.io/chassis/python/user-guide/performance/)
   describes.

## Library

chassisdb looks for `libchassis_ffi` in this order:

1. `CHASSIS_LIB_PATH` environment variable
2. Next to the Python package
3. `../target/release` (development)
4. System library paths

To set a custom location:
```bash
export CHASSIS_LIB_PATH=/path/to/lib
python your_script.py
```

## Examples

See the `examples/` directory for complete examples:
- `notes_search.py` - Notes in SQLite, their vectors in Chassis: search, "more like this", deletes
- `basic_usage.py` - Add and search
- `batch_insert.py` - Many vectors at once with `add_batch`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chassisdb --cov-report=html

# Run specific tests
pytest tests/test_index.py -k delete
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/tanvincible/chassis?tab=Apache-2.0-1-ov-file))
- MIT License ([LICENSE-MIT](https://github.com/tanvincible/chassis?tab=MIT-2-ov-file))

at your option.

## Contributing

See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for development guidelines.
