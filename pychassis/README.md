# PyChassis - Python Bindings for Chassis

Python bindings for Chassis, an embedded vector index for local semantic search: one file, in your process, no server.

## Features

- **Zero-copy operations**: Direct memory access via ctypes
- **NumPy integration**: Native support for NumPy arrays
- **Type hints**: Full type annotations for IDE support
- **Thread-safe**: Search from many threads while another adds or deletes; writes run one at a time
- **Pythonic API**: Context managers, properties, and familiar patterns

## Installation

```bash
pip install chassisdb
```

The package is named `chassisdb` and imported as `chassis`. Wheels are built for Python 3.13 and
later on Linux (x86-64 and ARM), macOS on Apple silicon and Windows (x86-64). Elsewhere, or from a
checkout of the repository, `pip install .` in this directory builds it, which needs Rust.

### Development Installation

```bash
# Build the library, then install with dev dependencies
cargo build --release -p chassis-ffi
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy pychassis

# Formatting
black pychassis tests
ruff check pychassis tests
```

## Quick Start

```python
import numpy as np
from chassis import VectorIndex

# Create or open an index
index = VectorIndex("embeddings.chassis", dimensions=768)

# Add vectors
vectors = np.random.rand(1000, 768).astype(np.float32)
for i, vec in enumerate(vectors):
    vector_id = index.add(vec)
    print(f"Added vector {vector_id}")

# Flush to disk
index.flush()

# Search
query = np.random.rand(768).astype(np.float32)
results = index.search(query, k=10)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance:.6f}")

# Index info
print(f"Total vectors: {len(index)}")
print(f"Dimensions: {index.dimensions}")
print(f"Empty: {index.is_empty()}")
```

## Advanced Usage

### Custom HNSW Parameters

```python
from chassis import VectorIndex, IndexOptions

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

### Context Manager

```python
# Automatic resource cleanup
with VectorIndex("vectors.chassis", dimensions=128) as index:
    index.add([0.1] * 128)
    index.flush()
# Index is automatically closed here
```

### Batch Operations

```python
import numpy as np

# Efficient batch insertion
vectors = np.random.rand(10000, 768).astype(np.float32)

with VectorIndex("large.chassis", dimensions=768) as index:
    for vec in vectors:
        index.add(vec)
    
    # Flush once at the end (much faster than flushing per insert)
    index.flush()
```

### NumPy Integration

```python
import numpy as np

# From NumPy array
vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
index.add(vec)

# From list
index.add([0.1, 0.2, 0.3])

# From tuple
index.add((0.1, 0.2, 0.3))

# All are automatically converted to float32
```

### Error Handling

```python
from chassis import VectorIndex, DimensionMismatchError, IndexLockedError

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

## API Reference

### `VectorIndex`

Main class for interacting with Chassis indexes.

#### Constructor

```python
VectorIndex(path: str | Path, dimensions: int, options: IndexOptions | None = None)
```

#### Methods

- **`add(vector: Sequence[float] | ndarray) -> int`**  
  Add a vector to the index. Returns the vector ID.

- **`add_batch(vectors: ndarray, ids: Iterable[int] | None = None) -> ndarray`**  
  Add a `(count, dimensions)` array at once, linking it on every core: much faster than `add` in a
  loop. Returns the ids. All or nothing: if any vector or id is rejected, none is added.

- **`search(query: Sequence[float] | ndarray, k: int = 10, allowed: Iterable[int] | ndarray | None = None) -> List[SearchResult]`**  
  Search for k nearest neighbors. Returns sorted list of results. With `allowed`, only those ids
  can be returned, e.g. `allowed=[row[0] for row in db.execute("SELECT id FROM docs WHERE owner = ?", (user,))]`;
  when walking the graph would cost more, as when few vectors match, every vector is checked
  instead and the results are exact.

- **`flush() -> None`**  
  Flush changes to disk. Call after batch insertions.

- **`warm() -> None`**  
  Read the index into memory on another thread, so that searches stop waiting for the disk while
  its file is not in memory yet, as after a reboot. Returns at once; searches go on meanwhile.
  `IndexOptions(warm=True)` does the same from the moment the index is opened.

- **`compact() -> None`**  
  Rewrite the index without its deleted vectors and with a rebuilt graph, reclaiming their space.
  Ids don't change. Takes as long as building the index and needs disk for a second copy.

- **`close() -> None`**  
  Close the index and free resources. Called automatically.

- **`__len__() -> int`**  
  Get number of vectors in the index.

- **`is_empty() -> bool`**  
  Check if index is empty.

#### Properties

- **`dimensions: int`** - Number of dimensions per vector
- **`metric: str`** - `"euclidean"` or `"cosine"`, as the index was created; opening with
  `read_only=True` and explicit `options` naming another raises `ChassisError`
- **`precision: str`** - `"full"` or `"half"`, as the index was created
- **`path: Path`** - Path to the index file
- **`options: IndexOptions`** - HNSW configuration

### `IndexOptions`

Configuration for HNSW algorithm.

```python
@dataclass
class IndexOptions:
    max_connections: int = 16      # M parameter
    ef_construction: int = 200     # Build-time search quality
    ef_search: int = 50            # Query-time search quality
    metric: str = "euclidean"      # or "cosine": 1 - cosine similarity, unit-length storage
    precision: str = "full"        # or "half": vectors kept as 16-bit floats, half the file and memory
    huge_pages: bool = False       # Linux: keep the vectors on 2 MB pages (faster search on a large index)
    warm: bool = False             # read the index into memory in the background from the start
```

### `SearchResult`

Search result with ID and distance.

```python
@dataclass
class SearchResult:
    id: int          # Vector ID in index
    distance: float  # Distance to query (lower = closer)
```

### Exceptions

Each says what was wrong, with the value, and on a line starting `help:` what to do instead. Each
is a `ChassisError` and also the built-in exception it is a kind of, so `except ValueError`
catches a bad argument. The [errors reference](https://github.com/tanvincible/chassis/blob/main/docs/src/guide/errors.md)
lists their causes and fixes.

- **`InvalidArgumentError`** (`ValueError`) - an argument out of range: dimensions, an option, an
  id, a vector with NaN or infinity. A wrong type is a plain `TypeError`.
- **`DimensionMismatchError`** (`ValueError`) - a vector or query of other dimensions than the index's
- **`OptionsMismatchError`** (`ValueError`) - reopened with another metric or precision
- **`IndexNotFoundError`** (`FileNotFoundError`) - no index at the path, for `read_only=True`
- **`NotAnIndexError`** (`ValueError`) - the file isn't a Chassis index
- **`IdInUseError`** (`ValueError`) - the id is taken; delete it first to replace it
- **`IndexLockedError`** - another writer has the index open
- **`ReadOnlyError`** - a `read_only=True` index asked to write
- **`CorruptIndexError`**, **`IndexFullError`**, **`InvalidPathError`** (`OSError`) - a damaged
  file, a full index, an operating-system error

## Thread Safety

Every method is safe to call from any thread. Searches run concurrently (ctypes releases the GIL
during each call); `add()`, `delete()` and `flush()` take the index's write lock, so they run one
at a time and searches wait for them. Don't call `close()` while other threads still use the index.

Other processes can search the file while one process writes it: open it with
`VectorIndex(path, dimensions, read_only=True)`. Such an index takes no lock, sees the writer's
newest `flush()` on every search, runs one search at a time, and raises `ChassisError` on writes.
A flush becomes visible just before its final fsync, so one that then fails, or is lost to power
loss, may already have been seen.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as pool:
    results = list(pool.map(lambda q: index.search(q, k=10), queries))
```

## Performance Tips

1. **Batch inserts before flushing:**
   ```python
   for vec in many_vectors:
       index.add(vec)
   index.flush()  # Once at the end
   ```

2. **Use NumPy arrays:**
   ```python
   # Faster (zero-copy)
   vec = np.array([0.1] * 768, dtype=np.float32)
   
   # Slower (creates copy)
   vec = [0.1] * 768
   ```

3. **Tune HNSW parameters:**
   - Increase `ef_construction` for better index quality
   - Increase `ef_search` for better search recall
   - Increase `max_connections` for better graph connectivity

4. **Use context managers:**
   ```python
   with VectorIndex(...) as index:
       # Automatic cleanup
   ```

## Library Location

PyChassis looks for `libchassis_ffi` in this order:

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
- `basic_usage.py` - Simple add and search
- `batch_insert.py` - Efficient bulk loading

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pychassis --cov-report=html

# Run specific test
pytest tests/test_index.py::test_add_and_search
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/tanvincible/chassis?tab=Apache-2.0-1-ov-file))
- MIT License ([LICENSE-MIT](https://github.com/tanvincible/chassis?tab=MIT-2-ov-file))

at your option.

## Contributing

See [CONTRIBUTING.md](https://github.com/tanvincible/chassis?tab=contributing-ov-file) for development guidelines.
