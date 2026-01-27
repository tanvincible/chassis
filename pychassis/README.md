# PyChasis - Python Bindings for Chassis

High-performance Python bindings for the Chassis vector storage engine.

## Features

- **Zero-copy operations**: Direct memory access via ctypes
- **NumPy integration**: Native support for NumPy arrays
- **Type hints**: Full type annotations for IDE support
- **Thread-safe**: Multi-reader support for concurrent searches
- **Pythonic API**: Context managers, properties, and familiar patterns

## Installation

### Prerequisites

1. Build the Chassis FFI library:
   ```bash
   cd ../chassis-ffi
   cargo build --release
   ```

2. Install PyChasis:
   ```bash
   cd ../pychassis
   pip install -e .
   ```

### Development Installation

```bash
# Install with dev dependencies
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
from pychassis import VectorIndex

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
from chassis import VectorIndex, DimensionMismatchError, ChassisError

try:
    index = VectorIndex("vectors.chassis", dimensions=128)
    
    # This will raise DimensionMismatchError
    index.add([0.1] * 64)
    
except DimensionMismatchError as e:
    print(f"Dimension error: {e}")
except ChassisError as e:
    print(f"General error: {e}")
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

- **`search(query: Sequence[float] | ndarray, k: int = 10) -> List[SearchResult]`**  
  Search for k nearest neighbors. Returns sorted list of results.

- **`flush() -> None`**  
  Flush changes to disk. Call after batch insertions.

- **`close() -> None`**  
  Close the index and free resources. Called automatically.

- **`__len__() -> int`**  
  Get number of vectors in the index.

- **`is_empty() -> bool`**  
  Check if index is empty.

#### Properties

- **`dimensions: int`** - Number of dimensions per vector
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

- **`ChassisError`** - Base exception for all errors
- **`DimensionMismatchError`** - Vector dimensions don't match
- **`InvalidPathError`** - Path is invalid or inaccessible
- **`IndexNotFoundError`** - Index file not found
- **`NullPointerError`** - FFI returned NULL pointer

## Thread Safety

| Operation | Thread Safety |
|-----------|---------------|
| `add()` | Single-writer (exclusive access required) |
| `flush()` | Single-writer (exclusive access required) |
| `search()` | Multi-reader (concurrent reads allowed) |
| `len()`, `is_empty()`, `dimensions` | Multi-reader |

**Safe:**
```python
# Thread 1: Writer
index.add(vector1)

# Thread 2: Reader (concurrent with writes is UNSAFE)
results = index.search(query, k=10)  # UNSAFE during writes
```

**Recommended:**
```python
# Separate read and write phases
# Phase 1: Write (single thread)
for vec in vectors:
    index.add(vec)
index.flush()

# Phase 2: Read (multiple threads OK)
with ThreadPoolExecutor() as executor:
    results = executor.map(lambda q: index.search(q, k=10), queries)
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

PyChasis looks for `libchassis_ffi` in this order:

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
