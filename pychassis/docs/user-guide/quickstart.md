# Quick Start

This guide will walk you through creating an index, adding vectors, and performing nearest neighbor searches.

## 1. Initialization

The `VectorIndex` is the main entry point. You must specify the dimensionality of your vectors when creating the index. This dimension is immutable for the life of the file.

```python
from chassis import VectorIndex

# Opens 'data.chassis'. If it doesn't exist, it is created.
# If it exists, it is opened and dimensions are validated.
index = VectorIndex("data.chassis", dimensions=768)
```

## 2. Adding Vectors

You can add vectors one by one. Chassis accepts standard Python lists or NumPy arrays.

```python
import numpy as np

# Using Python lists
index.add([0.1, 0.2, ...]) 

# Using NumPy (Recommended for performance)
data = np.random.rand(100, 768).astype(np.float32)

for vec in data:
    index.add(vec)
```

!!! warning "Important: Flush to Disk"
`add()` operations are buffered in memory for speed. You **must** call `index.flush()` to persist them to disk. If the program crashes before a flush, unwritten vectors will be lost.

```python
index.flush()
```

## 3. Searching

Search returns a list of `SearchResult` objects containing the `id` and `distance` (L2 Euclidean).

```python
query_vec = np.random.rand(768).astype(np.float32)

# Find top 10 nearest neighbors
results = index.search(query_vec, k=10)

for res in results:
    print(f"ID: {res.id}, Distance: {res.distance:.4f}")
```

## 4. Context Management

It is best practice to use the `with` statement to ensure resources are closed properly.

```python
with VectorIndex("data.chassis", dimensions=768) as index:
    index.add(vec)
    index.flush()
    # Index is automatically closed here
```
