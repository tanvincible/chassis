# Performance & Tuning

Chassis is built on HNSW (Hierarchical Navigable Small World) graphs. You can tune the trade-off between build speed, search speed, and recall accuracy using `IndexOptions`.

## Configuration Options

You can pass an `IndexOptions` object when opening the index.

```python
from chassis import VectorIndex, IndexOptions

options = IndexOptions(
    max_connections=32,    # 'M' in HNSW papers
    ef_construction=400,   # Build quality
    ef_search=100          # Search quality
)

index = VectorIndex("tuned.chassis", dimensions=128, options=options)
```

### Understanding Parameters

| Parameter | Description | Default | Impact |
| --- | --- | --- | --- |
| **`max_connections`** | Max edges per node in the graph. | 16 | Higher = Better recall, higher memory usage. |
| **`ef_construction`** | Size of the dynamic candidate list during build. | 200 | Higher = Slower build, higher quality graph. |
| **`ef_search`** | Size of the dynamic candidate list during search. | 50 | Higher = Slower search, better recall. |

## Batch Insertion Strategy

Calling `flush()` involves an `fsync` system call, which is expensive. For maximum write throughput:

1. Add vectors in batches (e.g., 1,000 to 10,000).
2. Call `flush()` only after the batch is complete.

```python
# Bad: Slow due to excessive syscalls
for vec in huge_dataset:
    index.add(vec)
    index.flush() 

# Good: High throughput
batch_size = 1000
for i, vec in enumerate(huge_dataset):
    index.add(vec)
    if i % batch_size == 0:
        index.flush()
index.flush() # Final flush
```
