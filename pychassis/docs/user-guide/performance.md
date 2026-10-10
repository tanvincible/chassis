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
| **`precision`** | `"full"` keeps 32-bit floats, `"half"` 16-bit floats. Fixed when the index is created. | `"full"` | `"half"` halves the vectors' size in the file and in memory, and on an index too large for the CPU's caches searches are a tenth to two thirds faster on most machines measured. Components are rounded to about three decimal digits; recall on the embeddings measured was the same. A component of 65,520 or more in magnitude is refused, and older releases can't open the file. |
| **`huge_pages`** | Ask Linux to keep the vectors on 2 MB pages. | `False` | On an index too large for the CPU's caches, up to a quarter faster search. Does nothing off Linux, or where the kernel and filesystem don't keep files on huge pages (ext4 on Linux 6.17 does). An index much larger than memory reads 2 MB per miss. |
| **`warm`** | Read the index into memory on another thread from the moment it is opened; `warm()` asks for it later. | `False` | While the file isn't in memory yet, as after a reboot, searches stop waiting for the disk one page at a time. On an Apple M5 the hundredth search over 99,000 vectors of 1,536 dimensions came 0.37 s after the process started, instead of 1.9 s. For an index that fits in memory. Linux 5.14 or later, or macOS; an older Linux reads in only some of the index, and Windows none. |

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
