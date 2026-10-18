# Performance

This page shows how fast Chassis is through Python, and the options that trade build speed, search
speed and recall.

On an Apple M5 in Low Power Mode (2026-10-15), through Python, a search over 20,000 random vectors
of 384 dimensions took a median of 0.16 ms, and `add_batch` of 5,000 of them into an empty index
took 0.57 s. Time your embedding model too: in most apps it, not the index, sets the speed.

- With several worker processes on one machine, limit each model's threads (fastembed's
  `threads=`, PyTorch's `torch.set_num_threads`) so they don't fight over the cores.
- Embed texts of similar length together, since a batch is padded to its longest text.

Chassis searches an HNSW graph (Hierarchical Navigable Small World). `IndexOptions` sets how it
trades build speed, search speed and recall.

## Options

You can pass an `IndexOptions` object when opening the index.

```python
from chassisdb import VectorIndex, IndexOptions

options = IndexOptions(
    max_connections=32,    # 'M' in HNSW papers
    ef_construction=400,   # Build quality
    ef_search=100          # Search quality
)

index = VectorIndex("tuned.chassis", dimensions=128, options=options)
```

### Parameters

| Parameter | Description | Default | Impact |
| --- | --- | --- | --- |
| **`max_connections`** | Max edges per node in the graph. | 16 | Higher = Better recall, higher memory usage. |
| **`ef_construction`** | Size of the dynamic candidate list during build. | 200 | Higher = Slower build, higher quality graph. |
| **`ef_search`** | Size of the dynamic candidate list during search. | 50 | Higher = Slower search, better recall. |
| **`precision`** | `"full"` keeps 32-bit floats, `"half"` 16-bit floats. Fixed when the index is created. | `"full"` | `"half"` halves the vectors' size in the file and in memory, and on an index too large for the CPU's caches searches are a tenth to two thirds faster on most machines measured. Components are rounded to about three decimal digits; recall on the embeddings measured was the same. A component of 65,520 or more in magnitude is refused, and older releases can't open the file. |
| **`huge_pages`** | Ask Linux to keep the vectors on 2 MB pages: `True` from opening, `False` never, `None` leaves it to `warm`, which asks under vectors of 2 KB or more. | `None` | On an index too large for the CPU's caches, faster search on most CPUs measured, by up to 27%; 2 to 5% slower on AMD's Zen 5 with vectors of 128 dimensions, and 10 to 20% slower on a Zen 4 with 384, or 768 in half precision. Does nothing off Linux, or where the kernel and filesystem don't keep files on huge pages (ext4 on Linux 6.17 does). Pages already in memory stay small. An index much larger than memory reads 2 MB per miss. |
| **`warm`** | Read the index into memory on another thread from the moment it is opened; `warm()` asks for it later. | `False` | While the file isn't in memory yet, as after a reboot, searches stop waiting for the disk one page at a time. On an Apple M5 the hundredth search over 99,000 vectors of 1,536 dimensions came 0.37 s after the process started, instead of 1.9 s. For an index that fits in memory. Linux 5.14 or later, or macOS; an older Linux reads in only some of the index, and Windows none. |

## Smaller files

An index is mostly its vectors: 4 bytes a dimension, 2 with `precision="half"`, and about 160 bytes
a vector for the graph. Besides half precision, the way to shrink it is fewer dimensions. Matryoshka
embeddings are trained so that the first part of a vector works on its own; OpenAI's
text-embedding-3 models take a `dimensions` argument for it. Keeping 512 of 1,536 dimensions makes
the index about a third of the size. Measure recall on your own queries first: OpenAI reports
text-embedding-3-large at 256 dimensions doing better than ada-002 at 1,536, but a model not trained
this way loses much more. Cut vectors yourself only with the `cosine` metric, or normalize them
again. Lower `max_connections` saves less. 12 instead of 16 makes a 128-dimension file in half
precision 7% smaller, and one of 1,536 dimensions 1%. Builds are 7 to 18% faster, but searches for
0.99 recall at 128 and 384 dimensions take up to 14% longer.

## Many queries

`search_batch(queries, k)` searches a `(count, dimensions)` array in one call and returns the ids
and distances as two `(count, k)` arrays. `search` adds 13 to 26 µs a query on the server CPUs
measured (checking the query, the call, the result objects), which `search_batch` pays once per
batch. On GitHub's server runners on 2026-10-17, a batch made 1.12 to 1.16 times the queries a
second of one call per query over a million vectors of 128 dimensions, and 1.00 to 1.09 times over
99,000 of 1,536 (0.95 recall). On 2026-10-10 a batch was within 5% of the same searches from Rust.

## Batches

A flush waits for the disk (`fsync`), so it takes milliseconds. To add many vectors quickly:

1. Add vectors with `add_batch`, which links them on every core, in chunks of, say, 10,000.
2. Call `flush()` once at the end, or every few chunks to bound what a crash can lose. A flush
   after every `add` waits for the disk once per vector.

```python
import numpy as np
from chassisdb import VectorIndex

vectors = np.random.rand(50_000, 384).astype(np.float32)  # stands for your embeddings

with VectorIndex("bulk.chassis", dimensions=384) as index:
    for start in range(0, len(vectors), 10_000):
        index.add_batch(vectors[start : start + 10_000])
    index.flush()
```
