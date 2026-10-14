# Quickstart

This page takes you from an empty file to your first search: open an index, add vectors, keep
them, and search them. The vectors here are random; in your app they come from your embedding
model. Each block runs after the one before it.

## Open

An index is one file. Give it a path and the number of dimensions your vectors have; the file is
created if it doesn't exist. The dimensions are fixed once the index is created, and you give the
same number to open it again.

```python
import numpy as np
from chassisdb import VectorIndex

index = VectorIndex("data.chassis", dimensions=768)
```

## Add

`add_batch` takes an array with one vector per row, links the vectors on every core, and returns
their ids. `add` takes one vector, as a list or an array. Each vector gets the next id unless you
pass your own: `add_batch(vectors, ids=...)`, `add(vector, id=...)`.

```python
vectors = np.random.rand(1000, 768).astype(np.float32)
ids = index.add_batch(vectors)  # 0 to 999

vector = np.random.rand(768).astype(np.float32)
index.add(vector)  # 1000
```

## Flush

!!! warning "Durability"
    A search finds a vector as soon as it is added, but the vector is on disk only after
    `flush()`. If the process ends before then, or you close the index without one, every add and
    delete since the last flush is gone, as after a crash. The file stays usable either way.

```python
index.flush()
```

A flush waits for the disk, so it takes milliseconds. Flush after a batch of changes, not after
each one.

## Search

`search` returns the `k` nearest vectors, nearest first, each with its `id` and `distance`. The
distance is Euclidean (L2) by default, or `1 - cosine similarity` for an index created with
`IndexOptions(metric="cosine")`. Smaller is nearer.

```python
query = np.random.rand(768).astype(np.float32)
for result in index.search(query, k=5):
    print(result.id, result.distance)
```

## Close

One process writes a file at a time, so close the index when you're done with it. A `with` block
closes it for you, and flushes first when the block ends normally. If an exception leaves the
block, it closes without flushing, so half-finished changes aren't kept.

```python
index.close()

with VectorIndex("data.chassis", dimensions=768) as index:
    index.add(np.random.rand(768).astype(np.float32))
# Flushed and closed
```

## Next

- [Performance](performance.md): the options for build speed, search speed and recall.
- [VectorIndex](../api/index_class.md): every method, with what it raises.
- [Next to your database](https://tanvincible.github.io/chassis/guide/with-a-database.html):
  keeping your text and metadata in SQLite under the same ids, and searching among the rows a
  query selects.
