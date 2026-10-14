# chassisdb

`chassisdb` is Chassis for Python: an embedded vector index for local semantic search. Your index
is one file, and Chassis runs inside your process, with no server to run or connect to. Store
embeddings under your own ids, search them, and keep the rest of your data where it already lives.

## Features

* **On disk first**: an index is one file, read in place, so opening it loads nothing and an index
  larger than memory still works.
* **Durable flushes**: if the process is killed, every add and delete up to the last `flush()` is
  kept, and the index stays usable.
* **NumPy**: vectors go in as arrays or lists, and come out as arrays.
* **No server**: nothing to deploy or keep running.

## Install

Chassis needs Python 3.11 or later, on Linux (x86-64 or ARM), macOS on Apple silicon, or Windows
(x86-64):

```bash
pip install chassisdb
```

Import it as `chassisdb`. To build it from a checkout of the repository instead, which needs
Rust:

```bash
pip install ./chassisdb
```

## Example

```python
import numpy as np
import chassisdb

index = chassisdb.VectorIndex("my_vectors.chassis", dimensions=128)  # created if missing

vector = np.random.rand(128).astype(np.float32)
vector_id = index.add(vector)
index.flush()  # on disk once this returns

results = index.search(vector, k=5)
print(results[0].id, results[0].distance)  # the vector itself, at distance 0
```

The [Quickstart](user-guide/quickstart.md) goes through each step, and
[Performance](user-guide/performance.md) through the options.
