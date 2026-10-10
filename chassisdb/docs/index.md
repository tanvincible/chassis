# chassisdb

**chassisdb** provides Python bindings for **Chassis**, an embedded vector index for local semantic search: one file, in your process, no server.

It combines the safety and raw speed of Rust with the ease of use of Python. Unlike many vector libraries that run primarily in-memory, Chassis is designed for **embedded, on-disk persistence** first.

## Key Features

* **Zero-Copy Search**: Vectors are memory-mapped, allowing instant access to datasets larger than RAM.
* **Crash Safety**: If the process is killed, every add and delete up to the last `flush()` is kept and the index stays usable.
* **Standard Interface**: Fully compatible with NumPy arrays.
* **No Server Required**: Runs entirely in-process. No Docker containers or external services to manage.

## Installation

Chassis needs Python 3.13 or later, on Linux (x86-64 or ARM), macOS on Apple silicon, or Windows
(x86-64):

```bash
pip install chassisdb
```

The package is named `chassisdb` and imported as `chassis`. To build it from a checkout of the
repository instead, which needs Rust:

```bash
pip install ./chassisdb
```

## The "Hello World" of Vector Search

```python
import chassis
import numpy as np

# 1. Open an index (creates file if not exists)
index = chassis.VectorIndex("my_vectors.chassis", dimensions=128)

# 2. Add some data
vector = np.random.rand(128)
vec_id = index.add(vector)

# 3. Persist to disk
index.flush()

# 4. Search
results = index.search(vector, k=5)
print(f"Found ID: {results[0].id} with distance: {results[0].distance}")
```
