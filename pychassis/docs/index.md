# PyChassis

**PyChassis** provides high-performance Python bindings for the **Chassis** vector storage engine.

It combines the safety and raw speed of Rust with the ease of use of Python. Unlike many vector libraries that run primarily in-memory, Chassis is designed for **embedded, on-disk persistence** first.

## Key Features

* **Zero-Copy Search**: Vectors are memory-mapped, allowing instant access to datasets larger than RAM.
* **Crash Safety**: ACID-compliant persistence guarantees your data is safe even if the process is killed.
* **Standard Interface**: Fully compatible with NumPy arrays.
* **No Server Required**: Runs entirely in-process. No Docker containers or external services to manage.

## Installation

PyChassis requires Python 3.8+ and a compatible OS (Linux, macOS, Windows).

```bash
pip install pychassis
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
