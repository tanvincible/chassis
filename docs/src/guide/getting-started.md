# Getting started

This page takes you from nothing to a first search, in Python or Rust. From C, start with the
[C API](c-api.md).

## Python

```sh
pip install chassisdb
```

```python
from chassisdb import VectorIndex

index = VectorIndex("quickstart.chassis", dimensions=3)  # created if it doesn't exist

index.add([1.0, 0.0, 0.0])  # id 0
index.add([0.0, 1.0, 0.0])  # id 1
index.add([0.0, 0.0, 1.0])  # id 2
index.flush()  # on disk once this returns

results = index.search([1.0, 0.1, 0.0], k=1)
print("nearest:", results[0].id)  # 0
```

The [Python docs](https://tanvincible.github.io/chassis/python/) go on from here: batches, your own
ids, options and every method.

## Rust

Add Chassis to your `Cargo.toml`:

```toml
[dependencies]
chassisdb = "0.8"
```

```rust
use chassisdb::{IndexOptions, VectorIndex};

fn main() -> chassisdb::Result<()> {
    // Created if it doesn't exist
    let mut index = VectorIndex::open("quickstart.chassis", 3, IndexOptions::default())?;

    index.add(&[1.0, 0.0, 0.0])?; // id 0
    index.add(&[0.0, 1.0, 0.0])?; // id 1
    index.add(&[0.0, 0.0, 1.0])?; // id 2
    index.flush()?; // on disk once this returns

    let results = index.search(&[1.0, 0.1, 0.0], 1)?;
    assert_eq!(results[0].id, 0);
    println!("nearest: {}", results[0].id);
    Ok(())
}
```

A debug build runs Chassis's arithmetic many times slower. To build Chassis optimized while your
own crate stays in debug:

```toml
[profile.dev.package.chassisdb]
opt-level = 3
```

[Basic usage](./basic-usage.md) goes on from here, and the [Rust API](./api-reference.md) has every
method.
