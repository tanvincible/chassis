# Getting Started

## Installation

Add Chassis to your `Cargo.toml`:

```toml
[dependencies]
chassis-core = "0.8"
```

A debug build runs Chassis's arithmetic many times slower. To build Chassis optimized while your
own crate stays in debug:

```toml
[profile.dev.package.chassis-core]
opt-level = 3
```

From Python, `pip install chassisdb` (see the [Python bindings](https://github.com/tanvincible/chassis/blob/main/chassisdb/README.md));
from C, see the [C API](c-api.md).

## Quick Start

```rust
use chassis_core::{VectorIndex, IndexOptions};
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Open the index (creates file if missing)
    let mut index = VectorIndex::open(
        "quickstart.chassis", 
        3, // 3D vectors for demo
        IndexOptions::default()
    )?;

    // 2. Add some data
    index.add(&[1.0, 0.0, 0.0])?; // ID 0
    index.add(&[0.0, 1.0, 0.0])?; // ID 1
    index.add(&[0.0, 0.0, 1.0])?; // ID 2
    
    index.flush()?;

    // 3. Search
    let query = vec![1.0, 0.1, 0.0];
    let results = index.search(&query, 1)?;

    assert_eq!(results[0].id, 0); // Should match ID 0 best
    println!("Nearest neighbor: ID {}", results[0].id);

    Ok(())
}
```