# Getting Started

## Installation

Add Chassis to your `Cargo.toml`:

```toml
[dependencies]
chassis-core = "0.4.0-alpha"
```

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