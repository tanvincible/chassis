# Getting Started

## Installation

Add Chassis to your `Cargo.toml`:

```toml
[dependencies]
chassis-core = "0.1"
```

## Opening an Index

Create or open a Chassis file by specifying a path and the number of dimensions your vectors will have:

```rust
use chassis_core::Storage;

let mut storage = Storage::open("vectors.chassis", 768)?;
```

If the file does not exist, Chassis creates it. If it exists, Chassis validates that the dimensions match what you specified. A dimension mismatch returns an error.

## File Locking

Chassis locks the file when you open it. Only one process can open a Chassis file at a time. If another process already has the file open, `Storage::open` returns an error.

The lock is released when the `Storage` object is dropped.

## Error Handling

All operations return `Result` types. Errors include context about what failed and why. For example, trying to open a file with mismatched dimensions produces an error message that states both the expected and actual dimensions.
