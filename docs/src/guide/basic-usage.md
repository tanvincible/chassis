# Basic Usage

## Inserting Vectors

Use the `insert` method to add a vector to storage. The vector must match the dimensions specified when opening the file.

```rust
let vector = vec![0.1, 0.2, 0.3]; // Must be 768 elements for a 768-dim index
let id = storage.insert(&vector)?;
```

The method returns the index of the inserted vector, starting from 0. Inserts are append-only. Each call to `insert` increments the internal counter.

If the vector has the wrong number of dimensions, `insert` returns an error.

## Retrieving Vectors

Retrieve a vector by its index:

```rust
let vector = storage.get_vector(id)?;
```

This returns an owned `Vec<f32>` containing a copy of the vector data. If the index is out of bounds, the method returns an error.

## Committing Changes

By default, `insert` does not guarantee durability. To ensure data survives a crash or power loss, call `commit`:

```rust
storage.insert(&vector1)?;
storage.insert(&vector2)?;
storage.commit()?; // Flush to disk
```

The `commit` method forces all pending writes to disk using `fsync`. This is slow but necessary for durability.

For batch operations, insert many vectors and call `commit` once at the end.

## Checking Count and Dimensions

Query the current state of the storage:

```rust
let count = storage.count();       // Number of vectors
let dims = storage.dimensions();   // Vector dimensions
```

These methods do not perform I/O. They read from the in-memory header.
