# Storage Layer

The storage layer manages the lifecycle of a Chassis file, including creation, validation, growth, and durability.

## Opening a File

When you call `Storage::open`, the following steps occur:

1. Open or create the file with read and write permissions
2. Acquire an exclusive lock on the file using `flock` (Linux/macOS) or equivalent
3. If the file is new or empty, initialize it with a header
4. Map the file into memory using `mmap`
5. Validate the header magic bytes, version, and dimensions
6. Return a `Storage` handle or an error if validation fails

## File Growth

The file grows as needed to accommodate new vectors. Growth happens in the `ensure_capacity` method, which is called before each insert.

Growth is page-aligned. If the file needs to grow by 100 bytes, it actually grows by 4096 bytes (one page). This wastes some disk space but reduces the number of `mmap` remap operations and aligns writes to hardware block boundaries.

When the file grows, the existing `mmap` is unmapped and a new one is created. All pointers into the old mapping become invalid. This is why `get_vector` returns an owned `Vec<f32>` instead of a reference.

## Durability

Inserts are not durable by default. They write to the memory-mapped region, which the OS flushes to disk at its discretion.

The `commit` method guarantees durability. It calls `mmap.flush()` to push changes to the kernel page cache, then calls `file.sync_data()` to force a write to physical storage. This ensures data survives a power loss or kernel panic.

On Linux, `sync_data` maps to `fdatasync`, which flushes file contents but not metadata. On macOS, it maps to `fcntl(F_FULLFSYNC)`. On Windows, it maps to `FlushFileBuffers`.

## Concurrency

The current implementation does not support concurrent access. Only one process can hold the file lock at a time.

If a second process tries to open the file, `Storage::open` returns an error immediately. It does not block or retry.

When the `Storage` object is dropped, the lock is released automatically.
