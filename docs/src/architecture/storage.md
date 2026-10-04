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

The file grows by 25% at a time, rounded to a page, so a growing index remaps O(log n) times. Growing
one page at a time remapped on every other insert.

When the file grows, the existing `mmap` is unmapped and a new one is created. All pointers into the old mapping become invalid. This is why `get_vector` returns an owned `Vec<f32>` instead of a reference.

## Durability

Inserts are not durable by default. They write to the memory-mapped region, which the OS flushes to disk at its discretion.

`commit` calls `mmap.flush()` (msync) and then `file.sync_all()`, which is `fsync` on Linux,
`fcntl(F_FULLFSYNC)` on macOS and `FlushFileBuffers` on Windows. Writes made after the last commit
can reach disk in any order.

## Concurrency

The current implementation does not support concurrent access. Only one process can hold the file lock at a time.

If a second process tries to open the file, `Storage::open` returns an error immediately. It does not block or retry.

When the `Storage` object is dropped, the lock is released automatically.
