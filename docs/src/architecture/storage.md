# Storage Layer

The storage layer (`storage.rs`) owns the index file: its lock, its regions, and the commit
protocol. The byte layout is in [File Format](./file-format.md).

## Opening a File

`Storage::open` and `VectorIndex::open`:

1. Open or create the file, and take an exclusive lock on it (`flock` on Unix, `LockFileEx` on
   Windows). On Unix they then check that the locked file is still the one at the path, since a
   migration may have replaced it in between.
2. Initialize an empty file: header copies A and B, fsync, then fsync the directory so power loss
   can't take the new file with it.
3. Migrate a version 1 or 2 file to version 3.
4. Read both header copies, take the newest valid one, map every committed region, and run recovery
   if a flush with deletes was interrupted.

## Growth

Slots fill segments; when a slot needs a segment that doesn't exist yet, the next region is
appended at the header's file end and mapped. Existing mappings never change, so a reference into
the file stays valid while the index grows, and nothing is ever copied or remapped. Regions past
the committed file end were left by a flush that never committed, and are reused.

## Durability

Writes are not durable until a commit. A commit calls `msync` on every mapped region, then
`sync_all` (`fsync` on Linux, `fcntl(F_FULLFSYNC)` on macOS, `FlushFileBuffers` on Windows), then
writes the header copy that does not hold the newest header, and syncs again. If any of those
fails, the storage refuses every later commit: after a failed fsync the OS may already have dropped
the dirty pages, so a retry that succeeds would commit data that is gone. Reopen the index instead.

## Concurrency

One writer at a time: a second writer's `open` of a locked file returns an error immediately. It
does not block or retry. The lock is released when the `Storage` is dropped.

Readers (`Storage::open_read_only`, behind `IndexReader`) take no lock and map the file read-only.
Before every search a reader takes a snapshot: it copies both header copies a word at a time,
keeps the newest valid one (never an older one than it already used), maps any region the header
or the live page shows, and routes through slots up to the live page's routing count. Search
returns only slots the header commits and no flush up to its epoch deleted. The writer stores
neighbor lists, delete marks, headers and the live page as atomic words, and rewrites a neighbor
list in place so that an entry in both the old and the new list keeps its slot.
