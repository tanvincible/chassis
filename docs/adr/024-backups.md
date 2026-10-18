# ADR-0024: Backups

**Date:** 2026-10-16, accepted 2026-10-18
**Status:** Accepted

## Context

The errors page tells users to restore a damaged index from a backup, and nothing says how to make
one. Copying the file is safe only sometimes:

- **Between flushes, a writer changes committed lists in place.** Before a committed node's lists
  first change after a flush, the old ones go to `<name>.undo`, and a writer that opens the index
  after a crash writes them back (ADR-0012). A copy of the index alone, taken between flushes,
  misses that file.
- **A commit writes the data, then the header.** A copy that reads the regions before a commit and
  the header after it holds a header that names vectors the copy never got: zeros, which no
  checksum covers, so nothing would notice.

## Decision

### 1. `backup(path)` on the writer

Flush, then copy the file to `path` while holding the writer, so nothing changes during the copy.
The copy goes to `<path>.partial`, is synced, and is renamed over `path`, so a crash never leaves
half a backup under the backup's name. An undo file left beside `path` is removed first: a writer
opening the backup would apply it. A `path` that is the index's own file, or its undo file, is
refused. In Rust `VectorIndex::backup`, in C `chassis_backup`, in Python `backup()`.

### 2. From a reader, the index and then its undo file

`IndexReader::backup` (`chassis_backup` on a reader's handle, `backup()` on a `read_only` index)
copies the writer's last flush while the writer keeps writing:

1. Read the checksum and sequence words of both header copies.
2. Copy the index file, through the reader's own handle, then `<name>.undo` if there is one.
3. Read the words again. If any changed, or the file is no longer the one at the path, a flush
   or a rollback came in between: start again. After five tries, fail with an error that says to
   back up from the writer.
4. Open the copy as a writer, which recovers it as after a crash: it writes back the lists the
   undo file saved and clears what the last flush didn't commit.

The copy is then exactly the last flush. A writer saves each committed list to the undo file before
it first changes it, and the undo file is copied second, so it holds every list the copy caught
changed. Only a commit or a rollback removes the undo file, and both write a header, which step 3
sees. SQLite's online backup restarts the same way when the database changes under it.

### 3. A rollback that puts lists back writes a header

A writer opening the index after a crash, or `rollback()`, writes back the saved lists and removes
the undo file; before this decision it wrote no header doing so. A backup that copied the index
before that and the undo file after it would hold changed lists and nothing to undo them with. So
once it has put lists back, the writer writes the same commit's header again, with the next
sequence number, before it removes the undo file.

### Left out

- **Exporting the vectors and building a new index.** Always sound, and as slow as a full build.
- **Asking the writer to pause.** Chassis has no channel between processes, and adding one for
  backups would be the first.
- **Keeping the file's holes.** The space past each region's last write is a hole; a backup writes
  it out as zeros. It costs disk, not correctness. Revisit if a backup's size becomes a complaint.

## Measured

On 2026-10-18, on an Apple M5, with `examples/backup_cost.rs`:

| Index | File, on disk | Backup, writer | Backup, reader | Backup on disk |
| --- | --- | --- | --- | --- |
| 100,000 × 384 | 221 MB, 175 MB | 0.35 s | 0.35 s | 221 MB |
| 1,000,000 × 128 | 713 MB, 688 MB | 1.15 s | 0.86 s | 713 MB |

With a writer adding ten vectors and flushing every 10 or 100 ms, none of ten backups from a reader
of the smaller index held a flush; with a flush every second, all ten did. At a million vectors, a
flush every second still let all ten hold one, in 1.6 s each with the tries. A backup from a reader
needs a gap between flushes as long as the copy.

The tests take a copy apart at its seam: a flush or a rollback between the two copies sends the
backup back to the start, and the copy then recovers to exactly the last flush. Copied the other
way round, undo file first, the same changes leave the copy wrong, so the tests can tell. With the
header check, the copy order or the rollback's header taken out, a test fails.

## Consequences

- An app backs up its index with one call, from the writer or from any process that can read it.
- A busy writer has to back up itself: a reader's copy needs a pause in its flushes.
- A backup keeps `user_version`, so an index restored next to a restored database tells the app
  whether to mend, whenever each backup was taken.
- Restoring is copying the backup into place while nothing has the index open.
