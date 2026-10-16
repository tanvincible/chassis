# ADR-0024: Backups

**Date:** 2026-10-16
**Status:** Proposed

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

Measured on 2026-10-16 on an Apple M5, with 20,000 clustered vectors of 384 dimensions flushed and
5,000 more added but not flushed, a plain copy of the index file without its undo file opened with
the 20,000, `check()` found nothing out of reach, and each of 500 sampled vectors was its own
nearest. The adds had only written links to vectors the header doesn't commit, which opening
passes by. So a copy between flushes is sound in practice. A copy across a commit is the real
hazard.

## Decision

### 1. `backup(path)` on the writer

Flush, then copy the file to `path` while holding the writer, so nothing changes during the copy:
after a flush there is no undo file, and the copy is exactly the last flush. Write to a temporary
name beside `path`, sync it, rename it into place and sync the folder, so a crash never leaves a
half copy under the backup's name. Holes in the file stay holes. In Rust `VectorIndex::backup`, in
C `chassis_backup`, in Python `backup()`.

### 2. From another process, a copy that starts again

A reader takes a snapshot, copies the file, and takes another. If a flush came in between, it
copies again; after a few tries it fails with an error that says to back up from the writer. Then
it opens the copy as a writer, which recovers it as after a crash, and checks it: anything out of
reach that the original didn't have means the copy caught lists mid-change, and it is compacted.
SQLite's backup restarts the same way when the database changes under it. The command line's
`backup` uses this when another process writes the index, and 1 when none does.

### Left out

- **Exporting the vectors and building a new index.** Always sound, and as slow as a full build.
  Compacting a copy already rebuilds the graph when it is needed.
- **Asking the writer to pause.** Chassis has no channel between processes, and adding one for
  backups would be the first.

## To measure before accepting

- How often a copy from a reader has to start again under a steady stream of flushes, and how long
  a backup of 1M vectors takes.
- That a copy from a reader, taken while a writer adds without flushing, recovers to the last
  flush across the crash tests' cut points, as the power-loss simulator does for a crash.
- Whether `std::fs::copy` keeps holes on Linux (`copy_file_range`) and Windows, or the copy has to
  skip zero blocks itself.

## Consequences

- An app backs up its index with one call from the writer, and a person with one command.
- A backup keeps `user_version`, so restoring the index next to a restored database tells the app
  whether to mend.
- Restoring is copying the backup into place while nothing has the index open.
