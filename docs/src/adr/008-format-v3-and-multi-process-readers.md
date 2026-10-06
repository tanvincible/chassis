# ADR-0008: File Format v3 and Multi-Process Readers

**Date:** 2026-10-04 **Status:** Proposed. Accept only after the experiments in "Before Accepting"
pass. Implemented on 2026-10-05, readers in other processes included; see "Implementation Status".

## Summary

Chassis aims to be to vector search what SQLite is to relational data: one embedded file that any
process can open, that survives crashes, and that stays readable for decades. Two things block that
today: one process at a time, and a file format that must change again before it can be promised
stable. This ADR proposes one last breaking change, file format v3, designed so that:

1. **Committed data never moves or changes.** The file grows only by appending segments, so graph
   relocation and the pointer invalidation that comes with remapping disappear. Operations that
   replace the graph wholesale (`rebuild_graph()`, a new `max_connections`) or reclaim space
   (vacuum, migration) write a new file and swap it in (decision 8). Apart from single-word edge
   updates (decision 5), nothing rewrites committed bytes a reader may be traversing; a writer that
   reopens the file after an unflushed exit reuses slots and regions past the committed counts,
   which readers only route through.
2. **Data and index are separate.** Vectors, ids, deletes and (later) metadata are the data and are
   never lost. The HNSW graph is derived from them and can always be rebuilt into a new file.
3. **Graph records shrink about 13×,** from 2,192 to about 164 bytes per vector at the default `M =
   16` (8× at `M = 4`, 15× at `M = 32`). Files shrink less, since vectors don't change: 5× on
   SIFT-1M and 1.6× on dbpedia, with bytes written within 3% of hnswlib's file. Smaller records
   don't by themselves make in-memory search faster: on identical graphs, record size and id width
   changed QPS by less than ±5%.
4. **Many processes can read while one writes,** without locks on the read path, by relying on HNSW
   tolerating slightly stale neighbor lists.
5. **The format carries SQLite-style read and write versions,** so later features never force a
   breaking change, and old releases refuse files they can't read instead of misreading them.

Multi-process readers can ship after v3. What v3 must get right is the *format properties* that
readers need, so that adding them later needs no format change.

## Context

Measured on 2026-10-04 (see the [Performance](../architecture/performance.md) page and the
`bench/ann` suite):

* Every node record reserves 2,192 bytes: room for 16 layers of u64 neighbor ids. 94% of nodes only
  use layer 0, which is 272 bytes. On SIFT-1M that made a 3.4 GB file against hnswlib's 660 MB, and
  4.2 GB peak memory against 0.93 GB.
* The vector zone and the graph zone share one file. When vectors outgrow their space, the whole
  graph is copied elsewhere in the file (ADR-0005 amendment). That copy is crash-safe now, but it
  means data moves, and anything holding an address into the old place, such as another process's
  mapping, breaks.
* Growing the file remaps it. That invalidates every pointer into the mapping, which is why searches
  and writes can't overlap without a lock, even within one process.
* An exclusive file lock keeps every other process out, readers included.
* Ids live in node records spread through the graph, so building the id table reads the whole graph
  zone.

SQLite's lesson is that the file format *is* the product: applications store data for years and
expect any future release to read it. Chassis can only make that promise once the format stops
changing, so the remaining breaking changes should all land together.

## Decision

### 1. Layout: one file of append-only segments

```text
0        [header A]
64 KiB   [header B]
128 KiB  [live page: routing count, see decision 5]
192 KiB  [segment 0][segment 1][segment 2] ... [heap chunks and table pages, interleaved]
```

The two header copies sit 64 KiB apart so that they never share a page (pages are 16 KiB on Apple
silicon and up to 64 KiB on Linux arm64). Every segment, heap chunk and table page also starts at a
multiple of 64 KiB, the Windows mapping granularity and a multiple of every supported page size; the
writer pads to the next boundary before appending one. Unpadded, segment k would start at `192 KiB +
4096·(40 + dims)·(2^k − 1)`, which `mmap` rejects (EINVAL) on Apple silicon whenever `dims` is not a
multiple of 4, and which is never 64 KiB-aligned for common dimensions.

A **segment** holds a fixed number of slots and contains, as three parallel arrays:

| Array | Per slot | Contents |
|-------|----------|----------|
| Slot headers | 24 bytes | id (u64), metadata reference (u64, 0 = none), deleted epoch (u64, 0 = live) |
| Vectors | `dims × 4` bytes | the vector, never rewritten after its flush commits |
| Level-0 records | `8 + M0 × 4` bytes | layer count, upper-layer reference, `M0` neighbor slots as u32 |

Segment `k < K` holds `1,024 · 2^k` slots. Every later segment holds `C = 1,024 · 2^K` slots, the
largest such power of two whose segment fits in 256 MiB: 262,144 slots (176 MB) at 128 dimensions,
32,768 (207 MB) at 1,536. The header stores `K`, so later releases never recompute it. With `B = C −
1,024`, slot `s < B` is in segment `k = ⌊log2(s + 1,024)⌋ − 10` at offset `s + 1,024 − 2^(k+10)`,
and slot `s ≥ B` is in segment `K + ((s − B) >> log2 C)` at offset `(s − B) & (C − 1)`. Its address
is that segment's file offset, plus the array's base, plus the offset times the array's stride.

Upper-layer neighbor lists (the 6% of nodes with more than one layer) go in an append-only **upper
heap**, and metadata will go in a **metadata heap**. Heap chunks grow by the same scheme, and an
entry never straddles two chunks. The segment and chunk tables don't fit in a header (2^32 slots
take 16,392 segments at 128 dimensions and 131,077 at 1,536), so they live in append-only table
chunks, each written before the header that first counts it. The header holds only the table chunks'
offsets.

The file only ever grows by appending a segment or a heap chunk. Nothing is copied or moved during
normal operation.

### 2. Data versus index

Slot headers and vectors are the data. A slot's id, vector and metadata reference are written once,
before the flush that commits them, and never change. Its deleted epoch is 0 until a flush deletes
it; if that flush never commits, recovery resets it to 0.

The graph (level-0 records, upper heap, entry point) is derived from the data. Readers traverse it
while the writer works, so it is never regenerated in place. An in-place rebuild under live readers,
tested on 2026-10-04, dropped their recall@10 from 0.987 to between 0.13 and 0.59, and with a
different `max_connections` most queries failed outright. `rebuild_graph()` instead writes a new
file: it copies the committed slots, builds the graph from the vectors with the requested
`max_connections`, and swaps the new file in (decision 8). Copying vectors takes about 1 s per 512
MB, against about 750 s to build SIFT-1M's graph. This is also how `max_connections` changes (it
changes the level-0 record size) and how the space of deleted vectors is reclaimed. Every flush
commits graph and slots together, so the graph always covers every committed slot.

### 3. Compact graph records

Neighbor ids become u32, with `u32::MAX` meaning "empty", which caps an index at about 4.29 billion
vectors. The level-0 record is `8 + 8M` bytes, 136 with the default `M = 16`. Upper layers cost `4M`
bytes each, and a node has `1/(M − 1)` of them on average: 4.3 bytes per node at `M = 16` (4.27 when
sampled from the code's own layer distribution). With the 24-byte slot header, overhead per vector
is `32 + 8M + 4M/(M − 1)` bytes: 164 at `M = 16`, against today's `16 + 136M` (2,192). Sizes on
2026-10-04 (v3 columns simulated, the rest measured):

| Dataset | Vectors | v3 overhead | v3 bytes written | v3 file size | hnswlib | Today |
|---------|---------|-------------|------------------|--------------|---------|-------|
| SIFT-1M | 512 MB | 164 MB | 676 MB | 712 MB | 660.5 MB | 3,379 MB |
| GloVe, 1.18M × 100 | 473 MB | 194 MB | 668 MB | 742 MB | 649.2 MB | 3,289 MB |
| dbpedia, 99k × 1536 | 608 MB | 16 MB | 625 MB | 820 MB | 623.0 MB | 982 MB |

File size also counts the unused end of the last segment (36, 74 and 196 MB here). hnswlib saves an
exact dump with no room to grow, so compare it with bytes written. v3 is 15.7 bytes per vector
larger: its 24-byte slot header and 8-byte level-0 head, against hnswlib's 8-byte label and 4-byte
list headers.

Ids sit in a 24-byte-per-slot array at the start of each segment, so building the id table for 1M
vectors reads 24 MB in 11 sequential runs instead of the whole graph.

`M` must be between 2 and 32,767. `M = 1` puts every node on all 16 layers, and `M0 = 2M` must fit
in a u16.

The 33-candidate cap on neighbor selection, which mattered more than any layout effect measured
(7–18% faster search at equal recall once lifted), was removed on its own on 2026-10-04 (see the
ADR-0004 amendment), so v3's benchmarks start from the uncapped code.

### 4. Double-buffered header

Each header copy carries a 64-bit sequence number and a 64-bit non-linear checksum (xxh3-64) over
the whole copy. CRC32 is not enough: it is affine, so a reader that catches a header mid-write can
assemble a mix of old and new words that passes it. LMDB alternates between two meta pages in the
same way, without a checksum. A new file starts with copy A at sequence 1 and copy B at 0 with equal
contents. Two valid copies with the same sequence number are corruption, and open fails.

A flush without deletes:

1. writes all slot data, vectors, graph records and any new segment or heap chunk;
2. fsyncs (`sync_all`, which is `F_FULLFSYNC` on Apple platforms);
3. writes the copy that does *not* hold the newest valid header, in one copy operation, with
   sequence number newest + 1;
4. fsyncs, and only then reports success.

The format assumes, as SQLite's "powersafe overwrite" does, that a torn sector write never changes
bytes outside the sectors being written.

Each header holds:
* magic, read version, write version and dimensions;
* metric, `M`, `M0` and maximum layers;
* committed slot count, entry point, max layer, delete epoch (u64), pending delete epoch (0 when
  none, see decision 6) and deleted count;
* references to the segment, upper-heap and metadata-heap chunk tables. The tables themselves live
  in append-only table pages: a 64 KiB header copy only has room for about 8,000 eight-byte entries,
  about 2 TiB of 256 MB segments, or 349M vectors at 1536 dimensions, well short of the u32 limit;
* the custom-id flag;
* reserved space.

### 5. Readers in other processes

A reader process opens the file read-only, with no lock. Per query it:

1. copies each header copy into local memory, verifies the checksum on the copy, takes the valid one
   with the higher sequence number, and issues an acquire fence. Every snapshot field comes from
   that one local copy: reading fields from the mapping after choosing a copy mixed fields from two
   commits 557 times in a measured run. A reader also re-checks the read version on every snapshot,
   never runs recovery, and ignores `pending_epoch`. That gives its *snapshot*: the committed slot
   count `N` and delete epoch `E`. A copy that fails its checksum while the writer is running is
   re-read, a bounded number of times, rather than skipped. A reader never adopts a snapshot older
   than one it already used: if the newest valid copy is older, it keeps its previous snapshot,
   which stays valid because committed data never changes;
2. maps any segment or heap chunk it hasn't mapped yet, and remaps an uncommitted one whose table
   entry changed: a writer that restarts after an unflushed exit reuses that space (committed
   mappings never change, because nothing committed moves);
3. searches, treating a slot as present only if it is below `N` and its deleted epoch is 0 or above
   `E`.

The writer guarantees only these things:

* **Committed data never changes.** Vectors, ids and metadata references are immutable once
  committed, so metadata changes by delete and re-add, never by rewriting a reference. The writer
  reuses a slot only if it is beyond every committed count (ghost slots after a crash). Readers
  never return those, except that after a power loss a reader may already have returned slots from
  the lost commit. They may route through them while a restarted writer rewrites them, vectors
  included, which costs routing quality, never results.
* **Graph updates are single aligned u32 stores made as relaxed atomics,** and readers load them as
  relaxed atomics too. Through a plain slice the compiler may re-read an id after its `< N` check,
  and a plain store is a data race in Rust's memory model; on arm64 relaxed atomics compile to the
  same `ldr` and `str`. A reader can see a neighbor list half old and half new, but every id it
  reads is a whole id. After commit, the writer touches only neighbor slots: never a committed
  record's layer count or upper-layer reference, even with the same value, and never with a
  whole-record copy. When pruning replaces neighbors, surviving entries keep their slots and only
  dropped entries' slots are overwritten, emptied slots last, so a reader can't miss an edge that is
  in both the old and the new list.
* **Data is written before the header that commits it,** with release ordering; readers load the
  header with acquire ordering. Both halves are needed: measured across processes on Apple silicon,
  dropping either let readers see a count ahead of its data.
* **The routing count is published after every insert.** Readers don't see *older* neighbor lists.
  They see the writer's *newer* ones, already pruned toward nodes the reader can't see yet. With
  readers limited to slots below `N`, measured recall on SIFT (100k growing to 200k, M=16) fell by
  0.03 to 0.16 points at an unflushed backlog of 1% of `N`, about 1 point at 5–10%, and 16 to 21
  points during a bulk load flushed once. A reader that could traverse every written node kept its
  recall (0.975 at ef 64). So after each insert's node and backlinks are written, the writer stores
  the routing count `R ≥ N` in the live page with release ordering. Readers traverse slots below `R`
  but return only slots below `N` that are live at `E`. `R` is not durable: recovery resets it to
  the committed count before reusing any ghost slot. A reader that loaded `R` before the reset, or
  is in the middle of a search, can still route through ghost slots the new writer is rewriting, and
  through uncommitted regions that now hold other data, so readers treat records at or past `N` as
  hints and skip any they can't decode. The same holds for a reader that opens the file after a
  power loss before any writer does: the live page it finds may describe records that never reached
  the disk. Auto-flushing every few hundred adds would also bound the loss, but it would make a
  caller's batch durable in pieces, which breaks ADR-0007's all-or-nothing flush.
* **The file never shrinks.** No v3 code path sets a smaller length: not recovery, not
  `rebuild_graph()`, not compaction. Space is reclaimed only by copy and rename (decision 8).
  Measured on macOS: after a 4 MiB to 1 MiB truncate, an attached reader died with SIGBUS, and after
  the file was extended again its pages read back as zero. The user docs warn that truncating or
  overwriting the file in place (`cp` over it, `truncate`) crashes attached readers.
* **Space is allocated before it is written.** Before writing into a new range, the writer allocates
  its blocks (`fallocate`, `F_PREALLOCATE`, or `SetEndOfFile` on Windows) in steps of at most 16 MiB
  ahead of its write position, not a whole segment at once, so a full disk fails the flush with an
  error while the unused end of the last segment stays sparse. Writing through a mapping into a
  sparse range kills the writer with SIGBUS on Linux; on macOS, measured on a full disk image,
  `msync` returned success and only `F_FULLFSYNC` reported the error, and 2,688 of 4,096 pages were
  lost.

Within one process, v3 removes the remap, but that is only one of four obstacles to dropping the
read side of the bindings' lock:

1. `add`, `delete` and `flush` take `&self` and run one at a time on an internal writer mutex. Rust
   rejects a `&mut` writer beside `&` readers, and aliasing them through the FFI pointer is
   undefined behavior (Miri flagged today's shape without the lock in 10 of 10 runs).
2. `Storage` never forms a `&mut [u8]` or `&[u8]` over bytes another thread may write.
3. Graph words, heap references and deleted epochs are read and written only as relaxed atomics
   through raw pointers (plain aligned stores raced under Miri in 4 of 10 runs).
4. The state searches read is published atomically: slot count, entry point with max layer, deleted
   count, pending deletes, the custom-id flag and the id table.

In-process searches keep read-your-writes: they use this published state, not the header snapshot,
so an add or delete is visible to them before the flush. Until all four are done, the bindings keep
their lock.

Only one process may write. On Unix the writer holds `flock(LOCK_EX)` on the file, as every release
since 0.1 does, so older releases stay excluded. On Windows the whole-file `LockFileEx` lock taken
today makes every `ReadFile` by another process fail (though not reads through a mapped view), so
the v3 writer locks a single byte at offset 2^62 instead. That byte still overlaps the whole-range
lock older releases request, so they still exclude each other. Readers hold no lock.

Readers see a commit as soon as its header is written, before its fsync finishes. After power loss
that commit may be gone, so a reader can have returned results the file no longer contains.

### 6. Crash safety

The commit point is the header write (decision 4), and adds, deletes and graph growth commit
together, as in ADR-0007. ADR-0007's dirty flag has no safe place in a double-buffered header:
written in place, a torn write can lose both copies' worth of state; placed in the commit header, it
is committed before the marks it guards. A flush with deletes therefore commits through two headers:

1. It writes its data, plus an *intent* header into the copy that does not hold the newest header:
   sequence newest + 1, the previous commit's state unchanged, and `pending_epoch = E + 1`. Then it
   fsyncs.
2. It writes `deleted_epoch = E + 1` into each deleted slot. Then it fsyncs.
3. It writes the commit header into the *other* copy, the one that held the previous commit, never
   over the intent: sequence newest + 2, the new slot count, epoch `E + 1`, `pending_epoch = 0`.
   Then it fsyncs.

Every add writes its slot's full 24-byte header with deleted epoch 0, so a crashed flush can't leave
a stale epoch in a slot that a later add reuses. On open with write access, if the newest valid
header has `pending_epoch ≠ 0`, recovery resets every deleted epoch above the header's epoch in
slots below the committed count, fsyncs, writes a header with `pending_epoch = 0` into the copy it
did not read, and fsyncs. Readers ignore `pending_epoch`. A power-loss model checker of this
protocol (1,194 crash continuations, including crashes inside recovery) found no violation; every
simpler placement of the flag failed it. The protocol assumes power loss never tears an aligned
8-byte write: recovery clears only marks above the committed epoch, so a mark torn at a byte
boundary (257 landing as 1) would survive it.

Recovery does not restore edges that an interrupted flush pruned from committed records, so some
committed vectors can be unreachable until `rebuild_graph()` (see the ADR-0005 amendment).

### 7. Versioning

As in SQLite's file header, there are two numbers:
* **Read version.** The oldest reader that can read this file correctly. A release refuses a file
  whose read version is newer than it knows.
* **Write version.** The oldest writer that may modify it. A release opens a newer-write-version
  file read-only.

v3 sets both to 3. A later feature that older readers can safely ignore, such as metadata they don't
query, bumps only the write version, so older releases can still read those files.

Both header copies hold `CHASSIS\0` at bytes 0–7, the read version at bytes 8–11 and the write
version at 12–15, as little-endian u32, at these offsets forever. Every release from 0.1 to v2 reads
bytes 8–11 as its version and refuses anything above what it knows before writing: tested on
2026-10-04 against the original code and the v2 branch, through `Storage::open`,
`VectorIndex::open`, the C API and Python, 64 of 64 opens of v3-shaped files were refused and every
file stayed byte-identical. A release checks both version numbers before the checksum, so a newer
header layout is reported as too new, not as corrupt.

Releases up to 0.6.3 initialize any file shorter than 4,096 bytes. v3 creates a new file in place
while holding its lock, which keeps those releases out until the first header is durable. A crash
before then leaves an empty file, which any release may initialize since it holds nothing, or a
header-sized file whose header copies are zero or torn and whose live page is zero, which v3 treats
as new: both copies are invalid at once only during creation. (This replaces a temporary name and a
rename, which added a path a concurrent creator could race without protecting more.)

Starting with 1.0, every release reads every file format since v3, forever. Releases only refuse
files, and only by these version checks; they never misread one.

### 8. Migration from v1 and v2

Migration is an explicit `migrate()`, also run on first open with write access. It converts the
graph (u64 to u32 neighbor ids, upper layers into the heap) instead of rebuilding it, and takes
O(n), once. It:

1. takes the exclusive lock on the original and reads only its committed state: slots below the
   graph header's node count, and a delete only where `0 < deleted_epoch ≤` the header's epoch.
   Copying a crashed file's raw epochs would turn its uncommitted deletes into committed ones;
2. creates `<name>.migrating`, replacing any left by an earlier crash, and takes its exclusive lock;
3. writes and converts, then fsyncs it;
4. renames it over the original;
5. fsyncs the directory (`F_FULLFSYNC` on Apple platforms, `MOVEFILE_WRITE_THROUGH` on Windows).
   Without this, a crash can bring the v2 file back after the first v3 flush was acknowledged;
6. keeps the already-locked new file open as the index.

It returns, and the first v3 flush can be acknowledged, only after step 5. File locks follow the
inode, not the path, so every open re-checks after taking its lock that its descriptor's inode is
still the one at the path, and reopens if not. v1 and v2 files have no lock-free readers. Any later
operation that replaces a v3 file by rename (`rebuild_graph()`, a `max_connections` change, vacuum)
must first commit a "superseded" header flag. A reader that sees it finishes its current query on
its snapshot, then reopens the path until the inode differs from the superseded file's. A reader
also reopens when `fstat` on its descriptor shows `st_nlink == 0`, which catches a replacement that
didn't go through the superseded flag.

On Unix the migrator keeps the original's descriptor and lock until it closes the index, which
narrows the window for releases before v3, since they don't re-check the inode. Windows refuses to
rename over a file that has any open handle, its own included. So on Windows the migrator unmaps and
closes the original before step 4, and its lock goes with it. It then calls
`MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)` directly, not `std::fs::rename`,
which on access denied silently retries with POSIX semantics and replaces the file under other
processes' handles. If another process opened the original in that window, the rename fails, the
original stays in use, and migration retries on the next open.

## Options Considered

| Option | Why not |
|--------|---------|
| Keep today's layout, add multi-process readers with a lock | Graph moves and remaps would make every write block readers in every process; graph records stay 13× larger than needed, and files 5× hnswlib's on SIFT-1M |
| Graph in a second file | Ends graph moves, but still remaps on growth, still large records, and two files to copy |
| SQLite-style write-ahead log | Needs a page table lookup on every record read, which breaks zero-copy search |
| LMDB-style copy-on-write pages | Every node lookup goes through a tree; a large redesign for guarantees HNSW doesn't need |
| A directory of segment files | Solves Windows growth (see below) but gives up the single file |
| Store each vector next to its graph record (hnswlib's layout) | Measured no faster than separate arrays in our search loop (0.89–1.02×), and mixes data with the derived index and makes rebuilding the graph a rewrite of every vector |

## Implementation Status

On 2026-10-05. Built:
* **Layout (decision 1):** header copies at 0 and 64 KiB, the live page, then 64 KiB-aligned
  segments, upper-heap chunks and table pages, as in [File Format](../architecture/file-format.md).
  A heap reference is `chunk << 32 | offset`, so chunks need no shared offset formula. No metadata
  heap is created yet; every slot's metadata reference is 0.
* **Compact records (decision 3):** u32 neighbor ids; `M` from 2 to 32,767.
* **Double-buffered header (decision 4):** xxh3-64 over the header's own bytes, which are 136 bytes
  plus 8 per table page, not the whole 64 KiB copy.
* **Crash safety (decision 6):** the intent header, recovery, and a full slot header on every add.
  One addition: after a failed fsync, every later commit fails until the index is reopened, since
  the OS may already have dropped the dirty pages a retry would claim to commit.
* **Versioning (decision 7)**, except that a newer write version is refused rather than opened
  read-only.
* **Migration (decision 8)** on first open, with the inode re-check on Unix. The Windows path
  (`MoveFileExW`) compiles but has not run.
* **Readers in other processes (decision 5):** `IndexReader` in Rust, `chassis_open_reader` in C and
  `read_only=True` in Python open the file read-only without a lock. Every search takes a snapshot
  as described above, copying header words atomically and re-reading a copy that fails its checksum
  while it keeps changing. The writer stores edges, delete marks, headers and the live page as
  relaxed atomic words with release fences before what publishes them, rewrites neighbor lists so
  surviving entries keep their slots, and publishes the routing count after each insert. One
  addition: the routing count can reach slots in a segment no header counts yet, so the live page
  also mirrors the writer's segment, heap chunk and table page counts and table page offsets.
  Readers treat a record at or past `N` that they can't decode as having no neighbors. A snapshot
  re-verifies the headers only when a copy's checksum or sequence word changed since both copies
  last read valid, and reads the file length only to map a new uncommitted region: about 60 ns when
  nothing changed, against about 1.2 µs (an `fstat`, two checksums, eight allocations).

The writer takes the single-byte lock at 2^62 on Windows (decision 5) and `flock` elsewhere.

Not built yet: the "superseded" flag and `st_nlink` reopen check (nothing replaces a v3 file by
rename yet), lock-free reads within one process (the bindings keep their lock), allocating space
before writing to it, an explicit `migrate()`, and `rebuild_graph()`.

Results of "Before Accepting", measured on the same Apple M5 as the
[Performance](../architecture/performance.md) page:

1. **Windows: passes on NTFS; exFAT and one migration step untested.** A probe on GitHub's
   `windows-latest` (branch `probe/windows-mmap`, 2026-10-06), each check run against another
   process:
   * (a) extending a file another process has mapped works, both with `SetEndOfFile` and with
     `WriteFile` past the end, and the other process can map the new range at a 64 KiB offset and
     read it;
   * (b) shrinking it fails with `ERROR_USER_MAPPED_FILE` (1224), as expected; v3 never shrinks;
   * (c) under a whole-range lock, another process's `ReadFile` fails (error 33); under a one-byte
     lock at 2^62 it works, and a whole-range lock request still conflicts with that byte, so
     releases up to 0.6.3 stay excluded;
   * (d) `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)` replaces a file nothing holds, and fails
     with access denied (5), leaving the original, while another process holds it open or mapped.
     `std::fs::rename` replaced it in every case, under open handles and mapped views alike, which
     is why migration doesn't use it.

   Not tested: exFAT, and renaming the migrated file while this process still holds it open and
   mapped, as migration does.
2. **Search speed: passes on paired measurements.** Today's SIFT-1M and dbpedia graphs, converted,
   return byte-identical results (ids and distances) for every query at every ef from 10 to 512.
   `search.rs` and `distance.rs` were unchanged when this was measured, so each layout ran the same
   loop. The machine was not quiet (load average 5 to 10 from other applications), and separate
   processes varied 2 to 5×. So one process opened all three files, each searched by its own library
   build, and alternated passes: 7 processes × 5 repetitions per ef, after a warm-up pass. Median
   v3/v2 QPS ratio per ef:

   | ef | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
   |----|----|----|----|----|-----|-----|-----|
   | SIFT-1M | 1.056 | 1.015 | 1.018 | 1.049 | 1.031 | 1.070 | 1.089 |
   | dbpedia | 1.087 | 1.095 | 1.097 | 1.052 | 1.099 | 1.109 | 1.117 |

SIFT at ef 512 counts only the 26 of 35 passes with no major page fault: v2's 3.4 GB file took
14,050 faults there and v3's 1. On the criterion as written, medians of each layout's own QPS, SIFT
at ef 32 came out at 0.950 (8,630 against 9,082), within that cell's noise: its paired ratio is
1.018, with an interquartile range of 0.946 to 1.131. A quiet machine should confirm it. The flat
control (one segment, no address arithmetic) puts segment addressing's cost at 2–5% on SIFT (v3/flat
0.950 to 0.981) and none measurable on dbpedia (0.985 to 1.077). Files: 713 MB for SIFT-1M (3,379 MB
in v2) and 821 MB for dbpedia (982 MB), as estimated above; allocated about 688 and 630 MB,
estimated from a fill without linking, against hnswlib's 661 and 623 MB. Migrating SIFT-1M took 9.4
s.

The reader work later changed the hot path: neighbor ids are loaded as relaxed atomics, record heads
with acquire ordering, and `search.rs` filters a reader's snapshot. Re-measured against the code
measured above, interleaving the layouts query by query and timing thread CPU time, on SIFT 300k and
dbpedia: the shipped code's paired ratio is 1.003 to 1.015 at every ef, with every 95% lower bound
at or above 0.997, and a relaxed head load is no faster. Against format 2 on the same SIFT 200k
graph, it is 1.06 to 1.085 (at least 1.051 at the 95% level). Reader searches, which take a snapshot
each time, run at 1.002 to 1.024 of the measured code.
3. **Readers tolerating a writer: passes for adds, deletes and flushes; incomplete, because
   `rebuild_graph()` is not built.** One writer and two reader processes on SIFT: 100,000 vectors,
   then 20 flushes of 1,000 adds and 50 deletes each, then 50,000 adds (42% of `N`) in one unflushed
   bulk load. The readers searched continuously, 200 queries at each ef in {16, 32, 64, 128}, and
   after every flush, with the writer idle, searched the same queries on the same snapshot; each
   query is paired with that idle run, and recall@10 is against the exact top 10 of the snapshot's
   live vectors. 587,200 searches, zero errors, zero returned ids that were not live in their
   snapshot, and no reader moving to an older snapshot. Mean paired loss in recall points (negative:
   better than idle) with 95% intervals from a bootstrap over queries, and pairs losing 0.5 or more:

   | ef | 16 | 32 | 64 | 128 |
   |----|----|----|----|-----|
   | Flushed batches | −0.04 [−0.10, +0.01], 0% | −0.02 [−0.05, 0.00], 0% | −0.01 [−0.02, 0.00], 0% | −0.01 [−0.01, 0.00], 0% |
   | Unflushed bulk load | −1.50 [−2.47, −0.56], 0.05% | −0.64 [−1.20, −0.09], 0% | +0.01 [−0.30, +0.34], 0% | −0.09 [−0.24, +0.02], 0% |

Two more runs gave −1.63 and −1.47 at ef 16 during the bulk load, and the batch row within 0.01.
Readers looked for the idle window only between passes of 800 searches, so 7% of the batch pairs
(17% at ef 128) ran with the writer already idle, where the loss is exactly zero; leaving them out
changes no cell by more than 0.01.

The bulk load's gain is not from concurrency. Its idle run came before the bulk load, and the
concurrent searches routed through up to 50,000 unflushed nodes they can't return; the filtered
search keeps expanding such nodes until it holds ef returnable results, so these searches computed
12–17% more distances. With the writer paused, without flushing, after every 10,000 bulk adds, idle
searches on the same snapshot gained 0.85 to 2.35 points at ef 16 (454 to 568 distances per query);
against those paused runs, concurrent searches lost −0.50 to +0.44 points, every interval below
+1.0. So writes in progress cost no measurable recall, but a criterion measured against the pre-bulk
idle run can't see up to about 2 points of such loss: this experiment should judge the bulk load
against paused runs.

With routing disabled in the readers, the bulk load lost 1.15 [0.37, 1.98], 1.10 [0.57, 1.68], 0.64
and 0.11 points, and 1.49, 1.18, 0.63 and 0.11 in a second run: failing at ef 16 and 32 both times,
but only by 0.1 to 0.5 points, with intervals reaching below 1. This bulk load is too small to
reproduce the 16 to 21 points decision 5 measured with a backlog as large as `N`; the control should
use a larger one.

The run had no writer restarts. A separate in-process stress test with the unit-test geometry (4 to
6 readers, a writer that adds, deletes and drops itself unflushed in up to 80% of batches, about
12.6 million searches, 1,970 restarts, files up to 135 segments and 34 table pages) found no wrong
result. It did find two interleavings, both deterministic, in which a search spanning a writer
restart failed on a record the new writer was reusing; readers now treat records at or past `N` as
hints (decision 5), with a regression test. Not covered: `rebuild_graph()`, a writer killed and
restarted with readers in other processes, a flush with deletes and no adds, a table page boundary
in another process, more than two reader processes, Linux and Windows.

Test coverage: the reader tests catch disabled routing, disabled snapshot filtering, search without
a new snapshot and `chassis_len` without one. Monotonic snapshots, header re-reads, remapping,
region publishing, the routing reset on open, slot-preserving rewrites, ignoring an uncommitted
delete's marks and publishing routing only after backlinks each have a targeted unit test. Memory
ordering (release and acquire on the routing count, the live page counts and record heads) is
covered by review only: neither the tests nor a litmus test on this machine can tell it from relaxed
ordering.
4. **Power loss:** `chassis-core/src/power_loss.rs` passes about 200 crash points: 50 in the
   workload (two between operations and three fsyncs per flush, ten flushes) plus about 150 inside
   the recovery a crash image triggers, about 510 crash images in all. Node levels come from an
   unseeded RNG, so the recovery count varies: runs on 2026-10-05 and 06 gave 198 to 214 crash
   points and 496 to 528 images. Unchanged sectors stay as they are; each changed one keeps its
   durable or its current contents, or is torn into a mix of their 8-byte words. Unit tests use
   16-slot segments, 1 KiB heap chunks and 4-entry table pages, so the workload crosses all three,
   and every flush deletes a vector added in that same flush. Each image must reopen to the last
   flush or the one in progress, then take one more add-and-delete flush and reopen to exactly that
   state. It catches each of eight protocol mutants, re-run against the reader-era storage code with
   3 runs each, every one failing an invariant at run time: no fsync before the commit header, no
   intent header, no fsync between the intent and the marks, header copies that don't alternate,
   recovery disabled, recovery that keeps the marks, an add that doesn't clear the slot header, and
   a checksum that isn't checked. The last two needed torn sectors: a header fits in one sector, so
   whole-sector choices never tear it.

(b): before any writer reopens it, a reader opens every crash image as found. It must show an
accepted state, never fail a search, and return only live ids of that state: in two runs it missed
the top hit for 0 and 144 of about 50,000 searches, from routing through slots the power loss
emptied. Before readers treated records past `N` as hints, about 7% of such opens failed a search
(`Invalid graph record`), because the crashed writer's live page reached the disk and records it
routes to did not. Creating a file is covered too: 100 crashes of its one fsync, which before the
fix left an unopenable file whenever both header copies tore. Not covered: orderings only readers
see within one sync interval (record head after its lists, routing after nodes, slot-preserving
rewrites), whose mutants pass the simulator; the unit tests above cover them.
5. **Id table build time: met with the page cache warm, barely, and only with a faster build.** The
   first `add_with_id` at 10M slots builds the table; the add itself is under 0.5 ms of it. Reading
   the 240 MB of slot headers takes 84 to 224 ms of CPU, depending on whether the pages are mapped,
   cached or on disk; filling the hash table is about 93% of the build, from its cache misses rather
   than hashing. Collecting the pairs first and filling a table keyed by a seeded multiply hash cut
   the build to 0.67× in paired runs (0.57 to 0.75×); either change alone did nothing. With that, at
   load average 6 to 15: 351 to 885 ms with the file in the page cache (median 468 ms, under 1
   second in 5 of 5), and 419 to 1,316 ms right after copying the file, which on macOS leaves it out
   of the page cache (median 913 ms, 3 of 5). Earlier runs of the original build, reported as warm,
   were in fact cold (742 to 1,039 ms). Under heavy load (load average 34 to 59), the original build
   took 1.1 to 1.8 s of CPU.

## Before Accepting

Each of these is an experiment with a clear pass criterion:

1. **Windows.** Windows refuses to shrink a file below a mapped view (`ERROR_USER_MAPPED_FILE`).
   Extending is expected to work: Microsoft's FastFAT source only checks on shrink, and SQLite's WAL
   index grows with `SetEndOfFile` under other processes' views. Test: (a) extend with
   `SetEndOfFile` and with `WriteFile` past the end while another process has views, then map the
   new range with a new section at a 64 KiB offset; (b) shrinking fails; (c) a `ReadFile` reader
   works under the single-byte writer lock; (d) migration step 4 on NTFS and exFAT, both with the
   migrator's own handles closed and with a third process holding the original.
2. **Search speed of the new layout.** Convert today's SIFT-1M and dbpedia graphs into a v3
   prototype (u32 records, segment addressing, upper heap) instead of rebuilding them, and confirm
   both files return identical result lists for every query, so recall is equal by construction.
   Time single-thread QPS for ef 10–512 with: one open per file for the whole run and an untimed
   warm-up pass after every open (a fresh mapping pays page faults on its first pass: measured at
   about half the warm rate on SIFT-1M at ef 16); no major faults during timing (wire the mappings
   or check rusage); at least 7 rounds alternating the files; the same search loop for both layouts
   (today's search code ran 0.86–0.96× a lean rewrite on the same bytes, so a new prototype would
   look faster for code reasons); and a flat u32 array control, so segment addressing's cost is
   reported apart from record size. Pass: v3 median QPS at least 0.97× today's at every ef. Report
   cold-open (first pass after open) QPS separately. Don't compare separately built graphs near the
   top of the recall curve, where build-to-build variance (0.996 against 0.998 at ef 512 on SIFT-1M)
   dominates. File size: report logical length and allocated blocks; pass if allocated bytes are
   within 10% of hnswlib's file and the logical length within 10% plus one segment.
3. **Readers tolerating a writer.** One writer and at least two reader processes, searching
   continuously while the writer adds (both in small flushed batches and as one bulk load of `N` new
   vectors with no flush), deletes, flushes and runs `rebuild_graph()`. Every query records its
   snapshot `(N, E)` and is compared with the same query on the same snapshot while the writer is
   idle, not with the final index: a measured run passed against the final index (0.9609 against
   0.9614) while the paired loss was 1.12 points. Pass: zero reader errors (invalid record, an id at
   or past `N` or deleted in the snapshot returned, panic, signal); at every ef in {16, 32, 64,
   128}, mean paired recall@10 loss of at most 1 point and at most 1% of queries losing 0.5 or more.
4. **Header commit under power loss.** A power-loss simulator that drops, reorders and tears
   unsynced writes, run against the real implementation of decisions 4, 6 and 8
   (`chassis-core/src/power_loss.rs`; results under "Implementation Status"). Crash points include
   inside recovery, a delete of a vector added in the same flush, and a flush that appends a
   segment. Pass: (a) every reopen shows the last acknowledged flush or the one in progress, never
   older; (b) a reader opening the crashed file without recovery sees that same state; (c) after
   every reopen, one more delete flush and a clean reopen show exactly that state plus the flush. A
   model checker (2026-10-04) showed that "every reopen shows exactly one committed state" alone
   accepts 6 of 8 broken protocols.
5. **Id table build time.** Building the id table at 10M slots stays under 1 second with the page
   cache warm.

## Consequences

### Positive

* Files 5× smaller on SIFT-1M and 1.6× on dbpedia (graph records 13× smaller), with bytes written
  within 3% of hnswlib's.
* No data moves and no remaps, which removes a whole class of crash and concurrency bugs and the
  code that handles graph relocation.
* Many readers across processes, with lock-free reads.
* Opening reads two header copies and maps one region per segment and chunk on first use: 11 for
  SIFT-1M, about 2,350 for 100M vectors at 1536 dimensions. Nothing is loaded up front, except that
  each process looking up custom ids builds its own id table (a 287 MB table at 10M ids, measured as
  peak memory footprint; peak resident memory, 527 MB, also counts the 240 MB of slot headers mapped
  from the page cache).
* Changing index parameters becomes a rebuild into a new file instead of an error.
* A format that can be frozen, with a SQLite-style compatibility promise.

### Negative

* One more breaking change, with a migration step for existing files.
* At most about 4.29 billion vectors per index (u32 neighbor ids).
* Segment addressing adds a lookup to every vector and record access: a prototype measured 1–9%
  lower QPS than a flat array on SIFT-1M.
* File size counts up to one segment (147–212 MB at 100–1,536 dimensions, never above 256 MiB) plus
  one chunk per heap that holds nothing yet. While segments are still doubling, file size is up to
  twice the bytes written (10k × 768: 50 MB against 32 MB). Disk use only grows in the writer's 16
  MiB allocation steps on APFS and ext4; NTFS allocates the whole extension, and tools that don't
  keep holes copy it in full.
* Readers never get wrong results, but while the writer adds they lose a little recall; the routing
  count keeps that loss small, and experiment 3 bounds it.
* `rebuild_graph()`, changing `max_connections` and reclaiming deleted space each need free disk for
  a second copy of the index while they run.
* Windows multi-process support depends on experiment 1.
* Releases up to 0.6.3 ignore the inode re-check, so one opening the path during a migration can
  lock and write to the replaced file, and that write is lost.
