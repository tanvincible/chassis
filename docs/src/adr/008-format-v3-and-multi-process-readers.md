# ADR-0008: File Format v3 and Multi-Process Readers

**Date:** 2026-10-04 **Status:** Proposed. Accept only after the experiments in "Before Accepting"
pass.

## Summary

Chassis aims to be to vector search what SQLite is to relational data: one embedded file that any
process can open, that survives crashes, and that stays readable for decades. Two things block that
today: one process at a time, and a file format that must change again before it can be promised
stable. This ADR proposes one last breaking change, file format v3, designed so that:

1. **Committed data never moves or changes.** The file grows only by appending segments, so graph
   relocation and the pointer invalidation that comes with remapping disappear. Operations that
   replace the graph wholesale (`rebuild_graph()`, a new `max_connections`) or reclaim space
   (vacuum, migration) write a new file and swap it in (decision 8). Apart from single-word edge
   updates (decision 5), nothing rewrites bytes a reader may be traversing.
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
2. maps any segment or heap chunk it hasn't mapped yet (existing mappings never change, because
   nothing moves);
3. searches, treating a slot as present only if it is below `N` and its deleted epoch is 0 or above
   `E`.

The writer guarantees only these things:

* **Committed data never changes.** Vectors, ids and metadata references are immutable once
  committed, so metadata changes by delete and re-add, never by rewriting a reference. The writer
  reuses a slot only if it is beyond every committed count (ghost slots after a crash). Readers
  never see those, except that after a power loss a reader may already have returned slots from the
  lost commit.
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
  the committed count before reusing any ghost slot. Auto-flushing every few hundred adds would also
  bound the loss, but it would make a caller's batch durable in pieces, which breaks ADR-0007's
  all-or-nothing flush.
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
simpler placement of the flag failed it.

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

Releases up to 0.6.3 initialize any file shorter than 4,096 bytes, so v3 creates a new file under a
temporary name and renames it into place once its first header is durable.

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
   unsynced writes, run against the real implementation of decisions 4, 6 and 8. One exists for
   today's format (`chassis-core/src/power_loss.rs`, see the ADR-0005 amendment); v3 extends it.
   Crash points include inside recovery, a delete of a vector added in the same flush, and a flush
   that appends a segment. Pass: (a) every reopen shows the last acknowledged flush or the one in
   progress, never older; (b) a reader opening the crashed file without recovery sees that same
   state; (c) after every reopen, one more delete flush and a clean reopen show exactly that state
   plus the flush. A model checker (2026-10-04) showed that "every reopen shows exactly one
   committed state" alone accepts 6 of 8 broken protocols.
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
  each process looking up custom ids builds its own id table (553 MB peak memory at 10M ids,
  measured).
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
