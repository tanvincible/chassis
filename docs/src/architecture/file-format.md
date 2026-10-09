# File Format

A Chassis file is one file in format version 3 ([ADR-0008](../adr/008-format-v3-and-multi-process-readers.md)),
or version 4 if it keeps its vectors in half precision ([ADR-0018](../adr/018-half-precision.md)).
It holds two header copies, then append-only **regions**: segments of slots, heap chunks of
upper-layer neighbor lists, and table pages that list where the others start. Each region is
mapped once, and committed bytes never move.

Opening a version 1 or 2 file migrates it to version 3 (see [Migration](#migration)).

## Top-Level Layout

| Offset | Size | Contents |
|--------|------|----------|
| `0` | 64 KiB | Header copy A |
| `64 KiB` | 64 KiB | Header copy B |
| `128 KiB` | 64 KiB | Live page, for readers in other processes |
| `192 KiB` | rest | Regions, in the order they were allocated |

The header copies sit 64 KiB apart so they never share a page. Every region starts at a multiple of
64 KiB, the Windows mapping granularity and a multiple of every supported page size, so each one
can be mapped on its own. The file only grows by appending a region, and never shrinks.

## Header

Each copy is a header of `144 + 8 × (segment table pages + heap table pages)` bytes, little-endian.
A later write version may add fields after these; the length covers them.
The valid copy with the higher sequence number is current.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | Magic | `CHASSIS\0` in every format version |
| 8 | 4 | Read version | Oldest format a reader must understand: `3`, or `4` in half precision |
| 12 | 4 | Write version | Oldest format a writer must understand: `3`, or `4` in half precision |
| 16 | 8 | Checksum | xxh3-64 of the header's bytes with this field zeroed |
| 24 | 8 | Sequence | Increases by one with every header written |
| 32 | 4 | Length | Header length in bytes, table page lists included |
| 36 | 4 | Dimensions | Components per vector, 1 to 4096 |
| 40 | 2 | M | Neighbors per upper-layer list |
| 42 | 2 | M0 | Neighbors per layer-0 list |
| 44 | 1 | Max layers | Layers a node can belong to |
| 45 | 1 | Metric | `0`: Euclidean; `1`: cosine, over vectors stored at unit length |
| 46 | 1 | Flags | Bit 0: some id differs from its slot. Bit 1: superseded, a compacted copy is replacing this file at its path ([ADR-0011](../adr/011-compaction.md)) |
| 47 | 1 | Table page log2 | Entries per table page, as a power of two |
| 48 | 1 | Segment base log2 | Slots in the first segment, as a power of two |
| 49 | 1 | Doubling segments | Segments that double in size before the size stays constant |
| 50 | 1 | Heap base log2 | Bytes in the first heap chunk, as a power of two |
| 51 | 1 | Doubling chunks | Heap chunks that double in size before the size stays constant |
| 52 | 1 | Precision | `0`: each component is an `f32`; `1`: an IEEE 754 16-bit float, in a file of version 4 |
| 56 | 8 | Count | Committed slots |
| 64 | 8 | Entry point | Slot of the graph's entry point, `u64::MAX` if empty |
| 72 | 4 | Max layer | Highest layer in the graph |
| 80 | 8 | Epoch | Last flush that committed deletes |
| 88 | 8 | Pending epoch | Nonzero while a flush with deletes is in progress |
| 96 | 8 | Deleted count | Deleted slots |
| 104 | 8 | File end | End of the last committed region |
| 112 | 4 | Segments | Committed segments |
| 116 | 4 | Heap chunks | Committed heap chunks |
| 120 | 8 | Heap used | Next free heap position, `chunk << 32 \| byte offset` |
| 128 | 4 | Segment table pages | Number of segment table pages |
| 132 | 4 | Heap table pages | Number of heap table pages |
| 136 | 8 each | Table page offsets | Segment table pages, then heap table pages |
| after the tables | 8 | Next id | One past the largest id used before the last compaction, which removed the deleted slots that showed it; `0` if never compacted. Absent from headers written before it existed, which read as `0` |

A release checks the read version before the checksum, so a newer layout is reported as too new,
not as corrupt. A newer write version only stops writers: `IndexReader` reads the file and ignores
the bytes after the next id. Releases up to 0.6.3 read bytes 8–11 as their version and refuse
anything above 2.

A file in half precision is version 4 to readers and writers alike, so that a release from before
half precision refuses it and doesn't read its vectors as `f32`s. A file in full precision is
written as version 3 by every release, with byte 52 zero as it always was.

## Live Page

What the writer has added since its last commit, so readers in other processes can route through
it. Every field is a little-endian u64. The counts and the routing count are written with release
ordering after the data, offsets and table entries they cover. It is not durable: a writer resets
it to its committed state when it opens the file, and a reader that opens the file before any
writer, after a crash, uses it as found, treating records it can't read past the committed count as
having no neighbors.

| Offset in page | Field |
|----------------|-------|
| 0 | Routing count: slots whose nodes and backlinks are written |
| 8, 16, 24, 32 | Segments, heap chunks, segment table pages, heap table pages |
| 64 | Segment table page offsets (512 entries) |
| 4,160 | Heap table page offsets (512 entries) |

## Segments

Segment `k` holds `2^(base + min(k, K))` slots, where `base` is the segment base log2 and `K` the
number of doubling segments. A new file uses `base = 10` and the largest `K` whose segment fits in
256 MiB, so every segment after the first `K` has the same `C = 2^(base + K)` slots. With
`B = C − 2^base`, the slots in all doubling segments together:

- slot `s < B` is in segment `⌊log2(s + 2^base)⌋ − base`, at index `s + 2^base − 2^⌊log2(s + 2^base)⌋`;
- slot `s ≥ B` is in segment `K + ((s − B) >> log2 C)`, at index `(s − B) & (C − 1)`.

A segment of `n` slots holds three arrays, each starting at a multiple of 64 bytes:

| Array | Per slot | Contents |
|-------|----------|----------|
| Slot headers | 24 bytes | id (u64), metadata reference (u64, `0`), deleted epoch (u64, `0` while live) |
| Vectors | `dims × 4` bytes, or `dims × 2` in half precision | the vector, little-endian |
| Level-0 records | `8 + 4 × M0` bytes | head (u64), then `M0` neighbor slots (u32) |

A record's head holds the node's layer count in bits 0–7 and, for a node above layer 0, the heap
position of its upper-layer lists in bits 8–63. Empty neighbor entries are `0xFFFFFFFF`. A slot is
deleted once its deleted epoch is nonzero; only a flush writes it.

## Upper Heap

Heap chunk `c` is `2^(heap base + min(c, doubling chunks))` bytes; a new file uses 64 KiB chunks
that double up to 256 MiB. A node on `L` layers has one heap entry of `L − 1` lists of `M` neighbor
slots (u32), for layers 1 to `L − 1`. An entry never straddles two chunks.

## Table Pages

A table page holds `2^(table page log2)` file offsets (u64), 8,192 in a new file, so a page is
64 KiB. Entry `i` of the segment table is in page `i >> log2`, at index `i & (2^log2 − 1)`; the heap
table works the same way. The header lists up to 512 pages per table, enough for 2^32 slots.

## Commits and Recovery

A flush writes data in place, fsyncs, then writes the header copy that does not hold the newest
header and fsyncs again. Slots, regions and table entries past the committed counts are leftovers
of a flush that never committed: the next writer reuses them, and until then readers may route
through them but never return them.

A flush with deletes commits through an *intent* header first: the previous commit with
`pending epoch = epoch + 1`, then the delete marks, then the commit header in the other copy, with
an fsync after each step. Opening a file whose newest header has a pending epoch clears every
deleted epoch above the header's epoch in committed slots, then writes a clean header. Process
kills are tested in `chassis-core/tests/crash_tests.rs`; power loss is simulated in
`chassis-core/src/power_loss.rs`, including crashes inside recovery.

## Size Example

10,000 vectors with 768 dimensions and default parameters take 3,232 bytes per slot (24-byte slot
header, 3,072-byte vector, 136-byte level-0 record) plus about 4 bytes of upper heap. The first four
segments hold 1,024, 2,048, 4,096 and 8,192 slots, so the file is about 50 MB, of which 32 MB is
written. The unused end of the last segment is never written, but whether the file system keeps it
sparse varies: on APFS, a real build of this example allocated 33.6 MB, while unused ranges of
16 MiB or less next to written data were sometimes allocated in full. ext4 is untested.

## The undo file

Adds change the neighbor lists of committed nodes in place before they are committed themselves.
Before a committed node's lists first change after a flush, they are appended to `<name>.undo`,
and a writer that opens the index after a crash writes them back (ADR-0012). The file exists
only between a flush and the next one that follows adds, or after a crash; a flush removes it.
An index moved or copied while the file exists has to take it along.

After the 8 bytes `CHSUNDO1`, each entry is the slot (u32), its layer count (u8), then per layer
a count (u16) and that many neighbor ids (u32), then an xxh3-64 of the entry, seeded with a hash
of the committed vector count, entry point, top layer, delete epoch, deleted count and id mark.
Entries saved from another commit fail the checksum and are ignored.

## Migration

Opening a version 1 or 2 file converts it. Chassis takes the original's lock, writes the committed
state into `<name>.migrating`: slots below the graph's node count, and only the delete marks the
graph header committed. It converts neighbor ids from u64 to u32 and moves upper layers into the
heap, then fsyncs the new file and renames it over the original, and fsyncs the directory. The
graph is converted, not rebuilt, so search results are unchanged. A failed or interrupted migration
leaves the original in place.

## Validation

On open, Chassis checks that:

- one header copy has the magic, versions it understands and a valid checksum; two valid copies
  with the same sequence number are corruption;
- the dimensions match, and the graph parameters and geometry are possible;
- the counts fit the tables, and every committed region lies inside the file;
- `VectorIndex::open` only: the graph parameters match the requested `max_connections`, and a file
  with vectors has a graph.

A failed check returns an error and never modifies the file.

## Stability

Format version 3 is the format ADR-0008 proposes to freeze. Until that ADR is accepted, it may
still change.
