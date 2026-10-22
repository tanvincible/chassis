# Limits

What one index can hold, and what it needs from the machine to hold it well. Going past a hard
limit is an error that names it ([Errors](./errors.md)), never a wrong answer.

## Poor fits

Some limits come with the design, and no option changes them. Reach for something else for:

- **An index much larger than memory.** Searches wait on the disk, over a thousand times slower
  ([Memory](#memory)). An index built for the disk, such as DiskANN, fits that better.
- **Many writers.** One process writes a file at a time; a server such as Qdrant or pgvector takes
  writes from many.
- **Heavy churn.** Deleted vectors slow searches until `compact()`, which builds the graph again:
  to about 70% of their speed with half the vectors deleted, and a fifth with 90%.
- **Metadata in the index.** Chassis keeps vectors and ids. Keep the rest in SQLite or a store of
  your own under the same ids, and pass the ids it selects to a filtered search.
- **Inner product.** There is Euclidean and cosine distance; for vectors of length 1, cosine
  orders results as inner product does, so only vectors of other lengths miss it.
- **Storage that fails reads.** The file is mapped into memory, so a read the disk fails ends the
  process (SIGBUS on Linux and macOS) instead of returning an error, as with every store that maps
  its file.

## Hard limits

| What | Limit |
| --- | --- |
| Vectors | 4,294,967,295 in a file, counting deleted ones until `compact()` |
| Ids | 0 to 2⁶⁴ − 2; 2⁶⁴ − 1 is reserved |
| Dimensions | 1 to 4,096 |
| `max_connections` | 2 to 32,767; 16 by default |
| Components | finite numbers; in half precision, under 65,520 in magnitude |

The file system's own limits apply too: FAT32 holds files up to 4 GiB. A full disk is an error,
except on Btrfs: it copies a page on every write, so the space Chassis allocates ahead doesn't
cover it, and a disk that fills as Chassis writes ends the process (SIGBUS)
([Verification](../architecture/verification.md#limits)).

## Size

Each vector takes its components, 4 bytes each in full precision or 2 in half, and about 164 bytes
for its slot header and links at the default `max_connections`; each step up adds about 8
([Size](../architecture/performance.md#size)).

| Dimensions | Full precision | Half precision | A million, full | A million, half |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 676 B | 420 B | 0.68 GB | 0.42 GB |
| 384 | 1,700 B | 932 B | 1.7 GB | 0.93 GB |
| 768 | 3,236 B | 1,700 B | 3.2 GB | 1.7 GB |
| 1,536 | 6,308 B | 3,236 B | 6.3 GB | 3.2 GB |
| 3,072 | 12,452 B | 6,308 B | 12.5 GB | 6.3 GB |

Measured on disk, 10 million vectors of 128 dimensions in full precision took 676.3 bytes each,
and 3 million of 1,536 in half precision 3,236.3. The file's length runs up to a segment, at most
256 MiB, ahead of what's written; where the file system keeps holes, that end takes no disk.

## Memory

Chassis reads the file through the operating system's cache, so an index searches at full speed
while the file fits in the memory the process can have. Past that, searches wait on the disk. Ten
million vectors of 128 dimensions, a 6.9 GB file, on a GitHub runner whose disk reads 4 KB at
random in 0.19 ms ([Scale](../architecture/performance.md#scale)):

| Where the file was | Queries a second, `ef` 64 |
| --- | --- |
| In memory | about 9,000 |
| The first 1,000 queries from a cold cache, with room to keep the file | 55 |
| In a 2 GB cgroup, a third of the file | 2.5 to 5 |

`warm()` reads an index in ahead of the first pass, when it fits; when it doesn't, it can't help.
Adds in the 2 GB went at 2 a second, so an index much larger than its memory can't be built or
compacted in useful time. Keep an index within the memory it has, or shrink it with half
precision. A faster disk helps by an amount not yet measured.

## Compaction

`compact()` builds a new file beside the index and renames it over the old one. It needs:

- **Disk** for a second file holding the live vectors, until the rename.
- **Time** about a build's, as it builds the graph again: 1,185 s for 10 million vectors of 128
  dimensions, where the build took 1,145 s; 789 s for 3 million of 1,536 in half precision, where
  it took 988 s.
- **Memory** for about one file, not two: it reads the old one in order, a piece at a time. In a
  cgroup of 1.2 times the file it ran at that speed; less wasn't measured.

## Crashes

The first writer to open an index after a crash reads its undo file whole, to put back the links
the unflushed adds changed: at most about 150 bytes for each flushed vector, at the default
`max_connections`. Killed partway through adding a million vectors to ten million, it left 203
MB, and the next writer opened in 8.4 s.
