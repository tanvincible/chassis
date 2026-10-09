# Roadmap

What is planned for Chassis and not yet built. Chassis is for local semantic search in one file;
the [non-goals](README.md#non-goals) say what it will not grow into.

Only what has been measured and found worth building is listed here. Ideas that have not been
validated are kept in `IDEAS.md` on the [`ideas`](https://github.com/tanvincible/chassis/blob/ideas/IDEAS.md)
branch, and move here when they have been.

## Planned

### `warm()`: read an index into memory in the background

An index opens in about a millisecond and answers its first query before hnswlib has finished
loading. But while its file is not yet in memory, every page a search touches is a separate
disk read, one at a time, and the first hundred queries or so are slow.

`warm()` will read the whole index in on another thread, at the disk's sequential speed, while
queries are already being answered, from Rust, C and Python. It will be something to ask for: an
index much larger than memory should not be read in whole.

Measured on 2026-10-09 with a prototype, from the start of a process, on 99,000 vectors of 1,536
dimensions whose file was not in memory, on an Apple M5:

| | First result | 100 results | 1,000 results |
| --- | --- | --- | --- |
| Today | 110 ms | 1,810 ms | 2,305 ms |
| With the background read | 103 ms | 324 ms | 782 ms |
| hnswlib | 794 ms | 982 ms | 2,795 ms |

On four Linux machines with slower disks, 100 results took 1.30 to 1.34 s with it, where they
took 1.46 to 2.38 s without and hnswlib took 1.35 to 1.45 s.
