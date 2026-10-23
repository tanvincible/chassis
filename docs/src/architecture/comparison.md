# Comparison

How Chassis compares with other ways to search vectors on one machine: what each one is, and how
each measured on the same vectors, on the same machines, on the same day. Pick by shape first;
the numbers matter only among tools that fit.

## Engines

| Engine | What it is | What it keeps |
| --- | --- | --- |
| **Chassis** | A library: an HNSW index in one file, read where it lies | Vectors under your integer ids; one writer, readers in other processes |
| **hnswlib** | A library: an HNSW index in memory, saved and loaded whole | Vectors under integer labels |
| **FAISS** | A library of in-memory indexes; HNSW here | Vectors; ids with an id map |
| **usearch** | A library: an HNSW index, in memory or viewed from its file | Vectors under integer keys |
| **Annoy** | A library: random-projection trees, built once, then read-only | Vectors under integer ids |
| **sqlite-vec** | A SQLite extension: vectors in SQL tables, searched by scanning them all (as of 0.1.9) | Vectors beside your rows, in your SQLite file |
| **LanceDB** | An embedded database of columns and vectors | Vectors with their columns, filters in the query, versions of the data |

## Measured

On GitHub's runners, 2026-10-10. Each engine through its Python package: built on every core
with `M` 16 and `ef_construction` 200 where it has such settings, saved to disk, opened, and
searched on one thread for the 10 nearest. Queries a second is the middle of three passes over the
queries; the recall beside it is recall@10 against the dataset's true neighbors, at the first
search setting that reached about 0.96 and 0.99. **First result, cold**: a fresh process opens the
file after it was put out of memory, and searches once.

Chassis was called once per query in this run, which costs it 15 to 38 µs a query in Python; the
other engines were given all their queries at once. A run of 0.10.0 on 2026-10-19, with every
engine called the same way, is on [Performance](./performance.md).

**SIFT**, 1M vectors of 128 dimensions, on an AMD EPYC 7763:

| Engine | Queries a second (recall) | Queries a second (recall) | Build | File | First result, cold |
| --- | --- | --- | --- | --- | --- |
| Chassis | 6,959 (0.967) | 3,619 (0.992) | 119 s | 713 MB | 317 ms |
| Chassis, half precision | 8,265 (0.967) | 5,835 (0.992) | 89 s | 444 MB | 273 ms |
| FAISS (HNSW) | 7,941 (0.964) | 4,270 (0.991) | 137 s | 656 MB | 1,469 ms |
| hnswlib | 7,338 (0.960) | 2,385 (0.997) | 136 s | 661 MB | 2,159 ms |
| usearch | 3,817 (0.956) | 896 (0.997) | 201 s | 661 MB | 1,417 ms |
| LanceDB | 323 (0.957) | 229 (0.997, re-ranked) | 114 s | 828 MB | 1,380 ms |
| Annoy | 331 (0.988) | 92 (0.999) | 22 s | 904 MB | 850 ms |
| sqlite-vec | 5 (0.999, every vector) | | 9 s | 536 MB | 1,003 ms |

**dbpedia**, 99,000 vectors of 1,536 dimensions (OpenAI embeddings), on an AMD EPYC 9V74:

| Engine | Queries a second (recall) | Queries a second (recall) | Build | File | First result, cold |
| --- | --- | --- | --- | --- | --- |
| Chassis | 1,214 (0.979) | 693 (0.993) | 92 s | 821 MB | 512 ms |
| Chassis, half precision | 1,929 (0.979) | 1,124 (0.993) | 61 s | 421 MB | 252 ms |
| FAISS (HNSW) | 2,565 (0.975) | 1,415 (0.991) | 70 s | 623 MB | 3,308 ms |
| hnswlib | 1,926 (0.967) | 614 (0.996) | 58 s | 623 MB | 2,067 ms |
| usearch | 1,224 (0.967) | 384 (0.996) | 72 s | 623 MB | 1,312 ms |
| LanceDB | 383 (0.963) | 224 (0.999, re-ranked) | 33 s | 777 MB | 1,139 ms |
| Annoy | 142 (0.962) | | 14 s | 647 MB | 1,362 ms |
| sqlite-vec | 4 (1.000, every vector) | | 37 s | 613 MB | 2,617 ms |

What these show:

- **Search speed among the HNSW libraries.** Here, on SIFT, Chassis searched at 0.85 to 0.9 times
  FAISS's speed in full precision and 1.04 to 1.37 times in half precision. On dbpedia, FAISS was
  about twice as fast as Chassis in full precision, and 1.3 times as fast in half. In the run of
  0.10.0, with every engine called alike, Chassis ran at 1.00 to 2.05 times FAISS's speed on SIFT,
  and on dbpedia at 0.86 to 0.99 times in full precision, 0.97 to 1.11 with `warm`, and 1.11 to
  1.26 in half ([Performance](./performance.md)).
- **First result from a cold file.** Chassis took 0.15 to 0.45 of the time the others did, since it
  reads only the pages a search touches instead of loading the index first.
- **File size.** Segments double up to 256 MiB, and the length counted here includes the unused
  end of the last one, which takes no disk on ext4, XFS and btrfs past the 1 MiB allocated ahead,
  nor on APFS from 16 MiB
  ([Performance](./performance.md#size)). On disk on APFS, Chassis's files took 4% more than
  hnswlib's on SIFT and 1% more on dbpedia. Half precision halves the vectors.
- **LanceDB and sqlite-vec are databases**, not just indexes: they keep your columns and filter in
  their queries, and here they searched one to three orders of magnitude more slowly. sqlite-vec
  checks every vector, so its results are exact.

## Choosing

- **Chassis** fits an app that embeds its own data and searches it on the same machine: one file,
  in your process, with your rows in your own store under the same ids
  ([With a database](../guide/with-a-database.md)).
- **sqlite-vec**, when the data is in SQLite already and the search should be part of SQL queries,
  over few enough vectors that a scan is fast enough.
- **LanceDB**, when you want one store for vectors and their columns, with filters in the query and
  versions of the data.
- **FAISS, hnswlib or usearch**, when the index fits in memory, is built in one go, and you handle
  saving it and keeping it in step with your data yourself.
- **A server** such as pgvector or Qdrant, when many machines write to one index, or other services
  query it over the network.

zvec was not measured. The harness is `bench/lab/incumbents.py` on the
[`lab/x86`](https://github.com/tanvincible/chassis/tree/lab/x86/bench/lab) branch, and every row it
wrote for this page is in `bench/results/engines-2026-10-10.csv`.
