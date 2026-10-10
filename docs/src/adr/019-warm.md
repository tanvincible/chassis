# ADR-0019: Reading an Index into Memory in the Background, on Request

**Date:** 2026-10-10
**Status:** Proposed

## Context

An index opens in about a millisecond, because opening maps the file and reads only its headers.
While the file is not in memory, as after a reboot, every page a search touches is read from the
disk when the search reaches it. A search waits for each of those reads before it knows what to
read next, so they come one at a time. On an Apple M5 the SSD serves 10,700 such reads a second
one at a time and 114,000 with 32 in flight; macOS reads a 16 KB page per fault and Linux 128 KB.
On 99,000 vectors of 1,536 dimensions the first result came 110 ms after the process started,
and the hundredth 1.8 s after: slower than hnswlib, which reads its whole file in before it
answers anything (794 ms for the first result, 982 ms for the hundredth).

A prototype on the experiment branch read the file in on another thread while searches went on,
and the hundredth result came after 324 ms (ROADMAP.md, 2026-10-09).

## Decision

### 1. An option and a method

`IndexOptions::warm`, off by default, reads the index in from the moment it is opened.
`VectorIndex::warm` and `IndexReader::warm` ask for the same later; in C, `chassis_warm`; in
Python, `IndexOptions(warm=True)` and `VectorIndex.warm()`. Asking turns the option on, so a reader
that opens the file again after a compaction reads the new one in too.

### 2. A thread that reads what was written

The thread is given the file ranges the index has written: each segment's slot headers, vectors
and level-0 lists up to the last slot, the heap chunks before the one in use, and that one up to
its last entry. What lies past them was never written, and in a sparse file is a hole. On the
99,000-vector file that is 625 MB of 821 MB. The ranges are read in file order. While a reading is
under way, asking again does nothing; once it is done, asking again reads what the index holds
then.

### 3. Through the index's own mappings, except a writer's off Linux

The thread gives the system advice about the index's own mappings, which then have the pages
mapped as well as in memory: on a file already in memory that alone made the first hundred
searches three times as fast on the M5. A reader's mappings are read-only. A writer's are
writable, and on macOS advice to read pages in through a writable shared mapping marks every page
it reads as changed, so the system writes them all back: 309 MB written for 320 MB read, on the
half-precision file. So a writer off Linux, where only Linux promises to leave such pages clean,
maps the file read-only for the thread and gives the advice about that view instead.

A reader unmaps a region when a restarted writer has laid the file out differently. It stops the
thread first: advice about an address that is no longer mapped is an error, but about one that
has been mapped again it acts on whatever is there now, and on macOS read untouched anonymous
memory in, 64 MB of it in a test. Dropping or replacing an index stops the thread before its
mappings go.

### 4. Each system asked in its own way

On Linux 5.14 and later the thread asks for `MADV_POPULATE_READ`, which reads the pages in before
it returns. Before 5.14 it falls back to `MADV_WILLNEED`, which the kernel honors only in part. On
macOS it asks for `MADV_WILLNEED`, which reads in all it is asked for. On Windows it does nothing
yet.

### 5. In steps, so that it can be stopped

The thread asks for 8 MiB at a time and checks between requests whether to stop; stopping waits
for at most one request. A forked child that drops an index whose parent was reading it in does
not wait for a thread it does not have.

### 6. Off unless asked for

An index much larger than memory would push everything else out, itself included, and an index
already in memory gains nothing. Whether a file is in memory is the application's to know.

## Measurements

On an Apple M5 in Low Power Mode, 2026-10-10, 99,000 OpenAI embeddings of 1,536 dimensions
(an 821 MB file in full precision, 421 MB in half), `ef` 64, each run a fresh process opening the
index as a reader and searching 1,000 queries; milliseconds from the start of the process, medians
of three to five rounds:

| File out of memory | First result | 100 results | 1,000 results |
| --- | --- | --- | --- |
| Full precision | 110 | 1,890 | 2,400 |
| Full, with `warm` | 110 | 365 | 864 |
| Half precision | 100 | 976 | 1,298 |
| Half, with `warm` | 81 | 198 | 517 |
| hnswlib | 499 | 705 | 2,571 |

| File in memory | First result | 100 results | 1,000 results |
| --- | --- | --- | --- |
| Full precision | 9 | 252 | 732 |
| Full, with `warm` | 9 | 85 | 571 |
| Half precision | 6 | 94 | 423 |
| Half, with `warm` | 6 | 53 | 370 |

A writer, on a copy of the full-precision file out of memory, returned its hundredth result after
523 ms with `warm` and 1,912 ms without; through a view of the file it gains nothing when the file
is already in memory. Once a search's pages are in, it takes as long either way: 0.52 to 0.55 ms.

What it reads in is what was written: 38,124 of the file's 50,088 pages of 16 KB, exactly the
pages that hold written bytes, where opening without it brought in 5. The file's hash and time of
change are the same after a writer read it in.

On Linux, GitHub's runners (EPYC 7763, 9V45 and 9V74, Xeon 8573C, Neoverse-N2) were measured in
two runs of four machines each, the first on an earlier commit that makes the same system calls.
Their disks read 260 to 660 MB/s in sequence and 5,600 to 11,600 random reads a second one at a
time. On the 1,536-dimension vectors read from their own disk, the hundredth result came after
1.3 to 1.7 s with it and 1.5 to 2.9 s without, and hnswlib's after 1.3 to 2.1 s; in half
precision after 0.55 to 1.1 s, from 0.62 to 1.3 s. On a million SIFT vectors of 128 dimensions it
came after 1.37 to 1.38 s, from 1.56 to 2.07 s, and in half precision after 0.75 to 0.76 s, from
0.94 to 1.3 s; hnswlib took 1.76 to 2.04 s. With the file already in memory the hundredth result
came 8 to 39% sooner.

On one runner whose disk read 263 MB/s in sequence, reading the 713 MB file in took longer than
the searches took without it: the hundredth result came after 2.7 s with it and 2.6 s without,
and the first 0.56 s later. Elsewhere the first result came up to 0.16 s later with it, while the
thread had the disk, and on one machine sooner.

## What It Does Not Do

* It does not keep the index in memory. The system may push its pages out again later, as it may
  any file's.
* It reads in file order, not the pages searches need first. The upper layers are in the heap,
  near the end.
* It does nothing on Windows, where `PrefetchVirtualMemory` would be the way to ask.
* It does not help an index larger than memory, nor one whose file is already in memory.

## What Was Left Out

* **Paging in a node's neighbors together**, from inside a search: the prototype's first result
  came in 31 ms instead of 110, but it slowed the searches after it on Linux and every search on a
  file already in memory. It is in `IDEAS.md` on the `ideas` branch.
* **Locking the index in memory** (`mlock`): it needs a privilege or a raised limit, and takes
  memory from everything else for as long as the index is open.
* **A signal for when the reading is done**: nothing measured needs one, and the share of a file in
  memory is the system's to report.

## Consequences

### Positive

* On a file out of memory the hundredth search returns five times as soon on the M5, and sooner
  than hnswlib's; on slower Linux disks, up to two fifths sooner.
* A reader's searches on a file already in memory start up to three times as fast.
* Nothing changes for an index opened without it: no thread is started and no advice given.

### Negative

* One more thread for the length of the reading, and a read-only mapping of the file while it
  lasts.
* On Linux before 5.14 it reads in only some of the index.
* On a disk that reads the file in more slowly than searches bring in what they need, it gains
  nothing and the first result comes later.
* An application that asks for it on an index much larger than memory pushes everything else out.
