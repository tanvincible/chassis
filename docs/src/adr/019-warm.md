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

### 3. Through a read-only view of its own

The thread maps the file read-only and gives the system advice about that mapping, never about the
index's own. On macOS, advice to read pages in through a writable shared mapping, which a writer
has, marks every page it reads as changed, and the system then writes them all back to the disk.
A view of its own also outlives any mapping the index replaces, as a reader does when a restarted
writer has laid the file out differently; advice about an address that is no longer mapped is an
error, but about one mapped again it acts on whatever is there.

### 4. Each system asked in its own way

On Linux 5.14 and later the thread asks for `MADV_POPULATE_READ`, which reads the pages in before
it returns. Before 5.14 it falls back to `MADV_WILLNEED`, which the kernel honors only in part. On
macOS it asks for `MADV_WILLNEED`, which reads in all it is asked for. On Windows it does nothing
yet.

### 5. In steps, so that it can be stopped

The thread asks for 8 MiB at a time and checks between requests whether to stop. Dropping or
replacing the index stops it and waits for it, which is at most one request.

### 6. Off unless asked for

An index much larger than memory would push everything else out, itself included, and an index
already in memory gains nothing. Whether a file is in memory is the application's to know.

## Measurements

⟪MEASUREMENTS⟫

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

* ⟪POSITIVE⟫
* Nothing changes for an index opened without it: no thread is started and no advice given.

### Negative

* One more thread for the length of the reading, and a read-only mapping of the file while it
  lasts.
* On Linux before 5.14 it reads in only some of the index.
* An application that asks for it on an index much larger than memory pushes everything else out.
