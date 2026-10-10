# Ideas

A dump of ideas for Chassis that have not been validated. This file lives on the `ideas` branch,
which is not merged.

An idea moves to `ROADMAP.md` on `main` once it has been measured and found worth building. One
that is measured and isn't worth it is recorded where it was measured, under "What Was Tried and
Left Out" in an ADR, and removed from here.

Each entry says what the idea is, what is known, and what would validate it. "Unmeasured" means
nobody has run it; an expectation is only that.

## Cold starts and the operating system's pages

### Ask for a node's neighbors' pages at once

A search knows a node's neighbors before it reads their vectors. On a file not in memory each
vector is then its own blocking disk read; asking the system for all of them first lets the disk
serve them together.

* **Known**, from a prototype on `lab/exp-cold` (99,000 vectors of 1,536 dimensions): the first
  result arrives at 31 ms where it takes 110 on an Apple M5, and at 47 to 68 ms where it takes
  204 to 435 on Linux runners. But on Linux the next hundred queries are about twice as slow,
  because the requests replace the kernel's reading of 128 KB around each fault; on a file
  already in memory the first hundred queries take 20 to 50% longer at 1,536 dimensions and up
  to twice as long at 128; and together with a background read it is slower than the background
  read alone.
* **To validate**: a rule for when to do it that keeps the gain and not the costs, such as only
  for a process's first query or only when the file is found out of memory at open. Measure on a
  laptop's NVMe under Linux, and on an index larger than memory, which is the case it is for.

### Lay neighbors out on the same pages when compacting

Compaction rewrites the file anyway. Ordering slots so that nodes linked in the graph sit near
each other would mean fewer page faults on a cold file and fewer TLB misses on a warm one.

* **Known**: nothing measured. It can only help where several vectors fit in a page: at 128
  dimensions eight do in 4 KB, and twice that in half precision; a full-precision vector of 1,536
  dimensions is a page and a half on its own.
* **To validate**: reorder an existing index in the experiment branch, then count faults a query
  on a cold file and measure warm search speed, at 128 and 768 dimensions.

### An index larger than memory

Nothing has been measured with one. Reading only what a query touches is the design's claim
there, and huge pages and `warm()` are both wrong for it.

* **To validate**: build an index several times the size of memory (or cap the page cache), and
  measure search speed, what the system reads per query, and what it keeps resident.

## Doing more at once

### `search_batch`: many queries across every core

A caller with a thousand queries has to start its own threads. Searches are independent and
read-only, and `add_batch` already has the thread pattern.

* **Known**: nothing measured; near-linear scaling is the expectation. It helps bulk jobs
  (deduplication, evaluation, re-ranking many chunks), not one interactive query.
* **To validate**: queries a second against cores, in cache and out of it, where memory
  bandwidth may be the limit before the cores are.

### Scan on every core when a filter passes few vectors

When walking the graph would cost more, a filtered search checks every slot, on one thread.

* **Known**: nothing measured.
* **To validate**: time the scan on a large index with a filter passing a thousandth of it, on
  one thread and on all.

### Build the id lookup on every core

With caller-chosen ids, the table from id to slot is built on first use by one thread reading
every slot.

* **Known**: nothing measured, at any size.
* **To validate**: time it at one and ten million ids; only worth doing if it is long enough to
  notice.

### Write a batch's vectors on every core

`add_batch` writes its vectors on one thread before linking on all of them, and in half
precision converts each component there too.

* **Known**: linking is nearly all of a build's time, so the expectation is that this gives
  little.
* **To validate**: the share of a batch build spent before linking starts.

## Smaller vectors

### Eight bits a component

Half precision halved the file with the same recall (ADR-0018). Eight bits would halve it again.

* **Known**: nothing measured here. It changes results by more than rounding does, so it would
  need the full vectors or a second pass to rank the final candidates, and it is another file
  format.
* **To validate**: quantize real embeddings in the experiment branch, as was done for half
  precision before it was built, and measure recall with and without re-ranking.

### Change an index's precision when compacting

The precision is fixed at creation; to change it the vectors are added to a new index.
Compaction already copies every live vector into a new file.

* **Known**: full to half is a rounding; half to full cannot bring back what was rounded away.
* **To validate**: little to measure. It is a question of whether anyone needs it.

### A faster half-precision kernel on ARM servers

On Neoverse-N2 the half-precision kernel takes 1.6 times as long as the `f32` one when nothing
waits for memory, and an index of long vectors that fits in cache searches 19% more slowly.

* **Known**: the kernel widens four components at a time with one instruction per four.
* **To validate**: try widening from wider loads, or SVE where a CPU has it, in the kernel
  benchmark.

## Builds of long vectors on x86

A batch build of 1,536-dimension vectors in full precision takes 1.4 to 1.7 times as long as
hnswlib's on x86 with four cores, where at 128 dimensions it takes half to nine tenths, and in
half precision it is level.

* **Known**, from a profile on 2026-10-10 (20,000 vectors, EPYC 7763 and 9V74): it is not the
  file. Kernel time is 1% of the build, and building on tmpfs takes as long as on disk. 64% of the
  time is the search for a new node's neighbors and 29% the distances computed while choosing
  which neighbors to keep; one thread spends 1.6 ms a vector where hnswlib spends 1.0 to 1.2. So
  it is the same memory-bound distance work as a query, where full precision trails hnswlib on
  these CPUs at the same `ef`. hnswlib builds 15 to 25% slower without its huge pages.
* **Since**: PR #42 computes a search's distances in groups, and builds of 384 dimensions and up
  take 7 to 17% less time.
* **To validate**: what remains of the gap against hnswlib after #42; then count distances per
  insert in both engines, to tell how much is more distances and how much is slower ones.

## Other engines from a cold start

* **Known**, from 2026-10-10 (every engine through Python on one thread, files verified out of
  memory first, `warm()` as in PR #40): from opening to the hundredth query, on 99,000 vectors of
  1,536 dimensions Chassis took 1.3 to 1.5 s in full precision either way, and 0.57 to 0.61 s in
  half precision with `warm` (0.70 to 0.76 s without); hnswlib 1.37 to 1.41 s, FAISS 1.31 to
  1.43 s, usearch 1.35 to 1.44 s, LanceDB 1.13 to 1.19 s. On a million SIFT vectors: Chassis
  1.38 to 1.41 s with `warm` (1.44 to 2.33 s without), 0.74 to 0.77 s in half precision; hnswlib
  2.01 to 2.17 s, FAISS 1.41 to 1.48 s. Chassis's first result came after 0.2 to 0.5 s;
  hnswlib's, FAISS's and usearch's, which load the file first, after 1.3 to 2.2 s; LanceDB's after
  0.9 to 1.4 s; usearch's memory-mapped view's after 0.2 to 0.8 s, with its hundredth after 1.4 to
  2.2 s. One Zen 4 runner was slow for every engine (FAISS 3.3 s), and there
  `warm` made Chassis slower: 2.8 s to 3.2 s in full precision, and its first result 0.5 s to
  1.2 s.
* **To validate**: a rule for when `warm` should hold back on a slow disk, for instance reading in
  only while searches aren't waiting on it.

## Results nobody has explained

* **FAISS's HNSW searched long vectors faster on x86**: 1.6 to 1.9 times Chassis's queries a
  second at 1,536 dimensions in full precision (2026-10-10, every engine through Python). Its
  distances four at a time were the likely reason; Chassis does the same in PR #42 (ADR-0021),
  1.25 to 1.42 times as fast there. Whether a gap remains is to be measured once it merges.
* **Zen 5 with huge pages at 128 dimensions** gained nothing or lost up to 7% (ADR-0016).
* **On a Xeon 6973P-C, hints into L1 beat L2 by 2 to 10%** up to 100,000 vectors (ADR-0014's
  runs): one machine, and nothing at a million.

## Not yet measured at all

* Anything past a million vectors, or a file past about a gigabyte.
* A consumer x86 laptop or desktop. Every x86 figure is from a server CPU.
* Apple silicon out of Low Power Mode, and more than one machine of it.
* Real power loss; it is only simulated.
