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
* **To validate**: count distances per insert in both engines, to tell how much is more distances
  and how much is slower ones; then try building with the vectors on huge pages from the first
  write, which today gave 5%.

## Prefetch depth in half precision

* **Known**, from sweeps on 2026-10-10 (half precision, 1,536 dimensions, 20,000 and 99,000
  vectors): on Zen 5 (three machines) asking for all 48 lines of a vector into L1 is 9 to 12%
  faster than today's eight; on Zen 4 (one machine, two runs) 48 lines into L2 is 21 to 26%
  faster than 32. Zen 3 is unmoved by depth, and so is 960 dimensions on Zen 3 and the Xeons
  except at 200,000 vectors. At 2,000 vectors, in cache, deeper is up to a fifth slower.
* **To validate**: whether the rule is "the whole vector, up to a number of bytes" rather than a
  number of lines, which would make the full-precision settings the same rule; check that it
  costs nothing in full precision at 960 and 1,536 dimensions and in cache.

## Results nobody has explained

* **FAISS's HNSW searches long vectors faster on x86.** On 2026-10-10, with every engine through
  its Python package on one thread (99,000 vectors of 1,536 dimensions, EPYC 7763 and 9V74), at
  95% recall FAISS answered 3,476 to 3,702 queries a second, hnswlib 2,634 to 2,693, Chassis
  1,924 to 2,132 in full precision and 2,563 to 2,979 in half; usearch 1,449 to 1,572. On
  Neoverse-N2, Chassis 1,578 (1,969 in half) against FAISS 2,305 and hnswlib 896. At 128
  dimensions and a million vectors Chassis led on Zen 5 and N2, and FAISS on Zen 3 by 1.05 to
  1.25. The other engines are asked for all the queries in one call and Chassis one call a query,
  which is worth little at half a millisecond a query. A likely cause is that FAISS computes a
  node's neighbors' distances four at a time, so that four vectors' memory is awaited together;
  untried here.
* **Zen 5 with huge pages at 128 dimensions** gained nothing or lost up to 7% (ADR-0016).
* **On a Xeon 6973P-C, hints into L1 beat L2 by 2 to 10%** up to 100,000 vectors (ADR-0014's
  runs): one machine, and nothing at a million.

## Not yet measured at all

* Other engines from a cold start: the first comparison's cold column for Chassis and usearch's
  view was taken while the measuring process still mapped the file, which kept it in memory.
* Anything past a million vectors, or a file past about a gigabyte.
* A consumer x86 laptop or desktop. Every x86 figure is from a server CPU.
* Apple silicon out of Low Power Mode, and more than one machine of it.
* Real power loss; it is only simulated.
