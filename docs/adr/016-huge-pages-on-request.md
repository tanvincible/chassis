# ADR-0016: Huge Pages for the Vectors, on Request

**Date:** 2026-10-09
**Status:** Proposed

## Context

ADR-0015 found that most of what hnswlib has over Chassis on a large index of long vectors on
x86 is the size of its pages. Its index is anonymous memory, which Linux backs with 2 MB pages;
Chassis's is a file mapping on 4 KB pages.

Linux can keep a file on huge pages too, where the filesystem supports it, and
`madvise(MADV_HUGEPAGE)` on the mapping asks for it. ADR-0015 measured what that gives, 6 to 30%
on a search, and what it costs: adding one vector and flushing went from about 3 ms to 47 to
156 ms with every region on huge pages, and to 3.5 to 41 ms with only the vectors on them. A page
written through a mapping is written back whole, so one changed neighbor list, or one appended
vector, became 2 MB to write.

Two more things about the kernel decide what can be asked for safely, both from how Linux 6.17
reads a file on a miss in a range that asked for huge pages (`do_sync_mmap_readahead`):

* **A huge page is a piece of the file, not of a mapping.** The miss is rounded down to a
  multiple of 2 MB in the file, and the page read there goes into the page cache, which every
  process that maps the file shares. So a huge page a reader caused is the page a writer writes
  through.
* **A miss reads ahead.** The kernel fetches that huge page and the next one, unless the range
  is marked as read at random.

## Decision

### 1. Off unless asked for

`IndexOptions::huge_pages` turns it on for a writer or a reader; `use_huge_pages()` on either
handle does the same after opening, as do `chassis_use_huge_pages` in C and
`IndexOptions(huge_pages=True)` in Python. Each process asks for itself.

It is a request. On Linux it is `madvise` on parts of the mapping; a kernel or filesystem that
doesn't keep files on huge pages ignores it, and on other systems nothing is asked.

### 2. Only the vectors, and only whole huge pages of them that will not be written again

Chassis asks for a 2 MB piece of the file only if every byte of it is a vector that is either
committed, and so never written again, or about to be written by a batch, which writes its
vectors once. A piece that also holds slot headers, neighbor lists, or the stretch of vectors
that single adds are appending to is never asked for.

So the graph stays on small pages, and so does the end of the vectors. Nothing a flush has to
write back is on a huge page, except a batch's own vectors, which it writes in full anyway.

### 3. Random access with it

Each range asked for is also marked `MADV_RANDOM`, so that a miss reads its own huge page and
not the next one, which may be a piece decision 2 left out. A search reads vectors in no order
anyway.

### 4. When it asks

At open, for the committed vectors. At each commit, and in a reader at each snapshot, for
vectors committed since. And before a batch writes its vectors, for the pieces it will fill, so
that an index built by `add_batch` is on huge pages from the start without being read back from
disk.

### 5. Readers ask for the same pieces as writers

A reader's mapping is never written back, but its huge pages are the writer's too. Asking only
for committed vectors, a reader can't slow the writer's flushes.

## Measurements

On 2026-10-09, on GitHub's runners (4 vCPUs, Linux 6.17, ext4), two runs of twelve jobs. The
code of ADR-0015 and this code with the option off and on, in the same job, on SIFT (128
dimensions, a million vectors), GIST (960, 200,000) and DBpedia with OpenAI embeddings (1,536,
99,000). `M = 16`, `ef_construction = 200`, search on one thread over 1,000 queries as the
median of five passes, three runs each. A range covers `ef` 32 to 256 and every machine of that
kind.

With the option on, 95 to 97% of the vectors were on huge pages, whether the file had just been
built by a batch or had been read back from disk. With it off, none were, and the new code ran
within 3% of the old in the median on all but one noisy machine.

Search with the option on against the same index with it off, as the median over `ef` and the
range:

| CPU (machines) | 128 × 1,000,000 | 960 × 200,000 | 1,536 × 99,000 |
| --- | --- | --- | --- |
| EPYC 7763, Zen 3 (7) | +3% (2 to 9) | +13% (11 to 17) | +13% (11 to 14) |
| EPYC 9V74, Zen 4 (5) | +11% (7 to 15) | +18% (16 to 18) | +27% (25 to 28) |
| EPYC 9V45, Zen 5 (5) | −2% (−7 to +5) | +13% (8 to 20) | +9% (8 to 10) |
| Xeon 8573C (1) | +5% (4 to 8) | | |
| Neoverse-N2 (6) | +14% (10 to 15) | +11% (4 to 15) | +5% (2 to 9) |

A reader that asked gained the same, within four points on every machine.

Against hnswlib, which has huge pages on these machines without asking:

| CPU | 960 × 200,000, off | On | 1,536 × 99,000, off | On |
| --- | --- | --- | --- | --- |
| EPYC 7763 (Zen 3) | 0.87 to 0.90 | 0.99 to 1.03 | 0.77 to 0.81 | 0.87 to 0.92 |
| EPYC 9V74 (Zen 4) | 0.92 to 0.94 | 1.08 to 1.11 | 0.65 to 0.67 | 0.82 to 0.86 |
| EPYC 9V45 (Zen 5) | 0.87 to 0.92 | 0.96 to 1.05 | 0.81 to 0.83 | 0.88 to 0.91 |
| Neoverse-N2 | 1.45 to 1.52 | 1.58 to 1.68 | 1.54 to 1.68 | 1.59 to 1.76 |

Adding one vector and flushing, as the median of 300 in milliseconds, over every machine:

| The file | Option off | On |
| --- | --- | --- |
| Just built by a batch | 1.7 to 7.2 | 1.5 to 5.5 |
| Read back from disk | 1.5 to 5.6 | 1.5 to 5.8 |
| Read back by a reader that asked, then written by a writer that didn't | | 1.3 to 5.5 |

Batch builds took 1 to 9% less time with the option on. The first pass of queries over a file
not yet in memory took up to 28% less, and on one machine 1% more.

* **Search over a large index is 3 to 27% faster in the median**, most where vectors are long
  and on Zen 4.
* **Except on Zen 5 at 128 dimensions**, where on two machines it gained nothing or lost up to
  7%. Why was not looked into.
* **A flush costs what it did.**
* **At 960 dimensions Chassis is level with hnswlib on x86 with the option on, or ahead.** At
  1,536 it is still 8 to 18% behind.

## What It Does Not Do

* **Vectors added one at a time go onto small pages**, and stay there while the file stays in
  memory: the kernel keeps the pages it has. They move to huge pages when the file is next read
  from disk.
* **The graph is never on huge pages.** ADR-0015 measured a third more gain at 128 dimensions
  with it there, and flushes 10 to 90 times slower.
* **3 to 5% of the vectors stay on small pages**: the 2 MB at each end of a segment's vectors
  that they share with something else.
* **A batch that fails, or is lost in a crash, leaves its slots asked for.** The adds that reuse
  them write into huge pages, and their flushes write 2 MB until those slots are full again.
* **It was measured on one kernel and one filesystem.** Elsewhere it may do nothing, or only
  make the kernel read 2 MB around a miss.

## What Was Tried and Left Out

* **Huge pages under everything**, and under the vectors without the rule of decision 2: the
  flush costs ADR-0015 measured.
* **Readers asking for everything.** Their mappings are never written back, but the pages are
  shared, so a reader would have put the writer's graph on huge pages.
* **On by default.** With it a page not yet in memory is read 2 MB at a time. An index that fits
  in memory reads the same bytes sooner; one much larger than memory reads far more than it
  uses, on every miss.

## Consequences

### Positive

* An index too large for the CPU's caches searches faster on every CPU measured but one, by a
  quarter on Zen 4 with long vectors, and builds and warms up a little faster.
* No cost to durability: flushes write what they wrote before.
* Nothing changes for anyone who doesn't ask.

### Negative

* It depends on the kernel and the filesystem, and Chassis can't tell whether it was granted;
  `FilePmdMapped` in `/proc/<pid>/smaps` shows it.
* An index built by single adds gains only after it has been read back from disk.
* On Zen 5 with short vectors it gave nothing or cost a little.
* The file's vectors are cached and evicted 2 MB at a time, which costs an index much larger
  than memory.
* One more option, in four APIs.
