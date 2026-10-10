# ADR-0018: Half Precision for the Vectors, on Request

**Date:** 2026-10-09
**Status:** Proposed

## Context

ADR-0017 found that a search over long vectors that don't fit in cache waits for memory, and that
what would move it is fewer bytes per vector. The same bytes are the size of the file: a million
vectors of 1,536 dimensions are 6.1 GB of `f32`s, on a laptop whose memory they share with
everything else. Chassis is for local semantic search in one file, so that is the tighter limit.

An IEEE 754 half-precision float is 16 bits: 11 significant bits, about three decimal digits,
and magnitudes up to 65,504. A component of an embedding is of order 0.01 to 1, and two vectors
that a search has to tell apart differ by far more than a part in two thousand.

Before anything was built, the experiment branch rounded real vectors to half precision and back
and searched them as before. Exact search over the rounded vectors found 100% of the true ten
nearest on SIFT (whose components are whole numbers, which half precision holds), 100% on 99,000
OpenAI embeddings of 1,536 dimensions, and 99.9% on 200,000 GIST vectors of 960. An index built
over them had the recall of one built over the originals, within what two builds of the same
index differ by.

## Decision

### 1. A precision, chosen when the index is created

`IndexOptions::precision` is `Precision::Full`, the default, or `Precision::Half`; in C the last
argument of `chassis_open_with_precision`, in Python `IndexOptions(precision="half")`. It is the
file's from then on, as the metric is: a writer that opens the file names it and is refused if it
names another, a reader takes it from the file, and `precision()` reports it. A compaction writes
its copy in the same precision.

### 2. The file keeps 16-bit floats

Each component is rounded to the nearest half, ties to even, and stored in two bytes, so a vector
is `dims × 2` bytes where it was `dims × 4`. Nothing else in the file changes size.

A finite component of 65,520 or more in magnitude would round to infinity. A vector with one is
refused, and a batch with one is refused whole, before any of it is written. A cosine index
stores unit vectors, which always fit.

### 3. A query stays in f32

A distance is from the query's `f32`s to a stored vector widened as it is read: eight components
at a time with F16C on x86, four with `fcvtl` on aarch64, one by one elsewhere. The sums are the
ones the `f32` kernel computes, in the same order, so over values that half precision holds the
two give the same bits.

The x86 kernel needs AVX2, FMA and F16C; a CPU with the first two and not the third gets the
scalar kernel for an index in half precision, and keeps the AVX2 kernel for one in full.

### 4. A vector is linked as kept

A new vector's neighbors are found from the vector the file kept, read back and widened, not
from the one given, so that a single add and a batch measure the same distances. Choosing
between two stored vectors widens one of them and measures as a search does. Reported distances
are to the vectors as kept.

### 5. Such a file is format 4

A release from before this one would read half-precision vectors as `f32`s, so it has to refuse
the file: its header says format 4 to readers and to writers. A full-precision file is written
as format 3, byte for byte as before, and older releases go on reading and writing it. The
precision is one byte of the header that was always zero.

### 6. Off unless asked for

The results are not quite the ones full precision gives, the file can't be read by older
releases, and on one CPU measured an index that fits in cache searches more slowly. Each is
small; none is for Chassis to choose on a user's behalf.

## Measurements

On 2026-10-09, on GitHub's runners (4 vCPUs, Linux 6.17, ext4), two runs of 17 jobs, and on one
Apple M5 in Low Power Mode. SIFT (128 dimensions), GIST (960) and DBpedia with OpenAI embeddings
(1,536). `M = 16`, `ef_construction = 200`, search on one thread over 1,000 queries as the median
of five passes, three runs each on the runners and five on the M5. Both precisions were searched
by the same build, each over its own index of the same vectors. A range covers `ef` 32 to 256 and
every machine of that kind.

**Recall.** At each `ef`, on each machine, half precision's recall less full precision's:

| Vectors | Least | Most | Mean |
| --- | --- | --- | --- |
| 128 × 1,000,000 | −0.0016 | +0.0031 | +0.0001 |
| 960 × 200,000 | −0.0028 | +0.0053 | +0.0003 |
| 1,536 × 99,000 | −0.0029 | +0.0015 | 0.0000 |

Two builds of one full-precision index differed by up to 0.004.

**Size.** The file is 51 to 52% of what it was at 960 and 1,536 dimensions, 821 MB to 421 MB for
the 99,000 embeddings, and 62 to 63% at 128, where the graph is a larger share.

**Search.** Queries per second in half precision over full, as the median over `ef`:

| CPU (machines) | 128 × 20,000 | 128 × 1,000,000 | 960 × 20,000 | 960 × 200,000 | 1,536 × 2,000 | 1,536 × 20,000 | 1,536 × 99,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EPYC 7763, Zen 3 (9) | 1.00 | 1.46 | 1.36 | 1.28 | 1.10 | 1.28 | 1.23 |
| EPYC 9V74, Zen 4 (6) | 1.00 | 1.09 | 2.37 | 2.27 | 1.13 | 1.63 | 1.62 |
| EPYC 9V45, Zen 5 (4) | 1.06 | 1.10 | 1.54 | 1.10 | 1.11 | 1.13 | 1.02 |
| Xeon 8370C (1) | | | | 1.42 | | | |
| Xeon 8573C (3) | 1.10 | 1.28 | | | | | 1.41 |
| Xeon 6973P-C (1) | | 1.50 | | | | | |
| Neoverse-N2 (10) | 1.01 | 1.09 | 1.11 | 1.41 | 0.81 | 1.28 | 1.26 |
| Apple M5 (1) | | | | | 1.25 to 2.39 | 1.46 to 1.74 | 1.52 to 1.69 |

**With huge pages** (ADR-0016), in the second run, each file read back from disk first. Against
full precision on small pages:

| CPU | Vectors | Huge pages | Half precision | Both |
| --- | --- | --- | --- | --- |
| Zen 3 | 128 × 1,000,000 | 1.06 | 1.45 | 1.49 |
| | 960 × 200,000 | 1.18 | 1.28 | 1.39 |
| | 1,536 × 99,000 | 1.15 | 1.24 | 1.30 |
| Zen 4 | 1,536 × 99,000 | 1.28 | 1.63 | 1.57 |
| Zen 5 | 128 × 1,000,000 | 0.99 | 1.07 | 1.10 |
| Xeon 8370C | 960 × 200,000 | 0.99 | 1.44 | 1.42 |
| Xeon 8573C | 128 × 1,000,000 | 1.05 | 1.27 | 1.42 |
| | 1,536 × 99,000 | 1.04 | 1.42 | 1.40 |
| Xeon 6973P-C | 128 × 1,000,000 | 1.00 | 1.50 | 1.64 |
| Neoverse-N2 | 128 × 1,000,000 | 1.07 | 1.09 | 1.31 |
| | 960 × 200,000 | 1.04 | 1.40 | 1.40 |
| | 1,536 × 99,000 | 1.06 | 1.29 | 1.32 |

**Against hnswlib** at its recall (ADR-0017), on small pages. hnswlib keeps `f32`s, and has huge
pages on these machines without asking:

| CPU | 128 × 1,000,000, full | Half | 960 × 200,000, full | Half | 1,536 × 99,000, full | Half |
| --- | --- | --- | --- | --- | --- | --- |
| Zen 3 | 1.23 to 1.32 | 1.75 to 1.96 | 0.91 to 1.03 | 1.16 to 1.27 | 0.94 to 1.11 | 1.13 to 1.40 |
| Zen 4 | 2.26 to 2.46 | 2.47 to 2.67 | 0.86 to 0.95 | 1.94 to 2.13 | 0.82 to 1.01 | 1.33 to 1.69 |
| Zen 5 | 2.16 to 2.56 | 2.54 to 2.94 | 0.97 to 1.01 | 1.02 to 1.09 | 0.99 to 1.13 | 1.00 to 1.11 |
| Xeon 8370C | | | 0.99 to 1.05 | 1.42 to 1.49 | | |
| Xeon 8573C | 1.52 to 1.69 | 1.95 to 2.17 | | | 0.95 to 1.09 | 1.38 to 1.53 |
| Xeon 6973P-C | 1.45 to 1.56 | 2.10 to 2.30 | | | | |
| Neoverse-N2 | 3.13 to 4.44 | 3.42 to 4.89 | 1.65 to 1.77 | 2.31 to 2.49 | 1.85 to 2.33 | 2.45 to 2.93 |

**Builds.** Time in half precision over full, as the range over machines of one batch build
each and of the median of two one-thread builds:

| CPU | One thread, 20,000 × 960 | One thread, 20,000 × 1,536 | Batch, 200,000 × 960 | Batch, 99,000 × 1,536 | Batch, 1,000,000 × 128 |
| --- | --- | --- | --- | --- | --- |
| Zen 3 | 0.75 to 0.81 | 0.77 to 0.79 | 0.62 to 0.68 | 0.74 to 0.76 | 0.72 to 0.73 |
| Zen 4 | 0.60 to 0.63 | 0.62 to 0.69 | 0.51 to 0.52 | 0.65 to 0.67 | 0.83 |
| Zen 5 | 0.67 | 0.77 | 0.75 | 0.86 | 0.89 to 0.90 |
| Xeon 8370C | 0.69 | | 0.69 | | |
| Xeon 8573C | | 0.73 | | 0.70 | 0.79 to 0.81 |
| Xeon 6973P-C | | | | | 0.69 |
| Neoverse-N2 | 0.97 to 1.07 | 1.05 to 1.15 | 0.80 to 0.84 | 0.99 to 1.04 | 0.95 to 1.00 |
| Apple M5 | | | | 0.69 | |

One-thread builds of 20,000 vectors of 128 dimensions took 0.93 to 1.04 of the time everywhere.

**Full precision.** The same indexes searched by this code and by the code before it, in full
precision: 0.998 in the median over 228 measurements, nine in ten within 3%.

**The kernels alone**, in an earlier run on eight machines, over vectors picked at random from
a gigabyte, sixteen at a time with their first eight lines asked for. In nanoseconds a distance
at 1,536 dimensions: 452 in `f32` and 357 in half on Zen 3, 543 and 391 on Zen 4, 623 and 521 on
Neoverse-N2. From a quarter megabyte, where nothing waits for memory: 90 and 90, 75 and 75, 176
and 280.

* **Recall is what it was.**
* **The file, and the memory it takes, is about half.**
* **A search over an index too large for the CPU's caches is faster**, by a tenth to two thirds
  on most machines and more than twice on Zen 4 at 960 dimensions, where the 32 lines it asks for
  ahead are now the whole vector. On Zen 5 at 1,536 dimensions and 99,000 vectors it gained
  nothing; why was not looked into.
* **In cache it costs nothing on x86 and a fifth on Neoverse-N2 with long vectors**, whose kernel
  alone takes 1.6 times as long there.
* **Huge pages add little to it where vectors are long**, from 3% less to 8% more at 960 and
  1,536 dimensions, and 3 to 20% at 128.
* **With long vectors, builds take 14 to 49% less time on x86.** On Neoverse-N2 a batch build
  takes up to a fifth less and a one-thread build up to 15% more.

## What It Does Not Do

* **It doesn't keep the vectors as given.** There is nothing to re-rank from, and a vector read
  back from the file is the rounded one.
* **It doesn't convert an index.** The precision is fixed at creation; to change it, the vectors
  are added to a new index.
* **It doesn't hold everything `f32` does.** Components of 65,520 or more in magnitude are
  refused, and ones below 3 × 10⁻⁸ become zero. Vectors of raw counts or unscaled features may
  not fit; embeddings do.
* **It isn't quantization.** Eight bits a component would halve the file again, and change
  results by more than rounding does. That is still not decided.

## What Was Left Out

* **On by default.** Decision 6.
* **Fewer level-0 neighbors for smaller files.** 24 instead of 32 (lab `m0.yml`, 2026-10-10, half
  precision, on Neoverse-N2, a Xeon 8573C and an EPYC 9V74): files 7% smaller at 128 dimensions and
  a million vectors, 3 to 4% at 384 and 1% at 1,536, and batch builds 9 to 18% faster, but searches
  for 0.99 recall 6 to 15% slower at 128 dimensions and 7 to 16% at 384, and 0.97 to 1.04 times as
  fast at 1,536. `max_connections: 12` gives the same, so there is nothing to build.
* **A kernel for two stored vectors.** It would widen both sides of every distance where one
  side widened once serves a whole selection.
* **Requiring F16C wherever AVX2 is used.** One code path fewer, but an x86 CPU or emulator with
  AVX2 and no F16C would have lost its fast kernel for every index, full precision included.

## Consequences

### Positive

* An index takes half the disk and half the memory, which on a laptop decides how large an index
  can be.
* Searches out of cache and builds of long vectors are faster on every x86 CPU measured and on
  the one Apple CPU.
* At equal recall, an index in half precision too large for cache was level with hnswlib, which
  keeps `f32`s, or ahead of it on every machine measured: 1.00 to 1.69 times its speed at 1,536
  dimensions on x86, where full precision was 0.82 to 1.13, and without huge pages.
* Nothing changes for an index in full precision: its file, its results, its speed.

### Negative

* A second file format version, and files that older releases refuse.
* A second set of kernels and of loops compiled around them.
* On Neoverse-N2 an index of long vectors that fits in cache searches a fifth more slowly, and
  one-thread builds of long vectors take up to 15% longer.
* Distances are to rounded vectors, so they differ a little from one precision to the other.
* One more option, in four APIs, and one that can't be changed after creation.
