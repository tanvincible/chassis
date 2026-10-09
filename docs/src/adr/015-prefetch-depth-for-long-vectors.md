# ADR-0015: Prefetch Depth for Long Vectors

**Date:** 2026-10-09
**Status:** Proposed

## Context

ADR-0013 and ADR-0014 were measured on SIFT, at 128 dimensions. A 128-dimension vector is 512
bytes, so the eight cache lines a search asks for ahead are the whole vector. Of a 1,536-dimension
vector they are a twelfth.

The same experiment branch ran both engines at 960 dimensions (GIST, 200,000 of its vectors) and
at 1,536 (DBpedia entities embedded with OpenAI's `text-embedding-3-large`, 99,000 vectors), with
hnswlib compiled on the machine with `-march=native`. ADR-0014's code against hnswlib:

| CPU | 960, 20,000 | 960, 200,000 | 1,536, 2,000 | 1,536, 20,000 | 1,536, 99,000 |
| --- | --- | --- | --- | --- | --- |
| Neoverse-N2 | 1.81 to 2.30 | 1.36 to 1.53 | 2.84 to 3.03 | 1.42 to 1.97 | 1.44 to 1.63 |
| EPYC 7763 (Zen 3) | 0.89 to 1.01 | 0.86 to 0.90 | 1.25 to 1.43 | 0.82 to 0.94 | 0.76 to 0.85 |
| EPYC 9V74 (Zen 4) | 0.73 to 0.82 | 0.76 to 0.87 | 1.07 to 1.10 | 0.66 to 0.67 | 0.62 to 0.73 |
| EPYC 9V45 (Zen 5) | 0.88 to 0.93 | 0.85 to 0.91 | | | 0.80 to 0.82 |
| Xeon 8573C | 0.87 to 0.90 | 0.86 to 0.88 | | | |
| Xeon 6973P-C | | 0.81 to 0.83 | | | |

Chassis leads on ARM, where hnswlib has no SIMD kernel, and in cache. Out of cache on x86 it
trails, by more than at 128 dimensions, and ADR-0014's changes gave about 5% there where on SIFT
they gave 15% or more nearly everywhere: they are to the loop around the kernel, and at these
sizes the kernel and the memory under it are most of the time. The kernels themselves are level,
within 8% of each other at 960 dimensions on every x86 machine.

Two things were found.

**Huge pages.** hnswlib's index is anonymous memory, which these runners back with 2 MB pages;
Chassis's is a file mapping on 4 KB pages. With huge pages turned off hnswlib is 4 to 9% slower
on the Xeons, 10 to 16% on Zen 5, and 17 to 28% on Zen 3 and Zen 4 out of cache. That is most of
the gap, and the last section is about it.

**Depth.** How much of a long vector to ask for depends on the CPU. Each depth against eight
lines, all into L2, as the median over `ef` and over index sizes of 20,000 and up:

| CPU | 16 lines | 32 lines | 64 lines | Whole vector |
| --- | --- | --- | --- | --- |
| Apple M5, 1,536 | +1 to 5% | +4 to 13% | +10 to 13% | +61 to 68% |
| Neoverse-N2 | +6 to 11% | +5 to 11% | +6 to 19% | −2 to +17% |
| EPYC 9V74 (Zen 4) | +1 to 3% | +3 to 9% | +5 to 21% | +6 to 23% |
| EPYC 7763 (Zen 3) | 0% | −3 to −4% | −9 to −17% | −10 to −17% |
| EPYC 9V45 (Zen 5) | 0 to +3% | −3 to 0% | −17% | |
| Xeon 8573C | −3 to +1% | −4 to −12% | −8 to −17% | −8 to −16% |
| Xeon 6973P-C | −4% | −7% | +2% | |

On the M5 only the whole vector pays: at 3,072 dimensions three quarters of it gave 5% and all of
it 27 to 31%. Zen 3, Zen 5 and the Xeons fetch the rest of a vector well enough themselves, and more
hints only get in the way.

## Decision

A search asks for more of a long vector where the CPU does better with it:

| CPU | Lines asked for |
| --- | --- |
| Apple silicon | The whole vector, up to 256 lines (4,096 dimensions) |
| Other aarch64 | 16 |
| AMD family 19h, a Zen 4 model | 32 |
| Everything else | 8, as before |

Into L2, and on Zen 5 into L1, as before. Zen 4 is told from Zen 3, which shares its CPUID
family, by the model ranges Linux uses: 10h to 1Fh and 60h to AFh. The two ARM policies are
chosen when Chassis is compiled.

A vector of 128 dimensions is eight lines, so nothing changes there.

## Measurements

On 2026-10-09, on GitHub's runners (4 vCPUs) and one Apple M5 in Low Power Mode. `M = 16`,
`ef_construction = 200`, search on one thread over 1,000 queries as the median of five passes
(three on the M5), each engine run four times per job on the runners and three times on the M5,
every version searching the same index. A range covers `ef` 32 to 256 and every machine of that kind.

The code of ADR-0014 and the code of this one, where the policy changed:

| CPU (machines) | Vectors | Before, against hnswlib | After | Change |
| --- | --- | --- | --- | --- |
| Neoverse-N2 (2) | 1,536 × 2,000 | 2.84 to 3.03 | 3.16 to 3.23 | +6 to 12% |
| | 1,536 × 20,000 | 1.64 to 1.97 | 1.68 to 1.94 | −8 to +16% |
| | 1,536 × 99,000 | 1.44 to 1.59 | 1.55 to 1.70 | +6 to 14% |
| Neoverse-N2 (2) | 960 × 20,000 | 1.82 to 2.30 | 1.91 to 2.47 | −2 to +10% |
| | 960 × 200,000 | 1.44 to 1.53 | 1.54 to 1.64 | +7 to 11% |
| EPYC 9V74 (1) | 1,536 × 2,000 | 1.07 to 1.10 | 1.04 to 1.10 | −2 to −1% |
| | 1,536 × 20,000 | 0.66 to 0.67 | 0.69 to 0.70 | +4 to 6% |
| | 1,536 × 99,000 | 0.63 to 0.64 | 0.66 to 0.68 | +4 to 5% |
| Apple M5 | 1,536 × 20,000 | | | +66 to 80% |
| | 1,536 × 99,000 | | | +54 to 58% |
| | 3,072 × 20,000 | | | +24 to 31% |

The 3,072-dimension vectors are pairs of the 1,536-dimension ones joined end to end. One of the
two Neoverse-N2 machines was noisy at 20,000 vectors, for every version it ran.

Where the policy did not change, three Zen 3 machines, a Zen 5 and a Xeon 8573C, and at 128
dimensions on all of them, the two versions were within 2% of each other in the median.

One-thread builds of 20,000 vectors took 28 to 35 s where they took 30 to 38 on Neoverse-N2 at
1,536 dimensions, 10.0 to 11.4 s against 10.8 to 12.1 at 960, and 27 s against 29 on Zen 4.

* **On ARM a search over long vectors is 4 to 9% faster in the median, and on Apple silicon at
  1,536 dimensions half as fast again or more.**
* **On Zen 4 it is 5% faster out of cache** and still well behind hnswlib there.
* **On Zen 3, Zen 5 and Intel nothing changes.** At 960 and 1,536 dimensions out of cache
  Chassis runs at 0.65 to 0.93 of hnswlib's speed on x86.

## Huge Pages: Measured, Not Decided

Asking for huge pages on the mapping, `madvise(MADV_HUGEPAGE)`, works on the runners: Linux 6.17
with ext4 put nearly the whole file on 2 MB pages, whether it had been read back from disk or
had just been built. It is not in this ADR's decision because of what it does to a flush.

Search, against ADR-0014's code in the same job, as the median over `ef`:

| CPU | SIFT, a million | 960 × 200,000 | 1,536 × 99,000 |
| --- | --- | --- | --- |
| EPYC 7763 (Zen 3) | | +16% | +11% |
| EPYC 9V74 (Zen 4) | +9% | +22% | +30% |
| EPYC 9V45 (Zen 5) | | +12% | +10% |
| Xeon 6973P-C | | +6% | |
| Neoverse-N2 | +22% | +7% | +7% |

With them Chassis is at 0.93 to 1.10 of hnswlib's speed at 960 dimensions on the three AMD
processors and 0.81 to 0.95 at 1,536. On the vectors alone, leaving the graph on small pages, the
gain is two thirds to all of that. Builds were 2 to 14% faster, and the first pass over a file
not yet in memory was no slower.

The cost is in adding one vector and flushing, as the median of 300:

| | Small pages | Huge pages on everything | On the vectors only |
| --- | --- | --- | --- |
| Milliseconds | 1.6 to 7.3 | 47 to 156 | 3.5 to 41 |

The figures fit a page written through a mapping being written back whole. One add changes a
few neighbor lists scattered over the graph, and each is then 2 MB to write. With only the
vectors on huge pages it is the page the new vector lands on.

What would keep the gain and not the cost: readers, whose mappings are never written back, ask
for huge pages on everything; a writer asks for them on the vectors only, and only on the 2 MB
stretches that a batch is about to fill or that are already committed, so that the stretch being
appended to stays on small pages; and an option turns it off, for an index much larger than
memory, where each miss would read 2 MB. That is its own change and its own ADR.

## What Was Tried and Left Out

* **Hints at the page starts inside a vector.** A 4 KB page boundary stops the hardware
  prefetcher, and most long vectors cross one. One to four lines at each later page start gave 2
  to 4% on Zen 4 and nothing consistent on Zen 3 or Neoverse-N2.
* **64 lines on Zen 4.** 10% at 20,000 vectors of 1,536 dimensions where 32 gave 5%, but 5 to 9%
  slower than eight lines at 2,000.
* **32 or 64 lines on Neoverse-N2.** 64 was the best setting at 960 dimensions and 200,000
  vectors, 8 to 17%, and gave nothing at 2,000 vectors of 1,536. Sixteen was the steadiest.
* **L1 for long vectors.** Within 2% of L2 on Zen 3, Zen 4 and Zen 5; 1 to 7% slower on the Xeon
  6973P-C.

## Consequences

### Positive

* Long vectors search faster on ARM, on Apple silicon by half or more.
* Nothing changes at 128 dimensions, or on a CPU whose policy did not change.

### Negative

* The policy is now four rules fitted to seven CPUs. A CPU that was not measured gets eight
  lines.
* Telling Zen 4 from Zen 3 depends on model numbers.
* Out of cache on x86, Chassis is still behind hnswlib at these dimensions until its mapping is
  on huge pages.
* Apple's policy was measured on one machine, and nothing here above 3,072 dimensions.
