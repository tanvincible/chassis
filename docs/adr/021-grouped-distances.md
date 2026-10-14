# ADR-0021: Distances Computed in Groups

**Date:** 2026-10-10
**Status:** Proposed

## Context

Out of cache, a search over long vectors waits for memory (ADR-0017). A search knows all the
unvisited neighbors of the node it expands before it computes any of their distances, and asks
the CPU to start loading their vectors (ADR-0013, ADR-0015), but it then computed the distances
one after another: each waited for its own vector, and the CPU had one vector's loads in flight at
a time. On x86, FAISS's HNSW answered 1.6 to 1.9 times as many queries a second as Chassis at
1,536 dimensions; it computes a node's neighbors' distances four at a time.

How deep to prefetch had been tuned per CPU (ADR-0015), and a sweep in half precision found more
depth paying on Zen 4 and Zen 5. Whether that should be decided at run time from what the index
is, or measured as it runs, was the open question.

## Decision

### 1. A search computes its distances in groups

Of the unvisited neighbors of the node a search expands, the distances are computed `W` at a time
by one kernel: each piece of the query is loaded once for all `W` vectors, and each vector keeps
its own sums, so the CPU works on all of them together and waits for their memory at once. What is
left over is computed one at a time. Each group's distances are taken in as soon as they are
known. Builds link through the same search.

### 2. The same sums as alone

Each vector in a group keeps the four partial sums a vector alone keeps, added in the same order,
so a distance is the same bits whether it was computed in a group or alone, and searches return
exactly what they did. One sum per vector was 3 to 10% faster in places, but would have given the
same vector distances a rounding apart by different paths, such as a search and an exact scan.

### 3. The width, decided once from what is known

`W` is 1 for vectors under a kilobyte, 8 on Zen 5 and later, and 4 everywhere else. It is decided
when a search starts, from the CPU and the vector's size in bytes, both fixed for an index; no
search measures another. The index's size against the cache does not enter the rule: in cache a
group cost Zen 3 and Zen 5 up to 8% at one `ef` in full precision (2 to 6% over all of them), and
gained elsewhere, against a tenth to three fifths out of cache. Timing settings while searches run was left out:
timings are noisy (other processes, clock speed, a file still arriving in memory), and tuning on
the first searches costs the time `warm()` saves.

### 4. Prefetch depth stays as it is

Once distances are grouped, deeper prefetch mostly cost speed: the groups already keep several
vectors' memory in flight, which is what more depth was buying.

## Measurements

The rule came from two sweeps on the experiment branch (2026-10-10, 31 jobs on GitHub's runners),
of group widths 1, 2, 4 and 8, with one sum or four per vector, against three prefetch depths, at
128, 960 and 1,536 dimensions, in cache and out, in both precisions. Then the code here was run
against `main` on the same runners (21 jobs: EPYC 7763 (Zen 3), 9V74 (Zen 4), 9V45 (Zen 5),
Neoverse-N2), queries a second at the same `ef`, the median over `ef` 32 to 256, the range over
CPUs of each CPU's median. Recall was identical at every `ef`. The 384- and 768-dimension vectors
are the first components of the 1,536-dimension OpenAI ones, which are trained to be embeddings of
their own.

| Vectors | Full precision | Half precision |
| --- | --- | --- |
| 128 × 1,000,000 | 1.00–1.04 | 1.00–1.04 |
| 384 × 99,000 | 1.19–1.55 | 0.99–1.02 (768 bytes: not grouped) |
| 768 × 99,000 | 1.17–1.30 | 1.19–1.58 |
| 960 × 200,000 | 1.11–1.27 | 1.18–1.52 |
| 1,536 × 99,000 | 1.25–1.42 | 1.14–1.60 |
| 1,536 × 2,000, in cache | 0.96–1.15 | 1.03–1.11 |

Zen 5 gained the most in half precision (1.52 to 1.60), Neoverse-N2 in full (1.42 to 1.55); the
one loss, 4%, was Zen 3 in cache at 1,536 dimensions. In the sweeps, which compute the same
distances, the Xeon 8573C and 6973P-C and Zen 4 gained 1.14 to 1.37 at 1,536 × 99,000 in full
precision and 1.31 to 1.35 in half. Batch builds of 384 dimensions and up took 7 to 17% less
time, and of 128 between 5% less and 3% more.

On the Apple M5 the comparison was too noisy to settle, with other processes loading the machine:
the best of three runs was 1.1 to 1.7 times as fast with groups, and the ranges overlapped.

Deeper prefetch, with groups of four or eight: asking for the whole vector into L2 or L1 was
slower than today's depth in 226 of the first sweep's 240 settings with long vectors (machine,
size, precision, width), by up to a quarter, and more than 2% faster in 10.

## Consequences

### Positive

* Searches over vectors of a kilobyte or more are a tenth to three fifths faster out of cache,
  and builds of them take 7 to 17% less time.
* Results are unchanged, bit for bit.

### Negative

* The search loop is compiled once per width: three times as much of it in the library.
* The rule is from the CPUs measured. A CPU not among them gets four.
