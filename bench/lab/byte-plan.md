# Eight bits a component: the kernel

Written and committed before any timing. The scheme is on the roadmap: each vector stored as its
difference from a center, the mean of the first vectors added, in 256 steps between that
difference's own smallest and largest component (LVQ-8; `quant-plan.md`, "The follow-up"). Its
recall was measured by rounding vectors and building Chassis on them in f32. What it costs to
search with was not, and it could still sink it: a byte has to be widened to a float before the
arithmetic, and the half-precision kernel it would replace widens with one instruction.

## The question

Is a distance from an f32 query to a vector kept in eight bits as fast as the half-precision
distance Chassis ships, in cache and out of it, on x86 and ARM?

## What is measured

The query stays in f32, as in the recall runs. `q' = q - c` is computed once per query. A stored
vector is `x = c + lo + step * k`, with `k` its bytes. Two kernels for the squared L2 distance,
which is what search ranks by:

| kernel | per component | per vector, stored |
|---|---|---|
| **D**, direct | widen `k`, `t = lo + step * k`, `d = q' - t`, sum `d * d` | `lo`, `step`: 8 bytes |
| **E**, expanded | widen `k`, sum `q' * k`; then `|q'|^2 - 2 lo sum(q') - 2 step sum(q' k) + |x - c|^2` | `lo`, `step`, `|x - c|^2`: 12 bytes |

`|q'|^2` and `sum(q')` are computed once per query. **H** is the half-precision kernel on `main`
(41c93bd), called as search calls it.

Each kernel is timed as a search runs it: one query, vectors picked at random 32 at a time (a
node's neighbors), each asked for ahead with the machine's prefetch policy on `main`, and their
distances computed in groups of the width `main` picks from the vector's size in bytes (1 under a
kilobyte, 8 on Zen 5, else 4). The timing loop is compiled for the kernel's instructions, so each
kernel is inlined into it, as in search. Nanoseconds a distance, the best of three rounds, at 384,
768 and 1,536 dimensions, from 256 KB of vectors (in cache) and from 1 GB of half-precision vectors
and the same vectors in eight bits (out of cache). The same picks for every kernel.

Recorded, not decided on: E in groups of four under a kilobyte (**E4**), since the width rule was
measured for f32 and half precision.

Machines: eight jobs on `ubuntu-latest` (whichever x86 CPUs they land on) and two on
`ubuntu-24.04-arm` (Neoverse-N2). Results by CPU model, the median over jobs on the same model.
The Apple M5 here runs in Low Power Mode with other work on it: reported, not decided on.

## Checks on the harness

* Every kernel against an f64 reference over the dequantized vectors: relative error at most 1e-4
  on every distance computed in the correctness pass (1,000 vectors, 10 queries, each dimension).
* A group's distances are the same bits as each computed alone.
* Planted, certain: **H2**, the half kernel timed again as if it were another kernel, within 5% of
  H everywhere; and **HH**, the half kernel run twice on each vector, at least 1.8 times H's time
  in cache. A machine where H2 is not within 5% of H is too noisy to decide on, and is reported as
  that.

## Predictions

1. In cache, E is within 10% of H's time on every x86 CPU at 768 and 1,536 dimensions.
2. In cache, D is at least 10% slower than E at 1,536 on every machine.
3. Out of cache, E is at least 1.15 times as fast as H at 768 and 1,536 on every machine.
4. On Neoverse-N2, in cache, E is faster than H at 1,536.
5. At 384, out of cache, E is at least 1.1 times as fast as H on every machine.
6. The decision is the first one below, with E.

## The decision

Speed is H's time over the kernel's: above 1 the kernel is faster. For the better of D and E, on
every CPU decided on and at every dimension:

* In cache at least 0.9, and out of cache at least 1.0: **the kernel is not what decides it.** The
  file format is next, with that kernel.
* Out of cache at least 1.0 everywhere, in cache below 0.9 somewhere but not below 0.75: **the
  format is next, with that cost stated**, as ADR-0018 stated half precision's on Neoverse-N2.
* Otherwise: **no format yet.** A kernel with the query in eight bits too, whose integer dot
  products are several times faster, would be tried first, and it needs a recall run of its own.

If D and E both qualify, E is the one, unless D is within 5% of it everywhere: D stores 4 bytes
less a vector.
