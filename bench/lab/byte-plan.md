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

## Amendment, before any run on the runners

The harness ran on the Apple M5 twice, after this plan was committed and before any job on the
runners. One planted check was a guess. HH at least 1.8 times H's time assumed the loop costs
nothing but the kernel. Picking the vectors and asking for them ahead are not doubled, and on the
M5 at 384 dimensions HH took 1.74 times H's time. The loop is now timed with no kernel too
(**Z**), and the check is on the kernel's own time: HH less Z at least 1.5 times H less Z, in
cache. On the M5 that was 1.95 to 2.06. The predictions and the decision are as committed.

What the M5 showed, not decided on: H2 was within 3% of H in cache and up to 17% from it out of
cache. E was 0.88 to 1.00 of H's speed in cache, and 0.95 to 1.10 out of it. D was 0.61 to 0.77.
E4 was 1.04 to 1.17 out of cache where the width rule gives 1.

## Results

Run 38242852959, at aff0d14, the branch at fd3ed34; `bytebench.py` over its ten artifacts. The
checks held in every job: every kernel within 4e-7 of the f64 reference, and groups the same bits
as alone. The jobs landed on Zen 3 (EPYC 7763, three), Zen 4 (9V74, one), Zen 5 (9V45, three), a
Xeon Platinum 8573C and two Neoverse-N2.

**Planted.** HH less Z was 1.61 to 2.00 times H less Z in cache, except in one Zen 3 job, where
it was 0.72: there H, and H2, took 98 and 187 ns at 768 and 1,536 dimensions, where the other two
Zen 3 jobs took 40 to 42 and 73 to 75, and HH and the eight-bit kernels took what they took
elsewhere. That job's H is not to be trusted, and it flatters eight bits: it is left out below,
and the decision is the same with it. H2 was more than 5% from H, out of cache only, in both Neoverse-N2 jobs (6.4%, 9.0%)
and on the Xeon (10.4%): not decided on.

**The decision: no format yet.** Speed is H's time over the kernel's, the median by CPU:

| CPU | regime | E at 384 | E at 768 | E at 1,536 | E4 at 384 | E4 at 768 | D at 1,536 |
|---|---|---|---|---|---|---|---|
| Zen 3 | cache | 1.03 | 0.88 | 0.71 | 0.94 | 0.72 | 0.57 |
| Zen 4 | cache | 1.04 | 0.92 | 0.74 | 0.96 | 0.78 | 0.55 |
| Zen 5 | cache | 1.14 | 1.18 | 1.20 | 1.07 | 1.03 | 0.64 |
| Zen 3 | memory | 1.18 | 0.83 | 1.07 | 1.58 | 1.05 | 0.85 |
| Zen 4 | memory | 1.75 | 0.91 | 1.21 | 2.10 | 1.30 | 0.97 |
| Zen 5 | memory | 1.30 | 0.65 | 1.32 | 1.76 | 1.09 | 1.02 |

E's worst was 0.71 in cache and 0.65 out of it; D's 0.55 and 0.54. Neither reaches the second
outcome's 0.75 and 1.0.

Not decided on: on Neoverse-N2, E was 0.92 to 0.93 in cache, where its H2 was within 1.5%, and
0.86 to 1.40 out of it; on the Xeon 0.86 to 0.99 in cache and 1.01 to 1.53 out of it.

The predictions:

1. Fails. In cache at 1,536 dimensions E was 0.71 on Zen 3 and 0.74 on Zen 4.
2. Holds. D was 0.55 to 0.64 where E was 0.71 to 1.20, and slower than E on every machine.
3. Fails. At 768 dimensions out of cache E was 0.65 to 0.91, and at 1,536 on Zen 3 1.07.
4. Not tested: both Neoverse-N2 jobs were too noisy out of cache. In cache E was 0.93 at 1,536,
   so it would have failed.
5. Holds. 1.18 to 1.75.
6. Fails.

**Two causes, one of them not the kernel.** Out of cache at 768 dimensions, E's vectors are 768
bytes and the width rule computes them one at a time, while H's 1,536 bytes go four or eight at a
time. In groups of four (E4) they were 1.05 to 1.30 times as fast as H. The width rule counts
bytes; for eight bits it should count components. In cache at 1,536, on Zen 3 and Zen 4, the
kernel is slower: each eight components take a widening load and a conversion to float, where H
takes one conversion. Its loop has the same shape as `main`'s: four sums a vector, in groups of
four, and, on x86, three of the sixteen spilled to the stack in both. Zen 5 was faster in cache
with eight bits, 1.20.

## What it means

As pre-registered: no file format yet. The out-of-cache shortfall goes with the width rule. The
cost in cache is the conversion, a quarter or more on Zen 3 and Zen 4, and a kernel that keeps the query
in f32 cannot avoid it. Next is the query in eight bits too, multiplied as integers, which needs
a recall run of its own first: its own rounding is added to the vectors'.

The one Zen 3 job where `main`'s grouped half-precision kernel ran 2.4 times as slow, and the
same kernel run twice did not, is not explained.
