# AVX-512, Chassis and FAISS (avx512 RUN 1)

Written and committed before any run.

## The question

With its vectors on huge pages, Chassis made 0.99 times FAISS's queries a second at 1,536
dimensions in full precision on Zen 3, and 0.97 on a Xeon 6973P-C (RUN 14, with `warm`). On Zen 5
FAISS led by more: 1.23 times Chassis with huge pages, and 1.20 times with both on 4 KB pages, where
on Zen 3 Chassis led (RUN 10, part 2, one job). FAISS ran at AVX-512 on Zen 5; Chassis has only an
AVX2 kernel.

What is known already:

- On a Zen 4, AVX-512 gave FAISS 2.6% over its own AVX2 (RUN 8, one job, `faiss-plan.md`).
- On Zen 5, FAISS's AVX2 kernel took 0.95 of Chassis's time a distance out of cache and 1.13 in
  cache (distbench RUN 1, part 1, two jobs). Its AVX-512 kernel was not timed.
- On Ice Lake Xeons at AVX-512, Chassis was even with FAISS or ahead (RUN 9).

So: is FAISS's lead on Zen 5 its AVX-512, and would an AVX-512 kernel give Chassis the same? If
neither, the lead is outside the distance. The same arms say what is left on the other AVX-512
CPUs, the 1 to 3% of RUN 14.

## What is measured

`avx512.yml` RUN 1, sixteen jobs on `ubuntu-latest`, on whatever x86 CPUs GitHub gives. Chassis at
`lab/exp-avx512` (3e011a7): `main` at 9f63dfb plus `LAB_K=avx512`, which has full-precision
searches use an AVX-512F kernel, in a search loop compiled for AVX-512F. Its kernel keeps two sums
of sixteen a vector, so that the eight vectors Zen 5 groups (ADR-0021) fit in the 32 registers.
The descent through the upper layers stays on AVX2. FAISS 1.15.1, which picks its SIMD level as it
starts; `FAISS_SIMD_LEVEL=AVX2` holds it to AVX2. Both parts run in the same job, so they see the
same machine.

**Part 1, one distance alone** (`distbench`, from `dist-plan.md`, cut down). On 99,000 random
vectors of 1,536 dimensions, nanoseconds a distance, 32 neighbors at random as a search computes
them, the best of five rounds with the arms in turn:

- `avx2`, `avx512`: Chassis's two kernels, inlined into a loop compiled for each, with this CPU's
  prefetch and group width (eight on Zen 5, four elsewhere). `avx2-again` is `avx2` timed last.
- `faiss-avx2`, `faiss-avx512`: FAISS's `fvec_L2sqr_batch_4` at `SIMDLevel` 1 and 2, called from
  its wheel's `libfaiss.so`, four at a time, nothing asked for ahead.

In three memories: `cache` (40 vectors, 240 KB), `anon` (anonymous memory, on huge pages under the
runners' THP `always`, as FAISS's vectors are) and `file-huge` (a file put out of memory, then
mapped with `MADV_HUGEPAGE` and read in, as Chassis's are with `warm`). The AVX-512 arms run only
on CPUs with AVX-512F.

**Part 2, whole searches** (`incumbents.py`, as RUN 14). dbpedia, 99,000 × 1,536, full precision.
Each engine in a process of its own, in this order:

- `chassis-py-batch`: builds the index on every core (`M` 16, `ef_construction` 200), then searches
  it from memory on small pages.
- `chassis-py-batch-warm`: the file put out of memory, opened with `warm`, searched once `warm` has
  read it in. On huge pages, as RUN 14.
- `chassis-py-batch-warm-512`: the same with `LAB_K=avx512`.
- `faiss`: `IndexHNSWFlat`, built and searched at the level FAISS picks.
- `faiss-avx2`: the same index with `FAISS_SIMD_LEVEL=AVX2`.
- `faiss-again`, `chassis-py-batch-warm-again`: `faiss` and `chassis-py-batch-warm` again, for
  drift.

One thread, every query in one call, `ef` 16 to 512, the median of three passes over 1,000 queries.
**Speed** is queries a second at 0.95 recall@10, interpolated linearly between the two `ef` around
it; 0.99 is reported too. Each ratio is taken within a job; a CPU's figure is the median of its
clean jobs.

Each AVX-512 CPU is decided on by itself: Zen 4 (EPYC 9V74), Zen 5 (EPYC 9V45) and each Xeon model.
Zen 3 has no AVX-512; it is reported, and decides only prediction 9.

## Checks on the harness

- Every arm in part 1 within 1e-4 of an f64 reference on 3,200 distances in each memory; the job
  stops otherwise. `anon` and `file-huge` at least 0.9 on huge pages, `cache` reported.
- `avx2-again` within 5% of `avx2` in each memory, or part 1 of that job is left out.
- `faiss-again` within 5% of `faiss`, and `chassis-py-batch-warm-again` within 5% of
  `chassis-py-batch-warm`, at every `ef`, or part 2 of that job is left out.
- `faiss` reports an AVX-512 level on CPUs with AVX-512F and `faiss-avx2` reports `AVX2`;
  `chassis-py-batch-warm-512` logs `inc-kernel avx512` on them, and `avx2, no avx512f` elsewhere.
  An arm that fails this is not decided on.
- The warm arms have at least 90% of the vectors' bytes on huge pages (`FilePmdMapped`).
- `chassis-py-batch-warm-512` finds recall within 0.002 of `chassis-py-batch-warm` at every `ef`:
  the sums are added in another order, which can only swap near-ties.

## Predictions

Part 1, per distance:

1. In `cache` on Zen 5, `avx512` takes at most 0.80 of `avx2`'s time.
2. In `cache` on Zen 4, `avx512` takes at least 0.90 of `avx2`'s time: Zen 4 runs a 512-bit
   operation as two halves.
3. In `file-huge`, `avx512` is within 10% of `avx2` on every CPU: out of cache a distance waits on
   memory.
4. In `cache`, `faiss-avx512` is within 10% of `avx512` on every AVX-512 CPU.

Part 2, whole searches at 0.95 recall:

5. On Zen 5, `chassis-py-batch-warm` makes under 0.95 times `faiss`'s queries a second.
6. On Zen 5, `faiss` makes at least 1.05 times `faiss-avx2`'s.
7. On Zen 5, `chassis-py-batch-warm-512` makes at least 1.05 times `chassis-py-batch-warm`'s.
8. On Zen 4 and the Xeons, `faiss` is within 5% of `faiss-avx2`.
9. On Zen 3, `chassis-py-batch-warm` is within 3% of `faiss`.

## The decision

For each AVX-512 CPU decided on, `warm` is `chassis-py-batch-warm` and `warm-512` its AVX-512 arm.
These can hold together; each gives its own next step:

- **No lead.** `warm` makes at least 0.97 times `faiss`'s queries a second: nothing to chase on that
  CPU.
- **Ours.** `warm-512` at least 1.05 times `warm`: next, a PR with the AVX-512 kernel for full
  precision and an ADR. It is chosen on every AVX-512 CPU if none drawn has `warm-512` under 0.98
  times `warm`, and otherwise only on the CPUs where it gained. Its intrinsics need Rust 1.89, one
  past the crate's minimum, which is the user's call.
- **FAISS's only.** `faiss` at least 1.05 times `faiss-avx2`, but `warm-512` under 1.05 times
  `warm`: FAISS's AVX-512 pays and Chassis's doesn't. Next: part 1's figures say whether it is the
  kernel; if `faiss-avx512` beats `avx512` by 10% or more in `cache`, a lab of FAISS's kernel shape
  (one sum a vector) against Chassis's.
- **Neither.** `faiss` under 1.05 times `faiss-avx2`, `warm-512` under 1.05 times `warm`, and `warm`
  still under 0.97 times `faiss`: the lead is outside the distance. Next: a profile of both searches
  on that CPU (`perf record`, as `prof.yml` does).
- **No Zen 5** among the clean jobs: RUN 2 with this plan, once.

## Not measured

Half precision, where Chassis already leads FAISS on every CPU (RUN 14); 960 dimensions; Arm. An
AVX-512 search loop lets the compiler use AVX-512 elsewhere in the loop too, which is what shipping
one would do.

## Results

[Run 39329804741](https://github.com/tanvincible/chassis/actions/runs/39329804741), `lab/exp-avx512`
at 3e011a7, FAISS 1.15.1. The sixteen jobs drew eight Zen 3 (EPYC 7763), four Zen 4 (9V74), two
Zen 5 (9V45), a Xeon 6973P-C and a Xeon Platinum 8573C. `avx512_report.py` reads the artifacts.

**Two jobs drew another runner image**: Linux 7.0 with transparent huge pages at `madvise`, where
every other job had Linux 6.17 at `always`. There FAISS's memory got no huge pages (2 to 4 MB of
676), and in part 1 neither did `anon`. One was Zen 5 (e), the other Zen 3 (o).

**Checks.**

- Every arm matched the f64 reference in every job.
- Part 1 left out: e and o, whose `anon` had no huge pages. `avx2-again` was within 2.7% of `avx2`
  everywhere.
- Part 2 left out for drift: e (FAISS 10.3%, Chassis 16.8%), b and m (Chassis 5.8%), n (Chassis
  5.1%). That leaves Zen 5 one job (i), Zen 4 three, each Xeon one, Zen 3 six.
- FAISS ran at `AVX512_VPOPCNT` on Zen 4 and 5, `AVX512_SPR` on both Xeons, `AVX2` on Zen 3, and
  `faiss-avx2` at `AVX2` everywhere. `inc-kernel` said `avx512` on every AVX-512 CPU and `avx2, no
  avx512f` on Zen 3.
- The warm arms had 94.5% of the vectors on huge pages in every job, on 6.17 and 7.0 alike.
- `chassis-py-batch-warm-512` found the same recall as `chassis-py-batch-warm` at every `ef` in every
  job.
- One Zen 4 job (k) took 183 ns a distance with `avx2` in `cache`, against 66 to 68 on the other
  three, while its `avx512` took 61 like theirs. Unexplained; its part 2 is in line with the others.

**Part 1**, nanoseconds a distance, the median of a CPU's jobs:

| CPU (jobs) | memory | `avx2` | `avx512` | `faiss-avx2` | `faiss-avx512` |
| --- | --- | --- | --- | --- | --- |
| Zen 5 (1) | `cache` | 53.8 | 30.3 | 49.9 | 30.7 |
| | `file-huge` | 201.6 | 148.9 | 182.2 | 151.4 |
| Zen 4 (4) | `cache` | 67.9 | 61.2 | 59.8 | 57.0 |
| | `file-huge` | 258.8 | 235.8 | 245.2 | 237.9 |
| Xeon 8573C (1) | `cache` | 69.4 | 39.2 | 68.2 | 38.6 |
| | `file-huge` | 371.7 | 335.9 | 364.0 | 335.7 |
| Xeon 6973P-C (1) | `cache` | 72.5 | 48.1 | 64.9 | 45.7 |
| | `file-huge` | 422.2 | 401.4 | 419.4 | 407.9 |
| Zen 3 (7) | `cache` | 73.8 | | 70.6 | |
| | `file-huge` | 243.6 | | 240.0 | |

**Part 2**, queries a second at 0.95 recall, the median of a CPU's clean jobs on Linux 6.17:

| CPU (jobs) | `warm` / `faiss` | `warm-512` / `faiss` | `warm-512` / `warm` | `faiss` / `faiss-avx2` | small pages / `faiss` |
| --- | --- | --- | --- | --- | --- |
| Zen 5 (1) | 0.853 | 1.026 | 1.203 | 1.151 | 0.819 |
| Zen 4 (3) | 0.933 | 1.014 | 1.088 | 1.038 | 0.742 |
| Xeon 8573C (1) | 0.936 | 1.039 | 1.110 | 1.088 | 0.946 |
| Xeon 6973P-C (1) | 0.972 | 1.034 | 1.064 | 1.083 | 0.939 |
| Zen 3 (5) | 0.978 | 0.976 | 0.998 | 1.003 | 0.861 |

On Linux 7.0 at `madvise`, the Zen 3 job (o) had `warm` at 1.209 times `faiss` and Chassis on small
pages at 1.029. The Zen 5 job there (e, left out for drift) had `warm-512` at 1.185 times `faiss`,
and `warm-512` 1.251 times `warm`, as job i.

**Predictions.** Seven of nine held.

1. **Held.** On Zen 5 in `cache`, `avx512` took 0.563 of `avx2`'s time.
2. **Held**, as a median: 0.915 on Zen 4. Two of its four jobs were under: 0.899, and k's 0.334.
3. **Failed on Zen 5.** In `file-huge`, `avx512` took 0.739 of `avx2`'s time there; 0.90 to 0.95
   elsewhere. On Zen 5 a distance out of cache still waits on the kernel.
4. **Held.** In `cache`, `faiss-avx512` took 0.93 to 1.01 of `avx512`'s time.
5. **Held.** On Zen 5, `warm` made 0.853 times `faiss`'s queries a second.
6. **Held.** `faiss` made 1.151 times `faiss-avx2`'s on Zen 5.
7. **Held.** `warm-512` made 1.203 times `warm`'s on Zen 5.
8. **Held on Zen 4 (1.038), failed on both Xeons** (1.083 and 1.088).
9. **Held.** On Zen 3, `warm` made 0.978 times `faiss`'s.

**The decision: ours, on every AVX-512 CPU.** `warm-512` made 1.06 to 1.20 times `warm`'s queries
a second on all four, and none was under 0.98. The Xeon 6973P-C also had no lead (0.972). Next: a
PR with the AVX-512 kernel for full precision and an ADR. Its intrinsics need Rust 1.89, one past
the crate's minimum, which is the user's call.

## What it means

- **FAISS's lead at long vectors was its pages and its AVX-512.** With both, Chassis makes 1.01 to
  1.04 times FAISS's queries a second on every AVX-512 CPU drawn; on Zen 3, which has none, 0.98.
- **RUN 8's "AVX-512 isn't why" held only for Zen 4**, where it is worth 4% to FAISS and 9% to
  Chassis. It is worth 8 to 9% to FAISS on the Xeons and 15% on Zen 5.
- **Chassis gained more from AVX-512 than FAISS on three of the four** (not on the Xeon 6973P-C,
  6% against 8%), because its AVX2 kernel is the slower one there: 1.08 to 1.14 times FAISS's AVX2
  time in cache on Zen 4, Zen 5 and the Xeon 6973P-C. At AVX-512 the two are within 8%.
- **Where Linux keeps huge pages to those who ask** (`madvise`, the newer runner image), FAISS gets
  none, and Chassis led it even on small pages in the one clean job.
- **Not measured**: 128 and 960 dimensions, where the kernel would run too, and half precision,
  whose widening AVX-512 has as well. The PR needs a run at 128 dimensions, where Zen 5 has lost
  before (ADR-0016).
