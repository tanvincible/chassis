# The AVX-512 kernel against main (avx512ship RUN 1)

Written and committed before any run.

## The question

avx512 RUN 1 (`avx512-plan.md`) found an AVX-512 kernel worth 6 to 20% to searches at 1,536
dimensions on every AVX-512 CPU drawn, through Python, after `warm`. PR #131 ships one, changed from
the lab's: it keeps the AVX2 kernel's sums lane for lane and hands them back to the AVX2 code for
the tail and the final adds, so a distance is the same bits on either kernel.

Before it ships: does it lose anywhere, at any size, on any AVX-512 CPU? Does it return exactly
what `main` does? And do builds gain, since they search too? 128 dimensions is the size to watch:
a distance there is four steps of the loop, and the lab never ran it.

## What is measured

`avx512ship.yml` RUN 1, 24 jobs on `ubuntu-latest`, on whatever x86 CPUs GitHub gives. Each job
builds the `lab` example from `main` (b009f39) and from `feat/avx512` (8a8d5c4, #131), and runs the
branch's distance tests (`cargo test --lib distance`), which compare the two kernels' bits on a
CPU with AVX-512.

| Data | Sizes | Jobs |
| --- | --- | --- |
| sift-128 | 128 × 1,000,000 | 10 |
| gist-960 | 960 × 200,000 | 6 |
| dbpedia, cut with `LAB_DIMS` | 384, 768 and 1,536 × 99,000; 1,536 × 2,000 (in cache) | 8 |

For each size, one index in full precision and one in half, built by `main`, then searched by a
reader (`LAB_READER=1`) of each binary: three rounds, each `main` then the branch, each arm in turn:

- `full`: the full-precision index on small pages.
- `half`: the half-precision index, whose code doesn't change: the A/A arm, for noise.
- `warm`: where `warm` asks for huge pages (2 KB a vector or more, so 768, 960 and 1,536 × 99,000),
  the full-precision index opened with `warm`, from a copy put out of memory first.

`ef` 32, 64, 128 and 256, one query a call, one thread, the median of five passes over 1,000
queries. With `LAB_DIGEST=1` each search prints a digest of every result's id and distance bits.

**Speed** is the branch's queries a second over `main`'s at the same `ef`, each the median of the
three rounds, then the median over the four `ef`. Results are the same, so the same `ef` is the
same recall. **Builds**: from 20,000 vectors, each binary builds the full-precision index twice, in
turn, on every core; `main`'s time over the branch's. The four files are compared byte for byte.

A CPU's figure is the median of its clean jobs. Each AVX-512 CPU is decided on by itself: Zen 4
(EPYC 9V74), Zen 5 (EPYC 9V45) and each Xeon model. Zen 3 (EPYC 7763) has no AVX-512 and takes the
AVX2 path in both binaries: it is the control.

`avx512ship_report.py` reads the artifacts; it caught a planted digest mismatch, a planted A/A off
by 5% and a planted `warm` arm without huge pages.

## Checks on the harness

- The branch's distance tests pass in every job.
- `half` within 3% of 1 in a job, or that size in that job is left out.
- The `warm` arm has at least 90% of the vectors on huge pages (`FilePmdMapped`) in every round, or
  that arm is left out. The `full` arm's huge pages are reported.
- A digest that differs between `main`'s own rounds means a search isn't repeatable; it is
  reported, and that size isn't decided on for results.

## Predictions

1. **Same bits.** The branch's digest equals `main`'s at every size, arm and `ef`, in every job.
   Where `main`'s two builds are the same file, the branch's are that file too.
2. **1,536 × 99,000, `full`**: Zen 5 at least 1.10; Zen 4 and both Xeons at least 1.03.
3. **1,536 × 99,000, `warm`**: Zen 5 at least 1.12; Zen 4 and the Xeons at least 1.04.
4. **1,536 × 2,000, in cache**: Zen 5 and the Xeon 8573C at least 1.20; Zen 4 at least 1.05.
5. **768 and 960, `full`**: at least 1.03 on every AVX-512 CPU.
6. **128 × 1,000,000**: within 3% of 1 on every CPU.
7. **Builds at 1,536 × 99,000**: on Zen 5, `main` takes at least 1.05 times the branch's time.
8. **Zen 3**: every size within 2% of 1.

## The decision

- **Bits differ.** A digest where the branch differs from a repeatable `main`, or builds where
  `main`'s two files match and the branch's differ: a bug. #131 stays a draft until it is found.
- **Ship.** On every AVX-512 CPU with a clean job, every size's `full` and `warm` at least 0.98,
  and `full` at 1,536 × 99,000 at least 1.03: ADR-0025, the changelog, and #131 ready for review.
- **By size.** Some size under 0.98 on an AVX-512 CPU, others gaining: AVX-512 only from the
  smallest vector size, in bytes, at and above which no AVX-512 CPU drawn is under 0.98. A commit on
  #131, then RUN 2 at the size it lost, once.
- **No gain.** `full` at 1,536 × 99,000 under 1.03 on every AVX-512 CPU: not shipped. Next, what
  differs from RUN 1's gain: Python, `warm`, or the kernel's change.
- **No Zen 5** among the clean jobs at 128 × 1,000,000 or at 1,536 × 99,000: RUN 2 of that data
  alone, once, before Zen 5 is decided on.
- **Zen 3 off** by more than 3% at a size: the harness is suspect there; RUN 2 of that data.

## Not measured

Half precision with AVX-512, which this PR doesn't ship; Arm, which doesn't change; FAISS, which
RUN 1 covered; Python, whose search calls the same loop.

## Results

[Run 39409718931](https://github.com/tanvincible/chassis/actions/runs/39409718931), `main` at
b009f39 against `feat/avx512` at 8a8d5c4. The 24 jobs drew ten Zen 3 (EPYC 7763), seven Zen 5
(9V45), four Zen 4 (9V74), two Xeon Platinum 8573C (Emerald Rapids) and one Xeon Platinum 8370C
(Ice Lake). Fourteen had THP at `always`, ten at `madvise`; `warm` got huge pages under both.

**Checks.**

- The branch's distance tests passed in all 24 jobs, on every AVX-512 CPU drawn among them.
- Every digest matched: the branch returned `main`'s results, ids and distance bits, at every size,
  arm and `ef`, in every job, and `main` matched itself across rounds.
- No build was ever the same file twice, not even `main`'s two: builds on every core aren't
  byte-repeatable, so the comparison of built files says nothing. The digests carry the claim.
- Left out for the A/A arm: 1,536 × 2,000 on three Zen 5 jobs (1.05 to 1.08) and on the Xeon
  8370C (0.954), and 384 on the Xeon 8370C (0.956).
- The `warm` arms had at least 90% of the vectors on huge pages in every round; the `full` arms
  had none.

**Speed**, the branch's queries a second over `main`'s at the same `ef`, the median of a CPU's
clean jobs (jobs), with `warm` after the comma; builds as `main`'s time over the branch's:

| Vectors | Zen 5 | Zen 4 | Xeon 8573C | Xeon 8370C | Zen 3 |
| --- | --- | --- | --- | --- | --- |
| 128 × 1,000,000 | 1.00 (2) | 1.03 (2) | 1.05 (1) | | 1.00 (5) |
| 384 × 99,000 | 1.14 (4) | 1.12 (1) | | | 1.00 (2) |
| 768 × 99,000 | 1.24, 1.22 (4) | 1.10, 1.10 (1) | | 0.95, 0.92 (1) | 1.01, 1.01 (2) |
| 960 × 200,000 | 1.20, 1.22 (1) | 1.05, 1.09 (1) | 1.09, 1.10 (1) | | 1.00, 1.01 (3) |
| 1,536 × 99,000 | 1.20, 1.19 (4) | 1.03, 1.08 (1) | | 0.98, 0.94 (1) | 1.00, 1.01 (2) |
| 1,536 × 2,000, in cache | 1.13 (1) | 1.12 (1) | | | 1.00 (2) |
| Builds | 1.01 to 1.10 | 1.03 to 1.11 | 1.04, 1.05 | 0.92 to 0.97 | 0.99 to 1.00 |

The Xeon 8370C took 3, 6 and 9% longer in both of its builds at 384, 768 and 1,536, and lost in
searches at every `ef`, by 3 to 6% at 768, where its A/A arm drifted 2% the same way.

**Predictions.** Three of eight held.

1. **Held.** The same bits everywhere. The builds' half of it couldn't be tested.
2. **Failed.** At 1,536 × 99,000 without `warm`, Zen 5 1.20, but Zen 4 1.028 in its one job, whose
   A/A arm was 1.027, and the Xeon 8370C 0.98.
3. **Failed on the Xeon 8370C** (0.94). Zen 5 1.19, Zen 4 1.08.
4. **Failed on Zen 5**: 1.13 in its one clean job, against at least 1.20. Zen 4 1.12.
5. **Failed on the Xeon 8370C** (0.95 at 768). Elsewhere 1.05 to 1.24.
6. **Failed the other way**: at 128, Zen 4 gained 3% and the Xeon 8573C 5%. Zen 5 and Zen 3 were
   even.
7. **Held.** Zen 5's builds at 1,536 took 0.94 of the time.
8. **Held** for `full`, `warm` and builds (0.99 to 1.01); the A/A arm reached 1.023 in cache.

**The decision.** The rule didn't foresee a CPU that loses at every size: **ship** needs every
AVX-512 CPU at 0.98 or more, and **by size** has no size above which the Xeon 8370C stops losing.
Read per CPU, as the plan decides each AVX-512 CPU by itself: AVX-512 on Zen 4, Zen 5, the Xeon
8573C and, from avx512 RUN 1, the Xeon 6973P-C; not on the Xeon 8370C. Its flags are the same as
the 8573C's (neither shows `avx512_fp16`, which Emerald Rapids has), so #131 tells them apart by CPUID model: Intel's
AVX-512 processors before Sapphire Rapids stay on AVX2, as the one measured lost and the older
ones slow their clocks more under 512-bit arithmetic. Zen 4 missed 1.03 at 1,536 × 99,000 by 0.002,
in one job whose A/A arm was off by as much; it gained 5 to 12% at every other long size, and 8%
there with `warm`.

## What it means

- **AVX-512 pays most on Zen 5**: 13 to 24% at 384 dimensions and up, and builds take 6 to 10%
  less time. At 128 dimensions it is even there and gains a little elsewhere.
- **On Ice Lake it costs**, 2 to 9% in searches and builds. One job; the other Intel CPUs drawn
  gained, and nothing in the flags tells the two apart.
- **Results don't change**, by construction and by every digest: a CPU with AVX-512 finds exactly
  what one without it does.
