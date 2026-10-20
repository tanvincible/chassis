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
