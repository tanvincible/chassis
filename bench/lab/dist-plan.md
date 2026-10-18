# What a distance costs, Chassis and FAISS

Written and committed before any run.

## The question

RUN 9 (`ndis-plan.md`) found that FAISS's HNSW computes no fewer distances than Chassis for 0.95
recall on dbpedia, yet each costs Chassis 1.21 to 1.22 times as long on Zen 3 at 1,536 dimensions
in full precision. What makes a distance cost more: the kernel, the prefetch, or the memory under
the vectors?

The memory became a suspect while this plan was written. GitHub's x86 runners have transparent
huge pages set to `always` (RUN 9's logs), so FAISS's vectors, in anonymous memory, sit on 2 MB
pages. Chassis's sit in a mapped file on 4 KB pages unless `huge_pages` is on, which it wasn't in
RUN 9. A 6 KB vector spans two or three 4 KB pages, and 608 MB of them is far past what the TLB
covers. The ARM runners have THP at `madvise`, so FAISS gets 4 KB pages there; on Neoverse-N2,
Chassis was ahead.

## What is measured

**Part 1: one distance alone.** `distbench.yml`, RUN 1, Chassis at `lab/exp-dist` (RUN 9's
code, `lab/exp-ndis`, plus this lab's harness and `LAB_PF`). Eight jobs on `ubuntu-latest`. On
99,000 random vectors of 1,536 dimensions, nanoseconds a distance, as a search computes them:

* 32 vectors picked at random, a node's neighbors.
* **Chassis's kernel**, as its search runs it (four vectors at a time, inlined into the loop),
  with each vector's first `N` cache lines asked for into L2 first: `pf0`, `pf8`, `pf16`,
  `pf32`, `pf96` (the whole vector). `now` is this CPU's policy: 8 lines on Zen 3, 32 on Zen 4.
  `now-again` is the same arm again, timed last.
* **FAISS's kernel**, `faiss`: its own `fvec_L2sqr_batch_4` at AVX2, called from its wheel's
  `libfaiss.so` (faiss-cpu 1.15.1), four vectors at a time, with nothing asked for ahead, as
  its search does. Its search calls it through a virtual call and a dispatcher; called here
  directly, it pays less per group of four than in a search, which favours it.

Each arm is timed over the vectors in five kinds of memory, each read into memory before any
timing:

| memory | what it stands for |
| --- | --- |
| `cache` | 40 vectors (240 KB), in L2 |
| `anon` | anonymous memory, no advice: FAISS's vectors |
| `anon-4k` | anonymous memory with `MADV_NOHUGEPAGE` |
| `file` | a file mapped shared and writable, no advice: Chassis's default |
| `file-huge` | the same with `MADV_HUGEPAGE` and `MADV_RANDOM`: Chassis's `huge_pages` |

The best of five rounds, the arms in turn within each round. Each memory's share on huge pages,
from `/proc/self/smaps`, is printed beside it.

**Part 2: whole searches.** `incumbents.yml`, RUN 10, Chassis at `lab/exp-dist`. dbpedia, 99,000
vectors of 1,536 dimensions, six jobs on `ubuntu-latest`. One Chassis index, built as in RUN 9,
searched by the lab's Rust harness six ways: `LAB_PF` at 0, 8 and 32 lines, each without and
with `huge_pages` (`pfN`, `pfN-huge`). Before each, the file is put out of memory, so each reads
it back under its own advice. Three rounds; queries a second the median. FAISS through
`incumbents.py`: `faiss`, `faiss-4k` (the same, with `PR_SET_THP_DISABLE`, so its memory gets no
huge pages) and `faiss-again`. Queries a second at 0.95 recall are interpolated linearly between
the two `ef` settings around it, as in RUNs 6 to 9.

Decided on Zen 3, and on Zen 4 if a job draws one, each CPU on its own, the median over its
jobs. Other CPUs are reported, not decided on.

## Checks on the harness

* Both kernels within 1e-4 of an f64 reference on 3,200 distances in each memory; the run
  stops otherwise.
* Pages, part 1: `anon` at least 0.9 on huge pages and `anon-4k` at most 0.05. `file` and
  `file-huge` count as 4 KB or huge only at most 0.05 or at least 0.9; otherwise they are
  reported as they are and not decided on.
* Pages, part 2: from `/proc/self/smaps_rollup` after each arm's searches, `pfN-huge` with at least
  500 MB of file on huge pages (`FilePmdMapped`) and `pfN` with at most 10 MB; `faiss` with at
  least 500 MB of anonymous huge pages and `faiss-4k` with at most 10 MB. An arm that fails is
  reported and not decided on.
* Noise: `now-again` within 5% of `now` in each memory; `faiss-again` within 5% of `faiss` at
  every `ef`. A job that fails one is not decided on.
* Every Chassis arm in part 2 finds the same recall at every `ef`: prefetch and pages don't
  change results.

## Predictions

On Zen 3:

1. Part 1: `now` takes at least 1.15 times as long on `anon-4k` as on `anon`.
2. Part 1: `now` on `file` within 5% of `now` on `anon-4k`.
3. Part 1: on `anon`, `pf0` within 10% of `faiss`.
4. Part 1: in `cache`, `pf0` takes at most 0.95 times `faiss`'s time.
5. Part 2: `pf8-huge` makes at least 1.10 times `pf8`'s queries a second.
6. Part 2: `faiss` makes at least 1.10 times `faiss-4k`'s.
7. Part 2: `pf8-huge` makes at least 0.95 times `faiss`'s.
8. Part 2: no `pfN` makes more than 1.05 times `pf8`'s queries a second, with the same pages.

## The decision

For each CPU decided on, `own` is its own policy (8 lines on Zen 3, 32 on Zen 4). These can hold
together; each gives its own next step, in this order:

* **Pages.** `own-huge` makes at least 0.95 times `faiss`'s queries a second, and `faiss` at
  least 1.05 times `faiss-4k`'s: the lead is the pages. Next: a plan for `huge_pages` on by
  default on Linux, which ADR-0016 kept opt-in, with a run on the cases where it cost (Zen 5 at
  128 dimensions, down to 0.93) before any change.
* **Prefetch.** Some `pfN` at least 1.05 times `own`'s queries a second with the same pages, and
  faster than `now` in part 1 on the matching memory: next, a PR moving that CPU to `N` lines,
  with ADR-0015 amended.
* **Kernel.** In part 1, `faiss` at most 0.90 times `pf0`'s time on `anon` or in `cache`: next,
  a lab of FAISS's loop shape (one sum a vector) against Chassis's (four).
* **None of these**, with `own-huge` still under 0.95 times `faiss`: the cost is outside the
  distance. Next: the search loop's other work (the heap, the visited list), counted.

## Results

Part 1, distbench RUN 1, [run 39139564685](https://github.com/tanvincible/chassis/actions/runs/39139564685);
part 2, incumbents RUN 10, [run 39139564621](https://github.com/tanvincible/chassis/actions/runs/39139564621).
Both built `lab/exp-dist` at 7430d30, with FAISS 1.15.1. Part 1 drew Zen 3 (EPYC 7763) five times,
Zen 5 (9V45) twice and a Xeon Platinum 8370C once; part 2, Zen 3 five times and Zen 5 once. No
Zen 4, so only Zen 3 is decided on. `dist_report.py` reads both runs' artifacts.

**Checks.** All held, except one noise check in part 2.

- Both kernels matched the f64 reference.
- In part 1 the memories' shares on huge pages were exactly as designed in every job: `anon` and
  `file-huge` 1.000, `anon-4k` and `file` 0.000.
- In part 2, every `pfN-huge` arm had 575 to 578 MB of file on huge pages and every `pfN` none.
  `faiss` had 606 to 616 MB of anonymous huge pages, and `faiss-4k` none.
- `now-again` was within 5% of `now` everywhere.
- `faiss-again` was within 5% of `faiss` on every job but one Zen 3 job, b, at 5.2%, which is left
  out.
- Every Chassis arm found the same recall at every `ef`. FAISS ran at AVX2 on Zen 3.

**Part 1 on Zen 3**, nanoseconds a distance, the median of five jobs:

| memory | `now` (8) | `pf0` | `pf16` | `pf32` | `pf96` | `faiss` |
| --- | --- | --- | --- | --- | --- | --- |
| `cache` | 75.2 | 73.8 | 76.8 | 80.8 | 99.7 | 72.3 |
| `anon` | 241.9 | 239.3 | 245.8 | 251.5 | 270.4 | 241.0 |
| `anon-4k` | 280.2 | 294.0 | 289.9 | 315.6 | 361.4 | 287.1 |
| `file` | 269.0 | 277.0 | 283.5 | 300.4 | 340.5 | 281.2 |
| `file-huge` | 245.7 | 240.2 | 247.1 | 252.9 | 270.5 | 241.3 |

**Part 2 on Zen 3**, queries a second at 0.95 recall, the median of four jobs:

| `pf0` | `pf8` | `pf32` | `pf0-huge` | `pf8-huge` | `pf32-huge` | `faiss` | `faiss-4k` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2,884 | 2,889 | 2,762 | 3,452 | 3,404 | 3,298 | 3,427 | 2,732 |

**Predictions.**

1. **Held.** `now` took 1.158 times as long on `anon-4k` as on `anon`.
2. **Held.** `now` on `file` was 0.960 of its time on `anon-4k`.
3. **Held.** On `anon`, `pf0` took 0.993 of `faiss`'s time.
4. **Failed.** In `cache`, `pf0` took 1.021 of `faiss`'s time, not 0.95 or less: the two kernels
   are even.
5. **Held.** `pf8-huge` made 1.178 times `pf8`'s queries a second.
6. **Held.** `faiss` made 1.255 times `faiss-4k`'s.
7. **Held.** `pf8-huge` made 0.993 times `faiss`'s.
8. **Held.** With 4 KB pages, `pf0` made 0.998 and `pf32` 0.956 of `pf8`'s queries a second; with
   huge pages, 1.014 and 0.969.

**The decision: pages.**

- `pf8-huge` made 0.993 times `faiss`'s queries a second, and `faiss` made 1.255 times
  `faiss-4k`'s.
- Prefetch: no setting reached 1.05 times `pf8`.
- Kernel: `faiss` took 0.98 of `pf0`'s time in cache and 1.01 out of it.

**Not decided on.**

- **Zen 5, one job in part 2.** Huge pages gave Chassis 1.165, but `faiss` at AVX-512 still made
  1.23 times `pf8-huge`'s queries a second, and `faiss-4k` 1.20 times `pf8`'s. In part 1, FAISS's
  AVX2 kernel took 0.95 of Chassis's time out of cache and 1.13 in cache.
- **The Xeon, one job in part 1.** Pages made no difference.

## What it means

- **FAISS's lead on Zen 3 is the pages.** On these runners the kernel gives anonymous memory huge
  pages and leaves a mapped file on 4 KB pages. With `huge_pages` on, Chassis makes 0.99 times
  FAISS's queries a second. With both on 4 KB pages, Chassis is ahead: FAISS makes 0.945 times
  Chassis's.
- **The kernel and the prefetch are not behind.** FAISS's own kernel and Chassis's take the same
  time a distance, in cache and out of it. Eight lines is still Zen 3's best prefetch setting.
- **A whole search gains more than one distance does**: 1.18 times, against 1.09 for one distance
  in a file. The vectors' huge pages leave the TLB to the rest of the search. That share wasn't
  measured on its own.
- **Next, by the decision rule:** a plan for `huge_pages` on by default on Linux, which ADR-0016
  kept opt-in. It needs a run on the cases where huge pages cost first: Zen 5 at 128 dimensions,
  down to 0.93. Whether to change the default is the user's call.
- **Zen 5 is open.** FAISS at AVX-512 leads there on one job, even with huge pages. It needs its
  own run.

## Huge pages made from small ones (distbench RUN 2)

Written before the run.

`huge_pages` is opt-in because a page not yet in memory is then read 2 MB at a time, which costs
an index much larger than memory on every miss (ADR-0016). Linux 6.1 and later can instead put
pages that are already in memory onto a huge page in place: `madvise(MADV_COLLAPSE)`. A file read
on small pages and collapsed once it is in memory would keep small reads from the disk and still
search on huge pages. Whether Linux does that for a regular file, and for one a writer has open,
is the question.

**What is measured.** `distbench.yml` RUN 2, `lab/exp-dist` with the collapse added, eight jobs.
The same timing as RUN 1, over five memories: `anon`, `file`, `file-huge` as before, and:

- `file-collapse`: the file mapped shared and writable, as a writer maps it, read on small pages,
  then `MADV_COLLAPSE` over all of it.
- `file-ro-collapse`: the same through a read-only mapping, with nothing holding the file open
  for writing, as a reader with no writer running.

Each collapse prints whether `madvise` returned 0 or its error, and how long it took. The share
on huge pages comes from `/proc/self/smaps`, as before.

**Predictions.**

1. `file-ro-collapse`: the collapse returns 0, and at least 0.9 of the memory is on huge pages.
2. `file-collapse`: the collapse fails, or leaves under 0.1 on huge pages, because the file is
   open for writing.
3. Wherever a collapse put at least 0.9 on huge pages, `now` there is within 3% of `now` on
   `file-huge`.
4. A collapse of the 608 MB that succeeds takes under one second.

**The decision.**

- **The writable collapse works** (at least 0.9 on huge pages): next, a plan for putting vectors
  on huge pages that way. Read on small pages, collapse the whole huge pages of committed vectors
  once they are in memory, and decide from it whether that can be on by default. Decision 2 of
  ADR-0016 still holds: only pieces that are never written again.
- **Only the read-only one works**: a reader with no writer running could do it, and a writer
  couldn't. That is reported, and whether a reader-only collapse is worth having is the user's
  call.
- **Neither works**: Linux 6.17 on ext4 won't do it for a file. What's left is choosing by size,
  huge pages when the index fits in memory, or keeping it opt-in, and that is the user's call.

### Results

[Run 39143314863](https://github.com/tanvincible/chassis/actions/runs/39143314863), `lab/exp-dist`
at 3f892a5. The jobs drew Zen 3 twice, Zen 4 once, Zen 5 four times and a Xeon Platinum 8370C
once.

**Linux refused both collapses on every job.** `madvise(MADV_COLLAPSE)` returned `EINVAL` at once,
for the writable mapping and for the read-only one with nothing holding the file open for
writing. Neither memory had any of it on huge pages afterwards, and both cost what `file` costs.
The other memories' shares were as in RUN 1, and `now-again` was within 5% of `now` everywhere.

`now` on 4 KB pages (`file`) against huge ones (`file-huge`), as the time a distance:

| CPU (jobs) | `file` / `file-huge` | `file-collapse` / `file-huge` | `file-ro-collapse` / `file-huge` |
| --- | --- | --- | --- |
| Zen 3 (2) | 1.107 | 1.104 | 1.100 |
| Zen 4 (1) | 1.402 | 1.400 | 1.413 |
| Zen 5 (4) | 1.054 | 1.072 | 1.063 |
| Xeon 8370C (1) | 1.010 | 0.994 | 0.966 |

**Predictions.**

1. **Failed.** The read-only collapse returned `EINVAL`.
2. **Held.** The writable collapse failed.
3. **Not tested.** No collapse succeeded.
4. **Not tested.** No collapse succeeded.

**The decision: neither works.** On Linux 6.17 with ext4, pages of a regular file already in
memory can't be put onto huge pages after the fact. The only way onto huge pages is to be read
onto them, which `huge_pages` asks for. What's left is choosing by size, or keeping it opt-in;
that is the user's call.

Not decided on: the one Zen 4 job took 1.40 times as long a distance on 4 KB pages, the largest
cost of small pages measured here.

## `warm` on huge pages (incumbents RUN 11)

Written before the run. `feat/warm-huge-pages` (PR #121) makes `warm` ask for huge pages, as
`huge_pages` does. This run checks that it gets them, and what they give.

**What is measured.** `incumbents.yml` RUN 11, dbpedia, four jobs. The branch's lab harness
(`--cfg lab_hp --cfg lab_warm`) searches one index four ways, three rounds:

- `off`: neither option.
- `warm-cached`: `warm`, right after `off`, with the file still in memory on small pages.
- `warm`: `warm`, the file read from disk.
- `huge`: `huge_pages`, the file read from disk.

Before `off`, `warm` and `huge` the file is put out of memory. Pages come from
`/proc/self/smaps_rollup`, queries a second at 0.95 recall as in RUN 10.

**Predictions.**

1. `warm` with at least 500 MB of file on huge pages, `warm-cached` and `off` with at most 10 MB.
2. `warm` within 3% of `huge`'s queries a second.
3. `warm-cached` within 3% of `off`'s.

**The decision.** With 1 and 2 holding, the PR's claim stands as written. If either fails, the PR
is changed before it is merged.

### Results

[Run 39148249932](https://github.com/tanvincible/chassis/actions/runs/39148249932),
`feat/warm-huge-pages` at b5b4b37. The jobs drew Zen 4 (EPYC 9V74) three times and Zen 3 once.
Every arm found the same recall at every `ef`, and no arm's three rounds were more than 8% apart.

| Job | CPU | `off` | `warm-cached` | `warm` | `huge` | `warm` / `huge` | `warm-cached` / `off` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | Zen 4 | 2,565 | 2,604 | 3,245 | 3,298 | 0.984 | 1.015 |
| b | Zen 3 | 3,170 | 3,186 | 3,645 | 3,664 | 0.995 | 1.005 |
| c | Zen 4 | 2,813 | 2,808 | 3,489 | 3,548 | 0.983 | 0.998 |
| d | Zen 4 | 2,836 | 2,839 | 3,494 | 3,565 | 0.980 | 1.001 |

Queries a second at 0.95 recall. File on huge pages: `warm` 553 to 563 MB, `huge` 573 to
578 MB, `off` and `warm-cached` none.

**Predictions.**

1. **Held.** `warm` had 553 to 563 MB on huge pages; `warm-cached` and `off`, none.
2. **Held.** `warm` made 0.980 to 0.995 of `huge`'s queries a second.
3. **Held.** `warm-cached` made 0.998 to 1.015 of `off`'s.

**The decision: the PR's claim stands.** `warm` keeps 15 to 20 MB fewer vectors on huge pages
than `huge_pages`, which may be the 0.5 to 2% it gives up; why was not looked into. Huge pages
were worth 1.16 on this Zen 3 and 1.26 to 1.29 on Zen 4.

## Where huge pages cost: `warm` at 128 dimensions (incumbents RUN 12)

Written before the run, after a review of PR #121. ADR-0016 found one case where huge pages didn't
pay: Zen 5 at 128 dimensions, from 7% slower to 5% faster, −2% in the median. RUN 10's
write-up said that case comes first, and #121 gives huge pages to every `warm` user without it.
This run measures it.

**What is measured.** `incumbents.yml` RUN 12, `feat/warm-huge-pages` at b5b4b37. SIFT, 1,000,000
vectors of 128 dimensions, on ten jobs (for Zen 5 draws), and dbpedia (1,536) on four. One index
per job, searched three ways, three rounds, the file put out of memory before each: `off`, `warm`
and `huge`. Queries a second at 0.95 recall, the median of the three rounds, as in RUN 10. Each
CPU's median over its jobs.

**Checks.** As RUN 11: `warm` and `huge` with most of their vectors on huge pages, `off` with none.
All arms have the same recall at every `ef`.

**Predictions.**

1. On Zen 5 at 128 dimensions, `warm` makes between 0.93 and 1.05 times `off`'s queries a second.
2. On Zen 3 at 1,536 dimensions, `warm` makes at least 1.10 times `off`'s.
3. `warm` is within 3% of `huge` on every job.

**The decision.** On Zen 5 at 128 dimensions:

- **`warm` at least 0.97 times `off`**: no cost worth an opt-out. #121 keeps `huge_pages` a plain
  bool. The docs say that `warm` asks for huge pages, and Python's `options` reports it.
- **`warm` under 0.97 times `off`**: the cost is real, and #121 gets a way to warm on small pages
  before it merges. How (a three-state `huge_pages`, as the review proposes, or a rule by vector
  size) is the user's call.
- **No Zen 5 drawn**: run again.

### Results

[Run 39149161712](https://github.com/tanvincible/chassis/actions/runs/39149161712),
`feat/warm-huge-pages` at b5b4b37. SIFT drew Zen 3 seven times, Zen 5 (EPYC 9V45) twice and a
Xeon 8573C once; dbpedia drew Zen 3 three times and a Xeon 6973P-C once. Every arm found the same
recall at every `ef`. File on huge pages: `warm` 452 to 462 MB on SIFT and 540 to 546 MB on
dbpedia, `huge` 464 to 470 and 564, `off` none.

| Data | Job | CPU | `off` | `warm` | `huge` | `warm` / `off` | `huge` / `off` | `warm` / `huge` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT | f | Zen 5 | 24,485 | 23,632 | 23,589 | 0.965 | 0.963 | 1.002 |
| SIFT | j | Zen 5 | 25,830 | 24,563 | 25,097 | 0.951 | 0.972 | 0.979 |
| SIFT | a | Zen 3 | 9,848 | 10,400 | 10,305 | 1.056 | 1.046 | 1.009 |
| SIFT | b | Zen 3 | 10,236 | 10,768 | 10,764 | 1.052 | 1.052 | 1.000 |
| SIFT | c | Zen 3 | 9,363 | 9,631 | 9,572 | 1.029 | 1.022 | 1.006 |
| SIFT | d | Zen 3 | 10,019 | 10,542 | 10,655 | 1.052 | 1.064 | 0.989 |
| SIFT | g | Zen 3 | 9,538 | 10,031 | 10,056 | 1.052 | 1.054 | 0.998 |
| SIFT | h | Zen 3 | 9,887 | 10,360 | 10,297 | 1.048 | 1.042 | 1.006 |
| SIFT | i | Zen 3 | 10,209 | 10,901 | 10,889 | 1.068 | 1.067 | 1.001 |
| SIFT | e | Xeon 8573C | 12,299 | 12,643 | 12,719 | 1.028 | 1.034 | 0.994 |
| dbpedia | a | Zen 3 | 2,645 | 2,907 | 2,923 | 1.099 | 1.105 | 0.995 |
| dbpedia | c | Zen 3 | 3,128 | 3,623 | 3,659 | 1.158 | 1.170 | 0.990 |
| dbpedia | d | Zen 3 | 2,812 | 3,396 | 3,297 | 1.207 | 1.173 | 1.030 |
| dbpedia | b | Xeon 6973P-C | 2,319 | 2,436 | 2,387 | 1.050 | 1.029 | 1.020 |

Queries a second at 0.95 recall. The medians over jobs of `warm` / `off`: Zen 5 on SIFT 0.958,
Zen 3 on SIFT 1.052, Zen 3 on dbpedia 1.158.

**Predictions.**

1. **Held.** On Zen 5 at 128 dimensions `warm` made 0.951 and 0.965 of `off`'s queries a second.
2. **Held.** On Zen 3 at 1,536 dimensions it made 1.158 in the median, from 1.099 to 1.207.
3. **Held.** `warm` made 0.979 to 1.030 of `huge`'s.

**The decision: the cost is real.** On Zen 5 at 128 dimensions huge pages cost 3 to 5%, through
`warm` or `huge_pages` alike, as ADR-0016 found (−2%, from −7 to +5%). At 0.99 recall, scored
after the decision, the cost was 2 to 3.5%: smaller, still there. #121 gets a way to warm on small
pages before it merges. At 128 dimensions Zen 3 and the Xeon 8573C gained 2 to 7%; at 1,536, Zen 3
and the Xeon 6973P-C gained 3 to 21%.

The user chose a three-state `huge_pages` whose unset state leaves it to `warm`, which asks under
vectors of 1 KB or more. A skeptic pass afterwards found the line less sure in half precision: in
single runs (ADR-0018), huge pages gained 3 to 20% under 128-dimension vectors, Zen 5's included,
and from 3% less to 8% more under long ones. A same-job run of `warm` in half precision would settle
it.

## The 1 KB line at 384 and 768 dimensions (incumbents RUN 13)

Written before the run. `warm` now asks for huge pages under vectors of 1 KB or more, unless
`huge_pages` says otherwise (#121, merged as 8676c4c). The line rests on 128 dimensions (512
bytes, Zen 5 lost) and 1,536 (6 KB, every CPU gained). Most text embeddings sit between: 384
and 768 dimensions. In half precision those are 768 and 1,536 bytes, one on each side of the line;
in full precision, 1,536 and 3,072 bytes, both above it. This run measures all four before v0.10.0
ships the line.

**What is measured.** `incumbents.yml` RUN 13, `main` at 8676c4c. dbpedia's 99,000 vectors cut to
their first 384 and first 768 components (`LAB_DIMS`; text-embedding-3 is trained so a prefix is an
embedding of its own), the true neighbors found again on the cut vectors. On each of 20 jobs, four
indexes: 384 and 768 dimensions, each in full and in half precision. Each is searched three ways,
three rounds, the file put out of memory before each:

- `off`: neither option.
- `warm`: `warm`, `huge_pages` unset: the rule decides.
- `huge`: `huge_pages` on.

`ef` from 16 to 512. Queries a second at 0.95 recall, the median of the three rounds, as in RUN 12.
Each CPU's median over its jobs of `huge` / `off`.

**Checks.** Every arm has the same recall at every `ef`. `off` has no file on huge pages. `huge` has
most of its file on them.

**Predictions.**

1. `warm` has at least 80% of `huge`'s file on huge pages at 384 full, 768 full and 768 half, and
   under 10 MB at 384 half.
2. `warm` is within 3% of `huge` where it asks, and within 3% of `off` at 384 half.
3. On Zen 3, `huge` makes at least 1.03 times `off`'s queries a second in all four, and more at
   768 full than at 384 half.
4. On Zen 5, `huge` makes at least 0.97 times `off`'s at 768 full.

**The decision.** A vector size belongs above the line if no CPU's median `huge` / `off` is under
0.97 there (RUN 12's bar).

- **Above the line, a CPU under 0.97** at 384 full, 768 full or 768 half: the line rises past that
  size, to the next power of two, in a PR before v0.10.0. If the sizes that fail aren't all below
  the ones that pass, no single line fits, and the choice goes to the user.
- **Below the line, 384 half:** if Zen 5 was drawn, no CPU is under 0.97, and the median over all
  jobs is at least 1.03, the line drops to 768 bytes before v0.10.0. Otherwise it stays.
- **Neither:** the line stays at 1 KB, and v0.10.0 is cut.
- **No Zen 5 drawn:** run once more. Without Zen 5 the second time, the rules above apply to the
  CPUs drawn, and ADR-0016 says Zen 5 wasn't measured between 128 and 1,536 dimensions.

### Results

[Run 39175932818](https://github.com/tanvincible/chassis/actions/runs/39175932818), `main` at
8676c4c on all 20 jobs. They drew Zen 3 twelve times, a Xeon 8573C three times, a Xeon 8370C three
times, Zen 4 (EPYC 9V74) once and Zen 5 (EPYC 9V45) once. Every arm found the same recall at every
`ef`. `off` had no file on huge pages; `huge` had 60 to 62 MB at 384 half, 128 to 130 at 384 full
and 768 half, 270 to 276 at 768 full.

`huge` / `off`, queries a second at 0.95 recall, each CPU's median over its jobs:

| Vector | Bytes | Zen 3 (12) | Xeon 8573C (3) | Xeon 8370C (3) | Zen 4 (1) | Zen 5 (1) |
| --- | --- | --- | --- | --- | --- | --- |
| 384 half | 768 | 1.008 | 1.061 | 1.039 | 0.904 | 1.030 |
| 384 full | 1,536 | 1.050 | 1.062 | 1.022 | 0.809 | 1.005 |
| 768 half | 1,536 | 1.038 | 1.002 | 1.009 | 0.802 | 0.961 |
| 768 full | 3,072 | 1.114 | 1.030 | 0.994 | 1.021 | 1.035 |

By round, the Zen 4 made 0.65, 0.85 and 0.82 at 384 full, 0.80, 0.79 and 0.80 at 768 half, and
0.91, 0.90 and 0.94 at 384 half; `warm` lost as much where it asked. The Zen 5 made 0.92, 0.99 and
1.05 at 768 half.

**Predictions.**

1. **Held.** `warm` had 92 to 99% of `huge`'s file on huge pages at 384 full, 768 full and 768
   half, and none at 384 half.
2. **Failed.** `warm` came within 3% of its match (`huge` where it asks, `off` at 384 half) in 68
   of 80 index-jobs, not all: outside on 0 of 20 at 384 full, 5 at 384 half (0.92 to 1.11), 3 at
   768 full and 4 at 768 half. The rounds were noisier than RUN 12's, up to 42% apart.
3. **Failed in part.** Zen 3 made 1.050, 1.038 and 1.114 where predicted, but 1.008 at 384 half,
   under 1.03. It did gain more at 768 full than at 384 half.
4. **Held.** The Zen 5 made 1.035 at 768 full, from one job.

**The decision: the line rises to 2 KB.** Above the line, the Zen 4 was under 0.97 at 384 full and
768 half, both 1,536 bytes, and the Zen 5 at 768 half. Every CPU passed at 3,072 bytes, so what
failed is all below what passed, and the line goes to the next power of two past 1,536: 2,048 bytes,
512 dimensions at full precision and 1,024 at half (#122). 384 half doesn't drop under the line:
the Zen 4 made 0.904 there.

What this leaves: the Zen 4 and the Zen 5 are one job each, so one machine moved the line; it moved
toward the setting before #121, where nobody got huge pages without asking. In half precision at
1,536 dimensions, 3,072 bytes and above the new line, a Zen 4 made 0.96 in a single run
(ADR-0018). The Zen 4's loss at 99,000 vectors runs against its gain of 1.26 to 1.29 at 1,536
dimensions full (RUN 11), on a file twice as large as 768 full's; vector size alone may not be the
whole story.
