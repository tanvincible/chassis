# Chassis v0.10.0 against hnswlib and FAISS (incumbents RUN 14)

Written and committed before any run.

## The questions

1. **Where Chassis stands now.** The performance page's "Against hnswlib" table (ADR-0017, ADR-0018)
   was measured from Rust before distances were computed in groups (ADR-0021), and its "Through
   Python" table (RUN 7, 2026-10-10) drew one machine per CPU. Both are measured again, on v0.10.0
   as users install it, against hnswlib and FAISS's HNSW through their Python packages.
2. **FAISS's Panorama.** FAISS now has `IndexHNSWFlatPanorama`: vectors kept in PCA order, and
   distances that stop once a bound rules the candidate out. On HNSW it made 3.91 times plain
   FAISS's queries a second at 1,536 dimensions for 0.99 recall, with no recall lost, on an AVX-512
   Xeon (Ramani et al., arXiv 2510.00566). Does it widen FAISS's lead at long vectors on the
   runners?
3. **The pages, or the rest.** At 1,536 dimensions in full precision Chassis made 0.74 to 0.85
   times FAISS's queries a second on Zen 3 and Zen 4 (RUNs 7, 8), and 0.99 times on Zen 3 with
   huge pages (RUN 10). Since v0.10.0 `warm` asks for them under long vectors. With `warm`, is what
   is left of the gap small on every CPU, so that asking at open is the one lever left?

## What is measured

`incumbents.yml` RUN 14. `chassisdb` 0.10.0 from PyPI (`main` at 1672c79), `faiss-cpu` 1.15.1,
`hnswlib` 0.8.0, Python 3.13. Three datasets: SIFT (1,000,000 × 128), GIST (200,000 × 960) and
dbpedia with OpenAI embeddings (99,000 × 1,536). Six x86 jobs each, on whatever CPUs GitHub gives,
and one Neoverse-N2 job each: 21 jobs.

Engines, each in a process of its own, in this order. Each builds on every core with `M` 16 and
`ef_construction` 200, saves, opens what it saved, and searches on one thread, every query in one
call:

- `chassis-py-batch`, `chassis-py-half-batch`: Chassis in full and in half precision, opened as
  readers, through `search_batch`. The file is in memory from its build, on small pages, as an app
  finds it without `warm`.
- `hnswlib`.
- `faiss`: `IndexHNSWFlat`.
- `faiss-pano`: `PCA{d},HNSW16,FlatPanorama{levels}`, the string FAISS's own
  `benchs/bench_hnsw_flat_panorama.py` uses, with its levels: 2 at 128 dimensions, 8 above. The PCA
  is trained on the vectors indexed.
- `faiss-again`: `faiss`'s index searched again, to see the runner drift.
- `chassis-py-batch-warm`, `chassis-py-half-batch-warm`: the same two Chassis files, put out of
  memory first, opened with `warm`. Searching starts once `warm` has read the file in (`pagetool
  resident` reaches the file's blocks on disk, or stops growing for 2 s). `warm` asks for huge pages
  under vectors of 2 KB or more: full precision at 960 and 1,536 dimensions, half at 1,536.

`ef` 16 to 512. Queries a second is the median of three passes over 1,000 queries. **Speed** is
queries a second at 0.95 and at 0.99 recall@10, interpolated linearly between the two `ef` around
it, as in RUNs 6 to 8. Each ratio is taken within a job; a CPU's figure is the median of its jobs.

## Checks on the harness

- Every engine logs its library: `chassisdb` 0.10.0 from site-packages, FAISS 1.15.1 and its SIMD
  level.
- `faiss-again` is within 5% of `faiss` at every `ef`. A job where it isn't is noisy and left out.
- The warm arms have at least 90% of their vectors' bytes on huge pages (`FilePmdMapped`) where
  `warm` asks, and none where it doesn't; the plain Chassis arms have none. A job where this fails is
  left out of question 3.
- Where `warm` doesn't ask (SIFT, and GIST in half precision), each warm arm is within 5% of its
  plain arm: a control on the warm arms themselves.
- An engine that doesn't reach a recall within its `ef` range has no figure there.

## Predictions

1. SIFT: `chassis-py-batch` makes at least 1.10 times hnswlib's queries a second at 0.95 recall on
   every x86 job, and at least 3 times on N2 (RUN 7: 1.13 on Zen 3, 1.33 on a Xeon, 3.46 on N2).
2. dbpedia, full precision: `chassis-py-batch` makes 0.70 to 0.95 times `faiss`'s at 0.95 recall on
   every x86 job.
3. dbpedia, full precision: the warm arm makes 1.10 to 1.35 times the plain arm's on x86 jobs, and
   at least 0.95 times `faiss`'s on every Zen 3 job.
4. dbpedia: `faiss-pano` makes at least 1.2 times `faiss`'s at 0.99 recall on every x86 job, and at
   least 1.1 times at 0.95.
5. dbpedia, half precision without `warm`: Chassis makes at least 1.0 times `faiss`'s at 0.95 recall
   on every x86 job (RUN 7: 1.09 to 1.40).
6. SIFT: `faiss-pano` is within 10% of `faiss`: at 128 dimensions two levels cut little.

## The decision

- **The docs.** The performance page's two tables become one from this run: per CPU, Chassis over
  hnswlib, over `faiss` and over `faiss-pano` at 0.95 recall, in full and in half precision, with
  `warm` beside them at 960 and 1,536 dimensions. The README's claim, faster than hnswlib at 128
  dimensions on every server CPU measured, stays if every CPU's median on SIFT is above 1.0, and
  otherwise says where it holds.
- **Panorama** (question 2). At 1,536 dimensions, if any x86 CPU's median `faiss-pano` over
  Chassis's fastest arm (any precision, with or without `warm`) is above 1.10 at 0.95 recall, the
  comparison page names Panorama, and a prototype goes on the lab's list with these numbers in
  IDEAS.md. Otherwise IDEAS.md gets the numbers and Panorama stays an idea.
- **The pages** (question 3). At 1,536 dimensions in full precision, per x86 CPU, warm over `faiss`
  at 0.95 recall:
  - At least 0.97 on every CPU, and the plain arm under 0.97 on some: the rest of the gap is the
    pages. Next is a plan for asking at open under long vectors, which first needs a run on an
    index larger than memory, the case that kept it off by default.
  - Under 0.97 on some CPU: pages aren't all of it there. That CPU gets a profiling run, as in RUNs
    9 and 10, before any default changes.
  - The plain arm already at least 0.97 on every CPU: nothing to do.
- **No Zen 3 or Zen 4 among a dataset's x86 jobs:** that dataset is run once more.

## Results

RUN 14, [run 39238399993](https://github.com/tanvincible/chassis/actions/runs/39238399993), all 21
jobs green. `compare_report.py` reads the artifacts. The x86 jobs drew:
- **SIFT:** five Zen 3 (EPYC 7763) and one Zen 5 (EPYC 9V45).
- **GIST:** three Zen 3, two Zen 4 (EPYC 9V74) and a Xeon Platinum 8370C.
- **dbpedia:** five Zen 3 and a Xeon 6973P-C.

FAISS ran at AVX2 on Zen 3, at AVX-512 on the others (AVX2 on one Zen 4, GIST a), and at SVE on N2.

**Checks.**
- Every job loaded `chassisdb` 0.10.0, `faiss-cpu` 1.15.1 and `hnswlib` 0.8.0.
- **`faiss-again` was off by more than 5% at some `ef` on 8 jobs, which are left out:**
  - dbpedia d and f;
  - GIST a, b, c, e and f;
  - SIFT e.

  That leaves GIST with one x86 job, a Zen 4.
- **Huge pages took where `warm` asks and nowhere else.** `warm` put 94.5% of the vectors on huge
  pages at 1,536 dimensions in full precision, 91.0% in half, and 94.8% at 960 in full, on every
  job. No arm had any where it doesn't ask. Each file was in memory before searching, 0.5 to 2.9 s
  after opening.
- **The control failed on Zen 3 at SIFT.** Where `warm` doesn't ask, its arms ran 5 to 9% slower
  than the plain ones on four of five Zen 3 jobs (warm half 0.906, 0.934 and 0.948 times; warm full
  0.944). They ran within 5% on Zen 5, N2 and in GIST's half precision. The warm arms run last, and
  their pages come from reading, not from the build's writes; which of the two it is wasn't looked
  into. It counts against `warm`, so the gains below are if anything low.
- **FAISS and hnswlib reached at most 0.986 and 0.988 recall on GIST at `ef` 512**, so GIST has no
  figure at 0.99.

**Each CPU's median at 0.95 recall** (jobs). Chassis over the other engine; "warm" is Chassis with
`warm`, on huge pages at 960 and 1,536 dimensions.

| Vectors | CPU | Full / hnswlib | Full / FAISS | Warm full / FAISS | Half / FAISS | Warm half / FAISS | Panorama / FAISS | Panorama / Chassis's fastest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 × 1M | Zen 3 (4) | 1.118 | 0.999 | 0.960 | 1.435 | 1.349 | 0.772 | 0.541 |
| | Zen 5 (1) | 2.494 | 2.047 | 2.057 | 2.178 | 2.167 | 0.752 | 0.345 |
| | N2 (1) | 4.011 | 1.891 | 1.828 | 1.987 | 1.951 | 0.891 | 0.449 |
| 960 × 200k | Zen 4 (1) | 1.180 | 0.852 | 0.971 | 1.495 | 1.479 | 1.357 | 0.907 |
| | N2 (1) | 1.958 | 1.059 | 1.138 | 1.465 | 1.472 | 2.396 | 1.628 |
| 1,536 × 99k | Zen 3 (3) | 1.153 | 0.864 | 0.990 | 1.112 | 1.202 | 0.922 | 0.769 |
| | Xeon 6973P-C (1) | 1.046 | 0.926 | 0.973 | 1.260 | 1.501 | 0.955 | 0.637 |
| | N2 (1) | 2.497 | 0.993 | 1.111 | 1.234 | 1.328 | 1.146 | 0.863 |

At 0.99 recall on dbpedia, Panorama over FAISS was 1.198 on Zen 3, 1.334 on the Xeon and 1.552 on
N2. Over Chassis's fastest it was 0.980, 0.858 and 1.098. Counting the noisy jobs too, Panorama on
GIST made 1.23 times Chassis's fastest on Zen 3 (three jobs) and 1.11 times on the Xeon 8370C.

**Predictions.**

1. **Failed, by one job.** Chassis over hnswlib on SIFT was 1.09 on one Zen 3 job (a). It was 1.12
   to 1.15 on the other Zen 3 jobs, 2.49 on Zen 5 and 4.01 on N2.
2. **Held:** 0.85 to 0.93.
3. **Failed on the Xeon.** `warm` gained 1.12 to 1.18 on Zen 3 but 1.05 on the Xeon 6973P-C. Warm
   over FAISS was 0.96 to 1.00 on Zen 3, so that half held.
4. **Failed.** At 1,536 dimensions Panorama made 1.19 to 1.33 times FAISS's speed at 0.99, under
   1.2 on two Zen 3 jobs. At 0.95 it was slower than plain FAISS on every x86 job (0.89 to 0.96).
   It gained most where a few components carry most of each vector: 1.36 to 2.40 times on GIST.
5. **Held:** 1.05 to 1.26.
6. **Failed.** At 128 dimensions Panorama was 11 to 25% slower than plain FAISS, not within 10%.

**Decisions.**

- **The docs** get one table from this run. The claim of beating hnswlib at 128 dimensions stays:
  1.12 on Zen 3, 2.49 on Zen 5 and 4.01 on N2.
- **Panorama stays an idea.** At 1,536 dimensions no x86 CPU's median beat Chassis's fastest setting
  by more than 1.10 (0.64 and 0.77). IDEAS.md gets these numbers, including GIST, where it led on N2
  (1.63) and, counting noisy jobs, on Zen 3.
- **The rest of the gap is the pages.** At 1,536 dimensions in full precision, `warm` brought Chassis
  to 0.99 of FAISS on Zen 3 and 0.97 on the Xeon, against 0.86 and 0.93 without it. Next is a plan
  for asking at open under long vectors. It first needs a run on an index larger than memory, the
  case that kept that off by default. Zen 4 and Zen 5 were not drawn at 1,536 dimensions. At 960,
  `warm` brought one Zen 4 job to 0.97 of FAISS.
- **No reruns.** Every dataset drew Zen 3, and GIST drew Zen 4 too.
