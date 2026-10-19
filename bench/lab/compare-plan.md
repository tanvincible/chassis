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
