# Distances a query, Chassis and FAISS

Written and committed before any run.

## The question

In RUN 8 FAISS's HNSW searched 1,536-dimension vectors in full precision at 1.17 to 1.26 times
Chassis's speed on Zen 3 and 1.31 times on Zen 4 with AVX2 forced, at 0.95 recall. AVX-512 isn't
why (`faiss-plan.md`). At 128 dimensions the two were even. Does FAISS compute fewer distances a
query for 0.95 recall (the graph or the search order), or does each distance cost Chassis more?

## What is measured

RUN 9 of `incumbents.yml`, Chassis from `lab/exp-ndis`: main (v0.9.1) with counters in the search
loops, compiled only with `--cfg lab --cfg lab_count`. The same jobs as RUN 8: dbpedia, 99,000
vectors of 1,536 dimensions, on four x86 jobs and one ARM; SIFT, 1M of 128, on two x86 and one ARM.

- **Chassis**, from Rust: queries a second from the build without counters, three rounds as
  before; distances and hops a query from the build with them, one pass over the same file.
- **FAISS** 1.15.1 from Python: queries a second, and `hnsw_stats.ndis` and `nhops` a query from
  the last of the three timed passes.

Both count every layer. Chassis counts the distances it computes; FAISS's upper layers count every
neighbor, visited or not, so its count runs a little high there: at most 16 a hop on a few upper
hops, against thousands at layer 0 on dbpedia.

At 0.95 recall, each engine's distances a query and queries a second are interpolated linearly
between the two `ef` settings around it, as in RUNs 6 to 8. **The time a distance** is
`1 / (queries a second × distances a query)`, all the search's time spread over its distances.

## Checks on the harness

- On 20,000 clustered vectors of 128 dimensions on an Apple M5, before this plan: at `ef` 32
  Chassis counted 410 distances and 40 hops a query, FAISS 394 and 39.5. The counters count the
  same things.
- The counted and the uncounted Chassis searches have to find the same recall at every `ef`: the
  same file, the same code but the counters.
- FAISS's level has to be AVX2 on Zen 3 (as in RUN 8).

## Predictions

1. On dbpedia, at 0.95 recall, Chassis computes within 10% of FAISS's distances a query, in full
   precision, on every job.
2. On SIFT, the same.
3. On Zen 3 at 1,536 dimensions in full precision, Chassis's time a distance is at least 1.15 times
   FAISS's.

## The decision

- **1 holds**: the lead is in what a distance costs at 6 KB a vector. Next: one distance alone at
  1,536 dimensions, from cache and from memory, Chassis's kernel and prefetch against FAISS's.
- **Chassis computes at least 10% more**: the lead is in the graph or the search order. Next: the
  two graphs side by side, their level-0 degrees and the order each search expands nodes in.
- **Chassis computes at least 10% fewer**: each distance costs it even more than the speeds say.
  Next as for 1 holding.
