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

## Results

RUN 9, [run 39075931632](https://github.com/tanvincible/chassis/actions/runs/39075931632), Chassis at
`lab/exp-ndis` 572c89d, FAISS 1.15.1. The x86 jobs drew AMD EPYC 7763 (Zen 3) three times, Intel
Xeon Platinum 8370C (Ice Lake, AVX-512) twice and EPYC 9V45 (Zen 5) once; no Zen 4.
`run9_report.py` interpolates at 0.95 recall.

**Checks.** The counted and uncounted Chassis searches found the same recall at every `ef` on every
job. FAISS ran at AVX2 on Zen 3, at AVX512_VPOPCNT on the Xeons and Zen 5, and at ARM_SVE on N2.

| Job | CPU | distances, Chassis / FAISS | ns a distance, Chassis / FAISS | FAISS's queries a second / Chassis's |
| --- | --- | --- | --- | --- |
| dbpedia a | Zen 3 | 1,044 / 1,087 = 0.960 | 376 / 309 = 1.220 | 1.171 |
| dbpedia b | Zen 3 | 1,034 / 1,087 = 0.951 | 307 / 253 = 1.212 | 1.153 |
| dbpedia c | Ice Lake | 1,063 / 1,087 = 0.978 | 421 / 478 = 0.881 | 0.861 |
| dbpedia d | Ice Lake | 1,054 / 1,087 = 0.970 | 460 / 462 = 0.996 | 0.966 |
| dbpedia | N2 | 1,057 / 1,087 = 0.972 | 372 / 410 = 0.907 | 0.882 |
| SIFT a | Zen 3 | 1,197 / 1,199 = 0.999 | 80 / 86 = 0.930 | 0.929 |
| SIFT b | Zen 5 | 1,211 / 1,199 = 1.010 | 33 / 68 = 0.480 | 0.485 |
| SIFT | N2 | 1,209 / 1,199 = 1.008 | 70 / 121 = 0.576 | 0.581 |

Hops a query were 51 or 52 for Chassis on dbpedia and 56 for FAISS, 67 to 69 against 69 on SIFT.
FAISS's upper layers count every neighbor, visited or not, so its distances run high by at most
16 a hop on its 5 to 8 upper hops: its true count on dbpedia is between about 960 and 1,087.

**Predictions.**

1. **Held.** On dbpedia Chassis computed 0.951 to 0.978 of FAISS's distances as counted, and at
   most 1.10 with FAISS's upper-layer count taken out whole.
2. **Held.** On SIFT, 0.999 to 1.010.
3. **Held as counted**: 1.212 and 1.220 on Zen 3. With FAISS's count at its lowest, 1.07 to 1.08,
   so the bound is wider than the prediction; the queries a second (1.15 to 1.17) don't depend on
   it.

## What it means

- **The graph isn't behind.** For the same recall Chassis computes no more distances than FAISS,
  and expands fewer nodes.
- **FAISS's lead is a cost per distance, on AMD Zen 3 (and Zen 4 in RUN 8) at 6 KB a vector.** On
  Ice Lake Xeons, at AVX-512, Chassis is even or ahead; on N2 and Zen 5 it is well ahead, and at
  128 dimensions it is even on Zen 3.
- **Next, by the decision rule:** one distance at 1,536 dimensions alone, from cache and from
  memory, Chassis's kernel and prefetch against FAISS's, on Zen 3 and Zen 4. Chassis prefetches 8
  lines of a vector into L2 on Zen 3 and FAISS none; the next lab should vary that first.
- **Half precision** beat FAISS's full precision on every job, 1.08 to 2.22 times.
