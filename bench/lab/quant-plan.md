# Vectors in fewer bits: the plan

Written and committed before any result. The harness is `quant.py`, run by
`.github/workflows/quant.yml`; the analysis and the decision rule are `report()` and `decide()`
there, in the same commit.

## The question

Half precision halved the file with the same recall (ADR-0018). Would eight bits a component
halve it again without losing recall, and if not, would re-ranking a few candidates by their
half-precision vectors win it back? And how far are four bits and one bit from that?

## What is measured

Vectors are rounded by each scheme and the rounded ones are searched, with each query as given
(32-bit). Nothing is a file format or a kernel, so nothing here says anything about speed; the
speed of 8-bit kernels is a question for later, and only if recall passes.

Schemes, each component rounded to:

| scheme | bits | range |
|---|---|---|
| i8-unit | 8 | fixed [-1, 1], for unit vectors; no calibration |
| i8-vec | 8 | each vector's own min and max; no calibration, 8 bytes more a vector |
| i8-global | 8 | ± the largest magnitude in the calibration set |
| i8-q99 | 8 | the calibration set's 0.5% and 99.5% quantiles over all components (Qdrant's default) |
| i8-dim | 8 | each dimension's min and max in the calibration set |
| i4-dim | 4 | as i8-dim, in 16 levels |
| b1-sign | 1 | the sign, times the vector's mean magnitude |
| b1-rabitq | 1 | RaBitQ: the sign after a random rotation about the calibration set's centroid, with its unbiased estimator; 8 bytes more a vector |

Calibration sets: `all` (every vector, an oracle Chassis cannot have), `first1k` (the first
1,000 added; a random sample here, since the data is shuffled) and `cluster1k` (the 1,000
nearest one random vector, as when the first notes added are all on one topic).

**Stage 1**, no graph: every vector ranked by the scheme's distance to each query, then the
first r×10 re-ranked by half-precision distance, r in 1, 2, 4, 10, 20, 50. Recall@10 against
the exact nearest 10 (float64). This is each scheme's ceiling.

**Stage 2**, Chassis: an index built (batch, M 16, ef_construction 200) on the rounded vectors in
full precision, so the graph is built on rounded distances too, and searched with the queries
as given at ef 16 to 256, re-ranking the first r×10 results, r in 1, 2, 4. Arms: full, half
(Chassis's own), i8-unit, i8-vec, i8-global/first1k, i8-q99/first1k, i8-dim/first1k and
i8-dim/cluster1k. Each arm is built twice (a, b) because batch builds differ from run to run.
Compared only at ef ≥ 64, where k = 40 does not raise the ef Chassis searches with.

Data: 1,000 held-out queries each, unit length.

| config | model | n | dims |
|---|---|---|---|
| openai3-1536 | text-embedding-3-large, DBpedia | 99,000 | 1536 |
| openai3-768, openai3-384 | its prefixes, renormalized | 99,000 | 768, 384 |
| ada-1536-99000, ada-1536-306000 | text-embedding-ada-002, DBpedia (8 of 26 shards) | 99,000 and 306,000 | 1536 |
| minilm-384 | all-MiniLM-L6-v2 on the same DBpedia texts as openai3 | 99,000 | 384 |
| bge-384 | bge-small-en-v1.5 on the same texts | 99,000 | 384 |
| sift-128 (stage 1 only) | SIFT, integer components | 200,000 | 128 |

Chassis is main at 2b3ccb2.

## Checks on the harness

`quant.py selftest`, run before anything else: each scheme's error is within half a step, 0..255
integers survive i8-dim exactly, a ten-vector calibration clips about 2/11 of components,
RaBitQ's estimate is unbiased, exact vectors find every neighbor, and noisy ones lose some that
re-ranking finds again. In the runs, two planted outcomes:

* **Known pass**: SIFT's components are integers, so i8-dim/all in stage 1 loses nothing:
  recall@10 at least 0.995.
* **Known loss**: every vector ranked under another's id ("shuffled"), in stage 1 at r=1 on
  every configuration: recall@10 at most 0.01.

If either fails, the harness is wrong and no result counts.

## Predictions

1. i8-dim/first1k is within 0.005 of half precision in stage 2 at ef 64 and 128, on every text
   configuration.
2. i8-unit loses more than 0.02 at 1536 dimensions: a unit vector's components there are
   around ±0.026, a few steps of 2/254.
3. i8-global loses more than i8-dim on minilm-384 and bge-384, whose dimensions differ in range.
4. i8-vec is within 0.005 of i8-dim.
5. cluster1k costs i8-dim no more than 0.005 more than first1k.
6. b1-sign in stage 1 at r=1 is between 0.55 and 0.75 at 1536 dimensions and below 0.5 at 384,
   and needs r ≥ 10 for 0.95 at 1536.
7. b1-rabitq beats b1-sign at every r.
8. The loss at 306,000 vectors is no more than at 99,000, for the same model.
9. i4-dim/all in stage 1 at r=1 on openai3-1536 is at most 0.90.

## The decision

Loss is half precision's recall@10 (r=1) minus the arm's, both averaged over the two builds,
with a 95% bootstrap interval over the 1,000 queries, paired. An arm passes when its loss is at
most 0.01, and the interval's upper end at most 0.02, at ef 64 and at ef 128, on every text
configuration.

* An arm with no calibration passes at r=1: **int8 alone, with no training step.**
* i8-dim passes at r=1 with first1k and with cluster1k: **int8 alone, calibrated on the first
  vectors added.**
* Some calibrated arm passes at r=1, but not cluster1k: **int8 alone only with a representative
  calibration**, which Chassis cannot promise.
* None passes at r=1, but one passes with r=2 or 4: **int8 with re-ranking**, which keeps
  half-precision copies: smaller in memory, not on disk.
* None passes even then: **not worth building**, recorded in ADR-0018's What Was Left Out.

Stage 1 and the 1-bit schemes inform, they do not decide: 1-bit in a graph would need its own
distance in Chassis, and is only worth that if stage 1 shows few re-ranked candidates win its
recall back.

A smoke test of the harness, on 8,000 vectors of one DBpedia shard on a laptop, ran after this
plan was written and before it was committed; its numbers are not results. One thing changed
after it. The known loss was first "i4-dim/all at r=1 on openai3-1536 is at most 0.90", which
is a guess and not a certainty, and the smoke test had it at 0.946. A planted failure has to be
certain, so it is now the shuffled ranking, and the i4 guess is prediction 9, unchanged, already
seen failing at 8,000 vectors. No other prediction, and not the decision rule, changed.
