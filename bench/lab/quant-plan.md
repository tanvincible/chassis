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

## Amendment, written after results were seen

The SIFT job finished first, and the known pass failed: i8-dim/all in stage 1 at r=1 was 0.9933,
not at least 0.995. Recall by distance was the same, so it was not ties. The expectation was
wrong, not the harness: i8-dim keeps a vector only if each dimension spans exactly 255, and
SIFT's 200,000 vectors span 138 to 212 (none 255), so its steps of 0.54 to 0.83 miss the integers
by up to 0.42. Run on the same split on a laptop, i8-dim/all gave 0.9933 again, and a mapping that
is lossless on these integers (a fixed [0, 255] in steps of 1) gave 1.0000. Half precision, which
holds integers up to 2,048 exactly, gave 1.0000 in the run itself. The known pass is now that:
half precision on SIFT, recall@10 exactly 1. This is the second planted check that was a guess
and not a certainty; both were replaced, and both are recorded here. Nothing else changed.

## Results

Run 38090885820, at b408e61; this report is `quant.py report` at f4dd9a1 over its eight
artifacts. Both planted outcomes held: half precision on SIFT 1.0000, shuffled at most 0.0001
everywhere.

**The decision: int8 only with re-ranking.** No arm was within 0.01 of half precision at r=1 at
ef 64 and 128 on all seven text configurations. Every arm was within it at r=2; the closest,
i8-q99/first1k on openai3-384 at ef 64, lost 0.0034 [0.0004, 0.0065]. Recomputed from this plan's
words, not `decide()`, the verdict is the same and all 252 losses agree within bootstrap noise.

The predictions:

1. Fails. i8-dim/first1k lost 0.0075 to 0.0210.
2. Holds. i8-unit lost 0.0286 to 0.0497 at 1,536 dimensions.
3. Fails. On minilm-384 i8-global lost less than i8-dim (0.0033 against 0.0116 at ef 64); on
   bge-384 more, within noise.
4. Fails. i8-vec lost 0.010 to 0.013 more than i8-dim on ada, and less on openai3 and minilm.
5. Fails. cluster1k cost 0.006 to 0.032 more than first1k.
6. Fails, all three parts, on the good side. b1-sign at r=1 was 0.770 to 0.796 at 1,536
   dimensions and 0.553 to 0.636 at 384, and reached 0.95 by r=4 everywhere.
7. Partly. With the all and first1k calibrations RaBitQ was never below b1-sign, and above it
   wherever neither had reached its ceiling; with cluster1k it was below on 10 of 42.
8. Not tested at this power. From 99,000 to 306,000 vectors (the same 1,000 queries, paired)
   losses moved by under 0.008, every interval spanning zero, except cluster1k's, whose cluster is
   drawn around a different vector at each size and so is not a comparison.
9. Fails. 0.9405.

## The skeptic pass

Each claim was attacked by running it again under the assumption most likely to break it.

* **Builds.** No verdict changes with any one build of an arm against any one of half's;
  build-to-build differences were at most 0.0046, under half of every deciding margin.
* **No configuration decides it alone.** Dropping any one, or both ada configurations, lets no
  arm pass at r=1. The two best arms fail in opposite places: i8-dim/first1k passes only on ada,
  i8-vec fails only on ada and bge.
* **The graph is not the cause.** Stage 2's loss at r=1 and ef 128 is 0 to 0.006 smaller than
  stage 1's, without a graph, and the same queries lose in both.
* **What is lost is near-ties** (openai3-384, Euclidean distance). The neighbors lost are the
  true 9th and 10th, and the result returned in their place is a median 0.12% farther for
  i8-dim/first1k and 0.03% for i8-vec; counting any result within 0.5% of the 10th distance,
  recall is at least 0.998. Half precision's own graph misses at ef 64 are twice as far.
* **i8-dim fails on its calibration, not on eight bits.** The first 1,000 vectors' ranges clip
  0.2% of components. A clipped vector is shorter, so it looks nearer to every query: 93% of the
  results that displaced a neighbor had a clipped component, against 55% of all vectors. Rounding
  with the same steps and no clipping gives 0.9947 in stage 1, against 0.9783. Built in Chassis on
  openai3-384, after the results and on that one configuration: i8-dim/all lost 0.0036 and 0.0058
  at ef 64 and 128, and first1k with each range widened by a quarter on each side 0.0082 and
  0.0076. Both are within the bar.
* **i8-vec and i8-global fail on ada because of an offset.** Every ada-002 vector has about -0.64
  in dimension 194 and +0.20 in 954, which set its minimum and maximum, so its steps are 4.4 times
  as coarse as openai3's. Leaving the nine offset dimensions out of the range, or subtracting
  their mean, takes i8-vec's loss in stage 1 from 0.021 to 0.003 (one ada shard, 30,000 vectors).
  i8-unit's larger loss on ada has another cause: its neighbors are 2.7 times as close together.

## What it means

As pre-registered: eight bits alone, with these calibrations, lose too much, and eight bits
re-ranked keep both copies, smaller in memory and not on disk. But the losses at r=1 come from
clipping and from ada's offset, and both have a fix this run did not include: a margin on the
first vectors' ranges, and centering. That is not decided. It needs its own run, planned first,
on all seven configurations and with cluster1k.

One bit did better than predicted. In stage 1, RaBitQ with 40 candidates re-ranked reached 0.998
at 1,536 dimensions, and with 100, 0.981 to 0.991 at 384, at a sixteenth of half precision's
size. Whether that holds when a graph is searched by those distances was not measured.

## The follow-up

Written and committed before any of its results. The harness is `quant2.py`, run by
`.github/workflows/quant2.yml`: quant.py's stages, data, calibrations, re-ranking and loss,
with three schemes and a decision of its own.

**The question.** Run 1's losses at r=1 came from clipping and from ada's offset. With ranges that
leave room, or with the offset taken out, does eight bits alone come within 0.01 of half
precision, calibrated on vectors Chassis would have?

| scheme | range |
|---|---|
| i8-dim+10, i8-dim+25 | each dimension's range in the calibration set, widened by 10% or 25% of it on each side |
| i8-vec-c | the calibration set's mean subtracted, then each vector's own min and max: LVQ-8 (Aguerrebere et al., VLDB 2023); 8 bytes more a vector |

Each with first1k and cluster1k in stage 2, and with all too in stage 1. Stage 2 also builds half
precision, the baseline, and run 1's i8-dim/first1k, to compare with run 1 and not decided on.
The same eight configurations, data from the same caches, the same queries.

**Checks on the harness.** `quant2.py selftest`: a margin clips nothing on its own calibration, and
on a narrow one less than no margin does; errors are within half a step, and centering takes an offset all
vectors share out of each vector's range. Then quant.py's own selftest. In the run, run 1's
planted outcomes, both certain: half precision on SIFT exactly 1, shuffled at most 0.01.

**Predictions.**

1. i8-vec-c/first1k passes on all seven text configurations.
2. i8-vec-c/cluster1k passes on all seven, within 0.003 of first1k everywhere.
3. i8-dim+25/first1k passes everywhere but bge-384, where it loses 0.009 to 0.013 at ef 128.
4. i8-dim+10/first1k passes on all seven, and loses less than i8-dim+25/first1k on at least five.
5. i8-dim+10/cluster1k and i8-dim+25/cluster1k each fail on at least one configuration.
6. i8-dim/first1k is within 0.005 of its run 1 loss everywhere in stage 2, and within 0.001 of
   its run 1 recall in stage 1.

**The decision.** The same bar as run 1: loss at r=1 at most 0.01 and its interval's upper end at
most 0.02, at ef 64 and 128, on all seven text configurations. Arms: the six new ones.

* A scheme passes with first1k and with cluster1k: **int8 alone, calibrated on the first vectors
  added.** It goes on the roadmap, saying what was measured (recall) and what was not (a kernel,
  its speed, the file format, the vectors added before there are 1,000).
* An arm passes with first1k only: **only with a representative calibration**. It stays an idea,
  needing ranges that are recomputed, at compaction for instance.
* None passes: **not worth building with these calibrations**, recorded in ADR-0018's What Was
  Left Out; eight bits re-ranked and one bit stay ideas.

An arm that passes with a loss within 0.002 of the limit anywhere is reported as a close call.

**What was seen before this was written.** Run 1's results and its skeptic pass, which built
i8-dim+25/first1k on openai3-384 (0.0082 and 0.0076) and centered i8-vec on one ada shard in stage
1 (0.0023). A smoke test of this harness on 4,000 openai3-384 vectors from one shard, not results:
every new arm was within 0.01, i8-vec-c lost least (0.0036 to 0.0050), and at that size cluster1k
is a quarter of the data, so it says nothing about clusters. Nothing of minilm or bge beyond run
1's numbers. 25% is the margin tried after run 1; 10% was added with only the smoke test seen.
