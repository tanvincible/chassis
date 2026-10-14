# Roadmap

What is planned for Chassis and not yet built. Chassis is for local semantic search in one file;
the introduction says [what it will not grow into](docs/src/introduction.md#what-chassis-does-not-do).

Only what has been measured and found worth building is listed here. Ideas that have not been
validated are kept in `IDEAS.md` on the [`ideas`](https://github.com/tanvincible/chassis/blob/ideas/IDEAS.md)
branch, and move here when they have been.

## Planned

### Eight bits a component

Half precision halved the file with the same recall ([ADR-0018](docs/adr/018-half-precision.md)).
Eight bits will halve the vectors again: 1,544 bytes each at 1,536 dimensions, against 3,072.

Each vector will be stored as its difference from a center, in 256 steps between that
difference's own smallest and largest component: LVQ (Aguerrebere et al., VLDB 2023). The center
is the mean of the first vectors added, and then stays fixed. With its own range no vector is
clipped, and the center takes out what every vector shares, such as ada-002's constant offset in
two dimensions.

Measured on 2026-10-11 by rounding the vectors and building Chassis on the result, on seven sets
of 99,000 or 306,000 text embeddings from four models, 384 to 1,536 dimensions, with 1,000 queries
each. The bar was a loss of recall@10 against half precision of at most 0.01 on every one, with
nothing re-ranked. The largest losses:

| Center from | ef 64 | ef 128 | Every vector ranked |
| --- | --- | --- | --- |
| The first 1,000 vectors | 0.004 | 0.005 | 0.006 |
| 1,000 vectors on one topic | 0.005 | 0.006 | 0.008 |

Ranges per dimension from the first vectors failed the bar when those vectors were on one topic:
they lost up to 0.041, and up to 0.024 with a margin.

Not measured yet: the kernel and its speed, the file format, and the vectors added before there is
a center. One vector was too few to center on, within 0.002 of the bar; ten were enough. The plan,
the results and the checks on them are in
[`bench/lab/quant-plan.md`](https://github.com/tanvincible/chassis/blob/lab/x86/bench/lab/quant-plan.md)
on the `lab/x86` branch.
