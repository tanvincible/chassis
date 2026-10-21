# Adds: why they slow as an index grows

Plan written 2026-10-21, before any run. Code: `chassis-core/examples/scale.rs` and the core on
`lab/exp-scale` (e93cdcc: `main` at 12963b0, the scale harness, and counters compiled only with
`--cfg lab`). Workflow: `.github/workflows/adds.yml` on this branch.

## Why

In scale RUNs 1 and 2, a million adds took 2.87 to 3.03 times as long into nine million vectors as
into an empty index, and the time levelled from 9M. A search's cost grows with the log of the size,
1.17 times from 1M to 10M. Huge pages cut the page tables to a third and left the slowdown as it
was (scale RUN 2), so it isn't the page tables. Two explanations are left:

- **More work:** the search for each add's neighbors computes more distances as the graph grows.
- **Slower work:** about as many distances, each waiting longer on memory.

Counting the distances an add computes tells them apart. A profile says where the time goes.

## What

The scale build: 10M vectors of 128 dimensions, the default options, a million at a time, each
flushed. Two jobs, each on its own `ubuntu-latest` runner:

- **`counted`** (`--cfg lab --cfg lab_count`): after each million, the distances computed
  searching (the greedy descent and `search_layer`), the nodes `search_layer` expanded, and the
  distances computed choosing neighbors (`distances_from` and `far_from_each_other`). Each search
  adds its counts once, so the threads don't contend over them. This job's times aren't used.
- **`profiled`** (no counters: the core as on `main`): the time of each million. `perf record -e
  cpu-clock` for 20 s from 8 s into the build, inside the first million, and for 40 s from the
  start of the tenth, reported by symbol. Then `perf stat` with cycles, instructions, dTLB-load
  and LLC-load misses for 10 s in the first million and 20 s in the tenth, if the runner has them.

Ratios below are the tenth million's over the first's, per add.

## Predictions

| # | Prediction |
| --- | --- |
| 1 | Distances searching: 1.1 to 1.6 |
| 2 | Distances choosing neighbors: 0.8 to 1.3 |
| 3 | Nodes expanded: 1.0 to 1.5 |
| 4 | Time, in `profiled`: 2.5 to 3.4 |
| 5 | Time per distance (4 over 1 and 2 together): 1.8 or more |
| 6 | The distance kernels' share of the profile changes by less than 10 points between the windows |

## Decision rule

- **More work**: if 1 and 2 together grow 2 times or more, the search does more work as the graph
  grows. Why (hops, the batch's links) goes to a plan before any change.
- **Slower work**: if they grow 1.6 times or less and prediction 5 holds, each distance costs
  more. A plan follows only for a cause the profile shows growing by 10 points or more, such as
  the visited filter or a heap; otherwise IDEAS.md, with the numbers.
- **Between**: both, reported as such, in IDEAS.md.
- Either way, the answer replaces "not looked into" in IDEAS.md's entry on adds.
