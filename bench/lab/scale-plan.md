# Scale: Chassis at ten million vectors

Plan written 2026-10-20, before any run. Code: `chassis-core/examples/scale.rs` on
`lab/exp-scale` (0905006). Workflow: `.github/workflows/scale.yml` on this branch.

## Why

Nothing had run past about 1 GB or a million vectors. A review: being the SQLite of something
means being boring at 50 GB on someone's laptop. This run covers what can break with size alone:
offsets past 4 GiB, ten times the vectors, and a file several times larger than the memory it
gets.

## What

One GitHub x86 runner (4 vCPUs, 16 GB). Ten million synthetic vectors of 128 dimensions, default
options (`M` 16, `ef_construction` 200, full precision): about 7 GB. The vectors are points around
1,000 centers in 16 dimensions, projected to 128 with noise, computed from their ids, so the exact
neighbors come from the generator and never from the file. Queries are ids from 2^40.

Steps, in order, each a process of its own:

1. **Build**: ids 0 to 10M, a million at a time, each flushed. Time per million, flush time,
   memory other than the file's pages (RssAnon), file size.
2. **Exact**: the 10 nearest of the first 200 queries, by brute force.
3. **Search**: a reader, `ef` 64 and 256, 1,000 queries, two passes: queries a second,
   recall@10 on the 200, and a digest of every id and distance bit. A 200-query pass at `ef` 64
   for the capped step to compare with.
4. **Check**: a writer's open time, `check`'s time and findings.
5. **Capped**: page cache dropped, then in a cgroup with `memory.max` 2 GB and no swap, a file
   about 3.5 times that (50 GB on a 16 GB laptop is 3.1): 200 queries at `ef` 64 with no warm,
   then with `warm`; then 1,000 adds, flushed. OOM kills and peak memory from the cgroup.
6. **Churn**: delete every hundredth id (100,000), add ids 10M+1,000 to 10M+100,000 in batches of
   10,000, each flushed; `check`; exact and search again over the live set.
7. **Backup**: a writer's `backup`; its time, apparent and allocated size; the backup searched.
8. **Crash**: a writer adds a million more without flushing and is killed with SIGKILL 20 s after
   it starts adding; a writer then opens the file: its length, open time and `check`.
9. **Compact**: time, `check`, search again.

## Predictions

| # | Prediction |
| --- | --- |
| 1 | Every step finishes without an error, panic or OOM kill, within the job's six hours. |
| 2 | The file is 6.5 to 9 GB; its last million take at most twice as long to add as its first. |
| 3 | RssAnon peaks below 2 GB in the build, of which 0.5 GB is the harness's million vectors. |
| 4 | A reader opens in under a second, a writer in under 5 s; `check` takes under 5 minutes. |
| 5 | `check` finds no vector that isn't finite and at most 0.01% (1,000) unreachable. |
| 6 | Recall@10 at `ef` 256 is 0.9 or more before churn and after compaction. |
| 7 | Under the cap, both 200-query passes give the uncapped digest; no OOM kill, with `warm` too. |
| 8 | Under the cap, searches are 5 to 100 times slower than in memory, and adds slower still. |
| 9 | The backup gives the digest of the index it came from, and takes under 3 minutes. |
| 10 | After the kill, the writer opens to the flushed length in under 30 s and `check` passes. |
| 11 | Compaction takes under 1.5 times the build, and recall stays within 0.02. |

## Decision rule

- An error, panic, OOM kill, wrong length, digest that should match and doesn't, or a `check`
  finding past prediction 5 is a bug: find the cause, fix it on `main` with a test, and run again.
- The crash step counts only if the kill landed mid-add (the harness prints a line if the add
  finished first).
- If every step passes, the times, sizes and memory go to the docs as the measured envelope, and
  the predictions that failed are reported as found. A recurring job on `main` is decided after,
  by how long it takes.

## Results

Not yet run.
