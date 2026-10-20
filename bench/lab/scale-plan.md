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

Run 39443115926, 2026-10-20: AMD EPYC 9V74 (Zen 4), 4 vCPUs, 15 GB, Linux 6.17, the index on the
runner's root disk (ext4). 55 minutes in all. Raw lines: the `scale-1` artifact.

| Step | Found |
| --- | --- |
| Build | 10M in 1,192 s; a million took 52 s first and 149 s last, 148 to 149 s from 9M; flushes 0.1 to 2.8 s |
| File | 6,936,985,600 bytes, 6.3 GiB on disk |
| Memory | RssAnon peak 534 MB, 512 MB of it the harness's million vectors |
| Search | `ef` 64: recall 0.988, 8,859 queries a second in memory (2,902 the first pass); `ef` 256: 0.999, 3,015 |
| Open, check | reader and writer under 0.01 s; `check` 4.0 s: 1 unreachable, none not finite |
| In 2 GB | 5 queries a second at `ef` 64, cold or after `warm`, both with the uncapped digest; 1,000 adds in 478 s (2.1 a second); peak 2 GiB, no OOM kill |
| Churn | 100,000 deletes in 0.94 s; 99,000 adds in 66.6 s (1,490 a second); `check` 1 unreachable; recall 0.988 and 0.999 |
| Backup | 16.8 s; the same 6.3 GiB on disk as the index; the index's digest |
| Kill | killed mid-add of a million; a writer opened in 6.0 s to 10,000,000 vectors; `check` 1 unreachable |
| Compact | 1,109 s (0.93 of the build); max RSS 12.5 GB, both files mapped; 0 unreachable; recall 0.988 and 0.9945 |

| # | Prediction | Held? |
| --- | --- | --- |
| 1 | No error, panic or OOM kill; within six hours | yes: 55 minutes |
| 2 | 6.5 to 9 GB; last million at most twice the first | **no**: 6.9 GB, but 2.8 times |
| 3 | RssAnon under 2 GB | yes: 534 MB |
| 4 | Reader under 1 s, writer under 5 s, `check` under 5 minutes | yes |
| 5 | None not finite, at most 1,000 unreachable | yes: 1 |
| 6 | Recall at `ef` 256 of 0.9 or more | yes: 0.999, 0.9945 after compaction |
| 7 | Capped digests match, no OOM kill | yes |
| 8 | Capped searches 5 to 100 times slower | **no**: 1,800 times slower than in memory (5 against 8,859 a second) |
| 9 | Backup has the digest, under 3 minutes | yes: 16.8 s |
| 10 | After the kill: flushed length, open under 30 s, `check` passes | yes: 6.0 s |
| 11 | Compaction under 1.5 times the build, recall within 0.02 | yes: 0.93, 0.0045 |

**Decision:** no bug, so the envelope goes to the docs. Two predictions failed:

- Adds slow as the index grows, 2.8 times from the first million to the tenth, then level from 9M:
  more than log n, so likely the graph outgrowing the caches. Not followed up here.
- An index 3.5 times the memory it gets searches at 5 queries a second and adds 2 vectors a second
  on this runner's disk, whose latency wasn't measured. A laptop's NVMe would do better by an
  unmeasured amount. `warm` doesn't help there, as documented. Compaction, which builds the graph
  again, would take months at that rate: an index has to fit in memory to be compacted in useful
  time.

Not measured: the disk's latency, a real laptop, macOS, an index past 7 GB, half precision.

A recurring job on `main`: not now. At 55 minutes it would fit a weekly run, but it would put this
harness on `main` to recheck a result nothing has changed since; this lab is run again after any
change to the file format or storage instead (IDEAS.md). Docs: #136.

## RUN 2

Plan written 2026-10-21, before the run, after a review of RUN 1 and #136. Code:
`chassis-core/examples/scale.rs` on `lab/exp-scale` (d342450, `main` at 12963b0 plus the harness).

### Why

- **The crash step checked too little.** A length and `check` can pass after a partial undo. On a
  300,000-vector index on a Mac, a planted bug that left 2% of the saved lists unrestored passed
  `check` and left 1,000 searches' digest unchanged; only a digest of the graph caught it. The
  same run, unplanted, found the graph's lists back as sets, 0.26% of them in another order:
  `store_list` keeps an entry that is in both lists where it is, so a reader never misses it,
  and rollback goes through it. ADR-0012 promises the neighbors, not their order.
- **1,800 times slower is a number, not a model.** Faults per query and the disk's latency would
  predict another disk's speed.
- **Linux may read around each fault**, up to 128 KB, filling the 2 GB with pages no search asked
  for. `MADV_RANDOM` stops that; the core doesn't use it.
- **Adds slowed 2.8 times from the first million to the tenth.** One guess: page tables. The
  file's are about 1.4 MB at a million vectors and 13.5 MB at ten, past what the caches hold, so a
  TLB miss costs a trip to memory. Huge pages make them 512 times smaller.
- **What compaction needs.** It peaked at 12.5 GB with both files mapped: does it need the memory
  of one file or two?
- **Longer vectors.** Users have 768 to 1,536 dimensions, not 128, often at half precision; nothing
  has run past 7 GB.

### What

Three jobs on `ubuntu-latest`, each its own runner, in parallel.

**`main`**: RUN 1's steps at 10M, with these changes.

1. Before the build, `fio` reads 4 KB at random from a 4 GB file on the index's disk for 30 s,
   one at a time, bypassing the cache: the disk's latency.
2. The build also records the writer's page tables (`VmPTE`) after each million.
3. After the build and `check`, `graph` digests every slot's id and lists, in order and as sets.
4. Under the 2 GB cap: 200 queries at `ef` 64 four times, the page cache dropped before each,
   alternating no advice and `MADV_RANDOM` (A B A B). Each pass records major faults and bytes
   read from the disk (`/proc/self/io`). Then, uncapped, the same pair with 1,000 queries, the
   cache dropped before each: what the advice costs a first pass that fits.
5. Churn as before; no backup.
6. **Crash**: `graph` before the kill; after `verify`, `graph` again and 1,000 queries at `ef` 64,
   whose digest is compared with the churned index's. The undo file's size at the kill.
7. **Compaction under a cap**: a cgroup of 1.2 times the file, the cache dropped, `compact` with an
   hour's limit. Then `check` and searches, or, if it ran out of time, `graph` and `check` to
   show the index is as it was.

**`pages`**: two builds of 10M on one runner, the default first, then `huge_pages` on
(`SCALE_HUGE=1`): time per million, page tables, and the file's memory mapped on huge pages
(`FilePmdMapped`) at the end.

**`long`**: 3M vectors of 1,536 dimensions at half precision, 100,000 at a time (`SCALE_DIMS`,
`SCALE_HALF`, `SCALE_CHUNK`), the same generator: build, exact, search at `ef` 64 and 256,
`check`, `compact` uncapped on the 16 GB runner (two files of about 10 GB), `check` and search.

### Predictions

| # | Prediction |
| --- | --- |
| 1 | After the kill, the graph's set digest is the one from before it, and so is the search digest. The order digest differs. The undo file isn't empty. |
| 2 | Under the cap, queries a second are within 2 times of 1 ÷ (major faults per query × `fio`'s mean latency + the uncapped time per query). |
| 3 | With no advice, bytes read per major fault are 32 KB or more; with `MADV_RANDOM`, 8 KB or less. |
| 4 | `MADV_RANDOM` makes capped searches at least 1.5 times faster, in both pairs. |
| 5 | `MADV_RANDOM` makes the uncapped first pass at least 1.5 times slower. |
| 6 | Each default build's last million takes 2.3 to 3.3 times its first. Page tables at 10M are 13.5 MB, give or take 30%. |
| 7 | With `huge_pages`, at least half the file is on huge pages at 10M, page tables are under 2 MB, the last million takes at most 2 times the first, and the build at most 0.85 times the default's on the same runner. |
| 8 | Compaction in 1.2 times the file's memory finishes in under 1.5 times RUN 1's 1,109 s, with no OOM kill; recall at `ef` 256 within 0.02 of the churned index's. |
| 9 | `long`: a file of 9.5 to 11 GB; recall@10 at `ef` 256 of 0.95 or more; `check` finds nothing not finite and at most 300 unreachable; compaction finishes in under 2 times the build. |

### Decision rule

- A set digest that differs after the kill, an error, a panic, an OOM kill, or a `check` finding
  past prediction 9 is a bug: find it, fix it on `main` with a test, run again.
- A search digest that differs while the set digest matches is the order: ADR-0012 and the docs
  then say a rollback can change results in rare ties.
- If prediction 2 holds, the formula and the faults per query go to the docs, for estimating
  other disks. If not, the gap is reported unexplained.
- If predictions 4 and 5 both hold, the advice is a trade, and a plan for when to give it
  (an index larger than the memory it has) comes before any change on `main`. If 4 holds and 5
  doesn't, a plan to advise random access on `main`. If 4 fails, it goes to IDEAS.md as tried.
- If prediction 7 holds, the slowdown is the page tables, and asking for huge pages while
  building joins the plan for huge pages at open. If the huge-page build slows as much, it isn't.
- Prediction 8's answer, either way, and the bytes per vector from `long` go to a Limits page.
