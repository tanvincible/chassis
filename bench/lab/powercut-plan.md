# Power cuts on the real I/O path

Plan written 2026-10-21, before any run. Code: `chassis-core/examples/scale.rs` (`powercut`,
`state`) and the `lab_nosync` mutant on `lab/exp-scale` (e93cdcc, `main` at 12963b0). Workflow:
`.github/workflows/powercut.yml` on this branch. `replay-log` from xfstests at 22348af.

## Why

Power loss is only simulated, inside the process (`chassis-core/src/power_loss.rs`). The simulator
drops, reorders and tears unsynced writes, under the assumptions it states, but never runs the real
path: mapped pages written back by the kernel, `msync`, `fsync`, ext4's journal, the block device.

Linux's dm-log-writes target records every write the kernel sends to a disk, with its flush and
FUA flags, and xfstests' `replay-log` replays the disk up to any write. A replayed disk, mounted,
is the disk after a power cut there, for a drive that keeps acknowledged writes in order. It's how
Linux file systems are tested for crash consistency. It doesn't test a drive that reorders writes
between flushes, or lies about flushing; the simulator covers reordering.

ADR-0012 sets what a power cut may leave. The length, the vectors and the slots are the last
commit's, or the one being written. The graph is the commit's, except that between flushes the
kernel may write a changed list before the undo entry that saved it, and that list then keeps the
lost adds' links.

## What

Two jobs on `ubuntu-latest`: `main`, the harness as built, and `mutant`, built with
`--cfg lab_nosync`, which skips the sync of the data before the header that commits it. Each:

1. Loads `dm-log-writes`, from `linux-modules-extra` if the kernel lacks it, and builds
   `replay-log`.
2. A 256 MiB data image and a 2 GiB log image on loop devices, the log-writes target over them,
   ext4 on it (no discards, inode tables written at `mkfs`), mounted.
3. The kernel told to write dirty pages back within 10 ms (`vm.dirty_expire_centisecs` and
   `vm.dirty_writeback_centisecs` at 1), so uncommitted changes reach the disk between flushes.
4. `scale powercut`: an empty index, flushed, then 9 batches of 1,000 vectors of 128 dimensions,
   each followed by a 100 ms pause and a commit. Batches 3, 6 and 9 also delete every seventh
   vector of the batch before; batch 4 commits with `compact()`. After each commit, a mark in the
   log (`s0` to `s9`) and the length and vectors digest the writer saw.
5. **Marks**: a fresh image replayed to each mark, mounted, and opened by a writer (`scale state`):
   length, vectors digest, `check`'s findings, slots, and the graph's lists as sets.
6. **Flushes**: from `s0`, the image checked at every flush the log holds.
7. **Writes**: from `s0`, the image checked at about 1,500 points evenly spaced: every
   `ceil(writes / 1,500)`th write.

Each point is a copy of the replayed image, mounted (ext4 replays its journal), opened by a writer,
and `state`. Points before `s0`, while the file is created, aren't checked.

## Predictions

| # | Prediction |
| --- | --- |
| 1 | `dm-log-writes` loads on the runner, from `linux-modules-extra` if need be |
| 2 | `main`: at every mark, the writer's length and vectors digest, and nothing not finite |
| 3 | `main`: at every flush and write point, the length, vectors digest and slots of one mark, nothing not finite, no error |
| 4 | `main`: the graph's lists as sets equal that mark's at every flush point and at 90% or more of write points, and no point finds more unreachable vectors than its mark |
| 5 | `mutant`: at least one write point breaks prediction 3 |
| 6 | Each job within 120 minutes |

## Decision rule

- **1 fails**: the run stops there. A plan for a virtual machine with its own kernel follows.
- **5 fails**: the harness can't see the failure it exists for, and is fixed before 2 to 4 count.
- **2 or 3 fails, with 5 holding**: a bug. Reproduce it from the replayed image, fix it on `main`
  with a test, run again.
- **4 fails**: lists that differ only as ADR-0012 allows go to ADR-0012 with how often. A point
  with more unreachable vectors than its mark is a vector searches can't return until compaction:
  a plan to keep the undo entries ahead of the lists they save.
- **2 to 5 hold**: verification.md's limits say what this run covers, and it reruns after any
  change to the file format or storage.

## Results

Run 39588830118, 2026-10-21: Linux 6.17.0-1022-azure, ext4, about 3 minutes a job. Raw lines: the
`powercut-1-main` and `powercut-1-mutant` artifacts. A first attempt (39588392143) stopped before
any check, on a harness bug: `replay-log`'s option is `--num-entries`, not the `--number-entries`
its help prints.

`dm-log-writes` was in the kernel's own modules. `main` logged 1,159 writes and `mutant` 941; with
fewer than 1,500 after `s0`, every write was a point.

| Job | Flush points | Write points | Matched a commit | Graph as its mark's | More unreachable |
| --- | --- | --- | --- | --- | --- |
| `main` | 215 | 1,050 | all | 164 of 172 flush, 332 of 801 write | none |
| `mutant` | 128 | 829 | all but 8 flush and 19 write | 85 of 93 flush, 234 of 616 write | none |

The marks matched the writer at every one of the ten in both jobs. In both, 292 and 221 points
matched no mark but the commit `compact()` makes before it rebuilds: s4's length and vectors in
4,000 slots. The harness didn't mark that commit, so those points have no mark's graph to compare
with; the plan missed it.

The mutant's 27 points are a header that commits data the disk doesn't have: a vectors digest
that matches no commit, or a file shorter than its header says.

| # | Prediction | Held? |
| --- | --- | --- |
| 1 | `dm-log-writes` loads | yes: the kernel's own module |
| 2 | Marks as the writer saw them | yes, all ten |
| 3 | Every point a mark's length, vectors and slots, nothing not finite, no error | yes, counting the commit compaction makes first, which the plan missed |
| 4 | Graph as the mark's at every flush point and 90% of write points; no more unreachable | **no**: 8 flush points and 59% of write points differ; none more unreachable |
| 5 | The mutant breaks 3 | yes: 8 flush and 19 write points |
| 6 | Within 120 minutes | yes: 3 minutes |

**Decision:** no bug in what RUN 1 can see: every power cut opened to a commit, with its vectors,
and the mutant shows the harness sees one that doesn't. Prediction 4 failed, and the rule sends
lists that differ only as ADR-0012 allows to ADR-0012. RUN 1 can't tell that they do: a set
digest says a graph differs, not how. RUN 2 compares each list.

## RUN 2

Plan written 2026-10-21, before the run, after RUN 1. Code: `lab/exp-scale` 87bc990.

### Why

RUN 1 found the graph different from its commit's at 59% of the write points and 8 flush points,
with no vector harder to reach. ADR-0012 allows one difference: a list the kernel wrote before the
undo entry that saved it keeps the lost adds' links, and lacks the links they pushed out. A list
that gained a link to a committed slot it never had is not that, and would be a bug.

### What

As RUN 1, with two changes:

1. The workload flushes and marks (`s4a`) before `compact()`, so every commit has a mark.
2. Each mark's lists are saved; each point's lists are compared, list by list, with those of the
   commit it matches: the lists that differ, the links gained to slots the commit has, the links
   gained past them, and the links lost.

### Predictions

| # | Prediction |
| --- | --- |
| 1 | `main`: every point matches a mark's length, vectors and slots, nothing not finite, no error |
| 2 | `main`: no list gains a link to a slot its commit has |
| 3 | `main`: no point has more unreachable vectors than its mark |
| 4 | `main`: points whose graph differs have at most 5% of their lists differing |
| 5 | `mutant`: at least one point breaks prediction 1 |

### Decision rule

- **5 fails**: the harness is fixed before the rest count.
- **1, 2 or 3 fails, with 5 holding**: a bug. Reproduce it from the replayed image, fix it on
  `main` with a test, run again.
- **4 fails, the rest holding**: as below, with how many lists.
- **1 to 3 hold**: ADR-0012 says how often a power cut left lists as it allows, on the real I/O
  path with writeback every 10 ms, and verification.md says what this lab covers. It reruns after
  any change to the file format or storage.

### Results

Run 39590958680, 2026-10-21: Linux 6.17, ext4, about 2 minutes a job. Raw lines: the
`powercut-2-main` and `powercut-2-mutant` artifacts. Every write after `s0` was a point.

| Job | Flush points | Write points | Matched a commit | Graph as its commit's | Lists differing, where any do |
| --- | --- | --- | --- | --- | --- |
| `main` | 231 | 1,028 | all | 223 flush, 560 write | median 89% at flushes, 74% at writes; at most 99.3% |
| `mutant` | 137 | 973 | all but 10 flush and 21 write | 119 flush, 372 write | median 89% and 72% |

Every list that differed had gained only links past its commit's slots and lost others: 3.1
million gained and 1.4 million lost across `main`'s write points. No list gained a link to a slot
its commit has, and no point found more unreachable vectors than its commit.

| # | Prediction | Held? |
| --- | --- | --- |
| 1 | Every point a commit's length, vectors and slots | yes: 1,259 of 1,259 |
| 2 | No list gains a committed link | yes |
| 3 | No point more unreachable | yes |
| 4 | Where the graph differs, at most 5% of its lists | **no**: median 74 to 89%, at most 99.3% |
| 5 | The mutant breaks 1 | yes: 10 flush and 21 write points |

**Decision:** no bug. Power cuts left graphs only as ADR-0012 allows, but far more of them than
its words suggest: where a graph differed, the rollback had put back almost nothing, so most lists
the batch had touched kept its lost adds' links. That is the undo file missing from the disk, not
a few lists ahead of their entries: it is created anew after each commit, and ext4 makes a new
file durable at its next journal commit, while the lists are overwritten in place and reach the
disk as soon as they're written back. ADR-0012 and verification.md say so, with these numbers.
Keeping the undo file on disk ahead of the lists goes to IDEAS.md.
