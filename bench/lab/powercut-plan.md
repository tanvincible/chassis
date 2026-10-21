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
