# Verification

How Chassis is tested, so you can judge how much to trust the file you keep your vectors in: what
each kind of test shows, how often it runs, and what none of them covers yet.
[Testing](../dev/testing.md) has the commands.

## Crashes

**Killed processes.** `crash_tests.rs` starts a writer in a child process and kills it at random
points, including during batch builds and compactions, then reopens the file. Every reopen has to
show exactly the last flush: no add or delete after it, none before it lost.

**Power loss.** A process that is killed keeps the page cache, so it can't show what a power cut
does. `power_loss.rs` takes the file's bytes at every fsync of every operation and builds the files
a power cut could leave, with each 512-byte sector as last synced, as last written, or torn between
the two. Each one has to reopen to the last flush or the one in progress, as a reader and as a
writer, and keep working after one more flush. On 2026-10-15 that was 210 crash points, 160 of them
inside the recovery a reopen runs, and 520 files. A model checker run while the protocol was
designed showed why every condition is there: "every reopen shows exactly one committed state" on
its own accepted 6 of 8 broken protocols
([ADR-0008](https://github.com/tanvincible/chassis/blob/main/docs/adr/008-format-v3-and-multi-process-readers.md)).

**The application's number.** Each simulated flush sets a new `user_version`, and every file the
simulator builds has to hold the number of the flush whose vectors it holds
([ADR-0023](https://github.com/tanvincible/chassis/blob/main/docs/adr/023-user-version.md)).

## Histories

`model_tests.rs` runs random sequences of every write: adds, batches, deletes, flushes,
compactions, and reopens without a flush. After each step it checks the index against a model of
what it should hold: every result is a live vector at its true distance, lookups agree, readers see
only what was flushed, and no search loses its way through the graph. CI runs 36 histories; run
2,000 with `CHASSIS_MODEL_SEEDS`.

## Readers

`reader_tests.rs` searches while a writer adds, deletes and flushes, in the same process and from
another one, and through a writer that crashes and restarts.
Before format 3 was accepted, two reader processes searched 587,200 times while a writer worked:
no errors, no result outside its snapshot, and no recall lost to the writer's changes
(ADR-0008).

## Memory

[Miri](https://github.com/rust-lang/miri) runs the tests that need no file on every pull request:
the header, node records, the search loop's parts, distances, half precision, and readers walking a
neighbor list while the writer rewrites it. It reports undefined behavior, such as a word read by
one thread and written by another other than atomically, which it found twice while format 3 was
designed. Miri can't map a file, so the code that reads the file through a mapping is covered by
the other tests instead. Every `unsafe` block in the core says why it is sound, and clippy refuses
one that doesn't.

## Damaged files

An embedded file meets bit rot, partial copies and sync tools. Fuzzing (`fuzz/`) opens files from
the compatibility set with bytes changed or cut off, and any bytes at all, as a reader and as a
writer, then searches and changes them: every outcome has to be an error, never a panic or a crash.
It runs a minute per target on each pull request and half an hour each night. It found that a file
cut off after its header crashed the process on its first search; that is now an error.

## Old files

`compat_tests.rs` opens indexes written by earlier releases, from v0.6.3's format 1 to v0.9.1, with
deletes, custom ids, cosine, half precision and a compaction, and one of 1,500 vectors, past the
first segment. Each has to show what the release that wrote it showed, and take one more vector
([ADR-0023](https://github.com/tanvincible/chassis/blob/main/docs/adr/023-user-version.md)).

## Platforms

Every pull request runs the tests on Linux (x86-64 and ARM64), macOS and Windows, with the newest
Rust and with 1.89, the oldest supported; the Python package on Python 3.11 with numpy 1.26 and on
3.14; and clippy with warnings as errors.

## Limits

- No test has cut the power to real hardware. The simulator stands in for it, under the
  assumptions `power_loss.rs` states: a write never tears inside an aligned 8-byte word.
- Readers were not tested during a compaction, nor with exFAT on Windows (ADR-0008).
- A full disk while a write first touches a page of the file can end the process (SIGBUS) instead
  of returning an error: the file grows without reserving its space.
