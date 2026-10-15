# ADR-0023: An Application's Number in the Commit, and Format 3 Frozen

**Date:** 2026-10-15
**Status:** Accepted

## Context

Apps keep their rows in their own database, usually SQLite, and their vectors in Chassis, under the
same ids ([With a database](../src/guide/with-a-database.md)). The two commit separately, so a
crash between the index's flush and the database's commit leaves them disagreeing. The guide's
`mend()` finds that by comparing every id in both, and nothing told an app whether it needed to,
so it had to on every start. Measured on 2026-10-15 on an Apple M5 in Low Power Mode, through
Python, with notes of 384 dimensions in SQLite, the median of 15 runs with nothing to mend:

| Notes | `mend()` |
| --- | --- |
| 20,000 | 45 ms |
| 100,000 | 213 ms |

SQLite solves the same problem for its own applications with `PRAGMA user_version`: a number in
the database header that the application sets and the database commits.

Meanwhile the format changed in v0.7.0 and the crate was renamed in v0.8.0, and 8-bit vectors on
the roadmap will need another change. The manifesto promises that data Chassis writes stays
readable; ADR-0008 proposed freezing format 3 for that, without saying how a later change gets in.

## Decision

### 1. `user_version`

A u64 in the header, after the next id, that the application sets and the next commit writes with
the data. In Rust `VectorIndex::set_user_version` and `user_version`, and
`IndexReader::user_version` from its snapshot; in C `chassis_set_user_version` and
`chassis_user_version`; in Python the `user_version` property. It is 0 until set. Compaction
carries it over. A delete's intent header copies the committed one, so a crash in a flush with
deletes rolls the number back with them.

The guide keeps a write counter in a one-row table, raises it in each write's transaction and sets
the index's `user_version` to it before the flush. On open, the two numbers differ only after a
crash between a flush and its commit, the one case that leaves anything to mend. Apps often use
SQLite's own `PRAGMA user_version` for schema versions, so the guide leaves it alone.

### 2. No new format version for it

The field follows the next id as the next id followed the tables: headers written before it have no
such bytes and read 0, and releases before it ignore bytes after the fields they know. A release
from before it that writes the file drops the number, which then reads 0: an app sees the numbers
differ and mends, which is always correct, provided it never uses 0 itself.

### 3. Format 3 is frozen

Format 3, and format 4, which is format 3 with half-precision vectors, are frozen: every later
release reads every file written in them. A change gets in one of two ways:

* **A field older releases can ignore** goes after the last field they know, as the next id and
  `user_version` did. If an older writer's dropping it would be unsafe, the change also raises the
  write version, so older releases open the file only to read it.
* **Anything older releases would misread**, such as another way of storing vectors, raises the read
  version, so they refuse the file before reading it, with a message saying a newer release reads
  it. 8-bit vectors will be read version 5, as half precision made 4.

Files that releases wrote are checked in, and CI opens each with the current code.

## Results

The same notes, with the number compared on open instead of mending:

| Notes | Comparing the numbers |
| --- | --- |
| 20,000 | 0.007 ms |
| 100,000 | 0.008 ms |

The power-loss simulator (`chassis-core/src/power_loss.rs`) now sets a new number before every
flush and requires every crash image, and every reader of one, to show the number of the flush
whose vectors it holds. Making the setter do nothing fails it.

## Consequences

### Positive

* An app mends only after the one crash that needs it, instead of reading every id on every start.
* A reader can tell which of the app's writes a snapshot holds.
* The format has a rule for what may change and how older releases react.

### Negative

* A release from before `user_version` that writes the file drops it, so the app mends once.
* 0 has to stay "never set": an app that counts from 0 would miss the crash before its first write
  is committed.
