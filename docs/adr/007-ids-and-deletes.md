# ADR-0007: Caller Ids and Deletes

**Date:** 2026-10-04
**Status:** Accepted

## Context

Chassis identified vectors only by insertion order and could not remove them. Applications built on
it need to store their own ids (a document id, a row id) and to delete or replace vectors as their
data changes.

Two constraints shaped the design:

* Files written before this change must keep working, with no migration.
* A crash must leave the index exactly as of the last completed `flush()`. Replacing a vector is a
  delete plus an add, and a crash must never apply one without the other.

## Decision

### Ids

Every node record already starts with an 8-byte `node_id`, which old files set to the node's slot
(its position in the graph zone). That field now holds the caller's id, so existing files read as
"id equals slot" without any rewrite. Records are addressed by slot, which is kept in memory
(`NodeRecord::slot`) and never stored.

* `add(vector)` assigns one past the largest id ever stored, so it never reuses an id.
* `add_with_id(id, vector)` fails if a live vector already has `id`. `u64::MAX` is reserved.
* While every id equals its slot (the graph header's custom-ids flag is clear), lookups need no
  table. Once an id differs, the first lookup in a process scans the node headers once to build an
  id-to-slot table in memory.

### Deletes

A delete is recorded in memory and only written by `flush()`. Search hides it immediately: deleted
nodes are still traversed, so the graph stays connected, but never returned.

The node header's unused padding holds `deleted_epoch` (0 means live). The graph header holds the
epoch of the last flush that committed deletes, a dirty flag and the deleted count. A flush with
pending deletes:

1. sets the dirty flag in the graph header and fsyncs, along with every record and vector written
   since the last flush;
2. writes `deleted_epoch = epoch + 1` into each deleted node's header and fsyncs;
3. writes the graph header (new node count, new epoch, deleted count, dirty flag cleared) and fsyncs.

Step 3 commits the flush's adds and deletes together. If a crash interrupts steps 1–3, the dirty
flag is still set at the next open, which clears every mark newer than the header's epoch. Adds are
rolled back by `node_count` as before (ADR-0005).

### Format version

Files that use either feature can't be read correctly by older releases: 0.6.x would misread ids as
slots and ignore deletes, and its open path silently resets a graph it can't read. So every flush
stamps the main file header with format version 2, which 0.6.x refuses with an error. Version 1
files open unchanged and become version 2 at their first flush.

## Consequences

### Positive

* No migration: version 1 files keep their ids and gain both features.
* Each flush is all-or-nothing for adds and deletes, including a delete and re-add of the same id.
  `chassis-core/tests/crash_tests.rs` checks this by killing a writer mid-batch.

### Negative

* An index that uses custom ids scans every node header once per process, on its first lookup.
* A flush with deletes costs three fsyncs instead of two.
* Deleted vectors keep their space, and search still walks through them. An index with many
  deletes gets larger and slower until it is rebuilt; compaction is future work.
* Releases before 0.7 can't open a file once a newer release has flushed it.
