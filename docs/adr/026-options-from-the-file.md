# ADR-0026: Options from the file

**Date:** 2026-10-21
**Status:** Accepted

## Context

An index's dimensions, metric, precision and `max_connections` are fixed when it is created, and
its file holds all four. Opening it again asked for them anyway, and failed with an
`OptionsMismatch` or `DimensionMismatch` error when one differed, so asking added only a way to
fail. A writer opened with the defaults couldn't open a cosine index at all. A program handed a
file it didn't create, such as an agent or the MCP server, had to guess four values to open it,
and a reader took the metric and precision from the file while a writer couldn't.

## Decision

### 1. What isn't given comes from the file

Opening an index takes each of the four from its file unless it's given, as a writer and as a
reader. One that is given is checked, with the error it had before. Creating an index takes the
defaults for any left out: Euclidean, full precision and 16. The dimensions have no default, so
creating an index needs them, and asking for an existing one's where there is no file is a
`NotFound` error that creates nothing.

### 2. How each language leaves one out

- **Rust**: `IndexOptions`' `metric`, `precision` and `max_connections` are `Option`s, `None` by
  default. Dimensions of 0 take the file's: an `impl Into<Option<u32>>` would refuse a bare
  integer literal, which every call passes.
- **C**: dimensions and `max_connections` of 0 take the file's, and `CHASSIS_FROM_FILE` does for a
  metric or precision, since 0 is already Euclidean and full precision there. A function without a
  metric or precision argument takes the file's. No function is added.
- **Python**: `dimensions`, `metric`, `precision` and `max_connections` default to `None`.

Values 0 and `CHASSIS_FROM_FILE` were invalid before, so every call that worked does the same,
except one: opening an index without naming a metric, precision or `max_connections` other than the
default, which failed, now opens it.

## Consequences

- `VectorIndex(path)` in Python, `open(path, 0, IndexOptions::default())` in Rust and
  `chassis_open(path, 0)` in C open any index with what it holds.
- Rust code that sets the three fields wraps them in `Some`: a breaking change.
- A typo'd path opened without dimensions is an error, not a new empty index. With dimensions, it
  still creates one, as before.
- Python's `options.max_connections` reports `None` when left out, since C has no call that
  returns it.
