# ADR-0020: Errors That Say What to Do

**Date:** 2026-10-10
**Status:** Proposed

## Context

Chassis is used from Rust, C and Python, more and more by coding agents, which read an error
and try again. An error that says what went wrong, with the value, and what to do instead gets
the next attempt right; one that doesn't sends the reader to search for causes. rustc is the model:
each of its errors shows the values involved and, after `help:`, a fix.

Seventy common mistakes were run through the Python package before this change. Most raised
something, but:

- The C layer passed on only the outermost message, so a missing directory, a path that is a
  directory, a denied permission and a reader opening a file that doesn't exist all read
  `Failed to open chassis file: <path>`.
- Messages named another layer's things: a Python reader was told it came from
  `chassis_open_reader`, and a precision mismatch said "Half" and "Full", Rust's names.
- A second writer in the same process was told the file was open "by another process".
- NaN and infinite vectors and queries were taken, and a NaN query returned NaN distances.
- A batch of queries passed to `search` was reported as a query of three dimensions, a negative
  dimension count as 4,294,967,293, and options given as a dict as an `AttributeError`.
- Python chose its exception class by searching the message for words such as "dimension".

## Decision

### 1. Every error has a kind

`VectorIndex` and `IndexReader` return `chassis_core::Error`, whose `kind()` is one of:
invalid argument, dimension mismatch, options mismatch, not found, not an index, id in use,
locked, read-only, corrupt, full, input or output, other. Inside, the core keeps `anyhow`; an
error is raised with its kind, and the conversion at the API finds it however much context was
added around it. `Error` converts into `anyhow::Error` and `Box<dyn Error>`, so code that uses
`?` keeps compiling; code that returned `VectorIndex::open(...)` as an `anyhow::Result` without
`?` needs one.

C has `chassis_last_error_code()`, one of the `CHASSIS_ERROR_` constants, beside the message.
Python's exception class is chosen by that code, and each class is also the built-in exception it
is a kind of: a dimension mismatch is a `ValueError`, a missing index a `FileNotFoundError`.

### 2. Every message says what was wrong, with the value, and what to do

A message is one line saying what was wrong and with what values, then a line starting
`help: ` with what to do instead. It is in words that fit every language, so the C layer and the
bindings pass it on as it is. The full chain of causes comes with it: the operating system's
words end a message about a file it refused.

### 3. Vectors and queries have to be finite

A component that is NaN or infinite gives no distance to anything, and usually comes from a bug
upstream, such as normalizing a zero vector. It is refused, with its position, at every metric
and precision.

### 4. Python checks what it can before calling into C

Types and shapes are checked in Python, where the message can use Python's terms: a 2-D array
where one vector belongs, a float for `k`, a string id, a function for `allowed`, an unknown
option (with the name it was likely meant to be, including other libraries' names for it).
`metric` and `precision` take any case and other libraries' names for them (`"l2"`,
`"float16"`), and options may be a dict.

### 5. A reference page for every kind

[Errors](../guide/errors.md) lists each kind with its causes and fixes, and its name in each
language: Chassis's `rustc --explain`.

## Consequences

### Positive

- Of the seventy mistakes, every one that should fail now raises the exception of its kind, with
  the value and a `help:` line; the Rust, C and Python tests pin them.
- A program can act on the kind instead of matching words in a message.

### Negative

- Rust code that returns `VectorIndex::open(...)` directly as an `anyhow::Result`, without `?`,
  no longer compiles.
- A NaN or infinite vector that was accepted before is now an error. An index holding one from
  before still opens and searches.
- Messages are longer.
