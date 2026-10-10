# Errors

Every error from Chassis says what was wrong, with the value that was wrong, and on a line
starting `help:` what to do instead:

```text
The index at vectors.chassis holds vectors of 768 dimensions, not 1536
help: open it with 768 dimensions, or create a new index at another path
```

Each also has a kind, the same in every language, for a program to act on: `Error::kind()` in
Rust, `chassis_last_error_code()` in C, and the exception's class in Python. Each Python class is
also the built-in exception it is a kind of, so `except ValueError` catches a bad argument
([ADR-0020](../adr/020-errors.md)).

| Kind | Rust `ErrorKind` | C code | Python exception |
| --- | --- | --- | --- |
| [Invalid argument](#invalid-argument) | `InvalidArgument` | `CHASSIS_ERROR_INVALID_ARGUMENT` (1) | `InvalidArgumentError` (`ValueError`), or `TypeError` |
| [Dimension mismatch](#dimension-mismatch) | `DimensionMismatch` | `CHASSIS_ERROR_DIMENSION_MISMATCH` (2) | `DimensionMismatchError` (`ValueError`) |
| [Options mismatch](#options-mismatch) | `OptionsMismatch` | `CHASSIS_ERROR_OPTIONS_MISMATCH` (3) | `OptionsMismatchError` (`ValueError`) |
| [Not found](#not-found) | `NotFound` | `CHASSIS_ERROR_NOT_FOUND` (4) | `IndexNotFoundError` (`FileNotFoundError`) |
| [Not an index](#not-an-index) | `NotAnIndex` | `CHASSIS_ERROR_NOT_AN_INDEX` (5) | `NotAnIndexError` (`ValueError`) |
| [Id in use](#id-in-use) | `IdInUse` | `CHASSIS_ERROR_ID_IN_USE` (6) | `IdInUseError` (`ValueError`) |
| [Locked](#locked) | `Locked` | `CHASSIS_ERROR_LOCKED` (7) | `IndexLockedError` |
| [Read-only](#read-only) | `ReadOnly` | `CHASSIS_ERROR_READ_ONLY` (8) | `ReadOnlyError` |
| [Corrupt](#corrupt) | `Corrupt` | `CHASSIS_ERROR_CORRUPT` (9) | `CorruptIndexError` |
| [Full](#full) | `Full` | `CHASSIS_ERROR_FULL` (10) | `IndexFullError` |
| [Input or output](#input-or-output) | `Io` | `CHASSIS_ERROR_IO` (11) | `InvalidPathError` (`OSError`) |
| Other | `Other` | `CHASSIS_ERROR_OTHER` (12) | `ChassisError` |

Every Python class is a `ChassisError`. In C, a call that succeeds sets the code back to
`CHASSIS_OK` (0).

## Invalid argument

An argument is out of range or malformed. The message names it and its value.

- **Dimensions** are 1 to 4,096: the length of the vectors the index will hold.
- **`max_connections`** is 2 to 32,767; 16 suits most indexes.
- **`k`** is at least 1. In Python it is an `int`; `5.0` is a `TypeError`.
- **A vector or query with NaN or infinity** is refused, at every metric and precision, with
  the component's position. NaN usually comes from dividing by zero, as in normalizing a zero
  vector.
- **A zero vector** in a cosine index has no direction, so it can't be compared by angle.
- **A component of 65,520 or more** in magnitude doesn't fit half precision, which holds up to
  65,504. Scale the vectors down, or use full precision.
- **Ids** are integers from 0 to 2⁶⁴ − 2. In Python a string id is a `TypeError`: keep a dict
  from your own keys to integer ids.
- **A path that is a directory**: give the path of a file in it.
- **In Python**, an unknown option is a `TypeError` that names the option it was likely meant
  to be, including other libraries' names for them (`ef` is `ef_search`, `M` is
  `max_connections`). `metric` takes any case, and `"l2"` for `"euclidean"`; `precision` takes
  `"float16"` and `"f16"` for `"half"`. There is no inner-product metric: for vectors of length
  1, cosine orders results the same way.

## Dimension mismatch

A vector or query has another number of components than the index's vectors, or the file holds
vectors of other dimensions than it was opened with. The usual cause is a vector made by another
embedding model than the index's.

- **Python, a 2-D array** where one vector belongs: `vector[0]` or `vector.ravel()` for one row,
  `add_batch` to add several, and one `search` call per query.
- **A batch** is its vectors back to back, `dimensions` floats each: a `(count, dimensions)`
  array in Python, `dim` equal to the index's in C.

## Options mismatch

The file was created with another metric or precision than the one asked for. Both are fixed
when the index is created, and a writer has to name the same ones again. A reader takes them from
the file. Open the index with the ones it was created with, or create a new index at another path.

## Not found

There is no index at the path a reader was given, or the directory of a path a writer was given
doesn't exist. A reader opens an index that exists: create it by opening it as a writer first.

## Not an index

The file at the path is not a Chassis index. Chassis never writes to such a file. Check the path;
to create a new index, give a path where no file exists yet.

## Id in use

`add_with_id` (or `ids` in Python) gave an id that a live vector already has. Delete that vector
first to replace it, or use another id. Within one batch, ids have to differ.

## Locked

Another writer has the index open, in this process or another. An index has one writer at a
time. Close the other writer first, or open a reader (`IndexReader`, `chassis_open_reader`,
`read_only=True`), which searches while the writer writes.

## Read-only

A reader was asked to add, delete, flush or compact. Open the index as a writer to change it.

## Corrupt

The file is damaged: its header or regions don't add up. Restore it from a backup, or rebuild it
from its vectors. A crash or power loss doesn't do this (the index recovers to its last
`flush()`); a disk error or another program writing the file can.

## Full

The index can hold no more: 4,294,967,295 vectors, or no room for another region in its tables.
Compact it to drop deleted vectors, or split the vectors across indexes.

## Input or output

The operating system refused: no permission, a read-only or full disk, a failed read or write.
The message ends with the system's own words. An earlier flush that failed makes later ones
refuse too, since what reached the disk is unknown: open the index again.
