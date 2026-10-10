"""The exceptions Chassis raises.

Each message says what was wrong, with the value, and on a line starting
``help:`` what to do instead. Each class is also the built-in exception it is a
kind of, so ``except ValueError`` catches a bad argument and
``except FileNotFoundError`` a missing index. ``code`` is the C API's error
code for the same error.
"""


class ChassisError(Exception):
    """Base class of every Chassis error."""

    code = 12


class InvalidArgumentError(ChassisError, ValueError):
    """An argument is out of range or malformed: dimensions, an option, an
    id, a vector with a component that isn't a finite number."""

    code = 1


class DimensionMismatchError(ChassisError, ValueError):
    """A vector or query has another number of components than the index's
    vectors, or the file holds vectors of other dimensions than asked for."""

    code = 2


class OptionsMismatchError(ChassisError, ValueError):
    """The file was created with another metric or precision than the one
    asked for."""

    code = 3


class IndexNotFoundError(ChassisError, FileNotFoundError):
    """There is no index at the path, or its directory doesn't exist."""

    code = 4


class NotAnIndexError(ChassisError, ValueError):
    """The file is not a Chassis index."""

    code = 5


class IdInUseError(ChassisError, ValueError):
    """The id is already in use."""

    code = 6


class IndexLockedError(ChassisError):
    """Another writer has the index open."""

    code = 7


class ReadOnlyError(ChassisError):
    """The index was opened with ``read_only=True``, which only searches."""

    code = 8


class CorruptIndexError(ChassisError):
    """The file is damaged."""

    code = 9


class IndexFullError(ChassisError):
    """The index can hold no more."""

    code = 10


class InvalidPathError(ChassisError, OSError):
    """The operating system refused: permissions, a full disk, a failed read
    or write."""

    code = 11


class NullPointerError(ChassisError):
    """The library failed without saying why."""

    code = 12


_BY_CODE = {
    cls.code: cls
    for cls in (
        InvalidArgumentError,
        DimensionMismatchError,
        OptionsMismatchError,
        IndexNotFoundError,
        NotAnIndexError,
        IdInUseError,
        IndexLockedError,
        ReadOnlyError,
        CorruptIndexError,
        IndexFullError,
        InvalidPathError,
    )
}


def from_code(code: int, message: str) -> ChassisError:
    """The exception for a C API error code."""
    return _BY_CODE.get(code, ChassisError)(message)
