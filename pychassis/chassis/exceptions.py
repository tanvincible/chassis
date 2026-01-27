"""Exception classes for PyChassis."""


class ChassisError(Exception):
    """Base exception for all Chassis errors."""

    pass


class DimensionMismatchError(ChassisError):
    """Raised when vector dimensions don't match index dimensions."""

    pass


class InvalidPathError(ChassisError):
    """Raised when the path is invalid or inaccessible."""

    pass


class IndexNotFoundError(ChassisError):
    """Raised when the index file cannot be found or opened."""

    pass


class NullPointerError(ChassisError):
    """Raised when a NULL pointer is returned from FFI."""

    pass
