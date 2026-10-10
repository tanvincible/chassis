"""
Chassis - Python bindings for Chassis, an embedded vector index

Chassis keeps an index of vectors in one file and searches it inside your
process, with no server. It is written in Rust; this package calls its C API.

Example:
    >>> import chassis
    >>> index = chassis.VectorIndex("vectors.chassis", dimensions=768)
    >>> vector_id = index.add([0.1] * 768)
    >>> results = index.search([0.1] * 768, k=10)
    >>> for result in results:
    ...     print(f"ID: {result.id}, Distance: {result.distance}")
    >>> index.flush()
"""

from chassis.index import VectorIndex, SearchResult, IndexOptions
from chassis.exceptions import (
    ChassisError,
    CorruptIndexError,
    DimensionMismatchError,
    IdInUseError,
    IndexFullError,
    IndexLockedError,
    IndexNotFoundError,
    InvalidArgumentError,
    InvalidPathError,
    NotAnIndexError,
    OptionsMismatchError,
    ReadOnlyError,
)

__version__ = "0.7.1"
__all__ = [
    "VectorIndex",
    "SearchResult",
    "IndexOptions",
    "ChassisError",
    "CorruptIndexError",
    "DimensionMismatchError",
    "IdInUseError",
    "IndexFullError",
    "IndexLockedError",
    "IndexNotFoundError",
    "InvalidArgumentError",
    "InvalidPathError",
    "NotAnIndexError",
    "OptionsMismatchError",
    "ReadOnlyError",
]
