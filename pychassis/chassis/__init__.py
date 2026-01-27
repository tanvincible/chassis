"""
Chassis - Python bindings for Chassis vector storage engine

This package provides a Pythonic interface to the Chassis vector storage engine,
which is built in Rust and exposed via FFI.

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
    DimensionMismatchError,
    InvalidPathError,
    IndexNotFoundError,
)

__version__ = "0.1.0"
__all__ = [
    "VectorIndex",
    "SearchResult",
    "IndexOptions",
    "ChassisError",
    "DimensionMismatchError",
    "InvalidPathError",
    "IndexNotFoundError",
]
