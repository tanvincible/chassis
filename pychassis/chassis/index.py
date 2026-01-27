"""High-level Pythonic interface to Chassis vector index."""

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from chassis import _ffi
from chassis.exceptions import (
    ChassisError,
    DimensionMismatchError,
    InvalidPathError,
    NullPointerError,
)


@dataclass
class IndexOptions:
    """Configuration options for HNSW index.

    Attributes:
        max_connections: Maximum connections per node (M parameter).
            Higher = better recall, more memory. Default: 16
        ef_construction: Construction quality parameter.
            Higher = better index quality, slower build. Default: 200
        ef_search: Search quality parameter.
            Higher = better search quality, slower search. Default: 50
    """

    max_connections: int = 16
    ef_construction: int = 200
    ef_search: int = 50

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if not 1 <= self.max_connections <= 65535:
            raise ValueError(
                f"max_connections must be 1-65535, got {self.max_connections}"
            )
        if self.ef_construction < 1:
            raise ValueError(
                f"ef_construction must be >= 1, got {self.ef_construction}"
            )
        if self.ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {self.ef_search}")


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        id: Vector ID in the index
        distance: Distance to the query vector (lower is closer)
    """

    id: int
    distance: float

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, distance={self.distance:.6f})"


class VectorIndex:
    """High-level interface to Chassis vector index.

    This class provides a Pythonic wrapper around the Chassis FFI layer,
    handling memory management, error handling, and type conversions.

    Thread Safety:
        - add() and flush() require exclusive access (single writer)
        - search(), len(), is_empty(), dimensions() allow concurrent
            access (multi reader)

    Example:
        >>> index = VectorIndex("vectors.chassis", dimensions=128)
        >>>
        >>> # Add vectors
        >>> vectors = np.random.rand(1000, 128).astype(np.float32)
        >>> for vec in vectors:
        ...     index.add(vec)
        >>> index.flush()
        >>>
        >>> # Search
        >>> query = np.random.rand(128).astype(np.float32)
        >>> results = index.search(query, k=10)
        >>> for result in results:
        ...     print(f"ID: {result.id}, Distance: {result.distance}")
    """

    def __init__(
        self,
        path: Union[str, Path],
        dimensions: int,
        options: Optional[IndexOptions] = None,
    ):
        """Open or create a vector index.

        Args:
            path: Path to the index file
            dimensions: Number of dimensions per vector
            options: Optional HNSW configuration. If None, uses defaults.

        Raises:
            InvalidPathError: If path is invalid or inaccessible
            NullPointerError: If index creation fails
            ChassisError: For other errors
        """
        self._path = Path(path)
        self._dimensions = dimensions
        self._options = options or IndexOptions()
        self._options.validate()
        self._ptr: Optional[_ffi.ChassisIndexPtr] = None
        self._closed = False

        # Encode path to UTF-8 bytes
        path_bytes = str(self._path).encode("utf-8")

        # Open index with options
        if options is None:
            # Use default options
            ptr = _ffi._lib.chassis_open(path_bytes, dimensions)
        else:
            # Use custom options
            ptr = _ffi._lib.chassis_open_with_options(
                path_bytes,
                dimensions,
                options.max_connections,
                options.ef_construction,
                options.ef_search,
            )

        if not ptr:
            error_msg = _ffi.get_last_error()
            if error_msg:
                if "dimension" in error_msg.lower():
                    raise DimensionMismatchError(error_msg)
                elif (
                    "path" in error_msg.lower() or "utf-8" in error_msg.lower()
                ):
                    raise InvalidPathError(error_msg)
                else:
                    raise ChassisError(error_msg)
            else:
                raise NullPointerError(
                    "Failed to open index (no error message)"
                )

        self._ptr = ptr

    def __del__(self):
        """Clean up resources when index is garbage collected."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def close(self) -> None:
        """Close the index and free resources.

        This is called automatically when the object is garbage collected
        or when used as a context manager. It's safe to call multiple times.
        """
        if not self._closed and self._ptr:
            _ffi._lib.chassis_free(self._ptr)
            self._ptr = None
            self._closed = True

    def _check_closed(self) -> None:
        """Check if index is closed and raise error if so."""
        if self._closed or not self._ptr:
            raise ChassisError("Index is closed")

    def add(
        self, vector: Union[Sequence[float], npt.NDArray[np.float32]]
    ) -> int:
        """Add a vector to the index.

        Args:
            vector: Vector to add (must match index dimensions)
                Can be a list, tuple, numpy array, or any sequence of floats

        Returns:
            Vector ID (0-based, sequential)

        Raises:
            ChassisError: If index is closed
            DimensionMismatchError: If vector dimensions don't match
            ChassisError: For other errors

        Note:
            This method does NOT guarantee durability. Call flush() to
            ensure data is written to disk.

        Thread Safety:
            Single-writer only. Do not call concurrently with other add()
            or flush() calls.
        """
        self._check_closed()

        # Convert to numpy array for consistent handling
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        elif vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        # Validate dimensions
        if len(vector) != self._dimensions:
            raise DimensionMismatchError(
                f"Vector has {len(vector)} dimensions, "
                f"but index expects {self._dimensions}"
            )

        # Ensure C-contiguous array
        if not vector.flags.c_contiguous:
            vector = np.ascontiguousarray(vector)

        # Call FFI
        vector_ptr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vector_id = _ffi._lib.chassis_add(self._ptr, vector_ptr, len(vector))

        # Check for error (UINT64_MAX)
        if vector_id == 2**64 - 1:
            error_msg = _ffi.get_last_error()
            if error_msg:
                if "dimension" in error_msg.lower():
                    raise DimensionMismatchError(error_msg)
                else:
                    raise ChassisError(error_msg)
            else:
                raise ChassisError("Failed to add vector")

        return int(vector_id)

    def search(
        self,
        query: Union[Sequence[float], npt.NDArray[np.float32]],
        k: int = 10,
    ) -> List[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector (must match index dimensions)
            k: Number of nearest neighbors to return (default: 10)

        Returns:
            List of SearchResult objects, sorted by distance (ascending)

        Raises:
            ChassisError: If index is closed
            DimensionMismatchError: If query dimensions don't match
            ValueError: If k < 1
            ChassisError: For other errors

        Thread Safety:
            Multi-reader safe. Can be called concurrently with other search()
            calls, but not with add() or flush().
        """
        self._check_closed()

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Convert to numpy array
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        elif query.dtype != np.float32:
            query = query.astype(np.float32)

        # Validate dimensions
        if len(query) != self._dimensions:
            raise DimensionMismatchError(
                f"Query has {len(query)} dimensions, "
                f"but index expects {self._dimensions}"
            )

        # Ensure C-contiguous
        if not query.flags.c_contiguous:
            query = np.ascontiguousarray(query)

        # Allocate output buffers
        out_ids = np.zeros(k, dtype=np.uint64)
        out_dists = np.zeros(k, dtype=np.float32)

        # Call FFI
        query_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ids_ptr = out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        dists_ptr = out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        count = _ffi._lib.chassis_search(
            self._ptr,
            query_ptr,
            len(query),
            k,
            ids_ptr,
            dists_ptr,
        )

        # Check for error (count == 0 could be error or empty index)
        if count == 0:
            error_msg = _ffi.get_last_error()
            if error_msg:
                if "dimension" in error_msg.lower():
                    raise DimensionMismatchError(error_msg)
                else:
                    raise ChassisError(error_msg)
            # Otherwise, just no results (empty index or no neighbors found)

        # Convert to SearchResult objects
        results = [
            SearchResult(id=int(out_ids[i]), distance=float(out_dists[i]))
            for i in range(count)
        ]

        return results

    def flush(self) -> None:
        """Flush all changes to disk.

        This method ensures durability by writing all pending changes to disk.
        It's expensive (1-50ms) so batch multiple add() calls before flushing.

        Raises:
            ChassisError: If flush fails

        Thread Safety:
            Single-writer only. Do not call concurrently with add() or other
            flush() calls.
        """
        self._check_closed()

        result = _ffi._lib.chassis_flush(self._ptr)

        if result != 0:
            error_msg = _ffi.get_last_error()
            raise ChassisError(f"Flush failed: {error_msg or 'unknown error'}")

    def __len__(self) -> int:
        """Get the number of vectors in the index.

        Returns:
            Number of vectors

        Thread Safety:
            Multi-reader safe.
        """
        self._check_closed()
        return int(_ffi._lib.chassis_len(self._ptr))

    def is_empty(self) -> bool:
        """Check if the index is empty.

        Returns:
            True if empty, False otherwise

        Thread Safety:
            Multi-reader safe.
        """
        self._check_closed()
        return bool(_ffi._lib.chassis_is_empty(self._ptr))

    @property
    def dimensions(self) -> int:
        """Get the dimensionality of vectors in this index.

        Returns:
            Number of dimensions

        Thread Safety:
            Multi-reader safe.
        """
        self._check_closed()
        return int(_ffi._lib.chassis_dimensions(self._ptr))

    @property
    def path(self) -> Path:
        """Get the path to the index file."""
        return self._path

    @property
    def options(self) -> IndexOptions:
        """Get the index configuration options."""
        return self._options

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (
            f"VectorIndex(path={self._path}, "
            f"dimensions={self._dimensions}, "
            f"len={len(self) if not self._closed else '?'}, "
            f"status={status})"
        )
