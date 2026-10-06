"""High-level Pythonic interface to Chassis vector index."""

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from chassis import _ffi
from chassis.exceptions import (
    ChassisError,
    DimensionMismatchError,
    InvalidPathError,
    NullPointerError,
)


_METRICS = {"euclidean": 0, "cosine": 1}


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
        metric: "euclidean" (L2) or "cosine" (1 - cosine similarity; vectors
            are stored at unit length and zero vectors are rejected). Fixed
            when the index is created. Default: "euclidean"
    """

    max_connections: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    metric: str = "euclidean"

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
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {list(_METRICS)}, got {self.metric!r}")


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
        Every method is safe from any thread. Searches run concurrently;
        add(), delete() and flush() run one at a time and searches wait for
        them. Don't call close() while other threads still use the index.
        An index opened with read_only=True runs one search at a time.

    Readers in other processes:
        Any number of processes can open the file with read_only=True while
        one process writes it. Each search sees the writer's last flush.

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
        read_only: bool = False,
    ):
        """Open or create a vector index.

        Args:
            path: Path to the index file
            dimensions: Number of dimensions per vector
            options: Optional HNSW configuration. If None, uses defaults.
            read_only: Open an existing index to search it while another
                process writes it. Takes no lock; add(), delete() and
                flush() raise ChassisError. The index keeps the metric it
                was created with; explicit options naming another raise.

        Raises:
            InvalidPathError: If path is invalid or inaccessible
            NullPointerError: If index creation fails
            ChassisError: For other errors
        """
        # Set before anything can raise: __del__ calls close(), which reads them.
        self._ptr: Optional[_ffi.ChassisIndexPtr] = None
        self._closed = False
        self._path = Path(path)
        self._dimensions = dimensions
        self._options = options or IndexOptions()
        self._options.validate()

        # Encode path to UTF-8 bytes
        path_bytes = str(self._path).encode("utf-8")

        # Open index with options
        if read_only:
            ptr = _ffi._lib.chassis_open_reader(
                path_bytes,
                dimensions,
                self._options.max_connections,
                self._options.ef_search,
            )
        elif options is None:
            # Use default options
            ptr = _ffi._lib.chassis_open(path_bytes, dimensions)
        else:
            # Use custom options
            ptr = _ffi._lib.chassis_open_with_metric(
                path_bytes,
                dimensions,
                options.max_connections,
                options.ef_construction,
                options.ef_search,
                _METRICS[options.metric],
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
        # A reader takes the file's metric; say so rather than search by another.
        if options is not None and options.metric != (metric := self.metric):
            self.close()
            raise ChassisError(f"Index was created with {metric} distance, not {options.metric}")

    @property
    def metric(self) -> str:
        """The distance metric the index was created with: "euclidean" or "cosine"."""
        self._check_closed()
        return {v: k for k, v in _METRICS.items()}[_ffi._lib.chassis_metric(self._ptr)]

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
        self,
        vector: Union[Sequence[float], npt.NDArray[np.float32]],
        id: Optional[int] = None,
    ) -> int:
        """Add a vector to the index.

        Args:
            vector: Vector to add (must match index dimensions)
                Can be a list, tuple, numpy array, or any sequence of floats
            id: Id to store the vector under, from 0 to 2**64 - 2. Search
                results report it. Default: one past the largest id used so far.

        Returns:
            The vector's id

        Raises:
            ChassisError: If index is closed
            DimensionMismatchError: If vector dimensions don't match
            ValueError: If id is out of range
            ChassisError: If a vector with this id already exists (delete it
                first to replace it), or for other errors

        Note:
            This method does NOT guarantee durability. Call flush() to
            ensure data is written to disk.

        Thread Safety:
            Safe from any thread; writes run one at a time.
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
        if id is not None:
            _check_id(id)
            if _ffi._lib.chassis_add_with_id(self._ptr, id, vector_ptr, len(vector)) != 0:
                raise ChassisError(_ffi.get_last_error() or "Failed to add vector")
            return id
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

    def delete(self, id: int) -> bool:
        """Delete the vector with this id.

        Search stops returning it immediately. The delete is durable after the
        next flush(); a crash before then rolls it back.

        Returns:
            True if it was deleted, False if no vector has this id

        Raises:
            ChassisError: If index is closed or the delete fails
            ValueError: If id is out of range

        Thread Safety:
            Safe from any thread; writes run one at a time.
        """
        self._check_closed()
        _check_id(id)
        result = _ffi._lib.chassis_delete(self._ptr, id)
        if result < 0:
            raise ChassisError(_ffi.get_last_error() or "Failed to delete vector")
        return result == 1

    def search(
        self,
        query: Union[Sequence[float], npt.NDArray[np.float32]],
        k: int = 10,
        allowed: Optional[Union[Iterable[int], npt.NDArray[np.uint64]]] = None,
    ) -> List[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector (must match index dimensions)
            k: Number of nearest neighbors to return (default: 10)
            allowed: If given, only these ids can be returned. When walking
                the graph would cost more, as when few vectors match, every
                vector is checked instead and the results are exact.

        Returns:
            List of SearchResult objects, sorted by distance (ascending)

        Raises:
            ChassisError: If index is closed
            DimensionMismatchError: If query dimensions don't match
            ValueError: If k < 1
            ChassisError: For other errors

        Thread Safety:
            Safe from any thread. Runs concurrently with other searches and
            waits while a write runs.
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

        if allowed is None:
            count = _ffi._lib.chassis_search(
                self._ptr, query_ptr, len(query), k, ids_ptr, dists_ptr
            )
        else:
            allowed = _allowed_ids(allowed)
            allowed_ptr = allowed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            count = _ffi._lib.chassis_search_filtered(
                self._ptr, query_ptr, len(query), k, allowed_ptr, len(allowed), ids_ptr, dists_ptr
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
            Safe from any thread; writes run one at a time.
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
            Safe from any thread.
        """
        self._check_closed()
        return int(_ffi._lib.chassis_len(self._ptr))

    def is_empty(self) -> bool:
        """Check if the index is empty.

        Returns:
            True if empty, False otherwise

        Thread Safety:
            Safe from any thread.
        """
        self._check_closed()
        return bool(_ffi._lib.chassis_is_empty(self._ptr))

    @property
    def dimensions(self) -> int:
        """Get the dimensionality of vectors in this index.

        Returns:
            Number of dimensions

        Thread Safety:
            Safe from any thread.
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


def _allowed_ids(allowed) -> npt.NDArray[np.uint64]:
    # Like _check_id: numpy and ctypes would silently wrap, truncate or flatten bad ids.
    ids = np.asarray(allowed if isinstance(allowed, np.ndarray) else list(allowed))
    if ids.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if ids.ndim != 1 or ids.dtype == np.bool_ or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"allowed must be a flat sequence of int ids, got {ids.dtype} {ids.shape}")
    if (ids < 0).any() or (ids == 2**64 - 1).any():
        raise ValueError("allowed ids must be between 0 and 2**64 - 2")
    return np.ascontiguousarray(ids, dtype=np.uint64)


def _check_id(id: int) -> None:
    # ctypes would silently wrap a negative or oversized id into a different one.
    if not 0 <= id < 2**64 - 1:
        raise ValueError(f"id must be between 0 and 2**64 - 2, got {id}")
