"""High-level Pythonic interface to Chassis vector index."""

import ctypes
import dataclasses
import difflib
import functools
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from chassis import _ffi
from chassis.exceptions import (
    ChassisError,
    DimensionMismatchError,
    InvalidArgumentError,
    OptionsMismatchError,
    ReadOnlyError,
)


_METRICS = {"euclidean": 0, "cosine": 1}
_PRECISIONS = {"full": 0, "half": 1}
# Other names for them, as other libraries spell them.
_METRIC_NAMES = {"l2": "euclidean", "euclid": "euclidean", "cos": "cosine", "angular": "cosine"}
_PRECISION_NAMES = {
    "f32": "full", "float32": "full", "fp32": "full", "single": "full",
    "f16": "half", "float16": "half", "fp16": "half",
}
_MAX_DIMENSIONS = 4096


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
        precision: "full" keeps each component of a vector as a 32-bit float,
            "half" as a 16-bit float: half the file and half the memory, each
            component rounded to about three decimal digits, and a vector with
            a component of 65,520 or more in magnitude refused. Fixed when the
            index is created; a read-only index takes it from the file.
            Default: "full"
        huge_pages: Ask the operating system to keep the vectors on huge
            pages. On an index too large for the CPU's caches, searches are
            up to a quarter faster. Only on Linux, and only where the kernel and
            filesystem keep files on huge pages (ext4 on Linux 6.17 does);
            elsewhere it does nothing. With it a page not yet in memory is
            read 2 MB at a time, which an index much larger than memory pays
            for on every miss. Default: False
        warm: Read the whole index into memory on another thread from the
            moment it is opened, so that searches stop waiting for the disk
            one page at a time. For an index that fits in memory. It takes
            Linux 5.14 or macOS: an older Linux reads only some of the
            index, and Windows none. Default: False
    """

    max_connections: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    metric: str = "euclidean"
    precision: str = "full"
    huge_pages: bool = False
    warm: bool = False

    def validate(self) -> None:
        """Check the options, and spell metric and precision as Chassis does:
        "L2" becomes "euclidean" and "float16" becomes "half".

        Raises:
            InvalidArgumentError: If an option is out of range or unknown
            TypeError: If an option has the wrong type
        """
        for name in ("max_connections", "ef_construction", "ef_search"):
            _check_int(name, getattr(self, name))
        if not 2 <= self.max_connections <= 32767:
            raise InvalidArgumentError(
                f"max_connections is {self.max_connections}, but has to be between 2 and "
                "32,767\nhelp: 16 suits most indexes; 32 gives higher recall for more memory"
            )
        for name in ("ef_construction", "ef_search"):
            if getattr(self, name) < 1:
                raise InvalidArgumentError(
                    f"{name} is {getattr(self, name)}, but has to be at least 1\nhelp: higher "
                    f"finds more of the true nearest for more time; the default is "
                    f"{getattr(IndexOptions(), name)}"
                )
        self.metric = _named("metric", self.metric, _METRICS, _METRIC_NAMES)
        self.precision = _named("precision", self.precision, _PRECISIONS, _PRECISION_NAMES)


# What other libraries call the options, for a hint when one of those names is given.
_OPTION_NAMES = {
    "M": "max_connections", "m": "max_connections", "ef": "ef_search", "efSearch": "ef_search",
    "efConstruction": "ef_construction", "ef_c": "ef_construction", "space": "metric",
    "distance": "metric", "dtype": "precision",
}


def _check_option_names(keys) -> None:
    """Raises TypeError for a key that isn't an option, saying which was likely meant."""
    names = [field.name for field in dataclasses.fields(IndexOptions)]
    for key in keys:
        if key in names:
            continue
        near = _OPTION_NAMES.get(key) or next(iter(difflib.get_close_matches(str(key), names, 1)), None)
        hint = f"did you mean {near!r}? " if near else ""
        raise TypeError(f"{key!r} is not an option\nhelp: {hint}the options are {', '.join(names)}")


def _named_options(init):
    """IndexOptions' __init__, saying which option was meant when one is misspelled."""

    @functools.wraps(init)
    def checked(self, *args, **kwargs):
        _check_option_names(kwargs)
        init(self, *args, **kwargs)

    return checked


IndexOptions.__init__ = _named_options(IndexOptions.__init__)


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
            IndexNotFoundError: If read_only and there is no index at path, or
                the path's directory doesn't exist
            IndexLockedError: If another writer has the index open
            DimensionMismatchError: If the index holds vectors of other
                dimensions
            OptionsMismatchError: If the index was created with another
                metric or precision
            NotAnIndexError: If the file isn't a Chassis index
            InvalidArgumentError: If dimensions or an option is out of range
            TypeError: If an argument has the wrong type
            ChassisError: For other errors; every class above is one too
        """
        # Set before anything can raise: __del__ calls close(), which reads them.
        self._ptr: Optional[_ffi.ChassisIndexPtr] = None
        self._closed = False
        self._read_only = read_only
        self._path = Path(path)
        self._dimensions = _check_dimensions(dimensions)
        self._options = _check_options(options)
        explicit = options is not None
        options = self._options

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
        elif not explicit:
            # Use default options
            ptr = _ffi._lib.chassis_open(path_bytes, dimensions)
        else:
            # Use custom options
            ptr = _ffi._lib.chassis_open_with_precision(
                path_bytes,
                dimensions,
                options.max_connections,
                options.ef_construction,
                options.ef_search,
                _METRICS[options.metric],
                _PRECISIONS[options.precision],
            )

        if not ptr:
            raise _ffi.last_error("The index couldn't be opened, and Chassis didn't say why")

        self._ptr = ptr
        if self._options.huge_pages:
            _ffi._lib.chassis_use_huge_pages(ptr)
        if self._options.warm:
            _ffi._lib.chassis_warm(ptr)
        # A reader takes the file's metric; say so rather than search by another.
        if explicit and options.metric != (metric := self.metric):
            self.close()
            raise OptionsMismatchError(
                f"The index at {self._path} was created with {metric} distance, not "
                f"{options.metric}\nhelp: open it with metric={metric!r}, or leave the metric "
                "out: a read_only index takes it from the file"
            )

    @property
    def metric(self) -> str:
        """The distance metric the index was created with: "euclidean" or "cosine"."""
        self._check_closed()
        return {v: k for k, v in _METRICS.items()}[_ffi._lib.chassis_metric(self._ptr)]

    @property
    def precision(self) -> str:
        """What the index keeps of each component of a vector: "full" or "half"."""
        self._check_closed()
        return {v: k for k, v in _PRECISIONS.items()}[_ffi._lib.chassis_precision(self._ptr)]

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
            raise ChassisError(
                f"The index at {self._path} is closed\nhelp: open it again with "
                "VectorIndex(path, dimensions), or use it inside its with block"
            )

    def _check_writable(self, action: str) -> None:
        self._check_closed()
        if self._read_only:
            raise ReadOnlyError(
                f"Can't {action}: the index at {self._path} was opened with read_only=True, which "
                "only searches\nhelp: open it without read_only to write it; an index has one "
                "writer at a time, and read_only indexes can search while it writes"
            )

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
        self._check_writable("add")
        vector = _as_vector(vector, self._dimensions, "vector")
        vector_ptr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if id is not None:
            id = _check_id(id)
            if _ffi._lib.chassis_add_with_id(self._ptr, id, vector_ptr, len(vector)) != 0:
                raise _ffi.last_error("The vector couldn't be added")
            return id
        vector_id = _ffi._lib.chassis_add(self._ptr, vector_ptr, len(vector))
        if vector_id == 2**64 - 1:
            raise _ffi.last_error("The vector couldn't be added")
        return int(vector_id)

    def add_batch(
        self,
        vectors: Union[Sequence[Sequence[float]], npt.NDArray[np.float32]],
        ids: Optional[Union[Iterable[int], npt.NDArray[np.uint64]]] = None,
    ) -> npt.NDArray[np.uint64]:
        """Add many vectors at once, linking them on every core.

        Much faster than calling add() in a loop for large batches; the
        graph then depends on thread timing. All or nothing: if any vector
        or id is rejected, none of the batch is added.

        Args:
            vectors: A (count, dimensions) array, or a sequence of vectors
            ids: Optional ids, one per vector, as for add(). Default: each
                vector gets the id add() would give it.

        Returns:
            The vectors' ids

        Raises:
            DimensionMismatchError: If the rows don't match the index dimensions
            ValueError: If ids are out of range or their count differs
            ChassisError: If an id repeats or already exists, or for other errors
        """
        self._check_writable("add")
        vectors = _as_batch(vectors, self._dimensions)
        count = len(vectors)
        vectors_ptr = vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if ids is None:
            out_ids = np.zeros(count, dtype=np.uint64)
            out_ptr = out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            added = _ffi._lib.chassis_add_batch(
                self._ptr, vectors_ptr, count, self._dimensions, out_ptr
            )
            if added != count:
                raise _ffi.last_error("The batch couldn't be added")
            return out_ids

        ids = [_check_id(i) for i in (ids.tolist() if isinstance(ids, np.ndarray) else ids)]
        if len(ids) != count:
            raise InvalidArgumentError(
                f"{len(ids)} ids were given for {count} vectors\nhelp: give one id per vector, "
                "or leave ids out to have them chosen"
            )
        id_array = np.array(ids, dtype=np.uint64)
        ids_ptr = id_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        if _ffi._lib.chassis_add_batch_with_ids(
            self._ptr, ids_ptr, vectors_ptr, count, self._dimensions
        ) != 0:
            raise _ffi.last_error("The batch couldn't be added")
        return id_array

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
        self._check_writable("delete")
        result = _ffi._lib.chassis_delete(self._ptr, _check_id(id))
        if result < 0:
            raise _ffi.last_error("The vector couldn't be deleted")
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
        k = _check_int("k", k)
        if k < 1:
            raise InvalidArgumentError(
                f"k is {k}, but has to be at least 1\nhelp: pass how many results to return, "
                "such as k=10"
            )
        query = _as_vector(query, self._dimensions, "query")

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

        # 0 results is an error only if one was set: an empty index has none either.
        if count == 0 and _ffi._lib.chassis_last_error_code() != 0:
            raise _ffi.last_error("The search failed")

        # Convert to SearchResult objects
        results = [
            SearchResult(id=int(out_ids[i]), distance=float(out_dists[i]))
            for i in range(count)
        ]

        return results

    def search_batch(
        self,
        queries: Union[Sequence[Sequence[float]], npt.NDArray[np.float32]],
        k: int = 10,
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.float32]]:
        """Search for the k nearest neighbors of many queries, in one call.

        Faster than calling search once per query: the queries cross from
        Python into the library together.

        Args:
            queries: A (count, dimensions) array, one query per row
            k: Number of nearest neighbors per query (default: 10)

        Returns:
            (ids, distances), two (count, k) arrays. Row i holds query i's
            results, nearest first; a row with fewer than k results is filled
            out with id 2**64 - 1, which no vector has, and distance inf.

        Raises:
            ChassisError: If index is closed
            DimensionMismatchError: If the queries aren't rows of the index's
                dimensions; the message names a query that is wrong
            InvalidArgumentError: If k < 1, or a query has a component that
                isn't a finite number
            ChassisError: For other errors

        Thread Safety:
            Safe from any thread. Runs concurrently with other searches, and a
            write waits for the whole batch.
        """
        self._check_closed()
        k = _check_int("k", k)
        if k < 1:
            raise InvalidArgumentError(
                f"k is {k}, but has to be at least 1\nhelp: pass how many results to return per "
                "query, such as k=10"
            )
        queries = _as_batch(queries, self._dimensions, "query", "search")
        count = len(queries)
        out_ids = np.empty((count, k), dtype=np.uint64)
        out_dists = np.empty((count, k), dtype=np.float32)
        if count == 0:
            return out_ids, out_dists

        done = _ffi._lib.chassis_search_batch(
            self._ptr,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count,
            self._dimensions,
            k,
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if done != count:
            raise _ffi.last_error("The search failed")
        return out_ids, out_dists

    def flush(self) -> None:
        """Flush all changes to disk.

        This method ensures durability by writing all pending changes to disk.
        It's expensive (1-50ms) so batch multiple add() calls before flushing.

        Raises:
            ChassisError: If flush fails

        Thread Safety:
            Safe from any thread; writes run one at a time.
        """
        self._check_writable("flush")
        if _ffi._lib.chassis_flush(self._ptr) != 0:
            raise _ffi.last_error("The flush failed")

    def warm(self) -> None:
        """Start reading the whole index into memory on another thread.

        Doesn't wait for the reading: searches go on meanwhile. While it
        is under way, calling again does nothing. Opening with
        ``IndexOptions(warm=True)`` does the same from the start.

        Raises:
            ChassisError: If the index can't be used any more
        """
        self._check_closed()

        if _ffi._lib.chassis_warm(self._ptr) != 0:
            raise _ffi.last_error("The index couldn't be read in")

    def compact(self) -> None:
        """Rewrite the index without its deleted vectors, with a newly built graph.

        Reclaims the space of deleted vectors and replaces the index file with
        the copy. Ids don't change. Like flush(), it makes every add and
        delete so far durable. Takes as long as building the index, on every
        core, and needs free disk for a second copy of the live vectors.
        Readers in other processes keep searching and move to the new file by
        themselves.

        Raises:
            ChassisError: If the copy can't be written or, on Windows, another
                process has the index open. The index is left as it was.
        """
        self._check_writable("compact")
        if _ffi._lib.chassis_compact(self._ptr) != 0:
            raise _ffi.last_error("The compaction failed")

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
    if callable(allowed):
        raise TypeError(
            "allowed is a function, but has to be the ids a result may have\nhelp: pass "
            "allowed=[i for i in candidate_ids if keep(i)], or a set or numpy array of ids"
        )
    ids = np.asarray(allowed if isinstance(allowed, np.ndarray) else list(allowed))
    if ids.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if ids.ndim != 1 or ids.dtype == np.bool_ or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(
            f"allowed has to be a flat sequence of int ids, but is {ids.dtype} of shape "
            f"{ids.shape}\nhelp: ids are ints; keep a dict from your own keys to them"
        )
    if (ids < 0).any() or (ids == 2**64 - 1).any():
        raise InvalidArgumentError(
            "allowed has an id below 0 or of 2**64 - 1\nhelp: ids go from 0 to 2**64 - 2"
        )
    return np.ascontiguousarray(ids, dtype=np.uint64)


def _check_int(name: str, value) -> int:
    """`value` as an int, or TypeError: a bool or a float isn't one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} is {value!r}, but has to be an int\nhelp: pass {name}=int(...)")
    return int(value)


def _check_id(id) -> int:
    # ctypes would silently wrap a negative or oversized id into a different one.
    if isinstance(id, (str, bytes)):
        raise TypeError(
            f"id is {id!r}, but ids have to be ints\nhelp: keep a dict from your own keys to int "
            "ids, or leave ids out to have them chosen"
        )
    id = _check_int("id", id)
    if not 0 <= id < 2**64 - 1:
        raise InvalidArgumentError(
            f"id is {id}, but ids go from 0 to 2**64 - 2\nhelp: pass an id in that range, or "
            "leave it out to have one chosen"
        )
    return id


def _check_dimensions(dimensions) -> int:
    dimensions = _check_int("dimensions", dimensions)
    if not 1 <= dimensions <= _MAX_DIMENSIONS:
        raise InvalidArgumentError(
            f"dimensions is {dimensions}, but has to be between 1 and {_MAX_DIMENSIONS}\nhelp: "
            "give the length of the vectors the index will hold, such as 384, 768 or 1536"
        )
    return dimensions


def _check_options(options) -> IndexOptions:
    """`options` as checked IndexOptions; a dict of their fields is taken too."""
    if options is None:
        options = IndexOptions()
    elif isinstance(options, Mapping):
        options = IndexOptions(**options)
    elif not isinstance(options, IndexOptions):
        raise TypeError(
            f"options is a {type(options).__name__}, but has to be IndexOptions or a dict\nhelp: "
            "pass options=IndexOptions(metric='cosine'), or options={'metric': 'cosine'}"
        )
    options.validate()
    return options


def _named(option: str, value, names: dict, aliases: dict) -> str:
    """`value` as one of `names`, taking any case and the aliases other libraries use."""
    key = value.lower() if isinstance(value, str) else value
    key = aliases.get(key, key)
    if key in names:
        return key
    known = ", ".join(repr(name) for name in names)
    hint = ""
    if option == "metric" and key in ("ip", "dot", "inner_product", "dot_product"):
        hint = "; for inner product on vectors of length 1, cosine orders results the same way"
    raise InvalidArgumentError(f"{option} is {value!r}, but has to be one of {known}\nhelp: pass "
                               f"{option}={next(iter(names))!r} or another of them{hint}")


def _as_vector(value, dims: int, what: str) -> npt.NDArray[np.float32]:
    """`value` as one contiguous float32 vector of `dims` components; `what` names it."""
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError(
            f"The {what} is {value!r:.40}, but has to be {dims} numbers\nhelp: pass a list or "
            f"numpy array of {dims} floats"
        )
    try:
        array = np.ascontiguousarray(value, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"The {what} has to be {dims} numbers, but can't be read as numbers: {e}\nhelp: pass "
            f"a list or numpy array of {dims} floats"
        ) from None
    if array.ndim == 1 and len(array) == dims:
        return array
    if array.ndim == 2 and array.shape[0] == 1:
        fix = f"pass {what}[0], or {what}.ravel() for an array"
    elif array.ndim == 2 and what == "vector":
        fix = "to add several, pass them to add_batch"
    elif array.ndim == 2:
        fix = "search takes one query; to search several in one call, pass them to search_batch"
    elif array.ndim == 1:
        raise DimensionMismatchError(
            f"The {what} has {len(array)} components, but this index holds vectors of {dims}\n"
            f"help: make the {what} with the same model as the index's vectors; this index was "
            f"created for {dims} dimensions"
        )
    else:
        fix = f"pass one vector of {dims} floats"
    raise DimensionMismatchError(
        f"The {what} is an array of shape {array.shape}, but has to be one vector of {dims} "
        f"components\nhelp: {fix}"
    )


def _as_batch(value, dims: int, what: str = "vector", one: str = "add") -> npt.NDArray[np.float32]:
    """`value` as a contiguous (count, dims) float32 array of `what`s, which `one` takes singly."""
    try:
        array = np.ascontiguousarray(value, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"The {what}s have to be numbers, but can't be read as numbers: {e}\nhelp: pass a "
            f"(count, {dims}) numpy array, or a list of lists of {dims} floats"
        ) from None
    if array.ndim == 2 and array.shape[1] == dims:
        return array
    if array.ndim == 1 and len(array) == dims:
        fix = f"for one {what}, pass [{what}], or call {one}"
    elif array.ndim == 2:
        fix = (f"each {what} has to have {dims} components: make them with the same model as the "
               "index's vectors")
    else:
        fix = f"pass a (count, {dims}) array, one {what} per row"
    raise DimensionMismatchError(
        f"The {what}s are an array of shape {array.shape}, but have to be (count, {dims})\n"
        f"help: {fix}"
    )
