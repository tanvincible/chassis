"""ctypes bindings to the C library, under `VectorIndex`."""

import ctypes
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chassisdb.exceptions import ChassisError


def _get_library_name() -> str:
    system = platform.system()
    if system == "Linux":
        return "libchassis_ffi.so"
    elif system == "Darwin":
        return "libchassis_ffi.dylib"
    elif system == "Windows":
        return "chassis_ffi.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def _find_library() -> Path:
    """The C library: in CHASSIS_LIB_PATH, beside this file (as a wheel has it), in the
    checkout's target/release, or on the system's library path."""
    lib_name = _get_library_name()

    if env_path := os.getenv("CHASSIS_LIB_PATH"):
        lib_path = Path(env_path) / lib_name
        if lib_path.exists():
            return lib_path

    this_dir = Path(__file__).parent
    lib_path = this_dir / lib_name
    if lib_path.exists():
        return lib_path

    dev_path = this_dir.parent.parent / "target" / "release" / lib_name
    if dev_path.exists():
        return dev_path

    from ctypes.util import find_library

    if lib_path_str := find_library("chassis_ffi"):
        return Path(lib_path_str)

    raise FileNotFoundError(
        f"Could not find {lib_name}. "
        "Set CHASSIS_LIB_PATH environment variable or ensure library is built."
    )


_lib_path = _find_library()
_lib = ctypes.CDLL(str(_lib_path))


class ChassisIndex(ctypes.Structure):
    """Opaque handle to a Chassis index (never accessed directly)."""

    pass


ChassisIndexPtr = ctypes.POINTER(ChassisIndex)


# Function signatures. Arrays go as `array.ctypes.data`: a pointer from `data_as` costs over a
# microsecond each, much of what a search spends outside the library.

_lib.chassis_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
_lib.chassis_open.restype = ChassisIndexPtr

_lib.chassis_open_with_options.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_uint32,  # dimensions
    ctypes.c_uint32,  # max_connections
    ctypes.c_uint32,  # ef_construction
    ctypes.c_uint32,  # ef_search
]
_lib.chassis_open_with_options.restype = ChassisIndexPtr

_lib.chassis_open_with_metric.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_uint32,  # dimensions
    ctypes.c_uint32,  # max_connections
    ctypes.c_uint32,  # ef_construction
    ctypes.c_uint32,  # ef_search
    ctypes.c_uint32,  # metric: 0 Euclidean, 1 cosine
]
_lib.chassis_open_with_metric.restype = ChassisIndexPtr

_lib.chassis_open_with_precision.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_uint32,  # dimensions
    ctypes.c_uint32,  # max_connections
    ctypes.c_uint32,  # ef_construction
    ctypes.c_uint32,  # ef_search
    ctypes.c_uint32,  # metric: 0 Euclidean, 1 cosine
    ctypes.c_uint32,  # precision: 0 full, 1 half
]
_lib.chassis_open_with_precision.restype = ChassisIndexPtr

_lib.chassis_open_reader.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_uint32,  # dimensions
    ctypes.c_uint32,  # max_connections
    ctypes.c_uint32,  # ef_search
]
_lib.chassis_open_reader.restype = ChassisIndexPtr

_lib.chassis_free.argtypes = [ChassisIndexPtr]
_lib.chassis_free.restype = None

_lib.chassis_add.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.chassis_add.restype = ctypes.c_uint64

_lib.chassis_add_with_id.argtypes = [
    ChassisIndexPtr,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.chassis_add_with_id.restype = ctypes.c_int

_lib.chassis_add_batch.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,  # vectors, count * dim, row-major
    ctypes.c_size_t,  # count
    ctypes.c_size_t,  # dim
    ctypes.c_void_p,  # out_ids
]
_lib.chassis_add_batch.restype = ctypes.c_size_t

_lib.chassis_add_batch_with_ids.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,  # ids
    ctypes.c_void_p,  # vectors, count * dim, row-major
    ctypes.c_size_t,  # count
    ctypes.c_size_t,  # dim
]
_lib.chassis_add_batch_with_ids.restype = ctypes.c_int

_lib.chassis_delete.argtypes = [ChassisIndexPtr, ctypes.c_uint64]
_lib.chassis_delete.restype = ctypes.c_int

_lib.chassis_search.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_lib.chassis_search.restype = ctypes.c_size_t

_lib.chassis_search_filtered.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,  # allowed_ids
    ctypes.c_size_t,  # allowed_len
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_lib.chassis_search_filtered.restype = ctypes.c_size_t

_lib.chassis_search_excluding.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,  # excluded_ids
    ctypes.c_size_t,  # excluded_len
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_lib.chassis_search_excluding.restype = ctypes.c_size_t

_lib.chassis_search_batch.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,  # queries
    ctypes.c_size_t,  # count
    ctypes.c_size_t,  # dim
    ctypes.c_size_t,  # k
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_lib.chassis_search_batch.restype = ctypes.c_size_t

_lib.chassis_flush.argtypes = [ChassisIndexPtr]
_lib.chassis_flush.restype = ctypes.c_int

_lib.chassis_rollback.argtypes = [ChassisIndexPtr]
_lib.chassis_rollback.restype = ctypes.c_int

_lib.chassis_compact.argtypes = [ChassisIndexPtr]
_lib.chassis_compact.restype = ctypes.c_int

_lib.chassis_backup.argtypes = [ChassisIndexPtr, ctypes.c_char_p]
_lib.chassis_backup.restype = ctypes.c_int

_lib.chassis_use_huge_pages.argtypes = [ChassisIndexPtr]
_lib.chassis_use_huge_pages.restype = ctypes.c_int

_lib.chassis_warm.argtypes = [ChassisIndexPtr]
_lib.chassis_warm.restype = ctypes.c_int

_lib.chassis_get.argtypes = [
    ChassisIndexPtr,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.chassis_get.restype = ctypes.c_int

_lib.chassis_contains.argtypes = [ChassisIndexPtr, ctypes.c_uint64]
_lib.chassis_contains.restype = ctypes.c_int

_lib.chassis_ids.argtypes = [ChassisIndexPtr, ctypes.c_void_p, ctypes.c_size_t]
_lib.chassis_ids.restype = ctypes.c_size_t

_lib.chassis_check.argtypes = [
    ChassisIndexPtr,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.chassis_check.restype = ctypes.c_size_t

_lib.chassis_deleted_count.argtypes = [ChassisIndexPtr]
_lib.chassis_deleted_count.restype = ctypes.c_uint64

_lib.chassis_user_version.argtypes = [ChassisIndexPtr]
_lib.chassis_user_version.restype = ctypes.c_uint64
_lib.chassis_set_user_version.argtypes = [ChassisIndexPtr, ctypes.c_uint64]
_lib.chassis_set_user_version.restype = ctypes.c_int

_lib.chassis_len.argtypes = [ChassisIndexPtr]
_lib.chassis_len.restype = ctypes.c_uint64

_lib.chassis_is_empty.argtypes = [ChassisIndexPtr]
_lib.chassis_is_empty.restype = ctypes.c_int

_lib.chassis_dimensions.argtypes = [ChassisIndexPtr]
_lib.chassis_dimensions.restype = ctypes.c_uint32

_lib.chassis_metric.argtypes = [ChassisIndexPtr]
_lib.chassis_metric.restype = ctypes.c_int

_lib.chassis_precision.argtypes = [ChassisIndexPtr]
_lib.chassis_precision.restype = ctypes.c_int

_lib.chassis_last_error_message.argtypes = []
_lib.chassis_last_error_message.restype = ctypes.c_char_p

_lib.chassis_last_error_code.argtypes = []
_lib.chassis_last_error_code.restype = ctypes.c_int

_lib.chassis_version.argtypes = []
_lib.chassis_version.restype = ctypes.c_char_p


def get_last_error() -> str | None:
    """The last error message on this thread, or None."""
    err_ptr: bytes | None = _lib.chassis_last_error_message()
    if err_ptr:
        return err_ptr.decode("utf-8")
    return None


def last_error(default: str) -> "ChassisError":
    """The exception for the last error on this thread, of the class its code names."""
    from chassisdb.exceptions import from_code

    return from_code(_lib.chassis_last_error_code(), get_last_error() or default)


def get_version() -> str:
    """The C library's version, such as "0.9.1"."""
    version_ptr: bytes = _lib.chassis_version()
    return version_ptr.decode("utf-8")


__all__ = [
    "_lib",
    "ChassisIndex",
    "ChassisIndexPtr",
    "get_last_error",
    "last_error",
    "get_version",
]
