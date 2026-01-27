"""Low-level FFI bindings to libchassis_ffi.

This module provides direct ctypes bindings to the Chassis C library.
Users should use the high-level VectorIndex class instead.
"""

import ctypes
import os
import platform
from pathlib import Path
from typing import Optional


# Determine library name based on platform
def _get_library_name() -> str:
    """Get the platform-specific library name."""
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
    """Find the Chassis FFI library.

    Search order:
    1. CHASSIS_LIB_PATH environment variable
    2. Next to this Python file (for development)
    3. ../target/release (for development)
    4. System library paths (TODO: for installed packages)

    Returns:
        Path to the library

    Raises:
        FileNotFoundError: If library cannot be found
    """
    lib_name = _get_library_name()

    # 1. Environment variable
    if env_path := os.getenv("CHASSIS_LIB_PATH"):
        lib_path = Path(env_path) / lib_name
        if lib_path.exists():
            return lib_path

    # 2. Next to this file
    this_dir = Path(__file__).parent
    lib_path = this_dir / lib_name
    if lib_path.exists():
        return lib_path

    # 3. Development location (../../target/release from chassis/)
    dev_path = this_dir.parent.parent / "target" / "release" / lib_name
    if dev_path.exists():
        return dev_path

    # 4. Try system paths (ctypes.util.find_library)
    from ctypes.util import find_library

    if lib_path_str := find_library("chassis_ffi"):
        return Path(lib_path_str)

    raise FileNotFoundError(
        f"Could not find {lib_name}. "
        "Set CHASSIS_LIB_PATH environment variable or ensure library is built."
    )


# Load the library
_lib_path = _find_library()
_lib = ctypes.CDLL(str(_lib_path))


# Opaque pointer type
class ChassisIndex(ctypes.Structure):
    """Opaque handle to a Chassis index (never accessed directly)."""

    pass


ChassisIndexPtr = ctypes.POINTER(ChassisIndex)


# Function signatures

# chassis_open
_lib.chassis_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
_lib.chassis_open.restype = ChassisIndexPtr

# chassis_open_with_options
_lib.chassis_open_with_options.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_uint32,  # dimensions
    ctypes.c_uint32,  # max_connections
    ctypes.c_uint32,  # ef_construction
    ctypes.c_uint32,  # ef_search
]
_lib.chassis_open_with_options.restype = ChassisIndexPtr

# chassis_free
_lib.chassis_free.argtypes = [ChassisIndexPtr]
_lib.chassis_free.restype = None

# chassis_add
_lib.chassis_add.argtypes = [
    ChassisIndexPtr,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.chassis_add.restype = ctypes.c_uint64

# chassis_search
_lib.chassis_search.argtypes = [
    ChassisIndexPtr,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float),
]
_lib.chassis_search.restype = ctypes.c_size_t

# chassis_flush
_lib.chassis_flush.argtypes = [ChassisIndexPtr]
_lib.chassis_flush.restype = ctypes.c_int

# chassis_len
_lib.chassis_len.argtypes = [ChassisIndexPtr]
_lib.chassis_len.restype = ctypes.c_uint64

# chassis_is_empty
_lib.chassis_is_empty.argtypes = [ChassisIndexPtr]
_lib.chassis_is_empty.restype = ctypes.c_int

# chassis_dimensions
_lib.chassis_dimensions.argtypes = [ChassisIndexPtr]
_lib.chassis_dimensions.restype = ctypes.c_uint32

# chassis_last_error_message
_lib.chassis_last_error_message.argtypes = []
_lib.chassis_last_error_message.restype = ctypes.c_char_p

# chassis_version
_lib.chassis_version.argtypes = []
_lib.chassis_version.restype = ctypes.c_char_p


# Helper functions


def get_last_error() -> Optional[str]:
    """Get the last error message from the Chassis library.

    Returns:
        Error message string, or None if no error
    """
    err_ptr = _lib.chassis_last_error_message()
    if err_ptr:
        return err_ptr.decode("utf-8")
    return None


def get_version() -> str:
    """Get the Chassis library version.

    Returns:
        Version string (e.g., "0.1.0")
    """
    version_ptr = _lib.chassis_version()
    return version_ptr.decode("utf-8")


# Export public interface
__all__ = [
    "_lib",
    "ChassisIndex",
    "ChassisIndexPtr",
    "get_last_error",
    "get_version",
]
