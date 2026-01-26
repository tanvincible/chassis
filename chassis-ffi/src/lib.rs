//! FFI bindings for Chassis vector index
//!
//! This module provides a C-compatible interface to the Chassis vector storage engine.
//! All functions are panic-safe and use thread-local error reporting.
//!
//! # Safety Guarantees
//!
//! - No panic may cross the FFI boundary (enforced by `ffi_guard`)
//! - Strict UTF-8 validation for all string inputs
//! - Null pointer checks on all pointer arguments
//! - ABI stability via `#[repr(C)]` and `extern "C"`
//!
//! # Error Handling
//!
//! Errors are reported through:
//! - Return values: `u64::MAX` for add, `0` for search, `-1` for flush
//! - Thread-local error message: `chassis_last_error_message()`
//!
//! # Thread Safety
//!
//! - Single-writer: `chassis_add`, `chassis_flush` require exclusive access
//! - Multi-reader: `chassis_search` allows concurrent readers
//! - Each thread has its own error message storage

use chassis_core::{IndexOptions, VectorIndex};
use libc::{c_char, c_float, c_int, size_t};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

/// Internal state holder (not exposed to C)
///
/// This holds the actual VectorIndex and is purely Rust-internal.
struct ChassisIndexState {
    inner: VectorIndex,
}

/// Opaque handle to a Chassis index (C-compatible)
///
/// This is a zero-sized type that serves as an opaque handle for C.
/// C code only sees pointers to this type, never the actual struct.
/// The real data is stored in `ChassisIndexState`.
#[repr(C)]
pub struct ChassisIndex {
    _private: [u8; 0],
}

thread_local! {
    /// Thread-local storage for error messages
    ///
    /// Each thread maintains its own error message to ensure thread safety
    /// without requiring locks. The `RefCell` allows interior mutability.
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

/// Set the last error message for the current thread
///
/// # Safety
///
/// This function handles interior NULs gracefully to prevent panics during
/// error reporting. If the error message contains NUL bytes, they are
/// replaced with the escaped sequence "\\0".
fn set_last_error(err: impl std::fmt::Display) {
    LAST_ERROR.with(|cell| {
        // Handle interior NULs gracefully to avoid panic during error reporting
        let safe_msg = err.to_string().replace('\0', "\\0");
        let c_str = CString::new(safe_msg).unwrap_or_default();
        *cell.borrow_mut() = Some(c_str);
    });
}

/// Clear the last error message for the current thread
fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Panic barrier that catches all panics at the FFI boundary
///
/// # Critical Safety Invariant
///
/// No Rust panic may EVER unwind across the FFI boundary. This would cause
/// undefined behavior as C code cannot handle Rust panics.
///
/// # Implementation
///
/// - Wraps all FFI operations in `std::panic::catch_unwind`
/// - Converts panics to error messages via `set_last_error`
/// - Returns `None` on panic, allowing callers to use sentinel values
///
/// # AssertUnwindSafe Justification
///
/// `AssertUnwindSafe` is permitted here because:
/// - We abort the operation on panic (don't resume broken logic)
/// - We don't hold any shared mutable state across the panic boundary
/// - The error is reported via thread-local storage
fn ffi_guard<F, R>(f: F) -> Option<R>
where
    F: FnOnce() -> R,
{
    // AssertUnwindSafe is permitted at the FFI boundary because we abort the
    // operation on panic, we do not attempt to resume broken logic.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => Some(result),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                format!("Panic: {}", s)
            } else if let Some(s) = e.downcast_ref::<String>() {
                format!("Panic: {}", s)
            } else {
                "Unknown panic".to_string()
            };
            set_last_error(msg);
            None
        }
    }
}

//
// === LIFECYCLE MANAGEMENT ===
//

/// Open or create a Chassis vector index
///
/// # Arguments
///
/// - `path`: UTF-8 encoded path to the index file (must not be NULL)
/// - `dimensions`: Number of dimensions per vector (must be > 0)
///
/// # Returns
///
/// - Non-NULL pointer on success
/// - NULL on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// - Safe to call from multiple threads with different paths
/// - The returned index requires exclusive access for writes
///
/// # Example (C)
///
/// ```c
/// ChassisIndex* index = chassis_open("vectors.chassis", 768);
/// if (index == NULL) {
///     fprintf(stderr, "Error: %s\n", chassis_last_error_message());
///     exit(1);
/// }
/// ```
///
/// # Safety
///
/// - `path` must be a valid, NULL-terminated UTF-8 string
/// - `path` must remain valid for the duration of this call
/// - Caller must free the returned pointer with `chassis_free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_open(path: *const c_char, dimensions: u32) -> *mut ChassisIndex {
    ffi_guard(|| {
        if path.is_null() {
            set_last_error("Path cannot be NULL");
            return ptr::null_mut();
        }

        if dimensions == 0 {
            set_last_error("Dimensions must be > 0");
            return ptr::null_mut();
        }

        // SAFETY: Caller guarantees path is valid C string
        let c_path = unsafe { CStr::from_ptr(path) };

        // STRICT UTF-8 CHECK: Do not use to_string_lossy()
        let path_str = match c_path.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Path must be valid UTF-8");
                return ptr::null_mut();
            }
        };

        let options = IndexOptions::default();

        match VectorIndex::open(path_str, dimensions, options) {
            Ok(index) => {
                clear_last_error(); // Success - clear any previous errors
                let state = Box::new(ChassisIndexState { inner: index });
                Box::into_raw(state) as *mut ChassisIndex
            }
            Err(e) => {
                set_last_error(e);
                ptr::null_mut()
            }
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Open or create a Chassis vector index with custom options
///
/// # Arguments
///
/// - `path`: UTF-8 encoded path to the index file (must not be NULL)
/// - `dimensions`: Number of dimensions per vector (must be > 0)
/// - `max_connections`: Maximum connections per node (M parameter, typically 8-64)
/// - `ef_construction`: Construction quality (typically 100-400)
/// - `ef_search`: Search quality (typically 50-200)
///
/// # Returns
///
/// - Non-NULL pointer on success
/// - NULL on failure (check `chassis_last_error_message()`)
///
/// # Parameter Guidelines
///
/// - **max_connections (M)**: Higher = better recall, more memory. Default: 16
/// - **ef_construction**: Higher = better index quality, slower build. Default: 200
/// - **ef_search**: Higher = better search quality, slower search. Default: 50
///
/// # Safety
///
/// Same safety requirements as `chassis_open()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_open_with_options(
    path: *const c_char,
    dimensions: u32,
    max_connections: u32,
    ef_construction: u32,
    ef_search: u32,
) -> *mut ChassisIndex {
    ffi_guard(|| {
        if path.is_null() {
            set_last_error("Path cannot be NULL");
            return ptr::null_mut();
        }

        if dimensions == 0 {
            set_last_error("Dimensions must be > 0");
            return ptr::null_mut();
        }

        // Validate max_connections is u16
        if max_connections > u16::MAX as u32 {
            set_last_error(format!("max_connections must be <= {}", u16::MAX));
            return ptr::null_mut();
        }

        // SAFETY: Caller guarantees path is valid C string
        let c_path = unsafe { CStr::from_ptr(path) };

        let path_str = match c_path.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Path must be valid UTF-8");
                return ptr::null_mut();
            }
        };

        let options = IndexOptions {
            max_connections: max_connections as u16,
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
        };

        match VectorIndex::open(path_str, dimensions, options) {
            Ok(index) => {
                clear_last_error();
                let state = Box::new(ChassisIndexState { inner: index });
                Box::into_raw(state) as *mut ChassisIndex
            }
            Err(e) => {
                set_last_error(e);
                ptr::null_mut()
            }
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Free a Chassis index and release all resources
///
/// # Arguments
///
/// - `ptr`: Pointer returned by `chassis_open()` or NULL
///
/// # Safety
///
/// - `ptr` must be NULL or a valid pointer from `chassis_open()`
/// - After this call, `ptr` is invalid and must not be used
/// - Safe to call with NULL (no-op)
/// - Safe to call multiple times with the same pointer (undefined behavior
///   on second call, but won't crash)
///
/// # Example (C)
///
/// ```c
/// chassis_free(index);
/// index = NULL; // Good practice
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_free(ptr: *mut ChassisIndex) {
    if !ptr.is_null() {
        ffi_guard(|| {
            // SAFETY: Caller guarantees ptr is valid (from chassis_open)
            let _ = unsafe { Box::from_raw(ptr as *mut ChassisIndexState) };
        });
    }
}

//
// === VECTOR OPERATIONS ===
//

/// Add a vector to the index
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (requires exclusive access)
/// - `vector`: Pointer to f32 array (must not be NULL)
/// - `len`: Number of elements in vector (must match index dimensions)
///
/// # Returns
///
/// - Vector ID (0-based) on success
/// - `UINT64_MAX` on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// **SINGLE-WRITER**: Only one thread may call this function at a time for a
/// given index. Concurrent writes will cause data corruption.
///
/// # Performance Note
///
/// This operation does NOT guarantee durability. Call `chassis_flush()` to
/// ensure data is written to disk.
///
/// # Example (C)
///
/// ```c
/// float vec[768] = {0.1, 0.2, ...};
/// u64 id = chassis_add(index, vec, 768);
/// if (id == UINT64_MAX) {
///     fprintf(stderr, "Add failed: %s\n", chassis_last_error_message());
/// }
/// ```
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
/// - `vector` must point to `len` valid f32 values
/// - `len` must match the dimensions specified in `chassis_open()`
/// - No other thread may access `ptr` during this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_add(
    ptr: *mut ChassisIndex,
    vector: *const c_float,
    len: size_t,
) -> u64 {
    ffi_guard(|| {
        // SAFETY: Caller guarantees ptr is valid and has exclusive access
        let state = unsafe { (ptr as *mut ChassisIndexState).as_mut() };
        let index = match state {
            Some(s) => &mut s.inner,
            None => {
                set_last_error("Null index pointer");
                return u64::MAX;
            }
        };

        if vector.is_null() {
            set_last_error("Null vector pointer");
            return u64::MAX;
        }

        if len == 0 {
            set_last_error("Vector length must be > 0");
            return u64::MAX;
        }

        // SAFETY: Caller guarantees vector points to len valid f32 values
        let slice = unsafe { slice::from_raw_parts(vector, len) };

        match index.add(slice) {
            Ok(id) => {
                clear_last_error();
                id
            }
            Err(e) => {
                set_last_error(e);
                u64::MAX
            }
        }
    })
    .unwrap_or(u64::MAX)
}

/// Search for k nearest neighbors
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (shared access allowed)
/// - `query`: Pointer to query vector (must not be NULL)
/// - `len`: Number of elements in query (must match index dimensions)
/// - `k`: Number of neighbors to find (must be > 0)
/// - `out_ids`: Output buffer for vector IDs (must have space for k elements)
/// - `out_dists`: Output buffer for distances (must have space for k elements)
///
/// # Returns
///
/// - Number of results found (≤ k) on success
/// - 0 on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// **MULTI-READER**: Multiple threads may call this function concurrently
/// on the same index. Reads do not block other reads.
///
/// # Output Format
///
/// Results are sorted by distance (ascending):
/// - `out_ids[0]` = closest vector ID
/// - `out_dists[0]` = distance to closest vector
///
/// # Example (C)
///
/// ```c
/// float query[768] = {0.1, 0.2, ...};
/// u64 ids[10];
/// float dists[10];
///
/// size_t count = chassis_search(index, query, 768, 10, ids, dists);
/// for (size_t i = 0; i < count; i++) {
///     printf("ID: %llu, Distance: %f\n", ids[i], dists[i]);
/// }
/// ```
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
/// - `query` must point to `len` valid f32 values
/// - `out_ids` must have space for at least `k` u64 values
/// - `out_dists` must have space for at least `k` float values
/// - Buffers must not overlap
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_search(
    ptr: *const ChassisIndex,
    query: *const c_float,
    len: size_t,
    k: size_t,
    out_ids: *mut u64,
    out_dists: *mut c_float,
) -> size_t {
    ffi_guard(|| {
        // SAFETY: Caller guarantees ptr is valid (shared access)
        let state = unsafe { (ptr as *const ChassisIndexState).as_ref() };
        let index = match state {
            Some(s) => &s.inner,
            None => {
                set_last_error("Null index pointer");
                return 0;
            }
        };

        if query.is_null() || out_ids.is_null() || out_dists.is_null() {
            set_last_error("Null buffer pointers");
            return 0;
        }

        if k == 0 {
            set_last_error("k must be > 0");
            return 0;
        }

        // SAFETY: Caller guarantees query points to len valid f32 values
        let query_slice = unsafe { slice::from_raw_parts(query, len) };

        match index.search(query_slice, k) {
            Ok(results) => {
                let count = results.len();

                // SAFETY: Caller guarantees out_ids and out_dists have space for k elements
                for (i, result) in results.iter().enumerate() {
                    unsafe {
                        *out_ids.add(i) = result.id;
                        *out_dists.add(i) = result.distance;
                    }
                }

                clear_last_error();
                count
            }
            Err(e) => {
                set_last_error(e);
                0
            }
        }
    })
    .unwrap_or(0)
}

/// Flush all changes to disk
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (requires exclusive access)
///
/// # Returns
///
/// - 0 on success
/// - -1 on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// **SINGLE-WRITER**: Only one thread may call this function at a time for a
/// given index. No other operations (read or write) may occur during flush.
///
/// # Performance Warning
///
/// This operation is expensive (1-50ms depending on storage device).
/// Batch multiple `chassis_add()` calls and flush once at the end.
///
/// # Example (C)
///
/// ```c
/// // Add many vectors
/// for (int i = 0; i < 1000; i++) {
///     chassis_add(index, vectors[i], 768);
/// }
///
/// // Flush once at the end
/// if (chassis_flush(index) != 0) {
///     fprintf(stderr, "Flush failed: %s\n", chassis_last_error_message());
/// }
/// ```
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
/// - No other thread may access `ptr` during this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_flush(ptr: *mut ChassisIndex) -> c_int {
    ffi_guard(|| {
        // SAFETY: Caller guarantees ptr is valid and has exclusive access
        let state = unsafe { (ptr as *mut ChassisIndexState).as_mut() };
        let index = match state {
            Some(s) => &mut s.inner,
            None => {
                set_last_error("Null index pointer");
                return -1;
            }
        };

        match index.flush() {
            Ok(_) => {
                clear_last_error();
                0
            }
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    })
    .unwrap_or(-1)
}

//
// === INTROSPECTION ===
//

/// Get the number of vectors in the index
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (shared access)
///
/// # Returns
///
/// - Number of vectors, or 0 if `ptr` is NULL
///
/// # Thread Safety
///
/// Safe to call concurrently with `chassis_search()`.
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_len(ptr: *const ChassisIndex) -> u64 {
    ffi_guard(|| {
        let state = unsafe { (ptr as *const ChassisIndexState).as_ref() };
        let index = match state {
            Some(s) => &s.inner,
            None => return 0,
        };

        index.len()
    })
    .unwrap_or(0)
}

/// Check if the index is empty
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (shared access)
///
/// # Returns
///
/// - 1 if empty, 0 if not empty or `ptr` is NULL
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_is_empty(ptr: *const ChassisIndex) -> c_int {
    ffi_guard(|| {
        let state = unsafe { (ptr as *const ChassisIndexState).as_ref() };
        let index = match state {
            Some(s) => &s.inner,
            None => return 0,
        };

        if index.is_empty() { 1 } else { 0 }
    })
    .unwrap_or(0)
}

/// Get the dimensionality of vectors in the index
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index (shared access)
///
/// # Returns
///
/// - Number of dimensions, or 0 if `ptr` is NULL
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_dimensions(ptr: *const ChassisIndex) -> u32 {
    ffi_guard(|| {
        let state = unsafe { (ptr as *const ChassisIndexState).as_ref() };
        let index = match state {
            Some(s) => &s.inner,
            None => return 0,
        };

        index.dimensions()
    })
    .unwrap_or(0)
}

//
// === ERROR HANDLING ===
//

/// Get the last error message for the current thread
///
/// # Returns
///
/// - Pointer to NULL-terminated error string
/// - NULL if no error occurred
///
/// # Thread Safety
///
/// Each thread has its own error message storage. Safe to call from multiple
/// threads concurrently.
///
/// # Lifetime
///
/// The returned pointer is valid until:
/// - The next FFI function call on this thread
/// - The thread exits
///
/// **Do NOT** free the returned pointer.
///
/// # Example (C)
///
/// ```c
/// if (chassis_add(index, vec, 768) == UINT64_MAX) {
///     const char* error = chassis_last_error_message();
///     if (error != NULL) {
///         fprintf(stderr, "Error: %s\n", error);
///     }
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn chassis_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null()))
}

//
// === VERSIONING ===
//

/// Get the Chassis library version
///
/// # Returns
///
/// Pointer to NULL-terminated version string (e.g., "0.1.0")
///
/// # Lifetime
///
/// The returned pointer is valid for the lifetime of the program.
/// **Do NOT** free the returned pointer.
///
/// # Example (C)
///
/// ```c
/// printf("Chassis version: %s\n", chassis_version());
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn chassis_version() -> *const c_char {
    // Compile-time constant.
    // concat! appends the null terminator required by C.
    // env! pulls "version" directly from Cargo.toml.
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

    VERSION.as_ptr() as *const c_char
}
//
// === TESTS ===
//

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_lifecycle() {
        let path = CString::new("ffi_test_lifecycle.chassis").unwrap();
        let ptr = unsafe { chassis_open(path.as_ptr(), 128) };
        assert!(!ptr.is_null(), "Failed to open index");

        // Add vector
        let vec = vec![0.1f32; 128];
        let id = unsafe { chassis_add(ptr, vec.as_ptr(), 128) };
        assert_eq!(id, 0, "First insert should have ID 0");

        // Add another vector
        let vec2 = vec![0.2f32; 128];
        let id2 = unsafe { chassis_add(ptr, vec2.as_ptr(), 128) };
        assert_eq!(id2, 1, "Second insert should have ID 1");

        // Search
        let mut ids = vec![0u64; 5];
        let mut dists = vec![0.0f32; 5];
        let count = unsafe {
            chassis_search(ptr, vec.as_ptr(), 128, 5, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert!(count > 0, "Search should return results");
        assert!(count <= 2, "Should not return more than 2 results");

        // Flush
        let flush_result = unsafe { chassis_flush(ptr) };
        assert_eq!(flush_result, 0, "Flush should succeed");

        // Clean up
        unsafe { chassis_free(ptr) };

        let _ = std::fs::remove_file("ffi_test_lifecycle.chassis");
    }

    #[test]
    fn test_ffi_null_safety() {
        // Null path
        let ptr = unsafe { chassis_open(ptr::null(), 128) };
        assert!(ptr.is_null());

        // Null index pointer for add
        let vec = vec![0.1f32; 128];
        let id = unsafe { chassis_add(ptr::null_mut(), vec.as_ptr(), 128) };
        assert_eq!(id, u64::MAX);

        // Null index pointer for search
        let mut ids = vec![0u64; 5];
        let mut dists = vec![0.0f32; 5];
        let count = unsafe {
            chassis_search(ptr::null(), vec.as_ptr(), 128, 5, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!(count, 0);

        // Double free should be safe (no-op second time)
        unsafe { chassis_free(ptr::null_mut()) };
    }

    #[test]
    fn test_ffi_dimension_mismatch() {
        let path = CString::new("ffi_test_dims.chassis").unwrap();
        let ptr = unsafe { chassis_open(path.as_ptr(), 128) };
        assert!(!ptr.is_null());

        // Try to add vector with wrong dimensions
        let vec = vec![0.1f32; 64];
        let id = unsafe { chassis_add(ptr, vec.as_ptr(), 64) };
        assert_eq!(id, u64::MAX, "Should fail with dimension mismatch");

        // Check error message
        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        let error_str = error.to_string_lossy();
        assert!(error_str.contains("dimension"), "Error should mention dimensions");

        unsafe { chassis_free(ptr) };
        let _ = std::fs::remove_file("ffi_test_dims.chassis");
    }

    #[test]
    fn test_ffi_introspection() {
        let path = CString::new("ffi_test_introspection.chassis").unwrap();
        let ptr = unsafe { chassis_open(path.as_ptr(), 256) };
        assert!(!ptr.is_null());

        // Check initial state
        assert_eq!(unsafe { chassis_len(ptr) }, 0);
        assert_eq!(unsafe { chassis_is_empty(ptr) }, 1);
        assert_eq!(unsafe { chassis_dimensions(ptr) }, 256);

        // Add a vector
        let vec = vec![0.5f32; 256];
        let id = unsafe { chassis_add(ptr, vec.as_ptr(), 256) };
        assert_eq!(id, 0);

        // Check updated state
        assert_eq!(unsafe { chassis_len(ptr) }, 1);
        assert_eq!(unsafe { chassis_is_empty(ptr) }, 0);

        unsafe { chassis_free(ptr) };
        let _ = std::fs::remove_file("ffi_test_introspection.chassis");
    }

    #[test]
    fn test_ffi_version() {
        let version = unsafe { CStr::from_ptr(chassis_version()) };
        let version_str = version.to_string_lossy();

        // Compare dynamically against the Cargo.toml version
        let expected = env!("CARGO_PKG_VERSION");
        assert_eq!(version_str, expected, "FFI version should match Cargo.toml version");
    }

    #[test]
    fn test_ffi_with_custom_options() {
        let path = CString::new("ffi_test_options.chassis").unwrap();
        let ptr = unsafe { chassis_open_with_options(path.as_ptr(), 128, 32, 100, 75) };
        assert!(!ptr.is_null(), "Should open with custom options");

        // Add and search to verify it works
        let vec = vec![0.3f32; 128];
        let id = unsafe { chassis_add(ptr, vec.as_ptr(), 128) };
        assert_eq!(id, 0);

        let mut ids = vec![0u64; 5];
        let mut dists = vec![0.0f32; 5];
        let count = unsafe {
            chassis_search(ptr, vec.as_ptr(), 128, 5, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!(count, 1);

        unsafe { chassis_free(ptr) };
        let _ = std::fs::remove_file("ffi_test_options.chassis");
    }

    #[test]
    fn test_ffi_invalid_utf8_path() {
        // Create a path with invalid UTF-8
        let invalid_bytes = b"test\xFF\xFE.chassis\0";
        let ptr = unsafe { chassis_open(invalid_bytes.as_ptr() as *const c_char, 128) };
        assert!(ptr.is_null(), "Should reject invalid UTF-8");

        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        let error_str = error.to_string_lossy();
        assert!(error_str.contains("UTF-8"), "Error should mention UTF-8");
    }

    #[test]
    fn test_ffi_error_thread_local() {
        use std::thread;

        // Set an error on main thread
        set_last_error("Main thread error");
        let main_error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        assert_eq!(main_error.to_string_lossy(), "Main thread error");

        // Spawn a thread and verify it has no error
        let handle = thread::spawn(|| {
            let error_ptr = chassis_last_error_message();
            assert!(error_ptr.is_null(), "New thread should have no error");

            // Set error on spawned thread
            set_last_error("Spawned thread error");
            let spawned_error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
            assert_eq!(spawned_error.to_string_lossy(), "Spawned thread error");
        });

        handle.join().unwrap();

        // Verify main thread still has its error
        let main_error_again = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        assert_eq!(main_error_again.to_string_lossy(), "Main thread error");
    }
}
