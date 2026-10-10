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
//! - Return values: `u64::MAX` for add, `size_t` insert count for `chassis_add_batch` (`0` on
//!   failure: a batch is added whole or not at all), `0` for search, `-1` for flush
//! - Thread-local error message, `chassis_last_error_message()`, which says what was wrong and,
//!   on a line starting `help:`, what to do instead; and a code to act on,
//!   `chassis_last_error_code()`, one of the `CHASSIS_ERROR_` constants
//!
//! # Thread Safety
//!
//! - Every function except `chassis_free` may be called from any thread on a shared handle.
//!   Searches run concurrently. Adds, deletes and flushes take an internal write lock, so they run
//!   one at a time and searches wait for them.
//! - A handle from `chassis_open_reader` runs one search at a time; open one per thread to search
//!   in parallel. Any number of readers, in any processes, can open the file a writer has open.
//! - `chassis_free` must not race with any other call on the same handle.
//! - Each thread has its own error message storage

use chassis_core::{
    DistanceMetric, ErrorKind, IndexOptions, IndexReader, Precision, SearchResult, VectorIndex,
};
use libc::{c_char, c_float, c_int, size_t};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;
use std::sync::{Mutex, RwLock, RwLockWriteGuard};

/// Internal state holder (not exposed to C)
///
/// This holds the actual index and is purely Rust-internal.
struct ChassisIndexState {
    inner: Kind,
}

/// A handle opens an index to write it, or to read it while another process writes it.
enum Kind {
    Writer(RwLock<VectorIndex>),
    Reader(Mutex<IndexReader>),
}

/// What searches and introspection need, from either kind of handle.
type Filter<'a> = Option<&'a dyn Fn(u64) -> bool>;

trait Readable {
    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        filter: Filter,
    ) -> chassis_core::Result<Vec<SearchResult>>;
    fn len(&mut self) -> u64;
    fn dimensions(&self) -> u32;
    fn metric(&self) -> DistanceMetric;
    fn precision(&self) -> Precision;
}

impl Readable for &VectorIndex {
    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        filter: Filter,
    ) -> chassis_core::Result<Vec<SearchResult>> {
        match filter {
            None => VectorIndex::search(self, query, k),
            Some(filter) => VectorIndex::search_filtered(self, query, k, filter),
        }
    }
    fn len(&mut self) -> u64 {
        VectorIndex::len(self)
    }
    fn dimensions(&self) -> u32 {
        VectorIndex::dimensions(self)
    }
    fn metric(&self) -> DistanceMetric {
        VectorIndex::metric(self)
    }
    fn precision(&self) -> Precision {
        VectorIndex::precision(self)
    }
}

impl Readable for IndexReader {
    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        filter: Filter,
    ) -> chassis_core::Result<Vec<SearchResult>> {
        match filter {
            None => IndexReader::search(self, query, k),
            Some(filter) => IndexReader::search_filtered(self, query, k, filter),
        }
    }
    /// The writer's latest flush, not the last search's snapshot.
    fn len(&mut self) -> u64 {
        if let Err(e) = self.refresh() {
            set_last_error(e);
        }
        IndexReader::len(self)
    }
    fn dimensions(&self) -> u32 {
        IndexReader::dimensions(self)
    }
    fn metric(&self) -> DistanceMetric {
        IndexReader::metric(self)
    }
    fn precision(&self) -> Precision {
        IndexReader::precision(self)
    }
}

/// Borrows the handle, or sets the last error if `ptr` is NULL.
///
/// Only shared references are made, so concurrent calls never alias a `&mut`; writes get
/// mutable access through the lock.
///
/// # Safety
///
/// `ptr` must be NULL or a live handle from `chassis_open`.
unsafe fn state<'a>(ptr: *const ChassisIndex) -> Option<&'a ChassisIndexState> {
    // SAFETY: Caller guarantees ptr is NULL or a live handle
    let state = unsafe { (ptr as *const ChassisIndexState).as_ref() };
    if state.is_none() {
        set_last_error("The index pointer is NULL\nhelp: pass the handle chassis_open returned");
    }
    state
}

const POISONED: Failure = Failure::other(
    "The index is unusable after an earlier panic\nhelp: free the handle and open the index again",
);

/// Runs `f` on the index for a search or other read, or sets the last error.
///
/// # Safety
///
/// Same as `state`.
unsafe fn read<T>(ptr: *const ChassisIndex, f: impl FnOnce(&mut dyn Readable) -> T) -> Option<T> {
    let result = match &unsafe { state(ptr) }?.inner {
        Kind::Writer(lock) => lock.read().map(|index| f(&mut &*index)).map_err(|_| ()),
        Kind::Reader(lock) => lock.lock().map(|mut reader| f(&mut *reader)).map_err(|_| ()),
    };
    result.map_err(|()| set_last_error(POISONED)).ok()
}

/// Locks the index for an add, delete or flush, or sets the last error.
///
/// # Safety
///
/// Same as `state`.
unsafe fn write_index<'a>(ptr: *const ChassisIndex) -> Option<RwLockWriteGuard<'a, VectorIndex>> {
    match &unsafe { state(ptr) }?.inner {
        Kind::Writer(lock) => lock.write().map_err(|_| set_last_error(POISONED)).ok(),
        Kind::Reader(_) => {
            set_last_error(Failure {
                code: CHASSIS_ERROR_READ_ONLY,
                message: "This index was opened as a reader, which only searches\nhelp: to \
                          add, delete, flush or compact, open it as a writer (chassis_open); an \
                          index has one writer at a time"
                    .into(),
            });
            None
        }
    }
}

/// Validates the arguments of an open call and opens the index, or sets the last error.
///
/// # Safety
///
/// `path` must be NULL or a NUL-terminated string.
unsafe fn open_handle(
    path: *const c_char,
    dimensions: u32,
    options: IndexOptions,
    read_only: bool,
) -> *mut ChassisIndex {
    if path.is_null() {
        set_last_error(
            "The path is NULL\nhelp: pass the index file's path as a NUL-terminated string",
        );
        return ptr::null_mut();
    }
    if dimensions == 0 {
        set_last_error(
            "dimensions is 0, but has to be between 1 and 4096\nhelp: pass the length of the \
             vectors the index will hold, such as 384, 768 or 1536",
        );
        return ptr::null_mut();
    }
    // SAFETY: Caller guarantees path is valid C string
    let c_path = unsafe { CStr::from_ptr(path) };
    // STRICT UTF-8 CHECK: Do not use to_string_lossy()
    let Ok(path_str) = c_path.to_str() else {
        set_last_error("The path isn't valid UTF-8\nhelp: pass the path encoded as UTF-8");
        return ptr::null_mut();
    };
    let inner = if read_only {
        IndexReader::open(path_str, dimensions, options).map(|r| Kind::Reader(Mutex::new(r)))
    } else {
        VectorIndex::open(path_str, dimensions, options).map(|i| Kind::Writer(RwLock::new(i)))
    };
    match inner {
        Ok(inner) => {
            clear_last_error(); // Success - clear any previous errors
            Box::into_raw(Box::new(ChassisIndexState { inner })) as *mut ChassisIndex
        }
        Err(e) => {
            set_last_error(e);
            ptr::null_mut()
        }
    }
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

/// No error: what `chassis_last_error_code` returns after a call that succeeded.
pub const CHASSIS_OK: c_int = 0;
/// An argument is out of range or malformed: a NULL pointer, dimensions, an option, an id, a
/// vector with a component that isn't a finite number.
pub const CHASSIS_ERROR_INVALID_ARGUMENT: c_int = 1;
/// A vector or query has another number of components than the index's vectors, or the file
/// holds vectors of other dimensions than asked for.
pub const CHASSIS_ERROR_DIMENSION_MISMATCH: c_int = 2;
/// The file was created with another metric or precision than the one asked for.
pub const CHASSIS_ERROR_OPTIONS_MISMATCH: c_int = 3;
/// There is no index at the path, or its directory doesn't exist.
pub const CHASSIS_ERROR_NOT_FOUND: c_int = 4;
/// The file is not a Chassis index.
pub const CHASSIS_ERROR_NOT_AN_INDEX: c_int = 5;
/// The id is already in use.
pub const CHASSIS_ERROR_ID_IN_USE: c_int = 6;
/// Another writer has the index open.
pub const CHASSIS_ERROR_LOCKED: c_int = 7;
/// The handle is a reader, which only searches.
pub const CHASSIS_ERROR_READ_ONLY: c_int = 8;
/// The file is damaged.
pub const CHASSIS_ERROR_CORRUPT: c_int = 9;
/// The index can hold no more.
pub const CHASSIS_ERROR_FULL: c_int = 10;
/// The operating system refused: permissions, a full disk, a failed read or write.
pub const CHASSIS_ERROR_IO: c_int = 11;
/// Anything else, a panic included.
pub const CHASSIS_ERROR_OTHER: c_int = 12;

/// An error as C sees it: a code and a message.
struct Failure {
    code: c_int,
    message: std::borrow::Cow<'static, str>,
}

impl Failure {
    const fn other(message: &'static str) -> Self {
        Self { code: CHASSIS_ERROR_OTHER, message: std::borrow::Cow::Borrowed(message) }
    }
}

impl From<chassis_core::Error> for Failure {
    fn from(e: chassis_core::Error) -> Self {
        let code = match e.kind() {
            ErrorKind::InvalidArgument => CHASSIS_ERROR_INVALID_ARGUMENT,
            ErrorKind::DimensionMismatch => CHASSIS_ERROR_DIMENSION_MISMATCH,
            ErrorKind::OptionsMismatch => CHASSIS_ERROR_OPTIONS_MISMATCH,
            ErrorKind::NotFound => CHASSIS_ERROR_NOT_FOUND,
            ErrorKind::NotAnIndex => CHASSIS_ERROR_NOT_AN_INDEX,
            ErrorKind::IdInUse => CHASSIS_ERROR_ID_IN_USE,
            ErrorKind::Locked => CHASSIS_ERROR_LOCKED,
            ErrorKind::ReadOnly => CHASSIS_ERROR_READ_ONLY,
            ErrorKind::Corrupt => CHASSIS_ERROR_CORRUPT,
            ErrorKind::Full => CHASSIS_ERROR_FULL,
            ErrorKind::Io => CHASSIS_ERROR_IO,
            _ => CHASSIS_ERROR_OTHER,
        };
        Self { code, message: e.to_string().into() }
    }
}

/// The C layer's own checks of its arguments.
impl From<&'static str> for Failure {
    fn from(message: &'static str) -> Self {
        Self { code: CHASSIS_ERROR_INVALID_ARGUMENT, message: message.into() }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self { code: CHASSIS_ERROR_INVALID_ARGUMENT, message: message.into() }
    }
}

thread_local! {
    /// The last error on this thread: its code and message.
    static LAST_ERROR: RefCell<Option<(c_int, CString)>> = const { RefCell::new(None) };
}

/// Sets the last error for the current thread. A NUL inside the message becomes "\\0", so that
/// reporting an error can't panic.
fn set_last_error(err: impl Into<Failure>) {
    let Failure { code, message } = err.into();
    let message = CString::new(message.replace('\0', "\\0")).unwrap_or_default();
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some((code, message)));
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
            let what = if let Some(s) = e.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "no message".to_string()
            };
            set_last_error(Failure {
                code: CHASSIS_ERROR_OTHER,
                message: format!(
                    "Chassis panicked: {what}\nhelp: this is a bug in Chassis; please report it, \
                     with this message"
                )
                .into(),
            });
            None
        }
    }
}

//
//  LIFECYCLE MANAGEMENT
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
/// - The returned handle may be shared across threads (see the crate's Thread Safety notes)
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
    ffi_guard(|| unsafe { open_handle(path, dimensions, IndexOptions::default(), false) })
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
        // Validate max_connections is u16
        let Ok(max_connections) = u16::try_from(max_connections) else {
            set_last_error(format!(
                "max_connections is {max_connections}, but has to be between 2 and 32,767\nhelp: 16 \
                 suits most indexes; 32 gives higher recall for more memory"
            ));
            return ptr::null_mut();
        };
        let options = IndexOptions {
            max_connections,
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
            ..Default::default()
        };
        unsafe { open_handle(path, dimensions, options, false) }
    })
    .unwrap_or(ptr::null_mut())
}

/// Open or create a Chassis vector index with custom options and a distance metric
///
/// # Arguments
///
/// - `metric`: `0` for Euclidean (L2) distance, `1` for cosine distance (1 - cosine similarity).
///   A cosine index stores vectors scaled to unit length, rejects zero vectors, and reports
///   distances from 0 to 2. The metric is fixed when the index is created; reopening an existing
///   index with another one fails.
/// - The other arguments are as for `chassis_open_with_options()`.
///
/// # Safety
///
/// Same safety requirements as `chassis_open()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_open_with_metric(
    path: *const c_char,
    dimensions: u32,
    max_connections: u32,
    ef_construction: u32,
    ef_search: u32,
    metric: u32,
) -> *mut ChassisIndex {
    // SAFETY: the caller's.
    unsafe {
        chassis_open_with_precision(
            path,
            dimensions,
            max_connections,
            ef_construction,
            ef_search,
            metric,
            0,
        )
    }
}

/// Open or create a Chassis vector index with custom options, a distance metric and a precision
///
/// # Arguments
///
/// - `precision`: `0` to keep each component of a vector as a 32-bit float, `1` as a 16-bit
///   float (half precision). Half precision halves the vectors' size in the file and in memory;
///   each component is rounded to about three decimal digits, and a vector with a component of
///   65,520 or more in magnitude is refused. The precision is fixed when the index is created;
///   reopening an existing index with another one fails. A release without this function can't
///   open a file in half precision.
/// - The other arguments are as for `chassis_open_with_metric()`.
///
/// # Safety
///
/// Same safety requirements as `chassis_open()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_open_with_precision(
    path: *const c_char,
    dimensions: u32,
    max_connections: u32,
    ef_construction: u32,
    ef_search: u32,
    metric: u32,
    precision: u32,
) -> *mut ChassisIndex {
    ffi_guard(|| {
        let Ok(max_connections) = u16::try_from(max_connections) else {
            set_last_error(format!(
                "max_connections is {max_connections}, but has to be between 2 and 32,767\nhelp: 16 \
                 suits most indexes; 32 gives higher recall for more memory"
            ));
            return ptr::null_mut();
        };
        let metric = match metric {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Cosine,
            other => {
                set_last_error(format!(
                    "metric is {other}, which is no metric\nhelp: pass 0 for euclidean distance \
                     or 1 for cosine"
                ));
                return ptr::null_mut();
            }
        };
        let precision = match precision {
            0 => Precision::Full,
            1 => Precision::Half,
            other => {
                set_last_error(format!(
                    "precision is {other}, which is no precision\nhelp: pass 0 for full (32-bit \
                     floats) or 1 for half (16-bit floats)"
                ));
                return ptr::null_mut();
            }
        };
        let options = IndexOptions {
            max_connections,
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
            metric,
            precision,
            ..Default::default()
        };
        unsafe { open_handle(path, dimensions, options, false) }
    })
    .unwrap_or(ptr::null_mut())
}

/// Open an index to search it while a writer, possibly in another process, adds to it
///
/// Takes no lock, so any number of readers can open the file next to one writer. Every search
/// first takes a new snapshot: it returns what the writer's last flush committed, and nothing that
/// flush deleted. Adds, deletes and flushes on the handle fail. The file must already exist in the
/// current format: open it once with `chassis_open` to create or migrate it.
///
/// # Arguments
///
/// - `path`: UTF-8 encoded path to the index file (must not be NULL)
/// - `dimensions`: Number of dimensions per vector (must be > 0)
/// - `max_connections`: The value the index was created with (16 by default)
/// - `ef_search`: Search quality parameter
///
/// # Returns
///
/// - Non-NULL pointer on success; free it with `chassis_free()`
/// - NULL on failure (check `chassis_last_error_message()`)
///
/// # Safety
///
/// Same safety requirements as `chassis_open()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_open_reader(
    path: *const c_char,
    dimensions: u32,
    max_connections: u32,
    ef_search: u32,
) -> *mut ChassisIndex {
    ffi_guard(|| {
        let Ok(max_connections) = u16::try_from(max_connections) else {
            set_last_error(format!(
                "max_connections is {max_connections}, but has to be between 2 and 32,767\nhelp: 16 \
                 suits most indexes; 32 gives higher recall for more memory"
            ));
            return ptr::null_mut();
        };
        let options =
            IndexOptions { max_connections, ef_search: ef_search as usize, ..Default::default() };
        unsafe { open_handle(path, dimensions, options, true) }
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
/// - Must not be called more than once with the same non-NULL pointer
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
//  VECTOR OPERATIONS
//

/// Add a vector to the index
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index
/// - `vector`: Pointer to f32 array (must not be NULL)
/// - `len`: Number of elements in vector (must match index dimensions)
///
/// # Returns
///
/// - The id assigned (one past the largest id used so far) on success
/// - `UINT64_MAX` on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// Safe from any thread. Writes run one at a time; searches wait for them.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_add(
    ptr: *mut ChassisIndex,
    vector: *const c_float,
    len: size_t,
) -> u64 {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return u64::MAX;
        };

        if vector.is_null() {
            set_last_error(
                "The vector pointer is NULL\nhelp: pass a pointer to the vector's floats",
            );
            return u64::MAX;
        }

        if len == 0 {
            set_last_error(
                "len is 0\nhelp: pass the number of floats in the vector: the index's dimensions",
            );
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

/// Add multiple vectors to the index in one call (row-major layout), linking them on every core
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index
/// - `vectors`: Contiguous `count * dim` floats: row `i` is
///   `vectors[i*dim .. (i+1)*dim]`
/// - `count`: Number of vectors to insert
/// - `dim`: Elements per vector (must match index dimensions)
/// - `out_ids`: Output buffer for assigned IDs, length at least `count` (if `count > 0`)
///
/// # Returns
///
/// - `count` on success
/// - `0` on failure, with none of the batch added; use `chassis_last_error_message()` for the
///   reason
/// - If `count == 0`, returns `0` and succeeds (pointers need not be valid)
///
/// # Thread Safety
///
/// Same as `chassis_add()`. The whole batch holds the write lock.
///
/// # Performance Note
///
/// Links the batch on every core, so a large batch builds many times faster than adding one
/// vector at a time; the graph then depends on thread timing. Does not by itself change
/// durability: call `chassis_flush()` when you need data on disk.
///
/// # Example (C)
///
/// ```c
/// float *batch; // count * dim elements, row-major
/// uint64_t ids[1000];
/// size_t n = chassis_add_batch(index, batch, 1000, 768, ids);
/// if (n == 0) {
///     fprintf(stderr, "Batch add failed: %s\n", chassis_last_error_message());
/// }
/// ```
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
/// - If `count > 0`, `vectors` and `out_ids` must be non-NULL; `vectors` must point
///   to `count * dim` valid floats
/// - `dim` must match dimensions passed to `chassis_open()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_add_batch(
    ptr: *mut ChassisIndex,
    vectors: *const c_float,
    count: size_t,
    dim: size_t,
    out_ids: *mut u64,
) -> size_t {
    ffi_guard(|| {
        if ptr.is_null() {
            set_last_error(
                "The index pointer is NULL\nhelp: pass the handle chassis_open returned",
            );
            return 0;
        }

        if count == 0 {
            clear_last_error();
            return 0;
        }

        if vectors.is_null() || out_ids.is_null() {
            set_last_error("A buffer pointer is NULL\nhelp: pass a pointer to each buffer");
            return 0;
        }

        if dim == 0 {
            set_last_error(
                "dim is 0\nhelp: pass the number of floats in each vector: the index's dimensions",
            );
            return 0;
        }

        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return 0;
        };

        let index_dim = index.dimensions() as usize;
        if dim != index_dim {
            set_last_error(Failure {
                code: CHASSIS_ERROR_DIMENSION_MISMATCH,
                message: format!(
                    "dim is {dim}, but this index holds vectors of {index_dim}\nhelp: pass \
                     dim = {index_dim}, with the vectors back to back, {index_dim} floats each"
                )
                .into(),
            });
            return 0;
        }

        let total = match dim.checked_mul(count) {
            Some(t) => t,
            None => {
                set_last_error("count × dim overflows\nhelp: pass the batch in smaller pieces");
                return 0;
            }
        };

        // SAFETY: Caller guarantees `vectors` points to at least `total` floats
        let data = unsafe { slice::from_raw_parts(vectors, total) };

        match index.add_batch(data) {
            Ok(ids) => {
                // SAFETY: Caller guarantees out_ids has room for `count` ids
                unsafe { slice::from_raw_parts_mut(out_ids, count) }.copy_from_slice(&ids);
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

/// Add a vector under the caller's id
///
/// # Returns
///
/// - `0` on success
/// - `-1` on failure, including when a live vector already has `id` or `id` is `UINT64_MAX`
///   (check `chassis_last_error_message()`)
///
/// # Safety
///
/// Same as `chassis_add()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_add_with_id(
    ptr: *mut ChassisIndex,
    id: u64,
    vector: *const c_float,
    len: size_t,
) -> c_int {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return -1;
        };

        if vector.is_null() || len == 0 {
            set_last_error(
                "The query pointer is NULL or len is 0\nhelp: pass the query's floats and their \
                 number, the index's dimensions",
            );
            return -1;
        }

        // SAFETY: Caller guarantees vector points to len valid f32 values
        let slice = unsafe { slice::from_raw_parts(vector, len) };

        match index.add_with_id(id, slice) {
            Ok(()) => {
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

/// Add multiple vectors under the caller's ids in one call, linking them on every core
///
/// `ids[i]` is the id of row `i` of `vectors` (`count * dim` floats, row-major).
///
/// # Returns
///
/// - `0` on success, including when `count == 0`
/// - `-1` on failure, with none of the batch added: an id repeats, already exists or is
///   `UINT64_MAX`, or the dimensions don't match (check `chassis_last_error_message()`)
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
/// - If `count > 0`, `ids` must point to `count` ids and `vectors` to `count * dim` floats
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_add_batch_with_ids(
    ptr: *mut ChassisIndex,
    ids: *const u64,
    vectors: *const c_float,
    count: size_t,
    dim: size_t,
) -> c_int {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return -1;
        };
        if count == 0 {
            clear_last_error();
            return 0;
        }
        if ids.is_null() || vectors.is_null() {
            set_last_error("A buffer pointer is NULL\nhelp: pass a pointer to each buffer");
            return -1;
        }
        let Some(total) = dim.checked_mul(count) else {
            set_last_error("count × dim overflows\nhelp: pass the batch in smaller pieces");
            return -1;
        };
        if dim != index.dimensions() as usize {
            let held = index.dimensions();
            set_last_error(Failure {
                code: CHASSIS_ERROR_DIMENSION_MISMATCH,
                message: format!(
                    "dim is {dim}, but this index holds vectors of {held}\nhelp: pass dim = {held}, \
                     with the vectors back to back, {held} floats each"
                )
                .into(),
            });
            return -1;
        }
        // SAFETY: caller guarantees both buffers are this long
        let (ids, data) =
            unsafe { (slice::from_raw_parts(ids, count), slice::from_raw_parts(vectors, total)) };
        match index.add_batch_with_ids(ids, data) {
            Ok(()) => {
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

/// Delete the vector with `id`
///
/// Search stops returning it immediately; the delete is durable after the next
/// `chassis_flush()`.
///
/// # Returns
///
/// - `1` if it was deleted
/// - `0` if no live vector has `id`
/// - `-1` on failure (check `chassis_last_error_message()`)
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_delete(ptr: *mut ChassisIndex, id: u64) -> c_int {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return -1;
        };

        match index.delete(id) {
            Ok(deleted) => {
                clear_last_error();
                c_int::from(deleted)
            }
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    })
    .unwrap_or(-1)
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
/// Safe from any thread. Searches run concurrently with each other and wait for writes.
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
        unsafe { read(ptr, |index| search_into(index, query, len, k, None, out_ids, out_dists)) }
            .unwrap_or(0)
    })
    .unwrap_or(0)
}

/// Search for the k nearest neighbors among the given ids
///
/// # Arguments
///
/// - `allowed_ids`: The ids results may have; ids not in the index are ignored. May be NULL when
///   `allowed_len` is 0, which matches nothing.
/// - `allowed_len`: Number of ids in `allowed_ids`
/// - The other arguments are as for `chassis_search()`.
///
/// When walking the graph would cost more, as when few vectors match, the search checks every
/// vector instead and returns the exact nearest.
///
/// # Returns
///
/// As for `chassis_search()`.
///
/// # Safety
///
/// As for `chassis_search()`, and `allowed_ids` must point to `allowed_len` valid u64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_search_filtered(
    ptr: *const ChassisIndex,
    query: *const c_float,
    len: size_t,
    k: size_t,
    allowed_ids: *const u64,
    allowed_len: size_t,
    out_ids: *mut u64,
    out_dists: *mut c_float,
) -> size_t {
    ffi_guard(|| {
        if allowed_ids.is_null() && allowed_len > 0 {
            set_last_error(
                "allowed_ids is NULL\nhelp: pass the ids a result may have, or call chassis_search \
                 to search them all",
            );
            return 0;
        }
        let allowed: std::collections::HashSet<u64> = if allowed_len == 0 {
            Default::default()
        } else {
            // SAFETY: caller guarantees allowed_ids points to allowed_len u64 values
            unsafe { slice::from_raw_parts(allowed_ids, allowed_len) }.iter().copied().collect()
        };
        let filter = |id| allowed.contains(&id);
        unsafe {
            read(ptr, |index| search_into(index, query, len, k, Some(&filter), out_ids, out_dists))
        }
        .unwrap_or(0)
    })
    .unwrap_or(0)
}

/// The body of `chassis_search`, once the index is locked.
///
/// # Safety
///
/// As for `chassis_search`.
unsafe fn search_into(
    index: &mut dyn Readable,
    query: *const c_float,
    len: size_t,
    k: size_t,
    filter: Filter,
    out_ids: *mut u64,
    out_dists: *mut c_float,
) -> size_t {
    {
        if query.is_null() || out_ids.is_null() || out_dists.is_null() {
            set_last_error("A buffer pointer is NULL\nhelp: pass a pointer to each buffer");
            return 0;
        }

        if k == 0 {
            set_last_error("k is 0\nhelp: pass how many results to return, at least 1");
            return 0;
        }

        // SAFETY: Caller guarantees query points to len valid f32 values
        let query_slice = unsafe { slice::from_raw_parts(query, len) };

        match index.search(query_slice, k, filter) {
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
    }
}

/// Flush all changes to disk
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index
///
/// # Returns
///
/// - 0 on success
/// - -1 on failure (check `chassis_last_error_message()`)
///
/// # Thread Safety
///
/// Safe from any thread. Writes run one at a time; searches wait for them.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_flush(ptr: *mut ChassisIndex) -> c_int {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return -1;
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

/// Ask the operating system to keep the index's vectors on huge pages
///
/// On an index too large for the CPU's caches, searches are up to a quarter faster and batch
/// builds a little faster. Only on Linux, and only where the kernel and filesystem keep files on huge
/// pages (ext4 on Linux 6.17 does); elsewhere this does nothing. It is off unless asked for: with
/// it a page not yet in memory is read 2 MB at a time, which an index much larger than memory
/// pays for on every miss. Call it right after opening, on a writer's handle or a reader's.
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index
///
/// # Returns
///
/// - 0 on success
/// - -1 on failure (check `chassis_last_error_message()`)
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_use_huge_pages(ptr: *mut ChassisIndex) -> c_int {
    ffi_guard(|| {
        let Some(state) = (unsafe { state(ptr) }) else {
            return -1;
        };
        let asked = match &state.inner {
            Kind::Writer(lock) => lock.write().map(|mut index| index.use_huge_pages()).is_ok(),
            Kind::Reader(lock) => lock.lock().map(|mut reader| reader.use_huge_pages()).is_ok(),
        };
        if asked {
            clear_last_error();
            0
        } else {
            set_last_error(POISONED);
            -1
        }
    })
    .unwrap_or(-1)
}

/// Start reading the whole index into memory on another thread
///
/// Doesn't wait for the reading: searches go on meanwhile and stop waiting for the disk one page
/// at a time as the index arrives. For an index that fits in memory: one much larger would push
/// everything else out. What is read is what the index holds when this is called; while that is
/// under way, calling again does nothing. A reader that opens the file again after a compaction
/// reads the new one in too. It takes Linux 5.14 or macOS: an older Linux reads only some of the
/// index, and Windows none. Call it on a writer's handle or a reader's, right after opening or
/// later.
///
/// # Arguments
///
/// - `ptr`: Non-NULL pointer to index
///
/// # Returns
///
/// - 0 once asked, whether or not this system can do it
/// - -1 on failure (check `chassis_last_error_message()`)
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_warm(ptr: *mut ChassisIndex) -> c_int {
    ffi_guard(|| {
        let Some(state) = (unsafe { state(ptr) }) else {
            return -1;
        };
        let asked = match &state.inner {
            Kind::Writer(lock) => lock.write().map(|mut index| index.warm()).is_ok(),
            Kind::Reader(lock) => lock.lock().map(|mut reader| reader.warm()).is_ok(),
        };
        if asked {
            clear_last_error();
            0
        } else {
            set_last_error(POISONED);
            -1
        }
    })
    .unwrap_or(-1)
}

/// Rewrite the index without its deleted vectors and with a newly built graph
///
/// Reclaims the space of deleted vectors and replaces the index file with the copy. Ids don't
/// change. Like `chassis_flush()`, it makes every add and delete so far durable. Takes as long
/// as building the index, on every core, and needs free disk for a second copy of the live
/// vectors. Readers in other processes keep searching and move to the new file by themselves.
///
/// # Returns
///
/// - `0` on success
/// - `-1` on failure, with the index left as it was (check `chassis_last_error_message()`). On
///   Windows it fails while another process has the index open.
///
/// # Thread Safety
///
/// Same as `chassis_flush()`: it holds the write lock, so searches on this handle wait.
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_compact(ptr: *mut ChassisIndex) -> c_int {
    ffi_guard(|| {
        let Some(mut index) = (unsafe { write_index(ptr) }) else {
            return -1;
        };
        match index.compact() {
            Ok(()) => {
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
//  INTROSPECTION
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
    ffi_guard(|| unsafe { read(ptr, |index| index.len()) }.unwrap_or(0)).unwrap_or(0)
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
    ffi_guard(|| unsafe { read(ptr, |index| c_int::from(index.len() == 0)) }.unwrap_or(0))
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
    ffi_guard(|| unsafe { read(ptr, |index| index.dimensions()) }.unwrap_or(0)).unwrap_or(0)
}

/// Get the distance metric the index was created with
///
/// # Returns
///
/// - `0` for Euclidean, `1` for cosine, as for `chassis_open_with_metric()`
/// - `-1` if `ptr` is NULL
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_metric(ptr: *const ChassisIndex) -> c_int {
    let metric = |index: &mut dyn Readable| match index.metric() {
        DistanceMetric::Cosine => 1,
        _ => 0,
    };
    ffi_guard(|| unsafe { read(ptr, metric) }.unwrap_or(-1)).unwrap_or(-1)
}

/// Get the precision the index keeps its vectors in
///
/// # Returns
///
/// - `0` for full precision, `1` for half, as for `chassis_open_with_precision()`
/// - `-1` if `ptr` is NULL
///
/// # Safety
///
/// - `ptr` must be non-NULL and valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chassis_precision(ptr: *const ChassisIndex) -> c_int {
    let precision = |index: &mut dyn Readable| match index.precision() {
        Precision::Half => 1,
        _ => 0,
    };
    ffi_guard(|| unsafe { read(ptr, precision) }.unwrap_or(-1)).unwrap_or(-1)
}

//
//  ERROR HANDLING
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
    LAST_ERROR
        .with(|cell| cell.borrow().as_ref().map_or(ptr::null(), |(_, message)| message.as_ptr()))
}

/// What the last error on this thread was, for a program to act on
///
/// # Returns
///
/// One of the `CHASSIS_ERROR_` constants, or `CHASSIS_OK` (0) if the last call succeeded or set
/// no error. `chassis_last_error_message` says the same in words, with what to do instead.
///
/// # Example (C)
///
/// ```c
/// ChassisIndex* index = chassis_open("vectors.chassis", 768);
/// if (index == NULL && chassis_last_error_code() == CHASSIS_ERROR_LOCKED) {
///     index = chassis_open_reader("vectors.chassis", 768, 16, 50);
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn chassis_last_error_code() -> c_int {
    LAST_ERROR.with(|cell| cell.borrow().as_ref().map_or(CHASSIS_OK, |&(code, _)| code))
}

//
//  VERSIONING
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
//  TESTS
//

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::TempDir;

    fn temp_index_path() -> (TempDir, CString) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.chassis");
        let path = CString::new(path.to_str().unwrap()).unwrap();
        (dir, path)
    }

    #[test]
    fn test_ffi_lifecycle() {
        let (_dir, path) = temp_index_path();
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
    }

    #[test]
    fn test_ffi_add_with_id_and_delete() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 8) };
        let vec = [0.5f32; 8];

        assert_eq!(unsafe { chassis_add_with_id(ptr, 42, vec.as_ptr(), 8) }, 0);
        assert_eq!(unsafe { chassis_add_with_id(ptr, 42, vec.as_ptr(), 8) }, -1);

        let mut ids = [0u64; 1];
        let mut dists = [0.0f32; 1];
        unsafe { chassis_search(ptr, vec.as_ptr(), 8, 1, ids.as_mut_ptr(), dists.as_mut_ptr()) };
        assert_eq!(ids[0], 42);

        assert_eq!(unsafe { chassis_delete(ptr, 42) }, 1);
        assert_eq!(unsafe { chassis_delete(ptr, 42) }, 0);
        assert_eq!(unsafe { chassis_delete(ptr::null_mut(), 42) }, -1);
        assert_eq!(unsafe { chassis_len(ptr) }, 0);

        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_cosine_metric() {
        let (_dir, path) = temp_index_path();
        assert!(unsafe { chassis_open_with_metric(path.as_ptr(), 2, 16, 200, 50, 2) }.is_null());
        let ptr = unsafe { chassis_open_with_metric(path.as_ptr(), 2, 16, 200, 50, 1) };
        assert!(!ptr.is_null());
        assert_ne!(unsafe { chassis_add(ptr, [3.0f32, 0.0].as_ptr(), 2) }, u64::MAX);
        assert_ne!(unsafe { chassis_add(ptr, [0.0f32, 0.5].as_ptr(), 2) }, u64::MAX);
        assert_eq!(unsafe { chassis_add(ptr, [0.0f32, 0.0].as_ptr(), 2) }, u64::MAX);
        let (mut ids, mut dists) = ([0u64; 2], [0.0f32; 2]);
        let n = unsafe {
            chassis_search(ptr, [1.0f32, 1.0].as_ptr(), 2, 2, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        // Both are 45 degrees from the query: 1 - cos 45° = 0.29, whatever their lengths.
        assert_eq!(n, 2);
        assert!(dists.iter().all(|d| (d - (1.0 - 0.5f32.sqrt())).abs() < 1e-5), "{dists:?}");
        assert_eq!(unsafe { chassis_metric(ptr) }, 1);
        unsafe { chassis_free(ptr) };
        assert!(
            unsafe { chassis_open(path.as_ptr(), 2) }.is_null(),
            "reopened with another metric"
        );
        let reader = unsafe { chassis_open_reader(path.as_ptr(), 2, 16, 50) };
        assert_eq!(unsafe { chassis_metric(reader) }, 1);
        unsafe { chassis_free(reader) };
    }

    #[test]
    fn test_ffi_half_precision() {
        let (_dir, path) = temp_index_path();
        let open = |precision| unsafe {
            chassis_open_with_precision(path.as_ptr(), 2, 16, 200, 50, 0, precision)
        };
        assert!(open(2).is_null());
        let ptr = open(1);
        assert!(!ptr.is_null());
        assert_eq!(unsafe { chassis_precision(ptr) }, 1);
        // A third is not a half; it is kept as the nearest one.
        assert_ne!(unsafe { chassis_add(ptr, [1.0f32 / 3.0, 0.0].as_ptr(), 2) }, u64::MAX);
        assert_eq!(unsafe { chassis_add(ptr, [1e6f32, 0.0].as_ptr(), 2) }, u64::MAX);
        assert_eq!(unsafe { chassis_len(ptr) }, 1);
        let (mut ids, mut dists) = ([0u64; 1], [0.0f32; 1]);
        let query = [1.0f32 / 3.0, 0.0];
        let n = unsafe {
            chassis_search(ptr, query.as_ptr(), 2, 1, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!((n, ids[0]), (1, 0));
        assert!(dists[0] > 0.0 && dists[0] < 1e-4, "{dists:?}");
        assert_eq!(unsafe { chassis_flush(ptr) }, 0);
        unsafe { chassis_free(ptr) };

        assert!(open(0).is_null(), "reopened with another precision");
        assert!(unsafe { chassis_open(path.as_ptr(), 2) }.is_null());
        let reader = unsafe { chassis_open_reader(path.as_ptr(), 2, 16, 50) };
        assert_eq!(unsafe { chassis_precision(reader) }, 1);
        unsafe { chassis_free(reader) };
        assert_eq!(unsafe { chassis_precision(ptr::null()) }, -1);
    }

    #[test]
    fn test_ffi_reader_handle() {
        let (_dir, path) = temp_index_path();
        let writer = unsafe { chassis_open(path.as_ptr(), 8) };
        let vec = [0.5f32; 8];
        assert_eq!(unsafe { chassis_add_with_id(writer, 7, vec.as_ptr(), 8) }, 0);
        assert_eq!(unsafe { chassis_flush(writer) }, 0);

        let reader = unsafe { chassis_open_reader(path.as_ptr(), 8, 16, 50) };
        assert!(!reader.is_null());
        let (mut ids, mut dists) = ([0u64; 1], [0.0f32; 1]);
        let found = unsafe {
            chassis_search(reader, vec.as_ptr(), 8, 1, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!((found, ids[0]), (1, 7));
        assert_eq!(unsafe { (chassis_len(reader), chassis_dimensions(reader)) }, (1, 8));

        assert_eq!(unsafe { chassis_add(reader, vec.as_ptr(), 8) }, u64::MAX);
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_READ_ONLY);
        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        assert!(error.to_str().unwrap().contains("open it as a writer (chassis_open)"));

        assert_eq!(unsafe { chassis_add_with_id(writer, 8, [0.9f32; 8].as_ptr(), 8) }, 0);
        assert_eq!(unsafe { chassis_len(reader) }, 1);
        assert_eq!(unsafe { chassis_flush(writer) }, 0);
        assert_eq!(unsafe { chassis_len(reader) }, 2);

        unsafe { chassis_free(reader) };
        unsafe { chassis_free(writer) };
    }

    #[test]
    fn test_ffi_warm_on_request() {
        let (_dir, path) = temp_index_path();
        let writer = unsafe { chassis_open(path.as_ptr(), 8) };
        let vec = [0.5f32; 8];
        assert_eq!(unsafe { chassis_add_with_id(writer, 7, vec.as_ptr(), 8) }, 0);
        assert_eq!(unsafe { chassis_flush(writer) }, 0);
        assert_eq!(unsafe { chassis_warm(writer) }, 0);

        let reader = unsafe { chassis_open_reader(path.as_ptr(), 8, 16, 50) };
        assert_eq!(unsafe { chassis_warm(reader) }, 0);
        let (mut ids, mut dists) = ([0u64; 1], [0.0f32; 1]);
        for handle in [reader, writer] {
            let found = unsafe {
                chassis_search(handle, vec.as_ptr(), 8, 1, ids.as_mut_ptr(), dists.as_mut_ptr())
            };
            assert_eq!((found, ids[0]), (1, 7));
        }
        assert_eq!(unsafe { chassis_warm(ptr::null_mut()) }, -1);

        unsafe { chassis_free(reader) };
        unsafe { chassis_free(writer) };
    }

    #[test]
    fn test_ffi_huge_pages_on_request() {
        let (_dir, path) = temp_index_path();
        let writer = unsafe { chassis_open(path.as_ptr(), 8) };
        assert_eq!(unsafe { chassis_use_huge_pages(writer) }, 0);
        let vec = [0.5f32; 8];
        assert_eq!(unsafe { chassis_add_with_id(writer, 7, vec.as_ptr(), 8) }, 0);
        assert_eq!(unsafe { chassis_flush(writer) }, 0);

        let reader = unsafe { chassis_open_reader(path.as_ptr(), 8, 16, 50) };
        assert_eq!(unsafe { chassis_use_huge_pages(reader) }, 0);
        let (mut ids, mut dists) = ([0u64; 1], [0.0f32; 1]);
        let found = unsafe {
            chassis_search(reader, vec.as_ptr(), 8, 1, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!((found, ids[0]), (1, 7));
        assert_eq!(unsafe { chassis_use_huge_pages(ptr::null_mut()) }, -1);

        unsafe { chassis_free(reader) };
        unsafe { chassis_free(writer) };
    }

    #[test]
    fn test_ffi_search_filtered() {
        let (_dir, path) = temp_index_path();
        let writer = unsafe { chassis_open(path.as_ptr(), 2) };
        for i in 0..20u64 {
            let v = [i as f32, 0.0];
            assert_eq!(unsafe { chassis_add_with_id(writer, 100 + i, v.as_ptr(), 2) }, 0);
        }
        assert_eq!(unsafe { chassis_flush(writer) }, 0);
        let reader = unsafe { chassis_open_reader(path.as_ptr(), 2, 16, 50) };

        let query = [0.0f32, 0.0];
        let allowed = [7u64, 119, 105, 110];
        for handle in [writer as *const ChassisIndex, reader] {
            let (mut ids, mut dists) = ([0u64; 4], [0.0f32; 4]);
            let n = unsafe {
                chassis_search_filtered(
                    handle,
                    query.as_ptr(),
                    2,
                    4,
                    allowed.as_ptr(),
                    allowed.len(),
                    ids.as_mut_ptr(),
                    dists.as_mut_ptr(),
                )
            };
            assert_eq!(
                (n, &ids[..3], &dists[..3]),
                (3, &[105, 110, 119][..], &[5.0, 10.0, 19.0][..])
            );
            let none = unsafe {
                chassis_search_filtered(
                    handle,
                    query.as_ptr(),
                    2,
                    4,
                    ptr::null(),
                    0,
                    ids.as_mut_ptr(),
                    dists.as_mut_ptr(),
                )
            };
            assert_eq!(none, 0);
            assert!(chassis_last_error_message().is_null());
        }
        unsafe { chassis_free(reader) };
        unsafe { chassis_free(writer) };
    }

    #[test]
    fn test_ffi_concurrent_search_while_adding() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 64) };
        let vec = [0.5f32; 64];
        unsafe { chassis_add(ptr, vec.as_ptr(), 64) };

        // Raw pointers aren't Send, but the handle is meant to be shared.
        let handle = ptr as usize;
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let (mut ids, mut dists) = ([0u64; 10], [0.0f32; 10]);
                    while !done.load(std::sync::atomic::Ordering::Relaxed) {
                        let found = unsafe {
                            let ptr = handle as *const ChassisIndex;
                            chassis_search(
                                ptr,
                                vec.as_ptr(),
                                64,
                                10,
                                ids.as_mut_ptr(),
                                dists.as_mut_ptr(),
                            )
                        };
                        assert!(found >= 1);
                    }
                });
            }
            // Past 1,024 records the file grows a second segment under the searches.
            for i in 0..1200 {
                let v = [i as f32; 64];
                let id = unsafe { chassis_add(handle as *mut ChassisIndex, v.as_ptr(), 64) };
                assert_ne!(id, u64::MAX);
            }
            done.store(true, std::sync::atomic::Ordering::Relaxed);
        });

        assert_eq!(unsafe { chassis_len(ptr) }, 1201);
        unsafe { chassis_free(ptr) };
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

        // Freeing NULL is safe.
        unsafe { chassis_free(ptr::null_mut()) };
    }

    #[test]
    fn test_ffi_dimension_mismatch() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 128) };
        assert!(!ptr.is_null());

        // Try to add vector with wrong dimensions
        let vec = vec![0.1f32; 64];
        let id = unsafe { chassis_add(ptr, vec.as_ptr(), 64) };
        assert_eq!(id, u64::MAX, "Should fail with dimension mismatch");

        // Check error message
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_DIMENSION_MISMATCH);
        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        assert!(
            error
                .to_string_lossy()
                .contains("has 64 components, but this index holds vectors of 128")
        );

        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_introspection() {
        let (_dir, path) = temp_index_path();
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
        let (_dir, path) = temp_index_path();
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
    }

    #[test]
    fn test_ffi_compact() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 2) };
        for i in 0..100u64 {
            assert_eq!(
                unsafe { chassis_add_with_id(ptr, 500 + i, [i as f32, 0.0].as_ptr(), 2) },
                0
            );
        }
        for i in 0..50u64 {
            assert_eq!(unsafe { chassis_delete(ptr, 500 + i * 2) }, 1);
        }
        assert_eq!(unsafe { chassis_compact(ptr) }, 0);
        assert_eq!(unsafe { chassis_len(ptr) }, 50);
        let (mut ids, mut dists) = ([0u64; 1], [0.0f32; 1]);
        let query = [41.0f32, 0.0];
        unsafe { chassis_search(ptr, query.as_ptr(), 2, 1, ids.as_mut_ptr(), dists.as_mut_ptr()) };
        assert_eq!(ids[0], 541);
        // Readers are read-only.
        let reader = unsafe { chassis_open_reader(path.as_ptr(), 2, 16, 50) };
        assert_eq!(unsafe { chassis_compact(reader) }, -1);
        unsafe { chassis_free(reader) };
        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_add_batch_with_ids() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 2) };
        let vectors: Vec<f32> = (0..100).flat_map(|i| [i as f32, 0.0]).collect();
        let ids: Vec<u64> = (0..100).map(|i| 1000 + i).collect();
        let added =
            unsafe { chassis_add_batch_with_ids(ptr, ids.as_ptr(), vectors.as_ptr(), 100, 2) };
        assert_eq!(added, 0);
        let (mut found, mut dists) = ([0u64; 1], [0.0f32; 1]);
        let query = [42.0f32, 0.0];
        unsafe {
            chassis_search(ptr, query.as_ptr(), 2, 1, found.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert_eq!(found[0], 1042);

        // An id already present rejects the whole batch.
        let again = [7u64, 1042];
        let rows = [1.0f32, 1.0, 2.0, 2.0];
        let rejected =
            unsafe { chassis_add_batch_with_ids(ptr, again.as_ptr(), rows.as_ptr(), 2, 2) };
        assert_eq!(rejected, -1);
        assert_eq!(unsafe { chassis_len(ptr) }, 100);
        assert_eq!(unsafe { chassis_add_batch_with_ids(ptr, ptr::null(), ptr::null(), 0, 2) }, 0);
        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_add_batch_success() {
        const DIM: usize = 128;
        const COUNT: usize = 3;
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), DIM as u32) };
        assert!(!ptr.is_null());

        let mut batch = Vec::with_capacity(COUNT * DIM);
        for row in 0..COUNT {
            let v = 0.1f32 + 0.1f32 * row as f32;
            batch.extend(std::iter::repeat_n(v, DIM));
        }

        let mut out_ids = vec![0u64; COUNT];
        let n = unsafe { chassis_add_batch(ptr, batch.as_ptr(), COUNT, DIM, out_ids.as_mut_ptr()) };
        assert_eq!(n, COUNT);
        assert_eq!(out_ids, vec![0u64, 1, 2]);
        assert_eq!(unsafe { chassis_len(ptr) }, COUNT as u64);

        let query = &batch[0..DIM];
        let mut ids = vec![0u64; 5];
        let mut dists = vec![0.0f32; 5];
        let n_search = unsafe {
            chassis_search(ptr, query.as_ptr(), DIM, 5, ids.as_mut_ptr(), dists.as_mut_ptr())
        };
        assert!(n_search > 0);

        let flush = unsafe { chassis_flush(ptr) };
        assert_eq!(flush, 0);
        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_add_batch_dimension_mismatch() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 128) };
        assert!(!ptr.is_null());

        let batch = vec![0.1f32; 64];
        let mut out_ids = vec![0u64; 1];
        let n = unsafe { chassis_add_batch(ptr, batch.as_ptr(), 1, 64, out_ids.as_mut_ptr()) };
        assert_eq!(n, 0);

        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_DIMENSION_MISMATCH);
        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        assert!(error.to_string_lossy().contains("help: pass dim = 128"));

        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_add_batch_null_out_ids() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 128) };
        assert!(!ptr.is_null());
        let batch = vec![0.1f32; 128];
        let n = unsafe { chassis_add_batch(ptr, batch.as_ptr(), 1, 128, ptr::null_mut()) };
        assert_eq!(n, 0);
        let error = unsafe { CStr::from_ptr(chassis_last_error_message()) };
        let s = error.to_string_lossy();
        assert!(!s.is_empty());
        assert!(s.to_lowercase().contains("null"));

        unsafe { chassis_free(ptr) };
    }

    #[test]
    fn test_ffi_add_batch_count_zero() {
        let (_dir, path) = temp_index_path();
        let ptr = unsafe { chassis_open(path.as_ptr(), 64) };
        assert!(!ptr.is_null());
        let n = unsafe { chassis_add_batch(ptr, ptr::null(), 0, 64, ptr::null_mut()) };
        assert_eq!(n, 0);
        assert_eq!(unsafe { chassis_len(ptr) }, 0);
        unsafe { chassis_free(ptr) };
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

    #[test]
    fn test_ffi_an_error_has_a_code_to_act_on_and_a_help_line() {
        let (dir, path) = temp_index_path();
        let writer = unsafe { chassis_open(path.as_ptr(), 8) };
        assert_eq!(chassis_last_error_code(), CHASSIS_OK);
        let message = || unsafe { CStr::from_ptr(chassis_last_error_message()) }.to_string_lossy();

        assert!(unsafe { chassis_open(path.as_ptr(), 8) }.is_null());
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_LOCKED);
        assert!(message().contains("\nhelp: "));

        let missing = CString::new(dir.path().join("none.chassis").to_str().unwrap()).unwrap();
        assert!(unsafe { chassis_open_reader(missing.as_ptr(), 8, 16, 50) }.is_null());
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_NOT_FOUND);

        let nan = [f32::NAN; 8];
        assert_eq!(unsafe { chassis_add(writer, nan.as_ptr(), 8) }, u64::MAX);
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_INVALID_ARGUMENT);
        assert!(message().contains("NaN at component 0"));

        assert_eq!(
            unsafe { chassis_open_with_metric(path.as_ptr(), 8, 16, 200, 50, 7) },
            ptr::null_mut()
        );
        assert_eq!(chassis_last_error_code(), CHASSIS_ERROR_INVALID_ARGUMENT);
        assert!(message().contains("help: pass 0 for euclidean distance or 1 for cosine"));

        // A call that succeeds clears it.
        assert_eq!(unsafe { chassis_add(writer, [0.5f32; 8].as_ptr(), 8) }, 0);
        assert_eq!(chassis_last_error_code(), CHASSIS_OK);
        unsafe { chassis_free(writer) };
    }
}
