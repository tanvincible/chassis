# Chassis FFI - C Bindings for Chassis Vector Storage

This crate provides a C-compatible shared library interface to Chassis, an embedded vector index.

## Features

- **Panic-safe**: All panics are caught at the FFI boundary
- **Thread-safe error handling**: Thread-local error storage
- **ABI stable**: `#[repr(C)]` guarantees
- **Zero-copy where possible**: Direct mmap access for search operations
- **Comprehensive documentation**: Every function documented with safety requirements

## Building

```bash
cd chassis-ffi
cargo build --release
```

This generates:
- **Linux**: `target/release/libchassis_ffi.so`
- **macOS**: `target/release/libchassis_ffi.dylib`
- **Windows**: `target/release/chassis_ffi.dll`

The C header is automatically generated at `include/chassis.h`.

## Usage

### Basic Example (C)

```c
#include "chassis.h"
#include <stdio.h>

int main(void) {
    // Open index
    ChassisIndex* index = chassis_open("vectors.chassis", 768);
    if (index == NULL) {
        fprintf(stderr, "Error: %s\n", chassis_last_error_message());
        return 1;
    }
    
    // Add vector
    float vec[768] = {0.1, 0.2, /* ... */};
    uint64_t id = chassis_add(index, vec, 768);
    if (id == UINT64_MAX) {
        fprintf(stderr, "Error: %s\n", chassis_last_error_message());
        chassis_free(index);
        return 1;
    }
    
    // Flush to disk
    if (chassis_flush(index) != 0) {
        fprintf(stderr, "Error: %s\n", chassis_last_error_message());
        chassis_free(index);
        return 1;
    }
    
    // Search
    float query[768] = {0.1, 0.2, /* ... */};
    uint64_t ids[10];
    float distances[10];
    
    size_t count = chassis_search(index, query, 768, 10, ids, distances);
    for (size_t i = 0; i < count; i++) {
        printf("ID: %llu, Distance: %f\n", ids[i], distances[i]);
    }
    
    // Clean up
    chassis_free(index);
    return 0;
}
```

### Compilation

```bash
# Linux
gcc -o myapp myapp.c -L./target/release -lchassis_ffi -lm

# Run (Linux)
LD_LIBRARY_PATH=./target/release ./myapp

# macOS
gcc -o myapp myapp.c -L./target/release -lchassis_ffi

# Run (macOS)
DYLD_LIBRARY_PATH=./target/release ./myapp

# Windows (MSVC)
cl myapp.c /I include /link /LIBPATH:target\release chassis_ffi.lib
```

## API Reference

### Lifecycle

#### `chassis_open`
```c
ChassisIndex* chassis_open(const char* path, uint32_t dimensions);
```
Open or create an index. Returns `NULL` on error, including when the file is a cosine index or
one in half precision: open those with `chassis_open_with_precision` or `chassis_open_reader`.

#### `chassis_open_with_options`
```c
ChassisIndex* chassis_open_with_options(
    const char* path,
    uint32_t dimensions,
    uint32_t max_connections,
    uint32_t ef_construction,
    uint32_t ef_search
);
```
Open with custom HNSW parameters.

#### `chassis_open_with_metric`
```c
ChassisIndex* chassis_open_with_metric(
    const char* path,
    uint32_t dimensions,
    uint32_t max_connections,
    uint32_t ef_construction,
    uint32_t ef_search,
    uint32_t metric
);
```
Like `chassis_open_with_options`, with a distance metric: `0` for Euclidean, `1` for cosine
(`1 - cosine similarity`; vectors are stored at unit length and zero vectors are rejected). The
metric is fixed when the index is created; reopening with another one fails.

#### `chassis_open_with_precision`
```c
ChassisIndex* chassis_open_with_precision(
    const char* path,
    uint32_t dimensions,
    uint32_t max_connections,
    uint32_t ef_construction,
    uint32_t ef_search,
    uint32_t metric,
    uint32_t precision
);
```
Like `chassis_open_with_metric`, with a precision: `0` keeps each component of a vector as a
32-bit float, `1` as a 16-bit float ([ADR-0018](../adr/018-half-precision.md)). Half precision
halves the vectors' size in the file and in memory, and a search over an index too large for the
CPU's caches is faster for it. Each component is rounded to about three decimal digits, and a
vector with a component of 65,520 or more in magnitude is refused. The precision is fixed when
the index is created; reopening with another one fails, and a release without this function
can't open a file in half precision.

#### `chassis_open_reader`
```c
ChassisIndex* chassis_open_reader(
    const char* path,
    uint32_t dimensions,
    uint32_t max_connections,
    uint32_t ef_search
);
```
Open an existing index to search it while a writer, possibly in another process, adds to it. Takes
no lock, so any number of readers can open the file next to one writer. Each search sees the
writer's newest flush, which becomes visible just before its final fsync; `chassis_len` takes a new
snapshot first. Adds, deletes and flushes on the
handle fail. A file in an older format must be opened once with `chassis_open` to migrate it.

#### `chassis_use_huge_pages`
```c
int chassis_use_huge_pages(ChassisIndex* index);
```
Ask the operating system to keep the index's vectors on huge pages
([ADR-0016](../adr/016-huge-pages-on-request.md)). On an index too large for the CPU's caches,
searches are up to a quarter faster. Linux only, where the kernel and filesystem keep files on huge pages
(ext4 on Linux 6.17 does); elsewhere it does nothing. Call it right after opening, on a writer's
handle or a reader's. Returns `0`, or `-1` for a NULL handle.

#### `chassis_warm`
```c
int chassis_warm(ChassisIndex* index);
```
Read the index into memory on another thread ([ADR-0019](../adr/019-warm.md)), so that searches
stop waiting for the disk one page at a time while its file is not in memory yet, as after a
reboot. Doesn't wait for the reading: searches go on meanwhile. What is read is what the index
holds when this is called; while that is under way, calling again does nothing. A reader that opens
the file again after a compaction reads the new one in too. For an index that fits in memory: one
much larger would push everything else out. Linux 5.14 or macOS; an older Linux reads only some of
the index, and Windows none. Call it on a writer's handle or a reader's, right after opening or
later. Returns `0`, or `-1` for a NULL handle.

#### `chassis_free`
```c
void chassis_free(ChassisIndex* index);
```
Free an index and release resources. Safe to call with `NULL`.

### Operations

#### `chassis_add`
```c
uint64_t chassis_add(
    ChassisIndex* index,
    const float* vector,
    size_t len
);
```
Add a vector. Returns the id assigned (one past the largest id used so far) or `UINT64_MAX` on error.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_add_with_id`
```c
int chassis_add_with_id(
    ChassisIndex* index,
    uint64_t id,
    const float* vector,
    size_t len
);
```
Add a vector under your own id; search reports it. Returns `0`, or `-1` if a live vector already
has `id`, `id` is `UINT64_MAX`, or the add fails.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_add_batch`
```c
size_t chassis_add_batch(
    ChassisIndex* index,
    const float* vectors,  // count * dim floats, row-major
    size_t count,
    size_t dim,
    uint64_t* out_ids      // room for count ids
);
```
Add many vectors at once, linking them on every core: a large batch builds many times faster than
`chassis_add` in a loop ([ADR-0010](../adr/010-parallel-batch-builds.md)). Writes their ids, assigned
as `chassis_add` assigns them, to `out_ids`. Returns `count`, or `0` if the batch fails, in which
case none of it is added.

#### `chassis_add_batch_with_ids`
```c
int chassis_add_batch_with_ids(
    ChassisIndex* index,
    const uint64_t* ids,   // count ids
    const float* vectors,  // count * dim floats, row-major
    size_t count,
    size_t dim
);
```
`chassis_add_with_id` for a whole batch, linked on every core. Returns `0`, or `-1` with none of the
batch added if an id repeats, already exists or is `UINT64_MAX`.

**Thread Safety** (both): a batch holds the write lock until it is linked, so searches on the same
handle wait for the whole batch.

#### `chassis_delete`
```c
int chassis_delete(ChassisIndex* index, uint64_t id);
```
Delete the vector with `id`. Returns `1` if deleted, `0` if no live vector has `id`, `-1` on error.
Durable after the next `chassis_flush()`; a crash before then rolls it back.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_search`
```c
size_t chassis_search(
    const ChassisIndex* index,
    const float* query,
    size_t len,
    size_t k,
    uint64_t* out_ids,
    float* out_dists
);
```
Search for k nearest neighbors. Returns number of results found.

**Thread Safety**: Safe from any thread; runs concurrently with other searches

#### `chassis_search_filtered`
```c
size_t chassis_search_filtered(
    const ChassisIndex* index,
    const float* query,
    size_t len,
    size_t k,
    const uint64_t* allowed_ids,
    size_t allowed_len,
    uint64_t* out_ids,
    float* out_dists
);
```
Like `chassis_search`, returning only vectors whose id is in `allowed_ids`. Ids not in the index
are ignored; an empty list matches nothing. When walking the graph would cost more, as when few
vectors match, every vector is checked instead and the results are exact.

#### `chassis_search_batch`
```c
size_t chassis_search_batch(
    const ChassisIndex* index,
    const float* queries,
    size_t count,
    size_t dim,
    size_t k,
    uint64_t* out_ids,
    float* out_dists
);
```
Search for the k nearest neighbors of each of `count` queries, given one after another, `dim`
floats each. `out_ids` and `out_dists` hold `count * k` values: row `i` is query `i`'s results,
nearest first, and a row with fewer than `k` is filled out with id `UINT64_MAX`, which no vector
has, and distance `INFINITY`. Returns `count`, or `0` on error or for an empty batch, which sets no
error. An error names the query that failed, as in `Query 3 of the batch has NaN at component 0`.

**Thread Safety**: Safe from any thread; runs concurrently with other searches, and a write waits
for the whole batch

#### `chassis_flush`
```c
int chassis_flush(ChassisIndex* index);
```
Flush changes to disk. Returns `0` on success, `-1` on error.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_compact`
```c
int chassis_compact(ChassisIndex* index);
```
Rewrite the index without its deleted vectors and with a newly built graph, then replace the file
with the copy ([ADR-0011](../adr/011-compaction.md)). Ids don't change. Like `chassis_flush`, it
makes every add and delete so far durable. Returns `0`, or `-1` with the index left as it was; on
Windows it fails while another process has the index open. It takes as long as building the index
and needs free disk for a second copy.

**Thread Safety**: holds the write lock for its whole run, so searches on this handle wait;
readers in other processes keep searching

### Introspection

#### `chassis_len`
```c
uint64_t chassis_len(const ChassisIndex* index);
```
Get number of vectors in the index.

#### `chassis_is_empty`
```c
int chassis_is_empty(const ChassisIndex* index);
```
Check if index is empty. Returns `1` if empty, `0` otherwise.

#### `chassis_dimensions`
```c
uint32_t chassis_dimensions(const ChassisIndex* index);
```
Get vector dimensionality.

#### `chassis_metric`
```c
int chassis_metric(const ChassisIndex* index);
```
The metric the index was created with: `0` Euclidean, `1` cosine, `-1` if `index` is `NULL`.

#### `chassis_precision`
```c
int chassis_precision(const ChassisIndex* index);
```
The precision the index keeps its vectors in: `0` full, `1` half, `-1` if `index` is `NULL`.

### Error Handling

#### `chassis_last_error_message`
```c
const char* chassis_last_error_message(void);
```
The last error on this thread: what was wrong, with the value, then a line starting `help:` with
what to do instead. Returns `NULL` if no error.

**Lifetime**: Valid until next FFI call on this thread.

#### `chassis_last_error_code`
```c
int chassis_last_error_code(void);
```
The last error's kind, for a program to act on: one of the `CHASSIS_ERROR_` constants
(`CHASSIS_ERROR_INVALID_ARGUMENT`, `CHASSIS_ERROR_DIMENSION_MISMATCH`, `CHASSIS_ERROR_LOCKED`, …),
or `CHASSIS_OK` (0) after a call that succeeded. [Errors](./errors.md) lists them, with their
causes and fixes.

### Versioning

#### `chassis_version`
```c
const char* chassis_version(void);
```
Get library version string.

## Thread Safety

Every function except `chassis_free` is safe to call from any thread on the same handle. Searches,
`chassis_len`, `chassis_is_empty` and `chassis_dimensions` run concurrently. `chassis_add`,
`chassis_add_with_id`, `chassis_add_batch`, `chassis_delete` and `chassis_flush` take the handle's
write lock, so they run one at a time and searches wait for them. `chassis_free` must not race with
any other call on the same handle. A handle from `chassis_open_reader` runs one call at a time,
introspection included; open one per thread to search in parallel.

### Concurrency Example

```c
// Thread 1: Writer
void* writer_thread(void* arg) {
    ChassisIndex* index = (ChassisIndex*)arg;
    
    // Each add takes the write lock; searches in other threads wait for it
    float vec[768];
    for (int i = 0; i < 1000; i++) {
        generate_vector(vec, i);
        chassis_add(index, vec, 768);
    }
    
    chassis_flush(index);
    return NULL;
}

// Thread 2: Reader
void* reader_thread(void* arg) {
    const ChassisIndex* index = (const ChassisIndex*)arg;
    
    // Runs concurrently with other searches, and waits while a write runs
    float query[768];
    uint64_t ids[10];
    float dists[10];
    
    while (keep_running) {
        generate_query(query);
        chassis_search(index, query, 768, 10, ids, dists);
        process_results(ids, dists);
    }
    
    return NULL;
}
```

## Error Handling Patterns

### Pattern 1: Check Return Value
```c
uint64_t id = chassis_add(index, vec, 768);
if (id == UINT64_MAX) {
    fprintf(stderr, "Error: %s\n", chassis_last_error_message());
    // Handle error
}
```

### Pattern 2: Check and Cleanup
```c
if (chassis_flush(index) != 0) {
    const char* error = chassis_last_error_message();
    fprintf(stderr, "Flush failed: %s\n", error ? error : "Unknown error");
    chassis_free(index);
    exit(1);
}
```

### Pattern 3: Thread-Local Errors
```c
// Thread A sets error
chassis_add(index, vec, 768);  // Fails
printf("Thread A: %s\n", chassis_last_error_message());

// Thread B has different error storage
// (no race condition)
```

## Safety Requirements

All functions document their safety requirements. Key rules:

1. **Null Checks**: Never pass `NULL` unless explicitly allowed
2. **Lifetime**: Pointers from `chassis_last_error_message()` are only valid until next FFI call
3. **Dimensions**: Vector length must match index dimensions
4. **Thread Safety**: Don't call `chassis_free` while other threads still use the handle
5. **Double Free**: Don't use pointers after `chassis_free()`

## Performance Tips

1. **Batch inserts**: Add many vectors before calling `chassis_flush()`
2. **Reuse buffers**: Allocate result buffers once, reuse for multiple searches
3. **Tune parameters**: Adjust `max_connections`, `ef_construction`, `ef_search` for your use case

## Panic Safety

All panics are caught at the FFI boundary and converted to errors:

```c
// Even if Rust code panics internally, this will return an error
uint64_t id = chassis_add(index, vec, 768);
if (id == UINT64_MAX) {
    // Error message will contain "Panic: ..." if a panic occurred
    fprintf(stderr, "%s\n", chassis_last_error_message());
}
```

**This is a critical safety guarantee** - undefined behavior will never occur due to unwinding across the FFI boundary.

## Examples

See `examples/example.c` for a complete working example.

## Testing

Run FFI tests:
```bash
cargo test
```

This includes:
- Lifecycle tests
- Null safety tests
- Dimension mismatch tests
- Thread-local error tests
- UTF-8 validation tests
