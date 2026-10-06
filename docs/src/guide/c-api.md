# Chassis FFI - C Bindings for Chassis Vector Storage

This crate provides a C-compatible shared library interface to the Chassis vector storage engine.

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
Open or create an index. Returns `NULL` on error, including when the file is a cosine index: open
those with `chassis_open_with_metric` or `chassis_open_reader`.

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

#### `chassis_flush`
```c
int chassis_flush(ChassisIndex* index);
```
Flush changes to disk. Returns `0` on success, `-1` on error.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

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

### Error Handling

#### `chassis_last_error_message`
```c
const char* chassis_last_error_message(void);
```
Get last error message for current thread. Returns `NULL` if no error.

**Lifetime**: Valid until next FFI call on this thread.

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
