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
Open or create an index. Returns `NULL` on error.

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
Add a vector. Returns ID or `UINT64_MAX` on error.

**Thread Safety**: Single-writer (exclusive access required)

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

**Thread Safety**: Multi-reader (shared access allowed)

#### `chassis_flush`
```c
int chassis_flush(ChassisIndex* index);
```
Flush changes to disk. Returns `0` on success, `-1` on error.

**Thread Safety**: Single-writer (exclusive access required)

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

| Function | Access Pattern | Concurrent Safety |
|----------|----------------|-------------------|
| `chassis_open` | N/A | Safe (different paths) |
| `chassis_free` | N/A | Safe (different indices) |
| `chassis_add` | Exclusive (`*mut`) | Single-writer only |
| `chassis_flush` | Exclusive (`*mut`) | Single-writer only |
| `chassis_search` | Shared (`*const`) | Multi-reader safe |
| `chassis_len` | Shared (`*const`) | Multi-reader safe |
| `chassis_is_empty` | Shared (`*const`) | Multi-reader safe |
| `chassis_dimensions` | Shared (`*const`) | Multi-reader safe |

### Concurrency Example

```c
// Thread 1: Writer
void* writer_thread(void* arg) {
    ChassisIndex* index = (ChassisIndex*)arg;
    
    // Exclusive access required
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
    
    // Shared access - can run concurrently with other readers
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
4. **Thread Safety**: Respect single-writer / multi-reader rules
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
