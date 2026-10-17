# C API

Chassis from C, or from any language that can call C: a shared library and its header,
`chassis.h`. This page shows how to get them and build against them, then describes every
function.

## Install

Each [release](https://github.com/tanvincible/chassis/releases) has the library and the header for
Linux (x86-64 and ARM64), macOS (ARM64) and Windows (x86-64). To build them from a checkout instead,
which needs Rust:

```bash
cd chassis-ffi
cargo build --release
```

That writes the library to `target/release`: `libchassis_ffi.so` on Linux, `libchassis_ffi.dylib`
on macOS, `chassis_ffi.dll` on Windows. The header is `chassis-ffi/include/chassis.h`.

## Example

```c
#include "chassis.h"
#include <inttypes.h>
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
        printf("id %" PRIu64 ", distance %f\n", ids[i], distances[i]);
    }
    
    // Clean up
    chassis_free(index);
    return 0;
}
```

### Compiling

```bash
# Linux and macOS, from the repository or an unpacked release
cc -o myapp myapp.c -Ichassis-ffi/include -Ltarget/release -lchassis_ffi \
  -Wl,-rpath,$PWD/target/release
./myapp

# Windows (MSVC)
cl myapp.c /I chassis-ffi\include /link /LIBPATH:target\release chassis_ffi.dll.lib
```

## Functions

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
Like `chassis_open_with_options`, with a distance metric: `CHASSIS_METRIC_EUCLIDEAN` (0) or
`CHASSIS_METRIC_COSINE` (1, `1 - cosine similarity`; vectors are stored at unit length and zero vectors are rejected). The
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
Like `chassis_open_with_metric`, with a precision: `CHASSIS_PRECISION_FULL` (0) keeps each
component of a vector as a 32-bit float, `CHASSIS_PRECISION_HALF` (1) as a 16-bit float ([ADR-0018](https://github.com/tanvincible/chassis/blob/main/docs/adr/018-half-precision.md)). Half precision
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
The metric and precision come from the file; `max_connections` has to be the one the index was
created with.

#### `chassis_use_huge_pages`
```c
int chassis_use_huge_pages(ChassisIndex* index);
```
Ask the operating system to keep the index's vectors on huge pages
([ADR-0016](https://github.com/tanvincible/chassis/blob/main/docs/adr/016-huge-pages-on-request.md)). On an index too large for the CPU's caches,
searches are up to a quarter faster. Linux only, where the kernel and filesystem keep files on huge pages
(ext4 on Linux 6.17 does); elsewhere it does nothing. Call it right after opening, on a writer's
handle or a reader's. Returns `0`, or `-1` for a NULL handle.

#### `chassis_warm`
```c
int chassis_warm(ChassisIndex* index);
```
Read the index into memory on another thread ([ADR-0019](https://github.com/tanvincible/chassis/blob/main/docs/adr/019-warm.md)), so that searches
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
Free an index and release resources. Safe to call with `NULL`. It doesn't flush: adds and deletes
since the last `chassis_flush` are discarded, as after a crash.

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
`chassis_add` in a loop ([ADR-0010](https://github.com/tanvincible/chassis/blob/main/docs/adr/010-parallel-batch-builds.md)). Writes their ids, assigned
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
Search for k nearest neighbors. Returns how many results were written, or `0` on error, which
`chassis_last_error_code()` tells apart from an index with nothing to return.

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
vectors match, every vector is checked instead and the results are exact. Ids in ascending order
are read where they lie; in any other order they are copied into a set first, a pass over all of
them per call.

#### `chassis_search_excluding`
```c
size_t chassis_search_excluding(
    const ChassisIndex* index,
    const float* query,
    size_t len,
    size_t k,
    const uint64_t* excluded_ids,
    size_t excluded_len,
    uint64_t* out_ids,
    float* out_dists
);
```
Like `chassis_search`, never returning a vector whose id is in `excluded_ids`, as "more like this"
leaves out the vector it starts from (see `chassis_get`). Ids not in the index are ignored; an
empty list leaves out nothing.

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
floats each. `out_ids` and `out_dists` hold `count * k` values: row `i` is query `i`'s results, nearest first. A row with fewer than `k` results is filled out with id `UINT64_MAX`, which no vector has, and distance `INFINITY`. Returns `count`, or `0` on error or for an empty batch, which sets no
error. An error names the query that failed, as in `Query 3 of the batch has NaN at component 0`.

**Thread Safety**: Safe from any thread; runs concurrently with other searches, and a write waits
for the whole batch

#### `chassis_flush`
```c
int chassis_flush(ChassisIndex* index);
```
Flush changes to disk. Returns `0` on success, `-1` on error.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_rollback`
```c
int chassis_rollback(ChassisIndex* index);
```
Discard every add, delete and user version since the last flush, as closing the index and opening
it again would, without letting go of the writer's lock. Call it where a write's transaction in
your own store rolls back, or the next `chassis_flush` commits what that write did to the index.
Returns `0`, or `-1` for a reader handle, a NULL one, or a file that can't be read or synced; after
that, later flushes fail too: close the index and open it again.

**Thread Safety**: Safe from any thread; writes run one at a time and searches wait for them

#### `chassis_compact`
```c
int chassis_compact(ChassisIndex* index);
```
Rewrite the index without its deleted vectors and with a newly built graph, then replace the file
with the copy ([ADR-0011](https://github.com/tanvincible/chassis/blob/main/docs/adr/011-compaction.md)). Ids don't change. Like `chassis_flush`, it
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

#### `chassis_get`
```c
int chassis_get(const ChassisIndex* index, uint64_t id, float* out, size_t len);
```
Copy the vector with `id` into `out`, which has space for `len` floats: the index's dimensions.
Returns `1` if a live vector has `id`, `0` if none has, `-1` on error. The vector comes back as the
index keeps it: scaled to length 1 in a cosine index, and rounded in half precision.

#### `chassis_contains`
```c
int chassis_contains(const ChassisIndex* index, uint64_t id);
```
Returns `1` if a live vector has `id`, `0` if none has, `-1` on error.

#### `chassis_ids`
```c
size_t chassis_ids(const ChassisIndex* index, uint64_t* out, size_t cap);
```
Copy the ids of the live vectors into `out`, in the order they were added, and return how many
there are. Only the first `cap` are written, so if the return is more than `cap`, call again with
more space; a reader can see a newer flush than a `chassis_len` just before. Returns `0` on error,
as for an empty index, which sets no error.

```c
size_t cap = chassis_len(index), count;
uint64_t* ids = NULL;
do {
    ids = realloc(ids, (cap ? cap : 1) * sizeof *ids);
    count = chassis_ids(index, ids, cap);
} while (count > cap && (cap = count));
```

#### `chassis_check`
```c
size_t chassis_check(const ChassisIndex* index, uint64_t* out, size_t cap, size_t* unreachable);
```
Check what opening can't, reading the whole graph and every vector, and return how many ids it
found; `0` means the index is in good order. The ids go into `out`, the first `cap` of them, as
with `chassis_ids`: first the live vectors that no path through the graph reaches, so no search
returns them, and `*unreachable` says how many those are; then the vectors with a component that
isn't a finite number, which only damage leaves. Damage can leave vectors out of reach, and
`chassis_compact` builds the graph again; so can data with no structure, or many exact copies of
one vector, rebuilt or not ([Rust API](./api-reference.md#check)). Returns `0` on error too;
check `chassis_last_error_code()`.

```c
size_t unreachable;
size_t found = chassis_check(index, NULL, 0, &unreachable);  // Then again with space for found
```

#### `chassis_deleted_count`
```c
uint64_t chassis_deleted_count(const ChassisIndex* index);
```
Deleted vectors the file still holds, until `chassis_compact` removes them.

#### `chassis_user_version`
```c
uint64_t chassis_user_version(const ChassisIndex* index);
```
Your application's number for the index, 0 until one is set: like SQLite's `user_version`. Each
`chassis_flush` commits the number set before it with the data, so after a crash it says which
flush the index holds; compare it on open and as each write begins with the same number kept in
your own store. A writer handle returns the number last set, flushed or not, and
`chassis_rollback` sets it back to the flushed one; a reader handle returns the number in its
newest snapshot. Returns `0` for a NULL handle, with an error set.

#### `chassis_set_user_version`
```c
int chassis_set_user_version(ChassisIndex* index, uint64_t version);
```
Set the number the next `chassis_flush` commits. Until then, a crash leaves the one flushed before.
Returns `0`, or `-1` for a reader handle or a NULL one.

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

### Errors

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

### Version

#### `chassis_version`
```c
const char* chassis_version(void);
```
Get library version string.

## Threads

Every function except `chassis_free` is safe to call from any thread on the same handle. Searches,
`chassis_len`, `chassis_is_empty` and `chassis_dimensions` run concurrently. `chassis_add`,
`chassis_add_with_id`, `chassis_add_batch`, `chassis_delete` and `chassis_flush` take the handle's
write lock, so they run one at a time and searches wait for them. `chassis_free` must not race with
any other call on the same handle. A handle from `chassis_open_reader` runs one call at a time,
introspection included; open one per thread to search in parallel.

In this example, `generate_vector`, `generate_query`, `process_results` and `keep_running` stand
for your own code.

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

## Error checks

A function's return value says whether it failed: `NULL`, `UINT64_MAX` or `-1`, or a count of `0`
with an error set, as its entry above says. `chassis_last_error_message` and
`chassis_last_error_code` then describe the failure. Each thread keeps its own last error, so
threads never see each other's.

```c
if (chassis_flush(index) != 0) {
    fprintf(stderr, "Flush failed: %s\n", chassis_last_error_message());
    chassis_free(index);
    exit(1);
}
```

A panic inside Chassis never unwinds into C. The call returns its error value with a message that
starts `Chassis panicked:`. After a panic in a write, the handle refuses later calls: free it and
open the index again.

## Safety

Each function's entry in `chassis.h` lists what it needs from you. These hold for all of them:

1. Never pass `NULL` where a function doesn't say it accepts one.
2. A vector's length must match the index's dimensions.
3. The message from `chassis_last_error_message` is valid until the next call on the same thread.
4. Don't call `chassis_free` while another thread still uses the handle, or use the handle after.

## Performance

1. Add many vectors before each `chassis_flush`: a flush waits for the disk.
2. Allocate result buffers once, and reuse them for every search.
3. Tune `max_connections`, `ef_construction` and `ef_search` as the
   [Rust API](./api-reference.md#indexoptions) describes; the options mean the same in C.

`chassis-ffi/examples/example.c` is a complete program, with a Makefile beside it.
