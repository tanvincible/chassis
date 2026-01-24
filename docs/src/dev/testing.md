# Testing Guide

Chassis has a comprehensive test suite covering unit tests, integration tests, and compile-time borrow checker validation.

## Running Tests

### All Tests

```bash
# Run all tests in the workspace
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture

# Run with specific number of threads
cargo test --workspace -- --test-threads=4
```

### Specific Test Suites

```bash
# Storage layer tests
cargo test --package chassis-core --test storage_tests

# Graph I/O tests
cargo test --package chassis-core --test graph_io_tests

# Linking tests
cargo test --package chassis-core --test link_tests

# Search tests
cargo test --package chassis-core --test search_tests

# Compile-time safety tests
cargo test --package chassis-core --test compile_fail
```

### Unit Tests (Embedded)

```bash
# Run unit tests in a specific module
cargo test --package chassis-core storage::tests
cargo test --package chassis-core distance::tests
cargo test --package chassis-core hnsw::node::tests
```

## Test Coverage

### Storage Layer Tests (`storage_tests.rs`)

Tests the memory-mapped storage engine:

**Basic Operations**:
- `test_create_new_storage()`: File creation and initialization
- `test_insert_and_retrieve()`: Vector insertion and retrieval
- `test_multiple_inserts()`: Sequential insertion
- `test_insert_wrong_dimensions()`: Dimension validation

**Zero-Copy Access**:
- `test_get_vector_slice()`: Slice-based access
- `test_vector_slice_lifetime()`: Borrow checker enforcement

**Durability**:
- `test_commit()`: Persistence guarantees
- `test_persistence()`: File reopen after commit
- `test_uncommitted_changes()`: Data loss on crash simulation

**File Growth**:
- `test_storage_growth()`: Automatic file expansion
- `test_remap_overhead()`: Performance of remapping

**Concurrency**:
- `test_exclusive_lock()`: File locking
- `test_concurrent_readers()`: Multi-reader support (if implemented)

### Graph I/O Tests (`graph_io_tests.rs`)

Tests HNSW graph persistence and addressing:

**Graph Header**:
- `test_graph_header_roundtrip()`: Serialization/deserialization
- `test_graph_header_validation()`: Magic bytes and version checks
- `test_graph_header_persistence()`: Survives reopening

**Node Records**:
- `test_node_record_write_read()`: Basic record I/O
- `test_sequential_node_writes()`: Batch writes
- `test_node_record_updates()`: Updating existing nodes

**Addressing**:
- `test_o1_addressing()`: Offset computation correctness
- `test_node_offset_consistency()`: Deterministic offsets
- `test_addressing_formula_correctness()`: Mathematical validation

**Neighbor Iteration**:
- `test_neighbors_iter_from_mmap()`: Zero-allocation iteration
- `test_neighbors_iter_empty_layer()`: Edge case handling
- `test_neighbors_iter_filters_invalid()`: `INVALID_NODE_ID` filtering

### Linking Tests (`link_tests.rs`)

Tests bidirectional graph linking and diversity heuristics:

**Basic Linking**:
- `test_link_single_node()`: Single node insertion
- `test_link_multiple_nodes()`: Sequential linking
- `test_forward_links_exist()`: Outgoing edges persisted

**Backward Links**:
- `test_backward_links_added()`: Reciprocal edges created
- `test_idempotency()`: Duplicate link prevention
- `test_backward_link_pruning()`: Overflow handling

**Diversity Heuristic**:
- `test_diversity_selection()`: Heuristic 2 application
- `test_starvation_fallback()`: Minimum degree guarantee
- `test_distance_cache_effectiveness()`: Cache hit rate

**Invariant Enforcement**:
- `test_self_links_filtered()`: No self-loops
- `test_future_neighbors_filtered()`: No forward references (Model A)
- `test_sequential_id_enforcement()`: Node ID monotonicity

**Crash Consistency**:
- `test_crash_during_linking()`: Partial updates are safe
- `test_ghost_nodes()`: Unreachable nodes ignored

### Search Tests (`search_tests.rs`)

Tests HNSW search algorithm:

**Basic Search**:
- `test_search_empty_graph()`: Empty graph returns empty results
- `test_search_single_node()`: Trivial case
- `test_search_k_neighbors()`: Correct k returned
- `test_search_ef_correction()`: `ef < k` handled

**Result Quality**:
- `test_search_accuracy()`: Distance sorting
- `test_search_recall()`: Finds true nearest neighbors
- `test_search_no_duplicates()`: Unique results only

**Visited Filter**:
- `test_visited_filter()`: Dense array correctness
- `test_visited_filter_out_of_bounds()`: Bounds checking

**Edge Cases**:
- `test_nan_safe_ordering()`: NaN distances don't panic
- `test_search_large_ef()`: High ef values
- `test_search_disconnected_graph()`: Unreachable nodes

**Performance**:
- `test_search_scales_logarithmically()`: Hop count grows slowly

### Compile-Time Safety Tests (`compile_fail.rs`)

Uses `trybuild` to verify borrow checker enforcement:

**UI Tests** (`tests/ui/`):

1. **`slice_then_insert.rs`** (Expected: Fails to compile)
   ```rust
   let slice = storage.get_vector_slice(0)?;
   storage.insert(&vec![1.0; 128])?;  // Mutable borrow while slice alive
   println!("{}", slice[0]);
   ```
   **Error**: Cannot borrow `storage` as mutable

2. **`slice_then_commit.rs`** (Expected: Fails to compile)
   ```rust
   let slice = storage.get_vector_slice(0)?;
   storage.commit()?;  // Mutable borrow (commit modifies mmap)
   println!("{}", slice[0]);
   ```
   **Error**: Cannot borrow `storage` as mutable

3. **`slice_escapes_scope.rs`** (Expected: Fails to compile)
   ```rust
   let slice = {
       let storage = Storage::open(...)?;
       storage.get_vector_slice(0)?
   };  // storage dropped, slice now dangling
   println!("{}", slice[0]);
   ```
   **Error**: Slice lifetime exceeds storage

4. **`return_slice_then_mutate.rs`** (Expected: Fails to compile)
   ```rust
   fn get_slice(storage: &Storage) -> &[f32] {
       storage.get_vector_slice(0).unwrap()
   }
   
   let slice = get_slice(&storage);
   storage.insert(&vec![1.0; 128])?;  // Mutable borrow
   ```
   **Error**: Cannot borrow `storage` as mutable

**Run UI Tests**:
```bash
cargo test --package chassis-core --test compile_fail
```

Expected output: All tests pass (compilation failures are expected).

## Unit Tests by Module

### `distance.rs` Tests

```rust
#[test]
fn test_euclidean_distance_basic()
fn test_cosine_distance()
fn test_simd_correctness_small()     // Tests tail loop
fn test_simd_correctness_large()     // Tests main loop
fn test_simd_random_vectors()        // Fuzz testing
fn test_simd_edge_cases()            // Zero, identical, sparse
fn test_avx2_specific()              // x86_64 only
fn test_neon_specific()              // aarch64 only
```

**Coverage**:
- SIMD vs scalar equivalence
- Edge cases (zeros, identical vectors, NaN)
- Dimension scaling (3D to 3072D)
- Architecture-specific paths

### `header.rs` Tests

```rust
#[test]
fn test_header_size()               // Compile-time size check
fn test_header_alignment()          // 4KB alignment
fn test_new_header_is_valid()
fn test_invalid_dimensions()
fn test_invalid_magic()
```

**Coverage**:
- Header validation logic
- Magic byte checking
- Dimension bounds (0, MAX_DIMENSIONS+1)

### `hnsw/node.rs` Tests

```rust
#[test]
fn test_node_header_size()
fn test_record_size_calculation()
fn test_layer_offset_calculation()
fn test_set_and_get_neighbors()
fn test_add_neighbor()
fn test_serialization_roundtrip()
fn test_node_to_record_conversion()
fn test_compute_node_offset()
fn test_addressing_formula_consistency()
```

**Coverage**:
- Fixed-size record layout
- Neighbor management
- Serialization correctness
- O(1) addressing formula

### `hnsw/graph.rs` Tests

```rust
#[test]
fn test_graph_header_roundtrip()
fn test_o1_addressing_no_hashmap()
fn test_graph_persistence()
fn test_insert_out_of_order_panics()    // Debug only
fn test_insert_duplicate_panics()       // Debug only
fn test_update_existing_node()
```

**Coverage**:
- Graph header persistence
- O(1) node addressing
- Sequential ID enforcement
- Update vs insert semantics

## Testing Best Practices

### 1. Use Temporary Files

```rust
use tempfile::NamedTempFile;

#[test]
fn my_test() -> anyhow::Result<()> {
    let temp_file = NamedTempFile::new()?;
    let storage = Storage::open(temp_file.path(), 128)?;
    // Test logic...
    Ok(())
}  // File automatically deleted
```

**Why**: Prevents test pollution and ensures isolation.

### 2. Test Invariants Explicitly

```rust
#[test]
fn test_node_id_invariant() {
    let graph = create_test_graph();
    
    // Invariant: node_count tracks reachable nodes
    assert_eq!(graph.node_count(), 10);
    
    // Invariant: all nodes have sequential IDs
    for i in 0..graph.node_count() {
        assert!(graph.read_node_record(i).is_ok());
    }
}
```

### 3. Test Edge Cases

```rust
#[test]
fn test_empty_graph() {
    let graph = create_empty_graph();
    assert!(graph.search(&query, 10, 50)?.is_empty());
}

#[test]
fn test_single_node() {
    let graph = create_graph_with_one_node();
    let results = graph.search(&query, 10, 50)?;
    assert_eq!(results.len(), 1);
}
```

### 4. Use Deterministic Random Data

```rust
#[test]
fn test_with_deterministic_random() {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    
    fn hash_to_f32(seed: u64) -> f32 {
        let state = RandomState::new();
        let mut hasher = state.build_hasher();
        seed.hash(&mut hasher);
        (hasher.finish() % 10000) as f32 / 10000.0
    }
    
    let vector: Vec<f32> = (0..128).map(hash_to_f32).collect();
    // Reproducible across test runs
}
```

### 5. Test Concurrency (Where Applicable)

```rust
#[test]
fn test_exclusive_lock() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let _storage1 = Storage::open(path, 128).unwrap();
    
    // Second open should fail
    let result = Storage::open(path, 128);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already open"));
}
```

## Running Tests with Coverage

Install `tarpaulin`:

```bash
cargo install cargo-tarpaulin
```

Generate coverage report:

```bash
cargo tarpaulin --workspace --out Html --output-dir coverage
```

Open `coverage/index.html` in browser.

**Target**: Aim for >80% line coverage, >90% for critical paths (storage, search).

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Run tests
        run: cargo test --workspace --verbose
      
      - name: Run compile-fail tests
        run: cargo test --package chassis-core --test compile_fail
      
      - name: Check formatting
        run: cargo fmt -- --check
      
      - name: Run clippy
        run: cargo clippy --workspace -- -D warnings
```
