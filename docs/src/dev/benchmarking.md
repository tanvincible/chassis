# Benchmarking Guide

Chassis uses [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical benchmarking with outlier detection and HTML reporting.

## Running Benchmarks

### All Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Takes ~15-20 minutes for full suite
```

### Specific Benchmarks

```bash
# Storage layer
cargo bench --bench storage_bench

# Distance computation
cargo bench --bench distance_bench

# HNSW search
cargo bench --bench search_bench

# Graph linking
cargo bench --bench link_bench

# Graph I/O
cargo bench --bench graph_io_bench

# Node operations
cargo bench --bench hnsw_node_bench
```

### Filtering Benchmarks

```bash
# Run only insert benchmarks
cargo bench --bench storage_bench -- insert

# Run only SIMD benchmarks
cargo bench --bench distance_bench -- simd

# Run only high_ef search tests
cargo bench --bench search_bench -- high_ef
```

## Benchmark Suites

### 1. Storage Benchmarks (`storage_bench.rs`)

#### Raw Insert Performance

```rust
raw_insert/single_insert_no_commit
```

Measures pure insertion speed without fsync.

**Result**: ~7.4 µs per vector (134K vectors/sec)

**What it tests**: Memory-mapped write performance, file growth overhead.

#### Durable Insert Performance

```rust
durable_insert/insert_with_commit
```

Measures insertion + fsync (full durability guarantee).

**Result**: ~6.0 µs per operation (166K vectors/sec)

**What it tests**: Kernel flush + disk sync latency.

#### Batch Insert Performance

```rust
batch_insert/10
batch_insert/100
batch_insert/1000
```

Measures batched insertions with single commit.

**Results**:
- 10 vectors: 174K/sec
- 100 vectors: 173K/sec
- 1000 vectors: 171K/sec

**What it tests**: Fsync amortization, optimal batch size.

#### Read Performance

```rust
hot_read/get_vector_cached        // L1/L2 cache hit
cold_read/get_vector_cold_start   // Page fault
sequential_read/read_1000_sequential
```

**Results**:
- Hot: 72 ns (13.8M vectors/sec)
- Cold: 15 µs (66K vectors/sec)
- Sequential 1000: 125 µs total (8M vectors/sec)

**What it tests**: Cache locality, prefetcher effectiveness, kernel page-in latency.

#### Remap Overhead

```rust
remap_overhead/grow_from_empty_to_1000
```

Measures file growth and remapping cost.

**Result**: ~5.9 ms total for 1000 vectors

**What it tests**: `truncate() + mmap()` overhead.

---

### 2. Distance Benchmarks (`distance_bench.rs`)

#### Distance by Dimension

```rust
distance_by_dimension/simd/{64,128,384,768,1536,3072}
distance_by_dimension/scalar/{64,128,384,768,1536,3072}
```

Compares SIMD vs scalar across dimensions.

**Key Results** (AVX2):
- 128D: 9.2 ns SIMD, 53.8 ns scalar → **5.8x speedup**
- 768D: 40 ns SIMD, 370 ns scalar → **9.2x speedup**
- 1536D: 70 ns SIMD, 750 ns scalar → **10.7x speedup**

**What it tests**: 4-way accumulator effectiveness, main loop vs tail loop ratio.

#### Batch Distances

```rust
batch_distances/simd_batch
batch_distances/scalar_batch
```

100 vectors × 768D per iteration.

**Results**:
- SIMD: 4.97 µs → 15.5 Gelem/s
- Scalar: 38.0 µs → 2.02 Gelem/s

**What it tests**: Real-world search pattern (query vs many candidates).

#### Unaligned Access

```rust
unaligned_access/simd_unaligned
```

Tests SIMD performance with misaligned memory.

**Result**: ~58 ns (minimal penalty)

**What it tests**: `_mm256_loadu_ps` vs `_mm256_load_ps`.

#### Small Vectors

```rust
small_vectors/dims/{3,7,15,31,63}
```

Tests scalar tail loop dominance.

**Results**:
- 3D: 4.67 ns
- 63D: 8.26 ns

**What it tests**: Scalar tail performance, setup overhead.

#### Sparse Vectors

```rust
sparse_vectors/simd_sparse_10pct
sparse_vectors/scalar_sparse_10pct
```

90% zeros, 10% non-zero.

**What it tests**: SIMD efficiency with sparse data (still faster despite wasted computation).

#### Memory Bandwidth

```rust
memory_bandwidth/simd/{4096,8192,16384}
```

Very large vectors to saturate DRAM bandwidth.

**Results**:
- 4096 floats: 505 GiB/s
- 8192 floats: 409 GiB/s
- 16384 floats: 394 GiB/s

**What it tests**: L3 cache miss → DRAM throughput limit.

---

### 3. Search Benchmarks (`search_bench.rs`)

#### Search Baseline

```rust
search_baseline/ef/{8,16,32,64,128}
```

Fixed k=10, varying ef (search quality).

**Results**:
- ef=8: 2.81 µs (356K queries/sec)
- ef=32: 5.85 µs (171K queries/sec)
- ef=128: 20.1 µs (50K queries/sec)

**What it tests**: ef scaling (logarithmic candidate pool growth).

#### High ef Stress Test

```rust
high_ef_stress/ef/{64,128,256,512}
```

5000-node graph, high ef values.

**Results**:
- ef=256: 45.3 µs
- ef=512: 101.6 µs

**What it tests**: Dense visited filter performance, heap operations.

#### Varying k

```rust
varying_k/k/{1,5,10,50,100}
```

Different result set sizes.

**Results**:
- k=1: 8.63 µs
- k=10: 8.98 µs
- k=100: 30.7 µs (ef=200)

**What it tests**: Result heap size impact (minimal for k < ef).

#### Graph Size Scaling

```rust
graph_size_scaling/nodes/{100,500,1000,5000}
```

Fixed ef=50, k=10.

**Results**:
- 100 nodes: 6.32 µs
- 5000 nodes: 9.07 µs

**What it tests**: Sub-linear scaling (HNSW's logarithmic hop count).

#### Dimension Scaling

```rust
dimension_scaling/dimensions/{64,128,384,768,1536}
```

**Results**:
- 64D: 7.38 µs
- 768D: 17.5 µs
- 1536D: 35.2 µs

**What it tests**: Distance computation dominates at high dimensions.

---

### 4. Linking Benchmarks (`link_bench.rs`)

#### Single Node Linking

```rust
link_single_node/neighbors/{4,8,16,32}
```

Link one node with varying neighbor counts.

**Results**:
- 4 neighbors: 39.5 µs
- 16 neighbors: 37.4 µs
- 32 neighbors: 68.0 µs (pruning kicks in)

**What it tests**: Neighbor slot filling, pruning threshold.

#### Pruning Pressure

```rust
link_with_pruning/full_list_pruning
```

Add node to hub with full neighbor list (triggers diversity heuristic).

**Result**: ~20.3 µs

**What it tests**: Lazy distance cache effectiveness.

#### Multi-Layer Linking

```rust
multilayer_linking/layers/{1,2,4,8}
```

**Results**:
- 1 layer: 22.1 µs
- 4 layers: 25.2 µs
- 8 layers: 29.1 µs

**What it tests**: Layer traversal overhead (~1.5 µs per additional layer).

#### Batch Construction

```rust
batch_linking/nodes/{10,50,100}
```

Sequential graph construction.

**Results**:
- 10 nodes: 37.5 µs total → 3.75 µs/node
- 100 nodes: 331 µs total → 3.31 µs/node

**What it tests**: Batching efficiency, graph growth patterns.

#### Worst Case: Identical Vectors

```rust
worst_case_clustering/identical_vectors
```

32 identical vectors (diversity heuristic degenerates).

**Result**: 394 µs total

**What it tests**: Cache still helps even when all distances equal.

---

### 5. Graph I/O Benchmarks (`graph_io_bench.rs`)

#### Graph Header

```rust
graph_header/read     // 8.29 ns
graph_header/write    // 2.00 ns
```

**What it tests**: 64-byte aligned header access (cache-resident).

#### Node Record I/O

```rust
node_record/read                 // 320 ns
node_record/read_sequential      // 279 ns per record
node_record/read_random          // 250 ns per record
node_record/write                // 283 ns
```

**What it tests**: Record deserialization, validation overhead, sequential vs random access.

#### Neighbor Iteration

```rust
neighbors_mmap/iter_layer0              // 43.3 ns (zero-alloc)
neighbors_mmap/iter_via_record_layer0   // 330 ns (with record read)
```

**What it tests**: Zero-allocation iteration vs deserialization overhead.

**Speedup**: 7.6x for direct mmap iteration.

#### Node Offset Computation

```rust
offset/compute_single        // 0.83 ns
offset/compute_batch_1000    // 398 ns total → 0.4 ns each
```

**What it tests**: O(1) addressing formula, instruction pipelining.

#### Search Pattern Simulation

```rust
search_pattern/visit_100_nodes       // 33.7 µs (with record read)
search_pattern/visit_100_nodes_mmap  // 2.96 µs (zero-alloc)
```

**What it tests**: Hot path allocation elimination.

**Speedup**: 11.4x with zero-copy iteration.

---

### 6. Node Benchmarks (`hnsw_node_bench.rs`)

#### Record Size Calculation

```rust
record_size/calculate/m{16,32,64}_l{8,16}
```

**Result**: ~320-340 ps (sub-nanosecond)

**What it tests**: Constant-time calculation (no runtime overhead).

#### Serialization

```rust
record_serialization/to_bytes    // 126 ns
record_serialization/from_bytes  // 194 ns
```

**What it tests**: Memory copy + validation cost.

#### Neighbor Access

```rust
neighbor_access/get_neighbors_layer0   // 188 ns (allocates Vec)
neighbor_access/neighbors_iter_layer0  // 10.7 ns (zero-alloc)
```

**Speedup**: 17.6x with iterator.

#### Node Conversion

```rust
node_conversion/node_to_record  // 73.6 ns
node_conversion/record_to_node  // 400 ns
```

**What it tests**: In-memory representation vs on-disk format.

---

## Interpreting Results

### Criterion Output

```
distance_by_dimension/simd/768
                        time:   [39.783 ns 40.207 ns 40.631 ns]
                        thrpt:  [18.902 Gelem/s 19.101 Gelem/s 19.305 Gelem/s]
Found 114 outliers among 1000 measurements (11.40%)
  30 (3.00%) high mild
  84 (8.40%) high severe
```

**Reading**:
- **time**: [lower bound, estimate, upper bound] at 95% confidence
- **thrpt**: Throughput (elements/second)
- **outliers**: Measurements discarded (system noise)

### Comparing Runs

```bash
# Baseline
cargo bench --bench distance_bench -- simd/768 > baseline.txt

# After optimization
cargo bench --bench distance_bench -- simd/768 > optimized.txt

# Compare
critcmp baseline.txt optimized.txt
```

**Example output**:
```
group          baseline              optimized
-----          --------              ---------
simd/768       40.2 ns (1.00x)       35.1 ns (1.15x faster)
```

---

## Best Practices

### 1. Warm Up System

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Drop caches (optional)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### 2. Reduce System Noise

```bash
# Close unnecessary apps
# Disable background services
# Run benchmarks when system is idle
```

### 3. Use Consistent Hardware

- Same machine for before/after comparisons
- Note CPU model, RAM, storage type in results
- Disable turbo boost for reproducibility (optional)

### 4. Increase Sample Size for Stability

```rust
group.sample_size(1000);  // Default: 100
group.measurement_time(Duration::from_secs(10));  // Default: 5s
```

### 5. Profile Hot Paths

```bash
# Install perf
sudo apt install linux-tools-generic

# Run with perf
cargo bench --bench search_bench --no-run
perf record --call-graph=dwarf target/release/deps/search_bench-*
perf report
```

---

## Regression Testing

Create a CI job to detect performance regressions:

```yaml
- name: Benchmark
  run: cargo bench --workspace -- --save-baseline main

- name: Compare
  run: |
    cargo bench --workspace -- --baseline main
    critcmp main current --threshold 5  # Fail if >5% slower
```

---

## Generating HTML Reports

```bash
# Run benchmarks
cargo bench --workspace

# Open reports
open target/criterion/report/index.html
```

**Report includes**:
- Performance graphs over time
- Distribution histograms
- Throughput charts
- Outlier analysis
