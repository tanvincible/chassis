# Performance Guide

## Benchmark Results

All benchmarks run on **12th Gen Intel i7-12650H** (10 cores, 16 threads, 4.7 GHz max) with **16 GB RAM** on Linux 6.16.8.

### Storage Performance

#### Insert Operations

| Operation | Throughput | Latency | Notes |
|-----------|-----------|---------|-------|
| Raw insert (no fsync) | 134,412/sec | 7.44 µs | Memory-speed writes |
| Durable insert (with fsync) | 165,840/sec | 6.03 µs | Full persistence guarantee |
| Batch 10 vectors | 174,216/sec | 57.4 µs | Amortized fsync cost |
| Batch 100 vectors | 172,998/sec | 578 µs | Optimal batch size |
| Batch 1000 vectors | 171,361/sec | 5.84 ms | Diminishing returns |

**Key Insight**: Batching provides ~23% throughput improvement by amortizing fsync overhead. Optimal batch size is 100-1000 vectors.

#### Read Operations

| Operation | Throughput | Latency | Cache State |
|-----------|-----------|---------|-------------|
| Hot read (L1 cache hit) | 13,820,000/sec | 72.4 ns | Resident in cache |
| Cold read (page fault) | 65,971/sec | 15.2 µs | Kernel page-in |
| Sequential 1000 reads | 7,996,805/sec | 125 µs total | Prefetcher-friendly |

**Key Insight**: ~200x difference between cached and uncached reads. Locality matters!

#### File Growth

| Operation | Throughput | Total Time |
|-----------|-----------|------------|
| Grow empty → 1000 vectors | 169,301/sec | 5.91 ms |

**Key Insight**: Remapping overhead is negligible when amortized over batch inserts.

---

### Distance Computation (SIMD)

#### Speedup by Dimension

| Dimensions | SIMD (ns) | Scalar (ns) | Speedup | Throughput |
|-----------|-----------|-------------|---------|-----------|
| 64 | 5.64 | 25.2 | **4.5x** | 11.4 Gelem/s |
| 128 | 9.22 | 53.8 | **5.8x** | 13.9 Gelem/s |
| 384 | 19.7 | 193.6 | **9.8x** | 19.5 Gelem/s |
| 768 | 40.2 | 369.6 | **9.2x** | 19.1 Gelem/s |
| 1536 | 70.2 | 750.0 | **10.7x** | 21.9 Gelem/s |
| 3072 | 141.2 | 1495 | **10.6x** | 21.8 Gelem/s |

**Architecture**: AVX2 with 4-way accumulator unrolling. Processes 32 floats/iteration (4 accumulators × 8 floats/vector).

**Key Insight**: Speedup scales with dimensionality. Peak efficiency at 768D+ where main loop dominates over scalar tail.

#### Batch Distance Computation (100 vectors × 768D)

| Implementation | Latency | Throughput |
|----------------|---------|-----------|
| SIMD | 4.97 µs | 15.5 Gelem/s |
| Scalar | 38.0 µs | 2.02 Gelem/s |

**Speedup**: 7.6x on realistic workloads (search computes distances to many candidates).

#### Memory Bandwidth

| Vector Size | Throughput | Bandwidth |
|-------------|-----------|-----------|
| 4096 floats | 241 ns | 505 GiB/s |
| 8192 floats | 598 ns | 409 GiB/s |
| 16384 floats | 1.24 µs | 394 GiB/s |

**Key Insight**: Saturates memory bandwidth at ~400 GiB/s for large vectors (L3 cache miss → DRAM fetch).

---

### HNSW Search

#### Search Performance by ef

| ef | Latency | Throughput | Use Case |
|----|---------|-----------|----------|
| 4 | 1.70 µs | 588K/sec | Fast, lower recall |
| 8 | 2.81 µs | 356K/sec | Balanced |
| 16 | 3.79 µs | 264K/sec | Default |
| 32 | 5.85 µs | 171K/sec | High recall |
| 64 | 10.7 µs | 93.7K/sec | Very high recall |
| 128 | 20.1 µs | 49.8K/sec | Maximum quality |
| 256 | 45.3 µs | 22.1K/sec | Exhaustive-like |
| 512 | 101.6 µs | 9.84K/sec | Extreme precision |

**Key Insight**: Logarithmic scaling with ef. 2x ef ≈ 2x latency.

#### Search Performance by k (neighbors returned)

| k | ef | Latency | Notes |
|---|----|---------| ------|
| 1 | 50 | 8.63 µs | Nearest neighbor |
| 5 | 50 | 9.00 µs | Minimal overhead |
| 10 | 50 | 8.98 µs | Default |
| 50 | 100 | 16.0 µs | Large result set |
| 100 | 200 | 30.7 µs | Very large result set |

**Key Insight**: k has minimal impact on latency (heap operations are cheap). ef dominates.

#### Graph Size Scaling

| Nodes | Latency (ef=50) | Throughput |
|-------|-----------------|-----------|
| 100 | 6.32 µs | 158K/sec |
| 500 | 8.21 µs | 122K/sec |
| 1,000 | 9.38 µs | 107K/sec |
| 5,000 | 9.07 µs | 110K/sec |

**Key Insight**: Sub-linear scaling. HNSW's logarithmic hop count means latency grows slowly with dataset size.

#### Dimension Scaling

| Dimensions | Latency | Throughput | Distance Cost |
|-----------|---------|-----------|---------------|
| 64 | 7.38 µs | 135K/sec | 5.6 ns |
| 128 | 8.50 µs | 118K/sec | 9.2 ns |
| 384 | 11.9 µs | 84K/sec | 19.7 ns |
| 768 | 17.5 µs | 57K/sec | 40 ns |
| 1536 | 35.2 µs | 28K/sec | 70 ns |

**Key Insight**: Search latency dominated by distance computation for high-dimensional vectors.

---

### HNSW Construction (Linking)

#### Single Node Linking

| Neighbors | Latency | Notes |
|-----------|---------|-------|
| 4 | 39.5 µs | Sparse graph |
| 8 | 39.5 µs | Low connectivity |
| 16 | 37.4 µs | Default M=16 |
| 32 | 68.0 µs | High connectivity (pruning activates) |

**Key Insight**: Linear scaling until neighbor list fills, then pruning overhead kicks in.

#### Pruning Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Full list pruning (M0=32) | 20.3 µs | Diversity heuristic + cache |
| Cache hit rate | ~70% | Most distances reused |

**Key Insight**: Lazy distance cache reduces pruning from O(N²) to ~O(N·M).

#### Multi-Layer Linking

| Layers | Latency | Throughput |
|--------|---------|-----------|
| 1 | 22.1 µs | 45.3K/sec |
| 2 | 23.8 µs | 84.1K/sec |
| 4 | 25.2 µs | 159K/sec |
| 8 | 29.1 µs | 275K/sec |

**Key Insight**: Overhead per layer is small (~1.5 µs). Higher layers use fewer neighbors (M vs M0).

#### Batch Construction

| Nodes | Total Time | Per-Node Latency |
|-------|-----------|------------------|
| 50 | 727 µs | 14.5 µs |
| 100 | 1.62 ms | 16.2 µs |
| 200 | 3.33 ms | 16.7 µs |

**Key Insight**: Construction is I/O bound. Per-node latency increases slightly as graph grows (more candidates to prune against).

#### Worst Case: Identical Vectors

| Operation | Latency |
|-----------|---------|
| 32 identical vectors | 394 µs total |

**Key Insight**: Diversity heuristic degenerates to k-nearest when all distances are equal. Still performant due to cache.

---

### HNSW Graph I/O

#### Header Operations

| Operation | Latency | Notes |
|-----------|---------|-------|
| Read graph header | 8.29 ns | 64-byte aligned read |
| Write graph header | 2.00 ns | Mmap dirty page |

**Key Insight**: Header access is cache-resident and virtually free.

#### Node Record I/O

| Operation | Latency | Notes |
|-----------|---------|-------|
| Read single record | 320 ns | Includes deserialization |
| Read 10 sequential | 2.79 µs | 279 ns/record (prefetch) |
| Read 10 random | 250 ns/record | Cache-dependent |
| Write record | 283 ns | Mmap dirty + validation |

**Key Insight**: Sequential access is ~15% faster due to hardware prefetching.

#### Neighbor Iteration

| Method | Latency | Notes |
|--------|---------|-------|
| `neighbors_iter_from_mmap()` (layer 0) | 43.3 ns | Zero-allocation |
| `neighbors_iter_from_mmap()` (layer 1) | 16.7 ns | Fewer neighbors |
| Via `NodeRecord::neighbors_iter()` | 330 ns | Requires record read |

**Key Insight**: Direct mmap iteration is **7.6x faster** than deserializing record first. Critical for search hot path.

#### Zero-Copy Patterns

| Operation | Latency | Benefit |
|-----------|---------|---------|
| Get node bytes | 10.6 ns | Pointer arithmetic only |
| Compute node offset | 0.83 ns | Single multiplication |
| Batch 1000 offsets | 398 ns | 0.4 ns/offset (pipelined) |

---

### Node Record Operations

#### Record Size Computation

| Parameters | Latency | Record Size |
|-----------|---------|-------------|
| M=16, layers=8 | 340 ps | 1,312 bytes |
| M=16, layers=16 | 342 ps | 2,576 bytes |
| M=32, layers=16 | 316 ps | 5,136 bytes |
| M=64, layers=16 | 307 ps | 10,256 bytes |

**Key Insight**: O(1) calculation. Larger M increases record size linearly.

#### Serialization

| Operation | Latency | Size |
|-----------|---------|------|
| Serialize to bytes | 126 ns | 656 bytes |
| Deserialize from bytes | 194 ns | 656 bytes |

**Key Insight**: Deserialization is slower due to validation (bounds checks, magic bytes).

#### Neighbor Access

| Method | Latency | Allocation |
|--------|---------|-----------|
| `get_neighbors()` (layer 0) | 188 ns | Allocates `Vec<NodeId>` |
| `get_neighbors()` (layer 1) | 156 ns | Allocates `Vec<NodeId>` |
| `neighbor_count()` | 18.8 ns | None (count-only) |
| `neighbors_iter()` (layer 0) | 10.7 ns | **Zero allocation** |
| `neighbors_iter()` (layer 1) | 7.57 ns | **Zero allocation** |

**Key Insight**: Iterator is **17.6x faster** than `get_neighbors()` by avoiding `Vec` allocation.

---

## Optimization Guidelines

### 1. Batching is Critical

```rust
// BAD: 1000x slower due to fsync per insert
for vector in vectors {
    storage.insert(vector)?;
    storage.commit()?;  // 6 µs per fsync!
}

// GOOD: Single fsync at the end
for vector in vectors {
    storage.insert(vector)?;  // 7.4 µs (memory speed)
}
storage.commit()?;  // One 6 µs fsync for all
```

**Impact**: 100-1000 vectors batched → ~23% throughput improvement.

### 2. Use Zero-Copy APIs

```rust
// BAD: Allocates Vec<f32>
let vector = storage.get_vector(id)?;
let distance = euclidean_distance(&query, &vector);

// GOOD: Zero-copy slice
let slice = storage.get_vector_slice(id)?;
let distance = euclidean_distance(&query, slice);
```

**Impact**: 72 ns (cached) vs 15 µs (page fault + allocation).

### 3. Use Iterator for Neighbors

```rust
// BAD: Allocates Vec, slower deserialization
let record = graph.read_node_record(node_id)?;
for neighbor in record.get_neighbors(layer) {
    process(neighbor);
}

// GOOD: Zero-allocation iteration
for neighbor in graph.neighbors_iter_from_mmap(node_id, layer)? {
    process(neighbor);
}
```

**Impact**: 43 ns vs 330 ns (7.6x speedup).

### 4. Choose Appropriate ef

| Use Case | Recommended ef | Latency | Recall |
|----------|---------------|---------|--------|
| Fast filtering | 8-16 | ~3-4 µs | ~85-90% |
| Balanced search | 32-64 | ~6-11 µs | ~95-98% |
| High precision | 128-256 | ~20-45 µs | ~99%+ |

**Rule of thumb**: Start with ef=32, increase if recall is insufficient.

### 5. Dimension-Specific Tuning

| Dimensions | SIMD Benefit | Recommendation |
|-----------|--------------|----------------|
| < 128 | Moderate (4-5x) | Still worthwhile |
| 128-768 | High (8-10x) | Default choice |
| 1536+ | Very High (10x+) | Critical for performance |

**Note**: SIMD overhead is ~2 ns regardless of dimension, so small vectors benefit less.

### 6. Memory Considerations

```rust
// Estimate index size
let vector_size = dims * 4 bytes;
let vector_zone = count * vector_size;

let node_record_size = 16 + (M0 * 8) + ((max_layers - 1) * M * 8);
let graph_zone = count * node_record_size;

let total_size = 4KB + vector_zone + graph_zone;
```

**Example**: 1M vectors, 768D, M=16, max_layers=16
- Vector zone: 1M × 768 × 4 = 3 GB
- Graph zone: 1M × 2,576 = 2.5 GB
- **Total: ~5.5 GB**

---

## Benchmark Reproduction

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench --bench distance_bench
cargo bench --bench storage_bench
cargo bench --bench search_bench
cargo bench --bench link_bench
cargo bench --bench graph_io_bench
cargo bench --bench hnsw_node_bench

# Generate HTML report
cargo install cargo-criterion
cargo criterion
```

Reports saved to `target/criterion/`.
