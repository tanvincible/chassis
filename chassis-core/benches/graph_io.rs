//! Benchmarks for Graph I/O operations. 
//!
//! These benchmarks measure the performance of:
//! - Graph header read/write
//! - Node record read/write
//! - Neighbor iteration from mmap

use chassis_core::hnsw::{
    compute_node_offset, HnswGraph, HnswParams, NodeId, NodeRecord, NodeRecordParams,
};
use chassis_core::Storage;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tempfile::NamedTempFile;

/// Setup helper: creates an HnswGraph with nodes
fn create_test_graph(node_count: usize, dims:  u32) -> (HnswGraph, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, dims).unwrap();

    // Insert vectors first
    let vector = vec![1.0f32; dims as usize];
    for _ in 0.. node_count {
        storage. insert(&vector).unwrap();
    }
    storage.commit().unwrap();

    // Create graph
    let params = HnswParams::default();
    let mut graph = HnswGraph:: open(storage, params).unwrap();

    // Insert nodes with varying layers
    for i in 0..node_count {
        let layer = if i % 10 == 0 {
            2
        } else if i % 3 == 0 {
            1
        } else {
            0
        };
        graph. insert(i as NodeId, layer).unwrap();
    }

    (graph, temp_file)
}

/// Setup helper: creates an HnswGraph with custom params
fn create_test_graph_with_params(
    node_count: usize,
    dims: u32,
    params: HnswParams,
) -> (HnswGraph, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, dims).unwrap();

    // Insert vectors first
    let vector = vec![1.0f32; dims as usize];
    for _ in 0.. node_count {
        storage. insert(&vector).unwrap();
    }
    storage.commit().unwrap();

    // Create graph with specified params
    let mut graph = HnswGraph::open(storage, params).unwrap();

    // Insert nodes with varying layers
    for i in 0.. node_count {
        let layer = if i % 10 == 0 {
            2
        } else if i % 3 == 0 {
            1
        } else {
            0
        };
        graph. insert(i as NodeId, layer).unwrap();
    }

    (graph, temp_file)
}

// ============================================================================
// Graph Header Benchmarks
// ============================================================================

fn bench_graph_header_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_header");

    // Create graph ONCE before benchmark loop
    let (graph, _temp_file) = create_test_graph(100, 128);

    group.bench_function("read", |b| {
        b.iter(|| {
            // Only benchmark the header read operation
            black_box(graph.read_graph_header().unwrap())
        })
    });

    group.finish();
}

fn bench_graph_header_write(c: &mut Criterion) {
    let mut group = c. benchmark_group("graph_header");

    // Create graph ONCE - we'll write to the same graph repeatedly
    // This is safe because write_graph_header just updates the header in place
    let (mut graph, _temp_file) = create_test_graph(10, 128);

    group.bench_function("write", |b| {
        b.iter(|| {
            // Benchmark:  write header (overwrites same location each time)
            black_box(graph.write_graph_header().unwrap())
        })
    });

    group.finish();
}

// ============================================================================
// Node Record I/O Benchmarks
// ============================================================================

fn bench_node_record_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_record");

    // Create graph with nodes ONCE
    let (graph, _temp_file) = create_test_graph(1000, 128);

    group.bench_function("read", |b| {
        let mut node_id = 0u64;
        b.iter(|| {
            // Cycle through different nodes to avoid cache effects
            let record = graph.read_node_record(node_id % 1000).unwrap();
            node_id += 1;
            black_box(record)
        })
    });

    group.bench_function("read_sequential", |b| {
        b.iter(|| {
            // Read multiple nodes sequentially (cache-friendly)
            for i in 0..10 {
                black_box(graph.read_node_record(i).unwrap());
            }
        })
    });

    group.bench_function("read_random", |b| {
        // Pre-generate random indices
        let indices: Vec<NodeId> = vec![42, 7, 999, 123, 456, 789, 0, 500, 250, 750];
        let mut idx = 0;

        b.iter(|| {
            let node_id = indices[idx % indices.len()];
            idx += 1;
            black_box(graph.read_node_record(node_id).unwrap())
        })
    });

    group.finish();
}

fn bench_node_record_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_record");

    // Create graph ONCE with default params
    let params = HnswParams::default();
    let (mut graph, _temp_file) = create_test_graph_with_params(100, 128, params);

    // Get the record params from the graph's params to ensure consistency
    let record_params = params.to_record_params();

    group.bench_function("write", |b| {
        let mut node_id = 0u64;
        b.iter(|| {
            // Create record with MATCHING params from the graph
            let mut record = NodeRecord::new(node_id % 100, 3, record_params);
            record.set_neighbors(0, &[1, 2, 3, 4, 5, 6, 7, 8]);
            record.set_neighbors(1, &[100, 200, 300]);
            record.set_neighbors(2, &[1000]);
            node_id += 1;
            black_box(graph.write_node_record(&record).unwrap())
        })
    });

    group.finish();
}

// ============================================================================
// Neighbor Iteration Benchmarks (Zero-Allocation)
// ============================================================================

fn bench_neighbors_from_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_mmap");

    // Create graph with populated neighbors
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file. path();

    let mut storage = Storage::open(path, 128).unwrap();
    for _ in 0..100 {
        storage.insert(&vec![1.0f32; 128]).unwrap();
    }
    storage.commit().unwrap();

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).unwrap();

    // Insert nodes and manually set neighbors for benchmark
    for i in 0..100u64 {
        let layer = if i < 10 { 2 } else { 0 };
        graph.insert(i, layer).unwrap();
    }

    // Update node 0 with many neighbors for realistic benchmark
    // Use the graph's record params to ensure consistency
    let record_params = params.to_record_params();
    let mut record = NodeRecord::new(0, 3, record_params);
    record.set_neighbors(0, &(1..=20).collect::<Vec<_>>());
    record.set_neighbors(1, &(100..=110).map(|x| x as u64).collect::<Vec<_>>());
    record.set_neighbors(2, &[50, 51, 52]);
    graph.write_node_record(&record).unwrap();

    group.bench_function("iter_layer0", |b| {
        b.iter(|| {
            let sum:  u64 = graph.neighbors_iter_from_mmap(0, 0).unwrap().sum();
            black_box(sum)
        })
    });

    group.bench_function("iter_layer1", |b| {
        b.iter(|| {
            let sum: u64 = graph.neighbors_iter_from_mmap(0, 1).unwrap().sum();
            black_box(sum)
        })
    });

    group.bench_function("iter_layer2", |b| {
        b.iter(|| {
            let sum: u64 = graph. neighbors_iter_from_mmap(0, 2).unwrap().sum();
            black_box(sum)
        })
    });

    // Compare with NodeRecord-based iteration (to show the win)
    group.bench_function("iter_via_record_layer0", |b| {
        b.iter(|| {
            let record = graph.read_node_record(0).unwrap();
            let sum: u64 = record.neighbors_iter(0).sum();
            black_box(sum)
        })
    });

    group.finish();
}

// ============================================================================
// Get Node Bytes (Zero-Copy) Benchmarks
// ============================================================================

fn bench_get_node_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_bytes");

    let (graph, _temp_file) = create_test_graph(1000, 128);

    group.bench_function("get_bytes", |b| {
        let mut node_id = 0u64;
        b.iter(|| {
            let bytes = graph.get_node_bytes(node_id % 1000).unwrap();
            node_id += 1;
            black_box(bytes)
        })
    });

    group.finish();
}

// ============================================================================
// Offset Computation Benchmark
// ============================================================================

fn bench_offset_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("offset");

    let params = NodeRecordParams::default();
    let record_size = params.record_size();
    let graph_start = 8192u64; // 2 pages

    group.bench_function("compute_single", |b| {
        let mut node_id = 0u64;
        b.iter(|| {
            let offset = compute_node_offset(graph_start, node_id, record_size);
            node_id = node_id. wrapping_add(1);
            black_box(offset)
        })
    });

    group.bench_function("compute_batch_1000", |b| {
        b.iter(|| {
            for node_id in 0..1000u64 {
                black_box(compute_node_offset(graph_start, node_id, record_size));
            }
        })
    });

    group.finish();
}

// ============================================================================
// Simulated Search Pattern Benchmark
// ============================================================================

fn bench_search_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_pattern");

    // Create a larger graph for realistic search simulation
    let (graph, _temp_file) = create_test_graph(10000, 128);

    // Simulate a search that visits ~100 nodes
    let visit_pattern:  Vec<NodeId> = (0..100).map(|i| (i * 97) % 10000).collect();

    group.bench_function("visit_100_nodes", |b| {
        b.iter(|| {
            for &node_id in &visit_pattern {
                // Read node and iterate neighbors (typical search operation)
                let record = graph. read_node_record(node_id).unwrap();
                for neighbor in record.neighbors_iter(0) {
                    black_box(neighbor);
                }
            }
        })
    });

    // Compare with mmap-based iteration
    group.bench_function("visit_100_nodes_mmap", |b| {
        b.iter(|| {
            for &node_id in &visit_pattern {
                // Use mmap-based neighbor iteration
                for neighbor in graph.neighbors_iter_from_mmap(node_id, 0).unwrap() {
                    black_box(neighbor);
                }
            }
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    header_benches,
    bench_graph_header_read,
    bench_graph_header_write,
);

criterion_group!(
    node_record_benches,
    bench_node_record_read,
    bench_node_record_write,
);

criterion_group!(neighbor_benches, bench_neighbors_from_mmap,);

criterion_group!(
    utility_benches,
    bench_get_node_bytes,
    bench_offset_computation,
);

criterion_group!(integration_benches, bench_search_pattern,);

criterion_main!(
    header_benches,
    node_record_benches,
    neighbor_benches,
    utility_benches,
    integration_benches,
);
