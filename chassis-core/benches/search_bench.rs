//! Production-grade benchmarks for hardened HNSW search.
//!
//! # Benchmark Goals
//!
//! 1. Measure HashSet → Dense Filter improvement
//! 2. Verify no regression at small ef
//! 3. Demonstrate scaling at high ef
//! 4. Validate NaN-safe ordering overhead
//!
//! # Reporting Requirements
//!
//! - Absolute timings (µs or ns)
//! - CPU architecture
//! - Dataset characteristics

use chassis_core::{HnswGraph, HnswParams, Storage};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tempfile::TempDir;

/// Build a realistic HNSW graph for benchmarking
fn build_benchmark_graph(
    num_vectors: usize,
    dims: u32,
    connectivity: usize,
) -> (HnswGraph, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("search_bench.chassis");

    let mut storage = Storage::open(&path, dims).expect("Failed to open storage");

    // Insert vectors with realistic distribution
    for i in 0..num_vectors {
        let mut vec = vec![0.0; dims as usize];

        // Create clustered structure
        let cluster = (i / 50) as f32;
        let offset = (i % 50) as f32;

        vec[0] = cluster + offset * 0.02;
        vec[1] = cluster * 0.7 + offset * 0.01;

        // Add variation
        for j in 2..dims.min(16) as usize {
            vec[j] = ((i * 13 + j * 7) as f32).sin() * 0.3 + 0.5;
        }

        storage.insert(&vec).expect("Failed to insert vector");
    }

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).expect("Failed to open graph");

    // Build graph with realistic connectivity
    for i in 0..num_vectors as u64 {
        let neighbors = if i > 0 {
            let num_neighbors = connectivity.min(i as usize);
            #[allow(unused_variables)]
            let start = i.saturating_sub(num_neighbors as u64);

            // Mix of recent and older neighbors (realistic HNSW pattern)
            let mut neighs = Vec::new();

            // Recent neighbors (better for navigability)
            for j in (i.saturating_sub(num_neighbors as u64 / 2))..i {
                neighs.push(j);
            }

            // Some older neighbors (long-range connections)
            let step = i / (num_neighbors as u64).max(1);
            if step > 0 {
                for j in (0..i).step_by(step as usize) {
                    if neighs.len() < num_neighbors {
                        neighs.push(j);
                    }
                }
            }

            neighs.truncate(connectivity);
            vec![neighs]
        } else {
            vec![vec![]]
        };

        graph.link_node_bidirectional(i, 1, &neighbors).expect("Failed to link");
    }

    (graph, temp_dir)
}

/// Benchmark: Baseline search at various ef values
fn bench_search_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_baseline");
    group.sample_size(50);

    let (graph, _temp) = build_benchmark_graph(1000, 128, 16);

    for ef in [8, 16, 32, 64, 128] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef_val| {
            let query = vec![0.5; 128];

            b.iter(|| black_box(graph.search(&query, 10, ef_val).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: High-ef stress test (tests visited filter performance)
fn bench_high_ef_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_ef_stress");
    group.sample_size(20);

    let (graph, _temp) = build_benchmark_graph(5000, 128, 16);

    for ef in [64, 128, 256, 512] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef_val| {
            let query = vec![0.5; 128];

            b.iter(|| black_box(graph.search(&query, 10, ef_val).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: Small ef (no regression check)
fn bench_small_ef_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_ef_regression");
    group.sample_size(100);

    let (graph, _temp) = build_benchmark_graph(1000, 128, 16);

    for ef in [4, 8] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef_val| {
            let query = vec![0.5; 128];

            b.iter(|| black_box(graph.search(&query, ef_val, ef_val).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: Varying k values
fn bench_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_k");
    group.sample_size(50);

    let (graph, _temp) = build_benchmark_graph(1000, 128, 16);

    for k in [1, 5, 10, 50, 100] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k_val| {
            let query = vec![0.5; 128];
            let ef = (k_val * 2).max(50);

            b.iter(|| black_box(graph.search(&query, k_val, ef).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: Different graph sizes
fn bench_graph_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_size_scaling");
    group.sample_size(30);

    for size in [100, 500, 1000, 5000] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("nodes", size), &size, |b, &sz| {
            let (graph, _temp) = build_benchmark_graph(sz, 128, 16);
            let query = vec![0.5; 128];

            b.iter(|| black_box(graph.search(&query, 10, 50).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: Different dimensions
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group.sample_size(30);

    for dims in [64, 128, 384, 768, 1536] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("dimensions", dims), &dims, |b, &d| {
            let (graph, _temp) = build_benchmark_graph(500, d, 16);
            let query = vec![0.5; d as usize];

            b.iter(|| black_box(graph.search(&query, 10, 50).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark: Greedy layer descent
fn bench_greedy_descent(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_layer_descent");
    group.sample_size(50);

    // Build multi-layer graph
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("multilayer.chassis");
    let mut storage = Storage::open(&path, 128).unwrap();

    for i in 0..1000 {
        let mut vec = vec![0.0; 128];
        vec[0] = i as f32 / 1000.0;
        storage.insert(&vec).unwrap();
    }

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).unwrap();

    // Build with varying layers
    for i in 0..1000u64 {
        let layers = if i % 10 == 0 {
            3
        } else if i % 5 == 0 {
            2
        } else {
            1
        };
        let neighbors = if i > 0 { vec![vec![i - 1]; layers] } else { vec![vec![]; layers] };
        graph.link_node_bidirectional(i, layers, &neighbors).unwrap();
    }

    group.bench_function("multi_layer_search", |b| {
        let query = vec![0.5; 128];

        b.iter(|| black_box(graph.search(&query, 10, 50).unwrap()));
    });

    group.finish();
}

/// Benchmark: Visited filter overhead
fn bench_visited_filter_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("visited_filter");
    group.sample_size(100);

    // Small graph to isolate filter overhead
    let (graph, _temp) = build_benchmark_graph(100, 128, 8);

    group.bench_function("filter_overhead", |b| {
        let query = vec![0.5; 128];

        b.iter(|| black_box(graph.search(&query, 10, 20).unwrap()));
    });

    group.finish();
}

/// Benchmark: Worst-case scenario (highly connected, large ef)
fn bench_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");
    group.sample_size(10);

    // Dense graph, high ef
    let (graph, _temp) = build_benchmark_graph(2000, 128, 32);

    group.bench_function("dense_graph_high_ef", |b| {
        let query = vec![0.5; 128];

        b.iter(|| black_box(graph.search(&query, 10, 200).unwrap()));
    });

    group.finish();
}

/// Benchmark: Best-case scenario (sparse graph, low ef)
fn bench_best_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_case");
    group.sample_size(100);

    // Sparse graph, low ef
    let (graph, _temp) = build_benchmark_graph(1000, 128, 4);

    group.bench_function("sparse_graph_low_ef", |b| {
        let query = vec![0.5; 128];

        b.iter(|| black_box(graph.search(&query, 5, 10).unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_search_baseline,
    bench_high_ef_stress,
    bench_small_ef_regression,
    bench_varying_k,
    bench_graph_size_scaling,
    bench_dimension_scaling,
    bench_greedy_descent,
    bench_visited_filter_overhead,
    bench_worst_case,
    bench_best_case,
);

criterion_main!(benches);
