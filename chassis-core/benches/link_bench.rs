//! Production-grade benchmarks for bidirectional linking with diversity heuristics.
//!
//! # Benchmark Philosophy
//!
//! - **No resource leaks**: Proper file descriptor management
//! - **Realistic workloads**: Patterns mirror actual HNSW construction
//! - **Statistical rigor**: Sufficient samples, stable measurements
//! - **Isolated tests**: Each benchmark measures one thing
//!
//! # Categories
//!
//! 1. **Throughput**: Single node, batch operations
//! 2. **Pruning**: Cache effectiveness under pressure
//! 3. **Scalability**: Dimensions, layers, graph size
//! 4. **Edge Cases**: Worst-case scenarios

use chassis_core::{HnswGraph, HnswParams, Storage};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tempfile::TempDir;

type NodeId = u64;

/// Create a graph with pre-built structure
fn create_prepared_graph(num_vectors: usize, dims: u32, build_to: usize) -> (HnswGraph, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("bench.chassis");

    let mut storage = Storage::open(&path, dims).expect("Failed to open storage");

    // Insert vectors with realistic clustering
    for i in 0..num_vectors {
        let mut vec = vec![0.0; dims as usize];
        let cluster = (i / 10) as f32;
        let offset = (i % 10) as f32;

        vec[0] = cluster + offset * 0.1;
        vec[1] = cluster * 0.5 + offset * 0.05;

        // Variation in high dimensions
        for j in 2..dims.min(16) as usize {
            vec[j] = ((i * 7 + j * 3) as f32).sin() * 0.5 + 0.5;
        }

        storage.insert(&vec).expect("Failed to insert vector");
    }

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).expect("Failed to open graph");

    // Build graph to target size
    for i in 0..build_to.min(num_vectors) {
        let node_id = i as u64;

        let neighbors = if i > 0 {
            let max_neighbors = graph.record_params().m0 as usize;
            let num_to_link = i.min(max_neighbors / 2);

            let mut neighs = Vec::new();
            for j in (i.saturating_sub(num_to_link))..i {
                neighs.push(j as u64);
            }
            vec![neighs]
        } else {
            vec![vec![]]
        };

        graph.link_node_bidirectional(node_id, 1, &neighbors).expect("Failed to link");
    }

    (graph, temp_dir)
}

/// Benchmark: Single node linking with varying neighbor counts
fn bench_single_link(c: &mut Criterion) {
    let mut group = c.benchmark_group("link_single_node");
    group.sample_size(100);

    for neighbor_count in [4, 8, 16, 32] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("neighbors", neighbor_count),
            &neighbor_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        // Fresh graph for each iteration
                        create_prepared_graph(100, 128, count + 10)
                    },
                    |(mut graph, _temp_dir)| {
                        let node_id = graph.node_count();
                        let neighbors: Vec<NodeId> =
                            (node_id.saturating_sub(count as u64)..node_id).collect();

                        black_box(graph.link_node_bidirectional(node_id, 1, &[neighbors]).unwrap());
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Pruning with full neighbor lists
fn bench_pruning_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("link_with_pruning");
    group.sample_size(50);
    group.throughput(Throughput::Elements(1));

    group.bench_function("full_list_pruning", |b| {
        b.iter_batched(
            || {
                // Setup: Hub with full neighbor list
                let (mut graph, temp_dir) = create_prepared_graph(200, 128, 0);
                let m0 = graph.record_params().m0 as usize;

                // Build hub at node 0
                graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

                for i in 1..=(m0 + 10) {
                    graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
                }

                (graph, temp_dir)
            },
            |(mut graph, _temp_dir)| {
                let next_id = graph.node_count();
                black_box(graph.link_node_bidirectional(next_id, 1, &[vec![0]]).unwrap());
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Benchmark: Multi-layer linking
fn bench_multilayer_linking(c: &mut Criterion) {
    let mut group = c.benchmark_group("multilayer_linking");
    group.sample_size(100);

    for num_layers in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(num_layers));

        group.bench_with_input(
            BenchmarkId::new("layers", num_layers),
            &num_layers,
            |b, &layers| {
                b.iter_batched(
                    || create_prepared_graph(100, 128, 20),
                    |(mut graph, _temp_dir)| {
                        let node_id = graph.node_count();
                        let neighbors: Vec<Vec<NodeId>> = (0..layers)
                            .map(|l| {
                                vec![
                                    node_id.saturating_sub(1 + l as u64),
                                    node_id.saturating_sub(2 + l as u64),
                                ]
                            })
                            .collect();

                        black_box(
                            graph
                                .link_node_bidirectional(node_id, layers as usize, &neighbors)
                                .unwrap(),
                        );
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch linking throughput
fn bench_batch_linking(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_linking");
    group.sample_size(20);

    for batch_size in [10, 50, 100] {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(BenchmarkId::new("nodes", batch_size), &batch_size, |b, &size| {
            b.iter_batched(
                || create_prepared_graph(size as usize + 10, 128, 0),
                |(mut graph, _temp_dir)| {
                    for i in 0..size as u64 {
                        let neighbors = if i > 0 { vec![vec![i - 1]] } else { vec![vec![]] };

                        black_box(graph.link_node_bidirectional(i, 1, &neighbors).unwrap());
                    }
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

/// Benchmark: Cache effectiveness
fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effectiveness");
    group.sample_size(50);

    for candidate_count in [8, 16, 32] {
        group.throughput(Throughput::Elements(candidate_count));

        group.bench_with_input(
            BenchmarkId::new("candidates", candidate_count),
            &candidate_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let (mut graph, temp_dir) = create_prepared_graph(200, 128, 0);

                        // Build hub
                        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();
                        for i in 1..=count {
                            graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
                        }

                        (graph, temp_dir)
                    },
                    |(mut graph, _temp_dir)| {
                        let next_id = graph.node_count();
                        black_box(graph.link_node_bidirectional(next_id, 1, &[vec![0]]).unwrap());
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Worst-case clustering
fn bench_worst_case_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case_clustering");
    group.sample_size(30);

    group.bench_function("identical_vectors", |b| {
        b.iter_batched(
            || {
                let temp_dir = TempDir::new().unwrap();
                let path = temp_dir.path().join("identical.chassis");
                let mut storage = Storage::open(&path, 128).unwrap();

                for _ in 0..50 {
                    storage.insert(&vec![0.5; 128]).unwrap();
                }

                let params = HnswParams::default();
                let graph = HnswGraph::open(storage, params).unwrap();
                (graph, temp_dir)
            },
            |(mut graph, _temp_dir)| {
                // Stress test diversity heuristic with identical vectors
                for i in 0..32u64 {
                    let neighbors = if i > 0 {
                        let max_prev = i.min(16);
                        let neighs: Vec<u64> = (i.saturating_sub(max_prev)..i).collect();
                        vec![neighs]
                    } else {
                        vec![vec![]]
                    };

                    black_box(graph.link_node_bidirectional(i, 1, &neighbors).unwrap());
                }
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Benchmark: High-dimensional vectors
fn bench_high_dimensional(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");
    group.sample_size(20);

    for dims in [384, 768, 1536] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("dimensions", dims), &dims, |b, &d| {
            b.iter_batched(
                || create_prepared_graph(50, d, 20),
                |(mut graph, _temp_dir)| {
                    let node_id = graph.node_count();
                    let neighbors =
                        vec![vec![node_id.saturating_sub(1), node_id.saturating_sub(2)]];

                    black_box(graph.link_node_bidirectional(node_id, 1, &neighbors).unwrap());
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

/// Benchmark: Idempotency overhead
fn bench_idempotency_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("idempotency");
    group.sample_size(100);

    group.bench_function("retry_existing_link", |b| {
        b.iter_batched(
            || create_prepared_graph(100, 128, 10),
            |(mut graph, _temp_dir)| {
                // Attempt to re-add existing link (should be fast - duplicate check)
                black_box(graph.add_backward_link_with_pruning(1, 0, 0).unwrap());
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Benchmark: Sequential construction (realistic HNSW building)
fn bench_sequential_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_construction");
    group.sample_size(10);

    for graph_size in [50, 100, 200] {
        group.throughput(Throughput::Elements(graph_size));

        group.bench_with_input(BenchmarkId::new("nodes", graph_size), &graph_size, |b, &size| {
            b.iter_batched(
                || create_prepared_graph(size as usize + 10, 128, 0),
                |(mut graph, _temp_dir)| {
                    // Realistic HNSW construction pattern
                    let m0 = graph.record_params().m0 as usize;

                    for i in 0..size as u64 {
                        let neighbors = if i > 0 {
                            // Link to some recent nodes
                            let num_to_link = i.min(m0 as u64 / 2);
                            let start = i.saturating_sub(num_to_link);
                            let neighs: Vec<u64> = (start..i).collect();
                            vec![neighs]
                        } else {
                            vec![vec![]]
                        };

                        black_box(graph.link_node_bidirectional(i, 1, &neighbors).unwrap());
                    }
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_link,
    bench_pruning_pressure,
    bench_multilayer_linking,
    bench_batch_linking,
    bench_cache_effectiveness,
    bench_worst_case_clustering,
    bench_high_dimensional,
    bench_idempotency_check,
    bench_sequential_construction,
);

criterion_main!(benches);
