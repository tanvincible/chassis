use chassis_core::hnsw::{HnswGraph, HnswParams, NodeRecord, NodeRecordParams};
use chassis_core::Storage;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

fn setup_graph_with_nodes(node_count: usize) -> (NamedTempFile, HnswGraph) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    for i in 0..node_count {
        let vector = vec![i as f32; 128];
        storage.insert(&vector).unwrap();
    }
    storage.commit().unwrap();
    drop(storage);
    
    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
    
    let params = NodeRecordParams::default();
    
    // Create and write node records with varying neighbor counts
    for node_id in 0..node_count as u64 {
        let layer_count = (node_id % 4 + 1) as u8; // 1-4 layers
        let mut record = NodeRecord::new(node_id, layer_count, params);
        
        // Fill layer 0 with some neighbors
        let layer0_neighbors: Vec<u64> = (0..10.min(node_count as u64))
            .map(|i| (node_id + i + 1) % node_count as u64)
            .collect();
        record.set_neighbors(0, &layer0_neighbors);
        
        // Fill layer 1 if exists
        if layer_count > 1 {
            let layer1_neighbors: Vec<u64> = (0..5.min(node_count as u64))
                .map(|i| (node_id + i * 2 + 1) % node_count as u64)
                .collect();
            record.set_neighbors(1, &layer1_neighbors);
        }
        
        // Fill layer 2 if exists
        if layer_count > 2 {
            let layer2_neighbors: Vec<u64> = (0..3.min(node_count as u64))
                .map(|i| (node_id + i * 3 + 1) % node_count as u64)
                .collect();
            record.set_neighbors(2, &layer2_neighbors);
        }
        
        graph.write_node_record(&record).unwrap();
    }
    
    (temp_file, graph)
}

// ============================================================================
// Graph Header Benchmarks
// ============================================================================

fn bench_graph_header_read(c: &mut Criterion) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
    
    // Write some state
    graph.insert(0, 2).unwrap();
    graph.commit().unwrap();
    
    c.bench_function("graph_header_read", |b| {
        b.iter(|| {
            // Read via the internal read method (by reopening)
            let storage = Storage::open(path, 128).unwrap();
            let graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
            black_box(&graph.entry_point);
            black_box(&graph.max_layer);
        });
    });
}

fn bench_graph_header_write(c: &mut Criterion) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    c.bench_function("graph_header_write", |b| {
        b.iter(|| {
            let storage = Storage::open(path, 128).unwrap();
            let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
            
            // Modify state
            graph.entry_point = Some(42);
            graph.max_layer = 5;
            
            // Write header (part of commit)
            graph.commit().unwrap();
            black_box(&graph);
        });
    });
}

// ============================================================================
// Node Record I/O Benchmarks
// ============================================================================

fn bench_node_record_read(c: &mut Criterion) {
    let (_temp_file, graph) = setup_graph_with_nodes(1000);
    
    c.bench_function("node_record_read", |b| {
        b.iter(|| {
            for node_id in (0..1000).step_by(10) {
                let record = graph.read_node_record(node_id).unwrap();
                black_box(&record);
            }
        });
    });
}

fn bench_node_record_write(c: &mut Criterion) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    for i in 0..100 {
        let vector = vec![i as f32; 128];
        storage.insert(&vector).unwrap();
    }
    storage.commit().unwrap();
    drop(storage);
    
    c.bench_function("node_record_write", |b| {
        b.iter(|| {
            let storage = Storage::open(path, 128).unwrap();
            let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
            
            let params = NodeRecordParams::default();
            let mut record = NodeRecord::new(0, 3, params);
            record.set_neighbors(0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            record.set_neighbors(1, &[11, 12, 13, 14, 15]);
            record.set_neighbors(2, &[21, 22, 23]);
            
            graph.write_node_record(&record).unwrap();
            black_box(&graph);
        });
    });
}

// ============================================================================
// Neighbor Iteration Benchmarks
// ============================================================================

fn bench_neighbors_from_mmap_layer0(c: &mut Criterion) {
    let (_temp_file, graph) = setup_graph_with_nodes(1000);
    
    c.bench_function("neighbors_from_mmap_layer0", |b| {
        b.iter(|| {
            for node_id in (0..100).step_by(10) {
                let sum: u64 = graph
                    .neighbors_iter_from_mmap(node_id, 0)
                    .unwrap()
                    .sum();
                black_box(sum);
            }
        });
    });
}

fn bench_neighbors_from_mmap_layer1(c: &mut Criterion) {
    let (_temp_file, graph) = setup_graph_with_nodes(1000);
    
    c.bench_function("neighbors_from_mmap_layer1", |b| {
        b.iter(|| {
            for node_id in (0..100).step_by(10) {
                let sum: u64 = graph
                    .neighbors_iter_from_mmap(node_id, 1)
                    .unwrap()
                    .sum();
                black_box(sum);
            }
        });
    });
}

// Compare mmap iteration vs record deserialization
fn bench_neighbors_comparison(c: &mut Criterion) {
    let (_temp_file, graph) = setup_graph_with_nodes(1000);
    let mut group = c.benchmark_group("neighbors_comparison");
    
    // Via mmap iterator (zero-allocation)
    group.bench_function("mmap_iter", |b| {
        b.iter(|| {
            let sum: u64 = graph
                .neighbors_iter_from_mmap(42, 0)
                .unwrap()
                .sum();
            black_box(sum);
        });
    });
    
    // Via full record deserialization (allocates)
    group.bench_function("full_record", |b| {
        b.iter(|| {
            let record = graph.read_node_record(42).unwrap();
            let neighbors = record.get_neighbors(0);
            let sum: u64 = neighbors.iter().sum();
            black_box(sum);
        });
    });
    
    group.finish();
}

// ============================================================================
// Random Access Pattern Benchmarks
// ============================================================================

fn bench_random_node_access(c: &mut Criterion) {
    use rand::Rng;
    
    let (_temp_file, graph) = setup_graph_with_nodes(10000);
    
    // Generate random access pattern (simulates search)
    let mut rng = rand::rng();
    let random_nodes: Vec<u64> = (0..1000).map(|_| rng.random_range(0..10000)).collect();
    
    let mut group = c.benchmark_group("random_node_access");
    
    group.bench_function("read_records", |b| {
        b.iter(|| {
            for &node_id in random_nodes.iter().take(100) {
                let record = graph.read_node_record(node_id).unwrap();
                black_box(&record);
            }
        });
    });
    
    group.bench_function("get_bytes", |b| {
        b.iter(|| {
            for &node_id in random_nodes.iter().take(100) {
                let bytes = graph.get_node_bytes(node_id).unwrap();
                black_box(bytes);
            }
        });
    });
    
    group.bench_function("neighbors_iter", |b| {
        b.iter(|| {
            for &node_id in random_nodes.iter().take(100) {
                let sum: u64 = graph
                    .neighbors_iter_from_mmap(node_id, 0)
                    .unwrap()
                    .sum();
                black_box(sum);
            }
        });
    });
    
    group.finish();
}

// ============================================================================
// Sequential Access Benchmarks
// ============================================================================

fn bench_sequential_access(c: &mut Criterion) {
    let (_temp_file, graph) = setup_graph_with_nodes(1000);
    
    let mut group = c.benchmark_group("sequential_access");
    
    group.bench_function("read_100_records", |b| {
        b.iter(|| {
            for node_id in 0..100 {
                let record = graph.read_node_record(node_id).unwrap();
                black_box(&record);
            }
        });
    });
    
    group.bench_function("iter_100_neighbors", |b| {
        b.iter(|| {
            for node_id in 0..100 {
                let sum: u64 = graph
                    .neighbors_iter_from_mmap(node_id, 0)
                    .unwrap()
                    .sum();
                black_box(sum);
            }
        });
    });
    
    group.finish();
}

// ============================================================================
// Record Size Variations
// ============================================================================

fn bench_different_record_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_sizes");
    
    for (m, max_layers) in [(8, 4), (16, 8), (32, 16), (64, 16)] {
        let params = HnswParams {
            max_connections: m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
            max_layers,
        };
        
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        let mut storage = Storage::open(path, 128).unwrap();
        for i in 0..100 {
            let vector = vec![i as f32; 128];
            storage.insert(&vector).unwrap();
        }
        storage.commit().unwrap();
        drop(storage);
        
        let storage = Storage::open(path, 128).unwrap();
        let mut graph = HnswGraph::open(storage, params).unwrap();
        
        let record_params = params.to_record_params();
        let mut record = NodeRecord::new(0, 2, record_params);
        record.set_neighbors(0, &vec![1u64; m as usize]);
        record.set_neighbors(1, &vec![2u64; m as usize / 2]);
        graph.write_node_record(&record).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("write", format!("m{}_l{}", m, max_layers)),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut rec = NodeRecord::new(1, 2, record_params);
                    rec.set_neighbors(0, &vec![1u64; m as usize]);
                    black_box(g);
                    black_box(&rec);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("read", format!("m{}_l{}", m, max_layers)),
            &graph,
            |b, g| {
                b.iter(|| {
                    let record = g.read_node_record(0).unwrap();
                    black_box(&record);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_header_read,
    bench_graph_header_write,
    bench_node_record_read,
    bench_node_record_write,
    bench_neighbors_from_mmap_layer0,
    bench_neighbors_from_mmap_layer1,
    bench_neighbors_comparison,
    bench_random_node_access,
    bench_sequential_access,
    bench_different_record_sizes,
);

criterion_main!(benches);
