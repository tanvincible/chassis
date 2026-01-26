use chassis_core::node::{Node, NodeRecord, NodeRecordParams, compute_node_offset};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_record_size_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_size");

    for (m, max_layers) in [(16, 8), (16, 16), (32, 16), (64, 16)] {
        let params = NodeRecordParams::new(m, m * 2, max_layers);

        group.bench_with_input(
            BenchmarkId::new("calculate", format!("m{}_l{}", m, max_layers)),
            &params,
            |b, params| {
                b.iter(|| black_box(params.record_size()));
            },
        );
    }

    group.finish();
}

fn bench_node_offset_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_offset");

    let params = NodeRecordParams::default();
    let record_size = params.record_size();
    let graph_start = 4096u64;

    group.bench_function("compute_offset", |b| {
        b.iter(|| {
            for node_id in 0..1000 {
                black_box(compute_node_offset(graph_start, node_id, record_size));
            }
        });
    });

    group.finish();
}

fn bench_record_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_serialization");

    let params = NodeRecordParams::new(16, 32, 8);
    let mut record = NodeRecord::new(0, 4, params);

    record.set_neighbors(0, &(1..=20).collect::<Vec<_>>());
    record.set_neighbors(1, &[100, 200, 300]);
    record.set_neighbors(2, &[1000, 2000]);
    record.set_neighbors(3, &[10000]);

    group.bench_function("to_bytes", |b| {
        b.iter(|| black_box(record.to_bytes()));
    });

    let bytes = record.to_bytes();

    group.bench_function("from_bytes", |b| {
        b.iter(|| black_box(NodeRecord::from_bytes(&bytes, params).unwrap()));
    });

    group.finish();
}

fn bench_neighbor_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_access");

    let params = NodeRecordParams::new(16, 32, 8);
    let mut record = NodeRecord::new(0, 4, params);

    record.set_neighbors(0, &(1..=20).collect::<Vec<_>>());
    record.set_neighbors(1, &(100..=110).collect::<Vec<_>>());

    group.bench_function("get_neighbors_layer0", |b| {
        b.iter(|| black_box(record.get_neighbors(0)));
    });

    group.bench_function("get_neighbors_layer1", |b| {
        b.iter(|| black_box(record.get_neighbors(1)));
    });

    group.bench_function("neighbor_count", |b| {
        b.iter(|| {
            black_box(record.neighbor_count(0));
            black_box(record.neighbor_count(1));
        });
    });

    // Add to the bench_neighbor_access function, after the existing benchmarks:

    group.bench_function("neighbors_iter_layer0", |b| {
        b.iter(|| {
            let sum: u64 = record.neighbors_iter(0).sum();
            black_box(sum)
        });
    });

    group.bench_function("neighbors_iter_layer1", |b| {
        b.iter(|| {
            let sum: u64 = record.neighbors_iter(1).sum();
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_node_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_conversion");

    let params = NodeRecordParams::new(16, 32, 8);

    let mut node = Node::new(42, 4);
    node.layers[0] = (1..=15).collect();
    node.layers[1] = (100..=108).collect();
    node.layers[2] = (1000..=1005).collect();
    node.layers[3] = vec![10000, 10001];

    group.bench_function("node_to_record", |b| {
        b.iter(|| black_box(node.to_record(params)));
    });

    let record = node.to_record(params);

    group.bench_function("record_to_node", |b| {
        b.iter(|| black_box(Node::from_record(&record)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_record_size_calculation,
    bench_node_offset_computation,
    bench_record_serialization,
    bench_neighbor_access,
    bench_node_conversion,
);

criterion_main!(benches);
