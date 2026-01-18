use chassis_core::Storage;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs;
use std::hint::black_box;
use tempfile::TempDir;

const DIMENSIONS: u32 = 768;

fn generate_vector(seed: u64) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|i| ((seed + i as u64) % 1000) as f32 / 1000.0)
        .collect()
}

fn bench_raw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_insert");
    group.sample_size(1000);

    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("bench.chassis");
    let mut storage = Storage::open(&path, DIMENSIONS).unwrap();

    let vector = generate_vector(0);

    group.bench_function("single_insert_no_commit", |b| {
        b.iter(|| {
            storage.insert(black_box(&vector)).unwrap();
        });
    });

    group.finish();
}

fn bench_durable_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("durable_insert");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("insert_with_commit", |b| {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("bench.chassis");
        let mut storage = Storage::open(&path, DIMENSIONS).unwrap();
        let vector = generate_vector(0);
        
        b.iter(|| {
            storage.insert(black_box(&vector)).unwrap();
            storage.commit().unwrap();
        });
    });

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    for batch_size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let temp_dir = TempDir::new().unwrap();
                let path = temp_dir.path().join("bench.chassis");
                let mut storage = Storage::open(&path, DIMENSIONS).unwrap();
                
                let vectors: Vec<Vec<f32>> = (0..batch_size)
                    .map(|i| generate_vector(i as u64))
                    .collect();

                b.iter(|| {
                    for vector in &vectors {
                        storage.insert(black_box(vector)).unwrap();
                    }
                    storage.commit().unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_hot_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_read");
    group.sample_size(1000);

    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("bench.chassis");
    let mut storage = Storage::open(&path, DIMENSIONS).unwrap();

    for i in 0..100 {
        storage.insert(&generate_vector(i)).unwrap();
    }
    storage.commit().unwrap();

    group.bench_function("get_vector_cached", |b| {
        b.iter(|| {
            let idx = black_box(50);
            let _vector = storage.get_vector(idx).unwrap();
        });
    });

    group.finish();
}

fn bench_cold_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_read");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("get_vector_cold_start", |b| {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("bench.chassis");
        
        {
            let mut storage = Storage::open(&path, DIMENSIONS).unwrap();
            for i in 0..10000 {
                storage.insert(&generate_vector(i)).unwrap();
            }
            storage.commit().unwrap();
        }

        b.iter(|| {
            drop(fs::File::open(&path).unwrap());
            
            let storage = Storage::open(&path, DIMENSIONS).unwrap();
            let idx = black_box(5000);
            let _vector = storage.get_vector(idx).unwrap();
        });
    });

    group.finish();
}

fn bench_sequential_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_read");
    group.sample_size(100);

    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("bench.chassis");
    let mut storage = Storage::open(&path, DIMENSIONS).unwrap();

    for i in 0..1000 {
        storage.insert(&generate_vector(i)).unwrap();
    }
    storage.commit().unwrap();

    group.bench_function("read_1000_sequential", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let _vector = storage.get_vector(black_box(i)).unwrap();
            }
        });
    });

    group.finish();
}

fn bench_remap_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("remap_overhead");
    group.sample_size(50);

    group.bench_function("grow_from_empty_to_1000", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("bench.chassis");
            let mut storage = Storage::open(&path, DIMENSIONS).unwrap();

            for i in 0..1000 {
                storage.insert(&generate_vector(i)).unwrap();
            }
            storage.commit().unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_insert,
    bench_durable_insert,
    bench_batch_insert,
    bench_hot_read,
    bench_cold_read,
    bench_sequential_read,
    bench_remap_overhead
);

criterion_main!(benches);
