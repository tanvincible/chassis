//! Cosine distance: ranking and distances match brute-force cosine; the metric is the file's.

use chassis_core::{DistanceMetric, IndexOptions, IndexReader, VectorIndex, cosine_distance};
use tempfile::tempdir;

const DIMS: u32 = 16;

/// Pseudo-random directions at lengths from 0.1 to 10, so L2 and cosine rank them differently.
fn vector(id: u64) -> Vec<f32> {
    let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let length = 0.1 + 9.9 * (next() + 0.5);
    let v: Vec<f32> = (0..DIMS).map(|_| next()).collect();
    let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    v.iter().map(|a| a / norm * length).collect()
}

fn cosine() -> IndexOptions {
    IndexOptions { metric: DistanceMetric::Cosine, ..IndexOptions::default() }
}

#[test]
fn test_cosine_index_ranks_and_reports_cosine_distance() {
    let dir = tempdir().unwrap();
    let mut index = VectorIndex::open(dir.path().join("cos.chassis"), DIMS, cosine()).unwrap();
    let n = 2000;
    for id in 0..n {
        index.add(&vector(id)).unwrap();
    }

    let (mut hits, mut euclidean_differs) = (0, 0);
    for q in 0..100 {
        let query = vector(10_000 + q);
        let mut exact: Vec<(f32, u64)> =
            (0..n).map(|id| (cosine_distance(&query, &vector(id)), id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10).unwrap();
        for result in &found {
            let want = cosine_distance(&query, &vector(result.id));
            assert!((result.distance - want).abs() < 1e-4, "{} vs {want}", result.distance);
        }
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|f| f.id == *id)).count();

        // Scaling the query changes nothing.
        let scaled: Vec<f32> = query.iter().map(|x| x * 7.5).collect();
        let again = index.search(&scaled, 10).unwrap();
        assert_eq!(
            found.iter().map(|r| r.id).collect::<Vec<_>>(),
            again.iter().map(|r| r.id).collect::<Vec<_>>()
        );

        let nearest_l2 = (0..n)
            .min_by(|&a, &b| {
                let d = |id| chassis_core::euclidean_distance(&query, &vector(id));
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        euclidean_differs += usize::from(nearest_l2 != exact[0].1);
    }
    assert!(hits >= 950, "recall@10 {} against brute-force cosine", hits as f64 / 1000.0);
    assert!(euclidean_differs > 50, "the data doesn't separate the metrics");
}

#[test]
fn test_cosine_handles_any_finite_vector_and_rejects_the_rest() {
    let dir = tempdir().unwrap();
    let mut index = VectorIndex::open(dir.path().join("cos.chassis"), 2, cosine()).unwrap();
    assert_eq!(index.metric(), DistanceMetric::Cosine);
    // Squared in f32, these would underflow to zero or overflow to infinity.
    for v in [[1e-30, 0.0], [f32::MIN_POSITIVE, 0.0], [3e20, 0.0], [0.0, 1e-38]] {
        index.add(&v).unwrap();
    }
    for v in [[f32::INFINITY, 1.0], [f32::NAN, 1.0]] {
        let error = index.add(&v).unwrap_err().to_string();
        assert!(error.contains("NaN or infinite"), "{error}");
    }
    let hits = index.search(&[-1e-25, 0.0], 4).unwrap();
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|h| h.distance <= 2.0), "antipodal distances stay within 2: {hits:?}");
}

#[test]
fn test_storage_insert_scales_vectors_in_a_cosine_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cos.chassis");
    VectorIndex::open(&path, 2, cosine()).unwrap().flush().unwrap();
    let mut storage = chassis_core::Storage::open(&path, 2).unwrap();
    let slot = storage.insert(&[10.0, 0.0]).unwrap();
    assert_eq!(storage.get_vector_slice(slot).unwrap(), [1.0, 0.0]);
}

#[test]
fn test_cosine_index_rejects_zero_vectors() {
    let dir = tempdir().unwrap();
    let mut index = VectorIndex::open(dir.path().join("cos.chassis"), DIMS, cosine()).unwrap();
    assert!(index.add(&[0.0; DIMS as usize]).is_err());
    index.add(&vector(1)).unwrap();
    assert!(index.search(&[0.0; DIMS as usize], 1).is_err());
    assert_eq!(index.len(), 1);
}

#[test]
fn test_metric_is_the_files() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cos.chassis");
    let mut index = VectorIndex::open(&path, DIMS, cosine()).unwrap();
    for id in 0..50 {
        index.add(&vector(id)).unwrap();
    }
    index.flush().unwrap();
    drop(index);

    assert!(VectorIndex::open(&path, DIMS, IndexOptions::default()).is_err());
    let mut reader = IndexReader::open(&path, DIMS, IndexOptions::default()).unwrap();
    let hit = &reader.search(&vector(7), 1).unwrap()[0];
    assert_eq!(hit.id, 7);
    assert!(hit.distance.abs() < 1e-5);
    let index = VectorIndex::open(&path, DIMS, cosine()).unwrap();
    assert_eq!(index.search(&vector(7), 1).unwrap()[0].id, 7);
}
