//! Batch adds link on every core (ADR-0010): the graph must be as good as one built a vector at a
//! time, and a failed batch must add nothing.

use chassis_core::{
    DistanceMetric, IndexOptions, IndexReader, VectorIndex, cosine_distance, euclidean_distance,
};
use tempfile::tempdir;

const DIMS: usize = 24;

fn vector(id: u64) -> Vec<f32> {
    let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn flat(ids: std::ops::Range<u64>) -> Vec<f32> {
    ids.flat_map(vector).collect()
}

fn options() -> IndexOptions {
    IndexOptions { ef_construction: 100, ef_search: 64, ..IndexOptions::default() }
}

/// Recall@10 over 100 queries against brute force over `0..n`.
fn recall(index: &VectorIndex, n: u64) -> f64 {
    let mut hits = 0;
    for q in 0..100 {
        let query = vector(1_000_000 + q);
        let mut exact: Vec<(f32, u64)> =
            (0..n).map(|id| (euclidean_distance(&query, &vector(id)), id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10).unwrap();
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|r| r.id == *id)).count();
    }
    hits as f64 / 1000.0
}

#[test]
fn test_batch_builds_a_graph_as_good_as_adding_one_at_a_time() {
    let dir = tempdir().unwrap();
    let n = 4000;
    let mut one =
        VectorIndex::open(dir.path().join("one.chassis"), DIMS as u32, options()).unwrap();
    for id in 0..n {
        one.add(&vector(id)).unwrap();
    }
    let mut batch =
        VectorIndex::open(dir.path().join("batch.chassis"), DIMS as u32, options()).unwrap();
    assert_eq!(batch.add_batch(&flat(0..n)).unwrap(), (0..n).collect::<Vec<_>>());
    assert_eq!(batch.len(), n);

    let (sequential, parallel) = (recall(&one, n), recall(&batch, n));
    assert!(parallel >= 0.95 && parallel >= sequential - 0.02, "{parallel} vs {sequential}");
    // Every vector is reachable: it finds itself.
    let found = (0..n).filter(|&id| batch.search(&vector(id), 1).unwrap()[0].id == id).count();
    assert!(found as u64 >= n - 2, "{found} of {n} vectors find themselves");
}

#[test]
fn test_batches_mix_with_single_adds_and_survive_reopening() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.chassis");
    let mut index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    for id in 0..300 {
        index.add(&vector(id)).unwrap();
    }
    assert_eq!(index.add_batch(&flat(300..2300)).unwrap(), (300..2300).collect::<Vec<_>>());
    assert_eq!(index.add(&vector(2300)).unwrap(), 2300);
    assert_eq!(index.add_batch(&[]).unwrap(), Vec::<u64>::new());
    index.flush().unwrap();
    drop(index);

    let index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    assert_eq!(index.len(), 2301);
    assert!(recall(&index, 2301) >= 0.95);
}

#[test]
fn test_batch_with_ids_is_all_or_nothing() {
    let dir = tempdir().unwrap();
    let mut index =
        VectorIndex::open(dir.path().join("ids.chassis"), DIMS as u32, options()).unwrap();
    let ids: Vec<u64> = (0..1000).map(|i| i * 10 + 7).collect();
    index.add_batch_with_ids(&ids, &flat(0..1000)).unwrap();
    assert_eq!(index.search(&vector(123), 1).unwrap()[0].id, 1237);

    let rejected = [
        (vec![5, 5], flat(2000..2002)),               // repeated in the batch
        (vec![5, 1237], flat(2000..2002)),            // already in the index
        (vec![5, u64::MAX], flat(2000..2002)),        // reserved
        (vec![5, 6], flat(2000..2003)),               // three vectors for two ids
        (vec![5, 6], flat(2000..2002)[1..].to_vec()), // not whole vectors
    ];
    for (ids, vectors) in rejected {
        assert!(index.add_batch_with_ids(&ids, &vectors).is_err(), "{ids:?}");
        assert_eq!(index.len(), 1000);
    }
    assert!(index.add_batch(&flat(0..1)[1..]).is_err());

    // Ids continue past the largest one used.
    assert_eq!(index.add_batch(&flat(3000..3002)).unwrap(), [9998, 9999]);
    assert_eq!(index.search(&vector(3001), 1).unwrap()[0].id, 9999);
}

#[test]
fn test_cosine_batch_rejects_a_zero_vector_and_adds_nothing() {
    let dir = tempdir().unwrap();
    let cosine = IndexOptions { metric: DistanceMetric::Cosine, ..options() };
    let mut index = VectorIndex::open(dir.path().join("cos.chassis"), DIMS as u32, cosine).unwrap();
    let mut vectors = flat(0..100);
    vectors[50 * DIMS..51 * DIMS].fill(0.0);
    assert!(index.add_batch(&vectors).is_err());
    assert!(index.is_empty());

    let scaled: Vec<f32> = flat(0..500).iter().map(|x| x * 3.0).collect();
    index.add_batch(&scaled).unwrap();
    let query = vector(1_000_001);
    for r in index.search(&query, 5).unwrap() {
        let want = cosine_distance(&query, &vector(r.id));
        assert!((r.distance - want).abs() < 1e-4, "{} vs {want}", r.distance);
    }
}

#[test]
fn test_reader_searches_while_a_batch_links() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.chassis");
    let mut writer = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    writer.add_batch(&flat(0..1000)).unwrap();
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS as u32, options()).unwrap();

    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            writer.add_batch(&flat(1000..6000)).unwrap();
            done.store(true, std::sync::atomic::Ordering::Release);
        });
        let mut searches = 0;
        while !done.load(std::sync::atomic::Ordering::Acquire) || searches < 50 {
            let found = reader.search(&vector(searches % 1000), 10).unwrap();
            assert!(found.iter().all(|r| r.id < 1000), "returned an unflushed vector");
            searches += 1;
        }
    });
    writer.flush().unwrap();
    assert_eq!(reader.search(&vector(5555), 1).unwrap()[0].id, 5555);
}

/// A batch lost to a crash leaves links to its slots in committed nodes, on every layer the lost
/// nodes were on. A batch reusing a slot must not follow them to a node that hasn't linked yet,
/// least of all to the node it is linking: it would find no neighbors but itself.
#[test]
fn test_a_batch_reusing_a_lost_batchs_slots_stays_connected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let (kept, n) = (500, 800);
    let mut index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    index.add_batch(&flat(0..kept)).unwrap();
    index.flush().unwrap();
    index.add_batch(&flat(kept..n)).unwrap();
    drop(index);

    let mut index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    assert_eq!(index.len(), kept);
    let stranded: Vec<u64> = (kept..n)
        .filter(|&id| {
            index.add_batch(&vector(id)).unwrap();
            index.search(&vector(id), 10).unwrap().len() < 10
        })
        .collect();
    assert!(stranded.is_empty(), "added {stranded:?} with too few neighbors to search from");
}
