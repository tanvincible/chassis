//! Filtered search (ADR-0009): only accepted ids come back, and they are the nearest ones.

use chassis_core::{
    DistanceMetric, IndexOptions, IndexReader, VectorIndex, cosine_distance, euclidean_distance,
};
use tempfile::tempdir;

const DIMS: u32 = 16;
const N: u64 = 4000;

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

/// Ids `id * 7 + 3`, so slot numbers and ids differ.
fn build(path: &std::path::Path, options: IndexOptions) -> VectorIndex {
    let mut index = VectorIndex::open(path, DIMS, options).unwrap();
    for slot in 0..N {
        index.add_with_id(slot * 7 + 3, &vector(slot)).unwrap();
    }
    index
}

fn exact(query: &[f32], k: usize, allow: impl Fn(u64) -> bool) -> Vec<u64> {
    let mut all: Vec<(f32, u64)> = (0..N)
        .filter(|&slot| allow(slot * 7 + 3))
        .map(|slot| (euclidean_distance(query, &vector(slot)), slot * 7 + 3))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn test_filtered_search_returns_the_nearest_accepted_ids() {
    let dir = tempdir().unwrap();
    let index = build(&dir.path().join("f.chassis"), IndexOptions::default());
    // Half, 5% and 0.5% of ids; the last is always scanned, so it is exact.
    for modulus in [2, 20, 200] {
        let allow = |id: u64| id % modulus == 3 % modulus;
        let (mut hits, mut total) = (0, 0);
        for q in 0..50 {
            let query = vector(100_000 + q);
            let found = index.search_filtered(&query, 10, allow).unwrap();
            let want = exact(&query, 10, allow);
            assert_eq!(found.len(), want.len());
            assert!(found.iter().all(|r| allow(r.id)), "returned an id the filter rejects");
            assert!(found.windows(2).all(|w| w[0].distance <= w[1].distance));
            hits += want.iter().filter(|id| found.iter().any(|r| r.id == **id)).count();
            total += want.len();
            if modulus == 200 {
                assert_eq!(found.iter().map(|r| r.id).collect::<Vec<_>>(), want, "a scan is exact");
            }
        }
        assert!(hits * 100 >= total * 97, "1/{modulus}: recall {hits}/{total}");
    }
    // All but the nearest: the graph search serves this one.
    let query = vector(100_000);
    let nearest = index.search(&query, 1).unwrap()[0].id;
    let rest = index.search_filtered(&query, 10, |id| id != nearest).unwrap();
    assert_eq!(rest.len(), 10);
    assert!(rest.iter().all(|r| r.id != nearest));

    assert!(index.search_filtered(&vector(1), 10, |_| false).unwrap().is_empty());
    let only = index.search_filtered(&vector(1), 10, |id| id == 7 * 1234 + 3).unwrap();
    assert_eq!(only.iter().map(|r| r.id).collect::<Vec<_>>(), [7 * 1234 + 3]);
}

#[test]
fn test_a_filter_may_search_the_index_itself() {
    let dir = tempdir().unwrap();
    let index = build(&dir.path().join("f.chassis"), IndexOptions::default());
    let (query, probe) = (vector(100_000), vector(100_001));
    // Accept an id only if it isn't the probe's nearest neighbor, found by a search made from
    // inside the filter, on the thread that is already searching.
    let banned = index.search(&probe, 1).unwrap()[0].id;
    let found = index
        .search_filtered(&query, 10, |id| index.search(&probe, 1).unwrap()[0].id != id)
        .unwrap();
    let want = index.search_filtered(&query, 10, |id| id != banned).unwrap();
    let ids =
        |results: &[chassis_core::SearchResult]| results.iter().map(|r| r.id).collect::<Vec<_>>();
    assert_eq!(ids(&found), ids(&want));
    assert_eq!(found.len(), 10);
    assert!(!ids(&found).contains(&banned));
}

#[test]
fn test_filtered_search_skips_deleted_vectors() {
    let dir = tempdir().unwrap();
    let mut index = build(&dir.path().join("f.chassis"), IndexOptions::default());
    let query = vector(5);
    // Every id (the graph search), half, and 0.5% (a scan).
    for modulus in [1, 2, 200] {
        let allow = |id: u64| id % modulus == 3 % modulus;
        let nearest = index.search_filtered(&query, 1, allow).unwrap()[0].id;
        assert!(index.delete(nearest).unwrap());
        let next = index.search_filtered(&query, 10, allow).unwrap();
        assert!(next.iter().all(|r| r.id != nearest && allow(r.id)));
        assert_eq!(next.len(), 10);
    }
}

#[test]
fn test_reader_filters_within_its_snapshot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.chassis");
    let mut writer = build(&path, IndexOptions::default());
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS, IndexOptions::default()).unwrap();
    writer.add_with_id(1, &vector(77)).unwrap();

    for modulus in [2, 200] {
        let allow = |id: u64| id == 1 || id % modulus == 3 % modulus;
        let found = reader.search_filtered(&vector(77), 10, allow).unwrap();
        assert!(found.iter().all(|r| r.id != 1), "returned an unflushed vector");
        assert_eq!(found.len(), 10);
    }
    writer.flush().unwrap();
    assert_eq!(reader.search_filtered(&vector(77), 1, |id| id == 1).unwrap()[0].id, 1);
}

#[test]
fn test_cosine_filtered_search_reports_cosine_distance() {
    let dir = tempdir().unwrap();
    let cosine = IndexOptions { metric: DistanceMetric::Cosine, ..IndexOptions::default() };
    let index = build(&dir.path().join("f.chassis"), cosine);
    let query = vector(100_001);
    for modulus in [2, 200] {
        for r in index.search_filtered(&query, 5, |id| id % modulus == 3 % modulus).unwrap() {
            let want = cosine_distance(&query, &vector((r.id - 3) / 7));
            assert!((r.distance - want).abs() < 1e-4, "{} vs {want}", r.distance);
        }
    }
}

/// Adds lost to a crash leave committed nodes with links to slots that are gone, unless the undo
/// file takes them out again (ADR-0012), which a power loss or an older release may not leave.
/// The graph search can then run out of nodes before it runs out of budget. A filter must still
/// find its ids.
#[test]
fn test_a_filter_finds_its_ids_in_a_graph_a_crash_thinned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let kept = 20;
    let mut index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    for id in 0..kept {
        index.add(&vector(id)).unwrap();
    }
    index.flush().unwrap();
    for id in kept..2000 {
        index.add(&vector(id)).unwrap();
    }
    drop(index);
    std::fs::remove_file(dir.path().join("index.chassis.undo")).unwrap();

    let index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), kept);
    for id in 0..kept {
        let found = index.search_filtered(&vector(id), 1, |other| other == id).unwrap();
        assert_eq!(found.iter().map(|r| r.id).collect::<Vec<_>>(), [id]);
        let all = index.search_filtered(&vector(id), kept as usize, |_| true).unwrap();
        assert_eq!(all.len(), kept as usize, "from {id}");
    }
}

/// A thread-local that searches in its destructor runs after the thread's visited filter is gone.
#[test]
fn test_a_search_may_run_while_its_thread_exits() {
    struct SearchOnExit(std::sync::Arc<VectorIndex>);
    impl Drop for SearchOnExit {
        fn drop(&mut self) {
            assert_eq!(self.0.search(&vector(1), 3).unwrap().len(), 3);
        }
    }
    thread_local!(static GUARD: std::cell::OnceCell<SearchOnExit> = const { std::cell::OnceCell::new() });

    let dir = tempdir().unwrap();
    let index =
        std::sync::Arc::new(build(&dir.path().join("index.chassis"), IndexOptions::default()));
    let worker = std::thread::spawn(move || {
        // Registered before the first search creates the filter, so destroyed after it.
        GUARD.with(|guard| drop(guard.set(SearchOnExit(index.clone()))));
        index.search(&vector(0), 1).unwrap().len()
    });
    assert_eq!(worker.join().unwrap(), 1);
}
