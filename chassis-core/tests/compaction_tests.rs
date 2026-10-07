//! `compact()` rewrites the index without its deleted vectors (ADR-0011): the same vectors under
//! the same ids, in a smaller file, with readers following along.

use chassis_core::{DistanceMetric, IndexOptions, IndexReader, VectorIndex, euclidean_distance};
use std::path::Path;
use tempfile::tempdir;

const DIMS: usize = 24;

fn vector(key: u64) -> Vec<f32> {
    let mut x = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn options() -> IndexOptions {
    IndexOptions { ef_construction: 100, ef_search: 64, ..IndexOptions::default() }
}

/// An index of `n` vectors; vector `key` has id `key * 3 + 1`.
fn build(path: &Path, n: u64) -> VectorIndex {
    let mut index = VectorIndex::open(path, DIMS as u32, options()).unwrap();
    let ids: Vec<u64> = (0..n).map(|key| key * 3 + 1).collect();
    index.add_batch_with_ids(&ids, &(0..n).flat_map(vector).collect::<Vec<_>>()).unwrap();
    index
}

/// Recall@10 over 100 queries against brute force over the keys `live` accepts.
fn recall(index: &VectorIndex, n: u64, live: impl Fn(u64) -> bool) -> f64 {
    let mut hits = 0;
    for q in 0..100 {
        let query = vector(1_000_000 + q);
        let mut exact: Vec<(f32, u64)> = (0..n)
            .filter(|&key| live(key))
            .map(|key| (euclidean_distance(&query, &vector(key)), key * 3 + 1))
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10).unwrap();
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|r| r.id == *id)).count();
    }
    hits as f64 / 1000.0
}

#[test]
fn test_compact_keeps_live_vectors_under_their_ids_in_a_smaller_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.chassis");
    let n = 4000;
    let mut index = build(&path, n);
    // Delete two of every three vectors.
    for key in (0..n).filter(|key| key % 3 != 0) {
        assert!(index.delete(key * 3 + 1).unwrap());
    }
    index.flush().unwrap();
    let before = std::fs::metadata(&path).unwrap().len();
    let live = |key: u64| key.is_multiple_of(3);

    index.compact().unwrap();
    assert_eq!(index.len(), n.div_ceil(3));
    let after = std::fs::metadata(&path).unwrap().len();
    // Segments double in size, so a third of the vectors isn't a third of the file.
    assert!(after * 3 < before * 2, "{after} bytes after, {before} before");
    assert!(!dir.path().join("c.chassis.compacting").exists());

    for reopened in [false, true] {
        if reopened {
            drop(index);
            index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
        }
        assert_eq!(index.len(), n.div_ceil(3));
        for key in (0..n).step_by(7) {
            let nearest = index.search(&vector(key), 1).unwrap()[0].id;
            assert_eq!(nearest == key * 3 + 1, live(key), "key {key} found {nearest}");
        }
        assert!(recall(&index, n, live) >= 0.95);
    }
    // It is a normal index afterwards.
    index.add_with_id(2, &vector(5_000_000)).unwrap();
    assert!(index.delete(1).unwrap());
    index.flush().unwrap();
    assert_eq!(index.search(&vector(5_000_000), 1).unwrap()[0].id, 2);
}

#[test]
fn test_compact_never_frees_an_id_for_add() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ids.chassis");
    let mut index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    for key in 0..100 {
        assert_eq!(index.add(&vector(key)).unwrap(), key);
    }
    // The largest ids go: after compaction no slot shows they were ever used.
    for id in 90..100 {
        index.delete(id).unwrap();
    }
    // Nothing was flushed: compacting makes the adds and deletes durable, as a flush would.
    index.compact().unwrap();
    assert_eq!(index.len(), 90);
    drop(index);

    let mut index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    assert_eq!(index.len(), 90);
    assert_eq!(index.add(&vector(100)).unwrap(), 100);
    assert_eq!(index.add(&vector(101)).unwrap(), 101);
    // A second compaction carries the mark on.
    index.delete(101).unwrap();
    index.compact().unwrap();
    assert_eq!(index.add(&vector(102)).unwrap(), 102);
    assert_eq!(index.search(&vector(42), 1).unwrap()[0].id, 42);
}

/// `u64::MAX` is reserved, so the id before it is the last `add` can give.
#[test]
fn test_add_fails_once_ids_run_out_and_compaction_keeps_it_so() {
    let dir = tempdir().unwrap();
    let mut index =
        VectorIndex::open(dir.path().join("top.chassis"), DIMS as u32, options()).unwrap();
    index.add_with_id(u64::MAX - 2, &vector(0)).unwrap();
    assert_eq!(index.add(&vector(1)).unwrap(), u64::MAX - 1);
    for _ in 0..2 {
        assert!(index.add(&vector(2)).is_err());
        assert!(index.add_batch(&vector(2)).is_err());
        assert_eq!(index.len(), 2);
        index.compact().unwrap();
    }
    index.add_with_id(7, &vector(2)).unwrap();
    assert_eq!(index.search(&vector(1), 1).unwrap()[0].id, u64::MAX - 1);
}

/// The copy replaces the file a symlink points to, not the link.
#[cfg(unix)]
#[test]
fn test_compaction_through_a_symlink_replaces_the_file_it_points_to() {
    let dir = tempdir().unwrap();
    let (file, link) = (dir.path().join("file.chassis"), dir.path().join("link.chassis"));
    build(&file, 300).flush().unwrap();
    std::os::unix::fs::symlink(&file, &link).unwrap();

    let mut index = VectorIndex::open(&link, DIMS as u32, options()).unwrap();
    for key in 0..200 {
        index.delete(key * 3 + 1).unwrap();
    }
    index.compact().unwrap();
    drop(index);
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    let index = VectorIndex::open(&file, DIMS as u32, options()).unwrap();
    assert_eq!(index.len(), 100);
}

#[test]
fn test_compact_with_nothing_deleted_keeps_slot_ids() {
    let dir = tempdir().unwrap();
    let mut index =
        VectorIndex::open(dir.path().join("p.chassis"), DIMS as u32, options()).unwrap();
    assert!(index.compact().is_ok(), "an empty index compacts");
    index.add_batch(&(0..500).flat_map(vector).collect::<Vec<_>>()).unwrap();
    index.compact().unwrap();
    assert_eq!(index.len(), 500);
    assert_eq!(index.add(&vector(500)).unwrap(), 500);
    assert_eq!(index.search(&vector(123), 1).unwrap()[0].id, 123);
}

#[test]
fn test_compact_keeps_a_cosine_index_cosine() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cos.chassis");
    let cosine = IndexOptions { metric: DistanceMetric::Cosine, ..options() };
    let mut index = VectorIndex::open(&path, DIMS as u32, cosine.clone()).unwrap();
    let scaled: Vec<f32> = (0..300).flat_map(vector).map(|x| x * 5.0).collect();
    index.add_batch(&scaled).unwrap();
    index.delete(7).unwrap();
    index.compact().unwrap();
    assert_eq!(index.metric(), DistanceMetric::Cosine);
    let hit = &index.search(&vector(8), 1).unwrap()[0];
    assert_eq!(hit.id, 8);
    assert!(hit.distance.abs() < 1e-5, "{}", hit.distance);
    drop(index);
    assert!(VectorIndex::open(&path, DIMS as u32, options()).is_err(), "still a cosine file");
}

#[test]
fn test_opening_removes_a_copy_a_crashed_compaction_left() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.chassis");
    build(&path, 10).flush().unwrap();
    let left = dir.path().join("s.chassis.compacting");
    std::fs::write(&left, b"half a copy").unwrap();
    let index = VectorIndex::open(&path, DIMS as u32, options()).unwrap();
    assert!(!left.exists());
    assert_eq!(index.len(), 10);
}

#[test]
fn test_failed_compaction_leaves_the_index_as_it_was() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.chassis");
    let mut index = build(&path, 300);
    index.delete(1).unwrap();
    // The copy's path is taken by a directory, so it can't be written.
    std::fs::create_dir(dir.path().join("f.chassis.compacting")).unwrap();
    assert!(index.compact().is_err());
    assert_eq!(index.len(), 299);
    assert_eq!(index.search(&vector(5), 1).unwrap()[0].id, 16);
    index.add_with_id(1, &vector(0)).unwrap();
    index.flush().unwrap();

    std::fs::remove_dir(dir.path().join("f.chassis.compacting")).unwrap();
    index.compact().unwrap();
    assert_eq!(index.len(), 300);
}

/// Readers in other processes hold no lock, so one in this process stands in for them.
#[cfg(unix)]
#[test]
fn test_reader_follows_a_compaction() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("r.chassis");
    let mut writer = build(&path, 2000);
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS as u32, options()).unwrap();
    assert_eq!(reader.search(&vector(10), 1).unwrap()[0].id, 31);
    let mut seen = vec![reader.snapshot()];

    for key in 0..1000 {
        writer.delete(key * 3 + 1).unwrap();
    }
    writer.flush().unwrap();
    reader.refresh().unwrap();
    seen.push(reader.snapshot());
    writer.compact().unwrap();

    // The reader moves to the new file: the deleted half is gone, the rest is where it was.
    assert_eq!(reader.search(&vector(1500), 1).unwrap()[0].id, 4501);
    assert_ne!(reader.search(&vector(10), 1).unwrap()[0].id, 31);
    assert_eq!(reader.len(), 1000);
    assert!(!seen.contains(&reader.snapshot()));
    seen.push(reader.snapshot());

    // It keeps following the writer there. Adding back to the old file's slot count must not
    // bring back the name of a snapshot of the old file, which held other vectors.
    let more: Vec<u64> = (0..1000).map(|key| key * 3 + 2).collect();
    writer
        .add_batch_with_ids(&more, &(0..1000).flat_map(|k| vector(k + 50_000)).collect::<Vec<_>>())
        .unwrap();
    writer.flush().unwrap();
    assert_eq!(reader.search(&vector(50_007), 1).unwrap()[0].id, 23);
    assert_eq!(reader.len(), 2000);
    assert!(!seen.contains(&reader.snapshot()), "{:?} names two contents", reader.snapshot());
}

/// Windows can't replace a file another handle has open.
#[cfg(windows)]
#[test]
fn test_compaction_fails_while_a_reader_has_the_index_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("r.chassis");
    let mut writer = build(&path, 500);
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS as u32, options()).unwrap();
    writer.delete(1).unwrap();
    assert!(writer.compact().is_err());
    assert_eq!(writer.len(), 499);
    assert_eq!(reader.search(&vector(10), 1).unwrap()[0].id, 31);
    drop(reader);
    writer.compact().unwrap();
    assert_eq!(writer.len(), 499);
}

/// ADR-0008's third experiment, for compaction: readers search without a pause while the writer
/// adds, deletes and compacts. No search may fail, and every result must be a real vector at its
/// true distance, whichever file the reader was on.
#[cfg(unix)]
#[test]
fn test_readers_search_through_repeated_compactions() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.chassis");
    let mut writer = build(&path, 1000);
    writer.flush().unwrap();
    let (done, searches) = (AtomicBool::new(false), AtomicU64::new(0));

    std::thread::scope(|scope| {
        for reader_number in 0..2u64 {
            let (path, done, searches) = (&path, &done, &searches);
            scope.spawn(move || {
                let mut reader = IndexReader::open(path, DIMS as u32, options()).unwrap();
                let mut q = reader_number;
                while !done.load(Ordering::Acquire) {
                    q += 2;
                    let query = vector(2_000_000 + q % 500);
                    let found = reader.search(&query, 10).unwrap();
                    assert_eq!(found.len(), 10, "search {q} on snapshot {:?}", reader.snapshot());
                    for r in &found {
                        let want = euclidean_distance(&query, &vector((r.id - 1) / 3));
                        assert!((r.distance - want).abs() < 1e-4, "id {} at {}", r.id, r.distance);
                    }
                    searches.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        // Each round replaces the oldest 300 vectors with 300 new ones, then compacts.
        for round in 0..10u64 {
            let added = 1000 + round * 300..1000 + (round + 1) * 300;
            let ids: Vec<u64> = added.clone().map(|key| key * 3 + 1).collect();
            writer.add_batch_with_ids(&ids, &added.flat_map(vector).collect::<Vec<_>>()).unwrap();
            for key in round * 300..(round + 1) * 300 {
                assert!(writer.delete(key * 3 + 1).unwrap());
            }
            writer.compact().unwrap();
            assert_eq!(writer.len(), 1000);
        }
        done.store(true, Ordering::Release);
    });
    assert!(searches.load(Ordering::Relaxed) > 100, "the readers barely ran");
}
