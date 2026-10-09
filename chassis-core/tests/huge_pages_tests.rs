//! `IndexOptions::huge_pages` (ADR-0016). Whether the kernel grants huge pages differs by machine
//! and can't be told from here; what must hold everywhere is that asking changes no result.

use chassis_core::{IndexOptions, IndexReader, SearchResult, VectorIndex};
use tempfile::tempdir;

/// 4 KiB a vector, so a thousand of them cover whole huge pages.
const DIMS: u32 = 1024;

/// Pseudo-random, so no two vectors tie.
fn vector(id: u64) -> Vec<f32> {
    (0..u64::from(DIMS))
        .map(|d| {
            let mut x = (id * u64::from(DIMS) + d).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x = (x ^ (x >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn options(huge_pages: bool) -> IndexOptions {
    IndexOptions { ef_construction: 32, ef_search: 32, huge_pages, ..IndexOptions::default() }
}

fn found(results: Vec<SearchResult>) -> Vec<(u64, u32)> {
    results.iter().map(|r| (r.id, r.distance.to_bits())).collect()
}

#[test]
fn test_asking_for_huge_pages_changes_no_result() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, DIMS, options(true)).unwrap();
    // A batch's vectors are asked for before they are written, a single add's once committed.
    index.add_batch(&(0..1100).flat_map(vector).collect::<Vec<f32>>()).unwrap();
    for id in 1100..1150 {
        index.add(&vector(id)).unwrap();
    }
    index.flush().unwrap();
    for id in 1150..1200 {
        index.add(&vector(id)).unwrap();
    }

    let mut with = IndexReader::open(&path, DIMS, options(true)).unwrap();
    let mut without = IndexReader::open(&path, DIMS, options(false)).unwrap();
    for id in (0..1150).step_by(37) {
        let wanted = found(without.search(&vector(id), 5).unwrap());
        assert_eq!(wanted[0].0, id);
        assert_eq!(found(with.search(&vector(id), 5).unwrap()), wanted);
    }
    // A reader that asked follows the writer's later flushes like any other.
    index.flush().unwrap();
    assert_eq!(with.search(&vector(1199), 1).unwrap()[0].id, 1199);
    assert_eq!(
        found(index.search(&vector(7), 5).unwrap()),
        found(with.search(&vector(7), 5).unwrap())
    );

    // Deletes, a compaction and a reopen keep the vectors, and the option goes with them.
    // Windows can't replace a file a reader has open.
    drop((with, without));
    for id in (0..1200).step_by(3) {
        assert!(index.delete(id).unwrap());
    }
    index.compact().unwrap();
    drop(index);
    let index = VectorIndex::open(&path, DIMS, options(true)).unwrap();
    let mut reader = IndexReader::open(&path, DIMS, options(true)).unwrap();
    assert_eq!((index.len(), reader.len()), (800, 800));
    for id in [1, 500, 1199] {
        assert_eq!(index.search(&vector(id), 1).unwrap()[0].id, id);
        assert_eq!(reader.search(&vector(id), 1).unwrap()[0].id, id);
    }
}
