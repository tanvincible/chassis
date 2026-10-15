//! Exact copies of one vector, which real data has (the same text embedded twice, empty strings):
//! each list keeps one copy, so the copies can't fill every list among themselves and leave other
//! vectors out of reach.

use chassisdb::{IndexOptions, VectorIndex};

const DIMS: usize = 32;

fn vector(i: u64) -> Vec<f32> {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn test_copies_leave_every_other_vector_within_reach() {
    for batch in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut index = VectorIndex::open(
            dir.path().join("copies.chassis"),
            DIMS as u32,
            IndexOptions::default(),
        )
        .unwrap();
        let mut distinct = Vec::new();
        let add = |index: &mut VectorIndex, vectors: &[Vec<f32>]| -> Vec<u64> {
            if batch {
                index.add_batch(&vectors.concat()).unwrap()
            } else {
                vectors.iter().map(|v| index.add(v).unwrap()).collect()
            }
        };
        for round in 0..20 {
            // Some distinct vectors, then 40 copies of another one.
            let fresh: Vec<Vec<f32>> = (0..20).map(|i| vector(round * 100 + i)).collect();
            distinct.extend(add(&mut index, &fresh).into_iter().zip(fresh));
            add(&mut index, &vec![vector(1_000_000 + round); 40]);
        }
        let lost = distinct.iter().filter(|(id, v)| index.search(v, 1).unwrap()[0].id != *id);
        assert_eq!(lost.count(), 0, "batch {batch}: distinct vectors out of reach");
    }
}
