//! Half precision (ADR-0018): an index that keeps 16-bit floats finds what brute force over the
//! vectors as given finds, in a smaller file of its own format version; the precision is the
//! file's.

use chassis_core::{
    DistanceMetric, IndexOptions, IndexReader, Precision, Storage, VectorIndex, cosine_distance,
    euclidean_distance,
};
use std::path::Path;
use tempfile::tempdir;

const DIMS: u32 = 96;

fn half() -> IndexOptions {
    IndexOptions { precision: Precision::Half, ..IndexOptions::default() }
}

fn random(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Components from −1 to 1, as an embedding's are.
fn vector(id: u64) -> Vec<f32> {
    let mut next = random(id);
    (0..DIMS).map(|_| next() * 2.0 - 1.0).collect()
}

/// Components that half precision holds exactly: multiples of 1/32 from −4 to 4.
fn exact_vector(id: u64) -> Vec<f32> {
    let mut next = random(id);
    (0..DIMS).map(|_| (next() * 256.0).floor() / 32.0 - 4.0).collect()
}

/// The format version in each header copy: what a release must know to read, then to write.
fn versions(path: &Path) -> [[u32; 2]; 2] {
    let bytes = std::fs::read(path).unwrap();
    let at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [0, 64 * 1024].map(|copy| [at(copy + 8), at(copy + 12)])
}

/// Of the true ten nearest of a hundred queries, by brute force over the vectors as given, how
/// many a search finds.
fn hits(index: &VectorIndex, n: u64) -> usize {
    let mut hits = 0;
    for q in 0..100 {
        let query = vector(1_000_000 + q);
        let mut exact: Vec<(f32, u64)> =
            (0..n).map(|id| (euclidean_distance(&query, &vector(id)), id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10).unwrap();
        for result in &found {
            // To the vector as kept: each component within 2⁻¹¹ of the one given.
            let want = euclidean_distance(&query, &vector(result.id));
            assert!((result.distance - want).abs() < 3e-3, "{} vs {want}", result.distance);
        }
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|f| f.id == *id)).count();
    }
    hits
}

#[test]
fn test_half_precision_finds_what_full_precision_finds() {
    let dir = tempdir().unwrap();
    let n = 3000;
    let [full, index] = [IndexOptions::default(), half()].map(|options| {
        let name = format!("{:?}.chassis", options.precision);
        let options = IndexOptions { ef_search: 200, ..options };
        let mut index = VectorIndex::open(dir.path().join(name), DIMS, options).unwrap();
        for id in 0..1000 {
            index.add(&vector(id)).unwrap();
        }
        index.add_batch(&(1000..n).flat_map(vector).collect::<Vec<f32>>()).unwrap();
        index
    });
    assert_eq!(index.precision(), Precision::Half);
    // Rounding can swap the tenth and eleventh nearest when they are all but equally far.
    let (full, half) = (hits(&full, n), hits(&index, n));
    assert!(full >= 980 && half + 10 >= full, "recall@10: {half} of 1000, {full} in full");

    // A vector finds itself, a rounding away, through the graph and through a scan.
    for id in (0..n).step_by(97) {
        let nearest = &index.search(&vector(id), 1).unwrap()[0];
        assert_eq!(nearest.id, id);
        assert!(nearest.distance < 3e-3);
        let scanned = &index.search_filtered(&vector(id), 1, |other| other == id).unwrap()[0];
        assert_eq!((scanned.id, scanned.distance.to_bits()), (id, nearest.distance.to_bits()));
    }
}

#[test]
fn test_values_half_precision_holds_are_measured_as_in_full_precision() {
    let dir = tempdir().unwrap();
    let n = 300;
    let mut indexes = [IndexOptions::default(), half()].map(|options| {
        let name = format!("{:?}.chassis", options.precision);
        let mut index = VectorIndex::open(dir.path().join(name), DIMS, options).unwrap();
        index.add_batch(&(0..n).flat_map(exact_vector).collect::<Vec<f32>>()).unwrap();
        for id in n..n + 50 {
            index.add(&exact_vector(id)).unwrap();
        }
        index
    });
    for q in 0..20 {
        // Every vector, so that the two graphs, which differ, don't matter.
        let query = vector(1_000_000 + q);
        let [full, half] = indexes.each_mut().map(|index| {
            let mut all = index.search(&query, 350).unwrap();
            all.sort_by_key(|r| r.id);
            all
        });
        assert_eq!((full.len(), half.len()), (350, 350));
        for (a, b) in full.iter().zip(&half) {
            assert_eq!(a.id, b.id);
            assert!((a.distance - b.distance).abs() <= 1e-5 * a.distance, "{a:?} and {b:?}");
        }
    }
}

#[test]
fn test_a_half_precision_file_is_about_half_the_size_and_format_4() {
    let dir = tempdir().unwrap();
    let dims = 768;
    let vectors: Vec<f32> = {
        let mut next = random(3);
        (0..dims * 400).map(|_| next()).collect()
    };
    let [full, half] = [IndexOptions::default(), half()].map(|options| {
        let path = dir.path().join(format!("{:?}.chassis", options.precision));
        let mut index = VectorIndex::open(&path, dims as u32, options).unwrap();
        index.add_batch(&vectors).unwrap();
        index.flush().unwrap();
        path
    });
    let size = |path: &Path| std::fs::metadata(path).unwrap().len() as f64;
    let ratio = size(&half) / size(&full);
    assert!((0.5..0.6).contains(&ratio), "half is {ratio} of full");

    // A release from before half precision reads and writes format 3, and must refuse the rest.
    assert_eq!(versions(&full), [[3, 3]; 2]);
    assert_eq!(versions(&half), [[4, 4]; 2]);
    // A compaction writes a new file, in the same precision.
    let mut index = VectorIndex::open(&half, dims as u32, self::half()).unwrap();
    assert!(index.delete(7).unwrap());
    index.compact().unwrap();
    assert_eq!((index.precision(), index.len()), (Precision::Half, 399));
    drop(index);
    assert_eq!(versions(&half), [[4, 4]; 2]);
}

#[test]
fn test_precision_is_the_files() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("half.chassis");
    let mut index = VectorIndex::open(&path, DIMS, half()).unwrap();
    index.add_batch(&(0..200).flat_map(vector).collect::<Vec<f32>>()).unwrap();
    index.flush().unwrap();
    drop(index);

    // A writer has to name it, as it names the metric; a reader takes it from the file.
    let error = VectorIndex::open(&path, DIMS, IndexOptions::default()).err().unwrap();
    assert!(error.to_string().contains("Half precision"), "{error}");
    let mut reader = IndexReader::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(reader.precision(), Precision::Half);
    let mut index = VectorIndex::open(&path, DIMS, half()).unwrap();
    for id in [0, 7, 199] {
        let (read, written) =
            (reader.search(&vector(id), 3).unwrap(), index.search(&vector(id), 3).unwrap());
        assert_eq!(read[0].id, id);
        assert_eq!(read, written);
    }
    index.add(&vector(200)).unwrap();
    index.flush().unwrap();
    assert_eq!(reader.search(&vector(200), 1).unwrap()[0].id, 200);

    let full = dir.path().join("full.chassis");
    VectorIndex::open(&full, DIMS, IndexOptions::default()).unwrap().flush().unwrap();
    let error = VectorIndex::open(&full, DIMS, half()).err().unwrap();
    assert!(error.to_string().contains("Full precision"), "{error}");
}

#[test]
fn test_a_vector_half_precision_cannot_hold_is_refused_and_leaves_nothing() {
    let dir = tempdir().unwrap();
    let mut index = VectorIndex::open(dir.path().join("half.chassis"), 4, half()).unwrap();
    index.add(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    // The largest half is 65,504, and from 65,520 a value would round to infinity.
    index.add(&[65_519.0, -65_519.0, 0.0, 0.0]).unwrap();

    let error = index.add(&[0.0, 65_520.0, 0.0, 0.0]).unwrap_err().to_string();
    assert!(error.contains("too large for a half-precision index"), "{error}");
    let batch = [[5.0; 4], [0.0, 0.0, -1e9, 0.0], [6.0; 4]].concat();
    assert!(index.add_batch(&batch).is_err());
    assert_eq!(index.len(), 2);
    assert_eq!(index.search(&[5.0; 4], 10).unwrap().len(), 2);

    // The index goes on as if they had never been offered.
    let id = index.add(&[5.0; 4]).unwrap();
    assert_eq!(id, 2);
    index.flush().unwrap();
    let nearest = &index.search(&[5.0; 4], 1).unwrap()[0];
    assert_eq!((nearest.id, nearest.distance), (2, 0.0));
    assert_eq!(index.search(&[65_504.0, -65_504.0, 0.0, 0.0], 1).unwrap()[0].distance, 0.0);
}

#[test]
fn test_storage_gives_back_what_half_precision_kept() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("half.chassis");
    let mut index = VectorIndex::open(&path, DIMS, half()).unwrap();
    index.add(&exact_vector(1)).unwrap();
    index.add(&vector(2)).unwrap();
    index.flush().unwrap();
    drop(index);

    let mut storage = Storage::open(&path, DIMS).unwrap();
    assert_eq!(storage.precision(), Precision::Half);
    assert_eq!(storage.get_vector(0).unwrap(), exact_vector(1));
    let kept = storage.get_vector(1).unwrap();
    for (kept, given) in kept.iter().zip(vector(2)) {
        assert!((kept - given).abs() <= given.abs() / 2048.0, "{kept} for {given}");
    }
    let error = storage.get_vector_slice(0).unwrap_err().to_string();
    assert!(error.contains("half precision"), "{error}");
    assert!(storage.insert(&[1e6; DIMS as usize]).is_err());
    assert_eq!(storage.count(), 2);
}

#[test]
fn test_cosine_in_half_precision() {
    let dir = tempdir().unwrap();
    let options = IndexOptions { metric: DistanceMetric::Cosine, ef_search: 200, ..half() };
    let mut index = VectorIndex::open(dir.path().join("cos.chassis"), DIMS, options).unwrap();
    let n = 2000;
    // Lengths from 0.5 to 50,000: a cosine index keeps directions, which always fit.
    let scaled = |id: u64| {
        vector(id).iter().map(|x| x * 10f32.powf(id as f32 % 6.0 - 0.3)).collect::<Vec<f32>>()
    };
    index.add_batch(&(0..n).flat_map(scaled).collect::<Vec<f32>>()).unwrap();

    let mut hits = 0;
    for q in 0..100 {
        let query = vector(1_000_000 + q);
        let mut exact: Vec<(f32, u64)> =
            (0..n).map(|id| (cosine_distance(&query, &scaled(id)), id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10).unwrap();
        for result in &found {
            let want = cosine_distance(&query, &scaled(result.id));
            assert!((result.distance - want).abs() < 1e-3, "{} vs {want}", result.distance);
        }
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|f| f.id == *id)).count();
    }
    assert!(hits >= 980, "recall@10 {} against brute-force cosine", hits as f64 / 1000.0);
}
