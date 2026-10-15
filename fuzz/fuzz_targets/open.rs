//! What both targets do with a file: open it as a reader and as a writer, then search it and
//! change it. Every outcome has to be an error or a result, never a panic, a crash or undefined
//! behavior: an embedded format meets bit rot, partial copies and sync tools.

use chassisdb::{DistanceMetric, IndexOptions, IndexReader, Precision, VectorIndex};

/// Every file in the compatibility set has vectors of 8 dimensions.
const DIMS: u32 = 8;

pub fn exercise(bytes: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    std::fs::write(&path, bytes).unwrap();
    // The options the header names, so that most opens get past checking them.
    let byte = |at: usize| bytes.get(at).copied().unwrap_or(0);
    let options = IndexOptions {
        metric: if byte(45) == 1 { DistanceMetric::Cosine } else { DistanceMetric::Euclidean },
        precision: if byte(52) == 1 { Precision::Half } else { Precision::Full },
        ..IndexOptions::default()
    };
    let query = [0.5f32; DIMS as usize];
    if let Ok(mut reader) = IndexReader::open(&path, DIMS, options.clone()) {
        let _ = reader.search(&query, 5);
        let _ = reader.search_filtered(&query, 5, |id| id % 2 == 0);
        let _ = reader.get(3);
        let _ = reader.ids();
    }
    if let Ok(mut index) = VectorIndex::open(&path, DIMS, options) {
        let _ = index.search(&query, 5);
        let _ = index.search_filtered(&query, 5, |id| id % 2 == 0);
        let _ = index.get(3);
        let _ = index.ids();
        let _ = index.add(&query);
        let _ = index.delete(3);
        let _ = index.flush();
    }
}
