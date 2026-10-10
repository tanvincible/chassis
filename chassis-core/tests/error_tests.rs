//! Each mistake gets an error of its kind, with the value that was wrong and a line starting
//! "help:" that says what to do instead (ADR-0020).

use chassis_core::{DistanceMetric, ErrorKind, IndexOptions, IndexReader, Precision, VectorIndex};
use tempfile::tempdir;

#[track_caller]
fn says(result: chassis_core::Result<impl std::fmt::Debug>, kind: ErrorKind, words: &str) {
    let error = result.unwrap_err();
    let message = error.to_string();
    assert_eq!(error.kind(), kind, "{message}");
    assert!(message.contains(words), "{message}");
    assert!(message.contains("\nhelp: "), "{message}");
}

#[test]
fn test_a_mistake_says_what_was_wrong_and_what_to_do() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, 4, IndexOptions::default()).unwrap();
    index.add_with_id(7, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    index.flush().unwrap();

    says(index.add(&[1.0, 2.0]), ErrorKind::DimensionMismatch, "The vector has 2 components");
    says(index.search(&[1.0; 3], 1), ErrorKind::DimensionMismatch, "The query has 3 components");
    says(index.add(&[1.0, f32::NAN, 0.0, 0.0]), ErrorKind::InvalidArgument, "NaN at component 1");
    says(index.search(&[f32::INFINITY; 4], 1), ErrorKind::InvalidArgument, "inf at component 0");
    says(index.add_with_id(7, &[1.0; 4]), ErrorKind::IdInUse, "Id 7 is already in use");
    says(index.add_with_id(u64::MAX, &[1.0; 4]), ErrorKind::InvalidArgument, "is reserved");
    says(index.add_batch(&[1.0; 6]), ErrorKind::DimensionMismatch, "6 floats");
    says(index.add_batch_with_ids(&[1, 2], &[1.0; 4]), ErrorKind::InvalidArgument, "2 ids");
    says(index.add_batch_with_ids(&[9, 9], &[1.0; 8]), ErrorKind::InvalidArgument, "twice");

    says(VectorIndex::open(&path, 4, IndexOptions::default()), ErrorKind::Locked, "one writer");
    says(
        IndexReader::open(&path, 8, IndexOptions::default()),
        ErrorKind::DimensionMismatch,
        "4 dimensions, not 8",
    );
    let missing = dir.path().join("none.chassis");
    says(
        IndexReader::open(&missing, 4, IndexOptions::default()),
        ErrorKind::NotFound,
        "There is no index",
    );
    let nowhere = dir.path().join("no").join("such.chassis");
    says(
        VectorIndex::open(&nowhere, 4, IndexOptions::default()),
        ErrorKind::NotFound,
        "directory doesn't exist",
    );
    says(
        VectorIndex::open(dir.path(), 4, IndexOptions::default()),
        ErrorKind::InvalidArgument,
        "is a directory",
    );
    says(
        IndexReader::open(dir.path(), 4, IndexOptions::default()),
        ErrorKind::InvalidArgument,
        "is a directory",
    );
    let text = dir.path().join("notes.txt");
    std::fs::write(&text, "not an index").unwrap();
    says(
        VectorIndex::open(&text, 4, IndexOptions::default()),
        ErrorKind::NotAnIndex,
        "not a Chassis index",
    );
    assert_eq!(std::fs::read_to_string(&text).unwrap(), "not an index");
    says(
        VectorIndex::open(dir.path().join("new.chassis"), 0, IndexOptions::default()),
        ErrorKind::InvalidArgument,
        "dimensions is 0",
    );
    let thin = IndexOptions { max_connections: 1, ..IndexOptions::default() };
    says(
        VectorIndex::open(dir.path().join("new.chassis"), 4, thin),
        ErrorKind::InvalidArgument,
        "max_connections is 1",
    );

    drop(index);
    let cosine = IndexOptions { metric: DistanceMetric::Cosine, ..IndexOptions::default() };
    says(
        VectorIndex::open(&path, 4, cosine.clone()),
        ErrorKind::OptionsMismatch,
        "created with euclidean",
    );
    let half = IndexOptions { precision: Precision::Half, ..IndexOptions::default() };
    says(VectorIndex::open(&path, 4, half), ErrorKind::OptionsMismatch, "in full precision");
    let mut index = VectorIndex::open(dir.path().join("cosine.chassis"), 2, cosine).unwrap();
    says(index.add(&[0.0, 0.0]), ErrorKind::InvalidArgument, "zero vector");
}

#[test]
fn test_an_error_goes_into_anyhow_with_its_message() {
    fn open(path: &std::path::Path) -> anyhow::Result<VectorIndex> {
        Ok(VectorIndex::open(path, 0, IndexOptions::default())?)
    }
    let dir = tempdir().unwrap();
    let error = open(&dir.path().join("x.chassis")).unwrap_err();
    assert!(error.to_string().starts_with("dimensions is 0"));
    assert_eq!(
        error.downcast_ref::<chassis_core::Error>().unwrap().kind(),
        ErrorKind::InvalidArgument
    );
    // And into a boxed std error.
    let boxed: Box<dyn std::error::Error + Send + Sync> =
        VectorIndex::open(dir.path().join("y.chassis"), 0, IndexOptions::default())
            .unwrap_err()
            .into();
    assert!(boxed.to_string().contains("\nhelp: "));
}
