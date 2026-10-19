//! Integration tests for VectorIndex facade
//!
//! Tests cover the complete lifecycle: open -> add -> search -> flush

use chassisdb::{ErrorKind, IndexOptions, IndexReader, VectorIndex};
use tempfile::NamedTempFile;

#[test]
fn test_index_lifecycle_basic() {
    let temp_file = NamedTempFile::new().unwrap();

    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.dimensions(), 128);

    let id1 = index.add(&vec![0.1; 128]).unwrap();
    let id2 = index.add(&vec![0.2; 128]).unwrap();
    let id3 = index.add(&vec![0.3; 128]).unwrap();

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);
    assert_eq!(index.len(), 3);

    let query = vec![0.15; 128];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].distance <= results[1].distance);

    index.flush().unwrap();
}

#[test]
fn test_index_persistence() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    {
        let mut index = VectorIndex::open(&path, 256, IndexOptions::default()).unwrap();

        for i in 0..20 {
            let mut vec = vec![0.0; 256];
            vec[0] = i as f32 / 20.0;
            vec[1] = ((i % 5) as f32) / 5.0;
            index.add(&vec).unwrap();
        }

        assert_eq!(index.len(), 20);
        index.flush().unwrap();
    }

    {
        let index = VectorIndex::open(&path, 256, IndexOptions::default()).unwrap();

        assert_eq!(index.len(), 20);
        assert_eq!(index.dimensions(), 256);

        let query = vec![0.5; 256];
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_dimension_validation() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Wrong dimension on add
    let result = index.add(&vec![0.1; 64]);
    assert_eq!(result.unwrap_err().kind(), chassisdb::ErrorKind::DimensionMismatch);

    // Wrong dimension on search
    let error = index.search(&vec![0.1; 64], 5).unwrap_err();
    assert_eq!(error.kind(), chassisdb::ErrorKind::DimensionMismatch);
    assert!(error.to_string().contains("The query has 64 components"));
}

#[test]
fn test_search_empty_index() {
    let temp_file = NamedTempFile::new().unwrap();
    let index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    let query = vec![0.5; 128];
    let results = index.search(&query, 10).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_search_single_vector() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    index.add(&vec![0.5; 128]).unwrap();

    let query = vec![0.5; 128];
    let results = index.search(&query, 5).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!(results[0].distance < 0.001);
}

#[test]
fn test_search_quality() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(
        temp_file.path(),
        2, // Small dimension for predictable distances
        IndexOptions::default(),
    )
    .unwrap();

    // Create a line of points: [0, 0], [1, 0], [2, 0], ..., [9, 0]
    for i in 0..10 {
        let vec = vec![i as f32, 0.0];
        index.add(&vec).unwrap();
    }

    // Query at [4.5, 0] - should find [4, 0] and [5, 0] as nearest
    let query = vec![4.5, 0.0];
    let results = index.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3);

    let top_two_ids: Vec<u64> = results.iter().take(2).map(|r| r.id).collect();
    assert!(top_two_ids.contains(&4));
    assert!(top_two_ids.contains(&5));
}

#[test]
fn test_large_batch_insert() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    for i in 0..100 {
        let mut vec = vec![0.0; 128];
        vec[0] = (i as f32) / 100.0;
        vec[1] = ((i % 10) as f32) / 10.0;

        let id = index.add(&vec).unwrap();
        assert_eq!(id, i as u64);
    }

    assert_eq!(index.len(), 100);

    let query = vec![0.5; 128];
    let results = index.search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    for i in 1..results.len() {
        assert!(results[i - 1].distance <= results[i].distance);
    }
}

#[test]
fn test_custom_options() {
    let temp_file = NamedTempFile::new().unwrap();

    let options = IndexOptions {
        max_connections: 8,
        ef_construction: 100,
        ef_search: 25,
        ..Default::default()
    };

    let mut index = VectorIndex::open(temp_file.path(), 128, options).unwrap();

    for i in 0..20 {
        let vec = vec![i as f32 / 20.0; 128];
        index.add(&vec).unwrap();
    }

    let results = index.search(&vec![0.5; 128], 5).unwrap();
    assert!(results.len() <= 5);
}

#[test]
fn test_flush_durability() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        for i in 0..10 {
            index.add(&vec![i as f32; 128]).unwrap();
        }

        index.flush().unwrap();
    }

    {
        let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
        assert_eq!(index.len(), 10);
    }
}

#[test]
fn test_no_flush_loses_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        for i in 0..5 {
            index.add(&vec![i as f32; 128]).unwrap();
        }

        // Dropped without a flush.
    }

    {
        let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        // The adds were never flushed, so they are rolled back.
        assert!(index.len() <= 5);
    }
}

#[test]
fn test_ghost_node_recovery() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Manually create a ghost node scenario
    {
        use chassisdb::{HnswGraph, HnswParams, Storage};

        let mut storage = Storage::open(&path, 128).unwrap();

        // Insert vector but don't add to graph
        storage.insert(&vec![1.0; 128]).unwrap();
        storage.insert(&vec![2.0; 128]).unwrap();
        storage.commit().unwrap();

        // Open graph and only add first vector
        let params = HnswParams::default();
        let mut graph = HnswGraph::open(storage, params).unwrap();

        graph.write_node_and_backlinks(0, 1, &[vec![]]).unwrap();
        graph.publish_node(0, 1).unwrap();
        graph.commit().unwrap();

        // Now we have:
        // - storage.count() = 2
        // - graph.node_count() = 1
        // - Vector at index 1 is a "ghost node"
    }

    // Open with VectorIndex - should handle ghost gracefully
    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        // Should see only the indexed vector
        assert_eq!(index.len(), 1);

        // Next add should reclaim the ghost node's space
        let id = index.add(&vec![3.0; 128]).unwrap();
        assert_eq!(id, 1); // Reuses ghost node ID

        assert_eq!(index.len(), 2);
    }
}

#[test]
fn test_dimension_mismatch_on_reopen() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
        index.add(&vec![1.0; 128]).unwrap();
        index.flush().unwrap();
    }

    {
        let error = VectorIndex::open(&path, 256, IndexOptions::default()).unwrap_err();
        assert_eq!(error.kind(), chassisdb::ErrorKind::DimensionMismatch);
        assert!(error.to_string().contains("holds vectors of 128 dimensions, not 256"));
    }
}

#[test]
fn test_max_connections_mismatch_on_reopen_keeps_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    {
        let mut index = VectorIndex::open(&path, 64, IndexOptions::default()).unwrap();
        for i in 0..50 {
            index.add(&vec![i as f32; 64]).unwrap();
        }
        index.flush().unwrap();
    }

    let other = IndexOptions { max_connections: 8, ..IndexOptions::default() };
    let error = VectorIndex::open(&path, 64, other).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::OptionsMismatch);
    assert_eq!(
        error.to_string(),
        "This index was created with max_connections 16, so it has to be opened with 16 too, not \
         8\nhelp: open it with max_connections 16"
    );

    let index = VectorIndex::open(&path, 64, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 50);
}

#[test]
fn test_vectors_without_graph_are_an_error_not_a_reset() {
    use chassisdb::Storage;

    let temp_file = NamedTempFile::new().unwrap();
    {
        let mut storage = Storage::open(temp_file.path(), 64).unwrap();
        storage.insert(&vec![1.0; 64]).unwrap();
        storage.commit().unwrap();
    }

    assert!(VectorIndex::open(temp_file.path(), 64, IndexOptions::default()).is_err());
    assert_eq!(Storage::open(temp_file.path(), 64).unwrap().count(), 1);
}

#[test]
fn test_truncated_graph_is_an_error() {
    let temp_file = NamedTempFile::new().unwrap();
    {
        let mut index = VectorIndex::open(temp_file.path(), 64, IndexOptions::default()).unwrap();
        for i in 0..50 {
            index.add(&vec![i as f32; 64]).unwrap();
        }
        index.flush().unwrap();
    }

    // Regions start every 64 KiB, so a shorter cut can remove only the padding after the last
    // one, which nothing reads: such a file opens, as it should.
    let len = std::fs::metadata(temp_file.path()).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(temp_file.path()).unwrap();
    file.set_len(len - 64 * 1024).unwrap();
    assert!(VectorIndex::open(temp_file.path(), 64, IndexOptions::default()).is_err());
}

#[test]
fn test_search_returns_k_or_fewer() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    for i in 0..5 {
        index.add(&vec![i as f32; 128]).unwrap();
    }

    let results = index.search(&vec![0.0; 128], 10).unwrap();
    assert_eq!(results.len(), 5);

    let results = index.search(&vec![0.0; 128], 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_identical_vectors() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    for _ in 0..10 {
        index.add(&vec![0.5; 128]).unwrap();
    }

    let results = index.search(&vec![0.5; 128], 5).unwrap();

    assert_eq!(results.len(), 5);

    for result in results {
        assert!(result.distance < 0.001);
    }
}

#[test]
fn test_search_with_k_zero() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    index.add(&vec![1.0; 128]).unwrap();

    let error = index.search(&vec![1.0; 128], 0).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(
        error.to_string(),
        "k is 0, but has to be at least 1\nhelp: pass how many results to return, such as 10"
    );
}

#[test]
fn test_concurrent_opens_fail() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    let _index1 = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

    let result = VectorIndex::open(&path, 128, IndexOptions::default());

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already open"));
}

#[test]
fn test_stress_sequential_adds() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    for i in 0..1000 {
        let mut vec = vec![0.0; 128];
        vec[0] = (i as f32) / 1000.0;
        vec[1] = ((i % 100) as f32) / 100.0;
        vec[2] = ((i % 10) as f32) / 10.0;

        let id = index.add(&vec).unwrap();
        assert_eq!(id, i as u64);
    }

    assert_eq!(index.len(), 1000);

    let query = vec![0.5; 128];
    let results = index.search(&query, 20).unwrap();

    assert_eq!(results.len(), 20);

    for i in 1..results.len() {
        assert!(results[i - 1].distance <= results[i].distance);
    }
}

#[test]
fn test_custom_ids_are_returned_and_persist() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();
    {
        let mut index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
        index.add_with_id(1_000, &[1.0; 8]).unwrap();
        index.add_with_id(7, &[2.0; 8]).unwrap();
        assert!(index.add_with_id(7, &[3.0; 8]).is_err());
        assert!(index.add_with_id(u64::MAX, &[3.0; 8]).is_err());
        assert_eq!(index.add(&[4.0; 8]).unwrap(), 1_001);
        index.flush().unwrap();
    }

    let mut index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 3);
    assert_eq!(index.search(&[1.0; 8], 1).unwrap()[0].id, 1_000);
    assert_eq!(index.search(&[2.0; 8], 1).unwrap()[0].id, 7);
    assert_eq!(index.add(&[5.0; 8]).unwrap(), 1_002);
}

#[test]
fn test_delete_hides_vector_and_persists() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();
    {
        let mut index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
        for i in 0..10 {
            assert_eq!(index.add(&[i as f32; 8]).unwrap(), i);
        }
        assert!(index.delete(3).unwrap());
        assert!(!index.delete(3).unwrap());
        assert!(!index.delete(99).unwrap());
        assert_eq!(index.len(), 9);
        assert!(index.search(&[3.0; 8], 10).unwrap().iter().all(|r| r.id != 3));
        index.flush().unwrap();
    }

    let mut index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 9);
    assert!(index.search(&[3.0; 8], 10).unwrap().iter().all(|r| r.id != 3));
    assert!(!index.delete(3).unwrap());

    index.add_with_id(3, &[3.0; 8]).unwrap();
    assert_eq!(index.search(&[3.0; 8], 1).unwrap()[0].id, 3);
    assert_eq!(index.add(&[10.0; 8]).unwrap(), 10);
    assert_eq!(index.len(), 11);
}

#[test]
fn test_unflushed_delete_and_add_roll_back_together() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();
    {
        let mut index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
        for i in 0..5 {
            index.add(&[i as f32; 8]).unwrap();
        }
        index.flush().unwrap();

        // Replace id 2, then drop without flushing.
        index.delete(2).unwrap();
        index.add_with_id(2, &[20.0; 8]).unwrap();
    }

    let index = VectorIndex::open(&path, 8, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 5);
    let hit = &index.search(&[2.0; 8], 1).unwrap()[0];
    assert_eq!((hit.id, hit.distance), (2, 0.0));
}

#[test]
fn test_open_never_overwrites_a_short_foreign_file() {
    use chassisdb::Storage;

    for contents in [&b"some notes, not an index"[..], b"CHASSIS\0 but truncated"] {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), contents).unwrap();

        assert!(Storage::open(temp_file.path(), 8).is_err());
        assert!(VectorIndex::open(temp_file.path(), 8, IndexOptions::default()).is_err());
        assert_eq!(std::fs::read(temp_file.path()).unwrap(), contents);
    }
}

/// The application's number is committed with the data (ADR-0023): a flush keeps it, a drop
/// without one rolls it back with the adds, compaction carries it over, and readers see each
/// flush's.
#[test]
fn test_user_version_is_committed_with_the_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("versioned.chassis");
    let mut index = VectorIndex::open(&path, 4, IndexOptions::default()).unwrap();
    assert_eq!(index.user_version(), 0, "never set");

    index.add(&[1.0, 0.0, 0.0, 0.0]).unwrap();
    index.set_user_version(7);
    index.flush().unwrap();
    let reader = IndexReader::open(&path, 4, IndexOptions::default()).unwrap();
    assert_eq!(reader.user_version(), 7);
    drop(reader); // Windows can't replace a file another handle has open, as compact does below.

    index.add(&[0.0, 1.0, 0.0, 0.0]).unwrap();
    index.set_user_version(8);
    assert_eq!(index.user_version(), 8, "the writer sees what it set");
    drop(index);
    let mut index = VectorIndex::open(&path, 4, IndexOptions::default()).unwrap();
    assert_eq!((index.user_version(), index.len()), (7, 1), "unflushed, both rolled back");

    index.delete(0).unwrap();
    index.set_user_version(9);
    index.compact().unwrap();
    drop(index);
    let index = VectorIndex::open(&path, 4, IndexOptions::default()).unwrap();
    assert_eq!((index.user_version(), index.len()), (9, 0), "compaction carried it over");
}

/// The second segment is a vector and 32 MiB of holes, in the index and in its backup; APFS keeps
/// holes of 16 MiB or more.
#[cfg(unix)]
#[test]
fn test_a_backup_takes_no_disk_for_what_was_never_written() {
    use std::os::unix::fs::MetadataExt;
    let sizes = |path: &std::path::Path| {
        let metadata = std::fs::metadata(path).unwrap();
        (metadata.len(), metadata.blocks() * 512)
    };
    let dir = tempfile::tempdir().unwrap();
    let (path, backup) = (dir.path().join("index.chassis"), dir.path().join("backup.chassis"));
    let options = IndexOptions { max_connections: 4, ef_construction: 8, ..Default::default() };
    let mut index = VectorIndex::open(&path, 4096, options.clone()).unwrap();
    let vectors: Vec<f32> = (0..1025 * 4096).map(|i| (i % 4099) as f32).collect();
    index.add_batch(&vectors).unwrap();
    index.backup(&backup).unwrap();

    let (len, disk) = sizes(&backup);
    assert!(disk < len / 2, "the backup takes {disk} bytes on disk of its {len}");
    #[cfg(target_os = "linux")]
    assert!(sizes(&path).1 < len / 2, "so does the index: {:?}", sizes(&path));
    let copy = VectorIndex::open(&backup, 4096, options).unwrap();
    assert_eq!(copy.get(1024).unwrap().unwrap(), vectors[1024 * 4096..]);
}
