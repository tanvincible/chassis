//! Integration tests for VectorIndex facade
//!
//! Tests cover the complete lifecycle: open -> add -> search -> flush

use chassis_core::{IndexOptions, VectorIndex};
use tempfile::NamedTempFile;

#[test]
fn test_index_lifecycle_basic() {
    let temp_file = NamedTempFile::new().unwrap();

    // Create index
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Verify initial state
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.dimensions(), 128);

    // Add vectors
    let id1 = index.add(&vec![0.1; 128]).unwrap();
    let id2 = index.add(&vec![0.2; 128]).unwrap();
    let id3 = index.add(&vec![0.3; 128]).unwrap();

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);
    assert_eq!(index.len(), 3);

    // Search
    let query = vec![0.15; 128];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].distance <= results[1].distance);

    // Flush
    index.flush().unwrap();
}

#[test]
fn test_index_persistence() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Create and populate
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

    // Reopen and verify
    {
        let index = VectorIndex::open(&path, 256, IndexOptions::default()).unwrap();

        assert_eq!(index.len(), 20);
        assert_eq!(index.dimensions(), 256);

        // Search should work
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
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dimension mismatch"));

    // Wrong dimension on search
    let result = index.search(&vec![0.1; 64], 5);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
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
    assert!(results[0].distance < 0.001); // Should be very close
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

    // The two nearest should be id 4 and id 5
    let top_two_ids: Vec<u64> = results.iter().take(2).map(|r| r.id).collect();
    assert!(top_two_ids.contains(&4));
    assert!(top_two_ids.contains(&5));
}

#[test]
fn test_large_batch_insert() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Insert 100 vectors
    for i in 0..100 {
        let mut vec = vec![0.0; 128];
        vec[0] = (i as f32) / 100.0;
        vec[1] = ((i % 10) as f32) / 10.0;

        let id = index.add(&vec).unwrap();
        assert_eq!(id, i as u64);
    }

    assert_eq!(index.len(), 100);

    // Search should work
    let query = vec![0.5; 128];
    let results = index.search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Verify sorted by distance
    for i in 1..results.len() {
        assert!(results[i - 1].distance <= results[i].distance);
    }
}

#[test]
fn test_custom_options() {
    let temp_file = NamedTempFile::new().unwrap();

    let options = IndexOptions { max_connections: 8, ef_construction: 100, ef_search: 25 };

    let mut index = VectorIndex::open(temp_file.path(), 128, options).unwrap();

    // Should work with custom options
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

    // Add vectors and flush
    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        for i in 0..10 {
            index.add(&vec![i as f32; 128]).unwrap();
        }

        index.flush().unwrap();
    }

    // Reopen - data should persist
    {
        let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
        assert_eq!(index.len(), 10);
    }
}

#[test]
fn test_no_flush_loses_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Add vectors without flush
    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        for i in 0..5 {
            index.add(&vec![i as f32; 128]).unwrap();
        }

        // Don't flush - drop index
    }

    // Reopen - some data might be lost
    {
        let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

        // Depending on OS buffering, we might have 0-5 vectors
        // The important thing is no corruption
        assert!(index.len() <= 5);
    }
}

#[test]
fn test_ghost_node_recovery() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Manually create a ghost node scenario
    {
        use chassis_core::{HnswGraph, HnswParams, Storage};

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

    // Create with 128 dimensions
    {
        let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
        index.add(&vec![1.0; 128]).unwrap();
        index.flush().unwrap();
    }

    // Try to open with different dimensions
    {
        let result = VectorIndex::open(&path, 256, IndexOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }
}

#[test]
fn test_search_returns_k_or_fewer() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Add 5 vectors
    for i in 0..5 {
        index.add(&vec![i as f32; 128]).unwrap();
    }

    // Search for k=10, should return only 5
    let results = index.search(&vec![0.0; 128], 10).unwrap();
    assert_eq!(results.len(), 5);

    // Search for k=3, should return 3
    let results = index.search(&vec![0.0; 128], 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_identical_vectors() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Add identical vectors
    for _ in 0..10 {
        index.add(&vec![0.5; 128]).unwrap();
    }

    let results = index.search(&vec![0.5; 128], 5).unwrap();

    // Should find 5 results
    assert_eq!(results.len(), 5);

    // All should have distance ~0
    for result in results {
        assert!(result.distance < 0.001);
    }
}

#[test]
fn test_search_with_k_zero() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    index.add(&vec![1.0; 128]).unwrap();

    let results = index.search(&vec![1.0; 128], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_concurrent_opens_fail() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Open first index
    let _index1 = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

    // Try to open second index on same file
    let result = VectorIndex::open(&path, 128, IndexOptions::default());

    // Should fail due to exclusive lock
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already open"));
}

#[test]
fn test_stress_sequential_adds() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

    // Add 1000 vectors sequentially
    for i in 0..1000 {
        let mut vec = vec![0.0; 128];
        vec[0] = (i as f32) / 1000.0;
        vec[1] = ((i % 100) as f32) / 100.0;
        vec[2] = ((i % 10) as f32) / 10.0;

        let id = index.add(&vec).unwrap();
        assert_eq!(id, i as u64);
    }

    assert_eq!(index.len(), 1000);

    // Verify search works on large index
    let query = vec![0.5; 128];
    let results = index.search(&query, 20).unwrap();

    assert_eq!(results.len(), 20);

    // Verify results are sorted
    for i in 1..results.len() {
        assert!(results[i - 1].distance <= results[i].distance);
    }
}
