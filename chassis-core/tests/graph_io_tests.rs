use chassis_core::Storage;
use chassis_core::{GraphHeader, HnswGraph, HnswParams, NodeRecord, NodeRecordParams};
use tempfile::NamedTempFile;

// Header persistence

#[test]
fn test_graph_header_roundtrip() {
    let params = NodeRecordParams::new(16, 32, 8);
    let mut header = GraphHeader::new(params);
    header.entry_point = 42;
    header.max_layer = 5;
    header.node_count = 100;

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 64);

    let restored = GraphHeader::from_bytes(&bytes).unwrap();
    assert_eq!(restored.entry_point, 42);
    assert_eq!(restored.max_layer, 5);
    assert_eq!(restored.node_count, 100);
    assert_eq!(restored.m, 16);
    assert_eq!(restored.m0, 32);
    assert_eq!(restored.max_layers, 8);
}

#[test]
fn test_graph_header_survives_reopen() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Create graph and insert some nodes
    {
        let storage = Storage::open(path, 128).unwrap();
        let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

        graph.insert(0, 0).unwrap();
        graph.insert(1, 1).unwrap();
        graph.insert(2, 2).unwrap();

        graph.commit().unwrap();

        // Verify state before closing
        assert_eq!(graph.entry_point, Some(2));
        assert_eq!(graph.max_layer, 2);
    }

    // Reopen and verify header was persisted
    {
        let storage = Storage::open(path, 128).unwrap();
        let graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

        assert_eq!(graph.entry_point, Some(2));
        assert_eq!(graph.max_layer, 2);
    }
}

#[test]
fn test_graph_header_magic_validation() {
    let params = NodeRecordParams::new(16, 32, 8);
    let header = GraphHeader::new(params);

    assert!(header.is_valid());

    let mut bytes = header.to_bytes();
    bytes[0] = b'X'; // Corrupt magic

    let corrupted = GraphHeader::from_bytes(&bytes).unwrap();
    assert!(!corrupted.is_valid());
}

// Node record I/O

#[test]
fn test_write_and_read_node_record() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    // Create a node record
    let params = NodeRecordParams::default();
    let mut record = NodeRecord::new(42, 3, params);
    record.set_neighbors(0, &[1, 2, 3, 4, 5]);
    record.set_neighbors(1, &[10, 20, 30]);
    record.set_neighbors(2, &[100, 200]);

    // Write it
    graph.write_node_record(&record).unwrap();

    // Read it back
    let read_record = graph.read_node_record(42).unwrap();

    assert_eq!(read_record.header.node_id, 42);
    assert_eq!(read_record.header.layer_count, 3);
    assert_eq!(read_record.get_neighbors(0), vec![1, 2, 3, 4, 5]);
    assert_eq!(read_record.get_neighbors(1), vec![10, 20, 30]);
    assert_eq!(read_record.get_neighbors(2), vec![100, 200]);
}

#[test]
fn test_node_record_addressing_formula() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    // Write a node first so we can test addressing
    let params = NodeRecordParams::default();
    let record = NodeRecord::new(5, 2, params);
    graph.write_node_record(&record).unwrap();

    // Now we can successfully get bytes for node 5
    let bytes_5 = graph.get_node_bytes(5);
    assert!(bytes_5.is_ok());

    // But node 100 (not written yet) should work too due to O(1) addressing
    // It just won't have valid data
    let bytes_100 = graph.get_node_bytes(100);
    // This might fail due to bounds checking, which is expected
    assert!(bytes_100.is_err());
}

#[test]
fn test_multiple_node_records() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    let params = NodeRecordParams::default();

    // Write multiple node records
    for node_id in 0..10 {
        let mut record = NodeRecord::new(node_id, 2, params);
        record.set_neighbors(0, &[node_id + 100, node_id + 200]);
        record.set_neighbors(1, &[node_id + 1000]);

        graph.write_node_record(&record).unwrap();
    }

    // Read them all back
    for node_id in 0..10 {
        let record = graph.read_node_record(node_id).unwrap();
        assert_eq!(record.header.node_id, node_id);
        assert_eq!(record.get_neighbors(0), vec![node_id + 100, node_id + 200]);
        assert_eq!(record.get_neighbors(1), vec![node_id + 1000]);
    }
}

#[test]
fn test_node_record_update_in_place() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    let params = NodeRecordParams::default();

    // Write initial record
    let mut record = NodeRecord::new(5, 2, params);
    record.set_neighbors(0, &[1, 2, 3]);
    graph.write_node_record(&record).unwrap();

    // Update in place
    let mut updated_record = NodeRecord::new(5, 2, params);
    updated_record.set_neighbors(0, &[10, 20, 30, 40]);
    updated_record.set_neighbors(1, &[100, 200]);
    graph.write_node_record(&updated_record).unwrap();

    // Read back and verify update
    let read_record = graph.read_node_record(5).unwrap();
    assert_eq!(read_record.get_neighbors(0), vec![10, 20, 30, 40]);
    assert_eq!(read_record.get_neighbors(1), vec![100, 200]);
}

// Neighbor access

#[test]
fn test_neighbors_iter_from_mmap() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    let params = NodeRecordParams::default();
    let mut record = NodeRecord::new(7, 3, params);
    record.set_neighbors(0, &[1, 2, 3, 4, 5, 6, 7, 8]);
    record.set_neighbors(1, &[10, 20, 30, 40]);
    record.set_neighbors(2, &[100, 200]);

    graph.write_node_record(&record).unwrap();

    // Test layer 0
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(7, 0).unwrap().collect();
    assert_eq!(neighbors, vec![1, 2, 3, 4, 5, 6, 7, 8]);

    // Test layer 1
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(7, 1).unwrap().collect();
    assert_eq!(neighbors, vec![10, 20, 30, 40]);

    // Test layer 2
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(7, 2).unwrap().collect();
    assert_eq!(neighbors, vec![100, 200]);
}

#[test]
fn test_neighbors_iter_empty_layer() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    let params = NodeRecordParams::default();
    let record = NodeRecord::new(3, 2, params); // Empty neighbors

    graph.write_node_record(&record).unwrap();

    // Empty layer should return no neighbors
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(3, 0).unwrap().collect();
    assert!(neighbors.is_empty());

    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(3, 1).unwrap().collect();
    assert!(neighbors.is_empty());
}

#[test]
fn test_neighbors_iter_out_of_bounds_layer() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

    let params = NodeRecordParams::default();
    let mut record = NodeRecord::new(4, 2, params); // Only 2 layers (0, 1)
    record.set_neighbors(0, &[1, 2, 3]);

    graph.write_node_record(&record).unwrap();

    // Layer 2 is out of bounds (node has layers 0 and 1)
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(4, 2).unwrap().collect();
    assert!(neighbors.is_empty());

    // Layer 5 is also out of bounds
    let neighbors: Vec<u64> = graph.neighbors_iter_from_mmap(4, 5).unwrap().collect();
    assert!(neighbors.is_empty());
}

// Integration Tests

#[test]
fn test_graph_with_multiple_layers() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Insert vectors first
    {
        let mut storage = Storage::open(path, 128).unwrap();
        for i in 0..10 {
            let vector = vec![i as f32; 128];
            storage.insert(&vector).unwrap();
        }
        storage.commit().unwrap();
    }

    // Now open graph and insert nodes
    {
        let storage = Storage::open(path, 128).unwrap();
        let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

        // Insert nodes with different layer counts
        graph.insert(0, 0).unwrap(); // Layer 0 only
        graph.insert(1, 1).unwrap(); // Layers 0-1
        graph.insert(2, 2).unwrap(); // Layers 0-2
        graph.insert(3, 0).unwrap(); // Layer 0 only

        graph.commit().unwrap();

        // Verify entry point and max layer
        assert_eq!(graph.entry_point, Some(2));
        assert_eq!(graph.max_layer, 2);
    }
}

#[test]
fn test_graph_persistence_with_vectors() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Create vectors and graph
    {
        let mut storage = Storage::open(path, 128).unwrap();
        for i in 0..5 {
            let vector = vec![i as f32; 128];
            storage.insert(&vector).unwrap();
        }
        storage.commit().unwrap();
        drop(storage); // Explicitly drop to release lock

        let storage = Storage::open(path, 128).unwrap();
        let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();

        for i in 0..5 {
            graph.insert(i, i as usize % 3).unwrap();
        }

        graph.commit().unwrap();
    }

    // Reopen and verify both vectors and graph
    {
        let storage = Storage::open(path, 128).unwrap();
        assert_eq!(storage.count(), 5);

        for i in 0..5 {
            let vector = storage.get_vector(i).unwrap();
            assert_eq!(vector[0], i as f32);
        }

        let graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
        assert_eq!(graph.entry_point, Some(2)); // Node 2 has layer 2
        assert_eq!(graph.max_layer, 2);
    }
}

#[test]
fn test_node_record_with_partial_neighbors() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Create graph with custom params
    let custom_params = HnswParams {
        max_connections: 4,
        ef_construction: 200,
        ef_search: 50,
        ml: 1.0 / (4.0_f32).ln(),
        max_layers: 3,
    };

    let storage = Storage::open(path, 128).unwrap();
    let mut graph = HnswGraph::open(storage, custom_params).unwrap();

    let params = NodeRecordParams::new(4, 8, 3); // Matches custom_params
    let mut record = NodeRecord::new(10, 3, params);

    // Fill layer 0 partially (max is 8, we add 5)
    record.set_neighbors(0, &[1, 2, 3, 4, 5]);

    // Fill layer 1 fully (max is 4)
    record.set_neighbors(1, &[10, 11, 12, 13]);

    // Leave layer 2 empty

    graph.write_node_record(&record).unwrap();

    // Read back and verify
    let read_record = graph.read_node_record(10).unwrap();
    assert_eq!(read_record.get_neighbors(0), vec![1, 2, 3, 4, 5]);
    assert_eq!(read_record.get_neighbors(1), vec![10, 11, 12, 13]);
    assert!(read_record.get_neighbors(2).is_empty());
}
