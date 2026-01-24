//! Integration tests for bidirectional linking.
//!
//! # Model A: Sequential Construction Only
//!
//! All tests must respect the invariant: nodes can only link to already-existing nodes.
//! This means we must build graphs sequentially (node 0, then 1, then 2, etc.).

use chassis_core::{HnswGraph, HnswParams, Storage};
use tempfile::NamedTempFile;

fn create_test_graph(num_vectors: usize, dims: u32) -> (HnswGraph, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let mut storage = Storage::open(temp_file.path(), dims).unwrap();

    for i in 0..num_vectors {
        let mut vec = vec![0.0; dims as usize];
        vec[0] = (i as f32) / (num_vectors as f32);
        vec[1] = ((i % 10) as f32) / 10.0;
        for j in 2..dims.min(10) as usize {
            vec[j] = ((i + j) as f32) / (num_vectors as f32);
        }
        storage.insert(&vec).unwrap();
    }

    let params = HnswParams::default();
    let graph = HnswGraph::open(storage, params).unwrap();

    (graph, temp_file)
}

#[test]
fn test_forward_links_written_before_backward_links() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Model A: Node 0 cannot link to non-existent nodes
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    // Node 1 can now link to node 0
    graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();

    // Node 2 can link to nodes 0 and 1
    graph.link_node_bidirectional(2, 1, &[vec![0, 1]]).unwrap();

    // Verify forward links exist
    let node2 = graph.read_node_record(2).unwrap();
    assert_eq!(node2.get_neighbors(0), vec![0, 1]);
    assert_eq!(graph.node_count(), 3);
}

#[test]
fn test_empty_neighbor_list() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    let record = graph.read_node_record(0).unwrap();
    assert!(record.get_neighbors(0).is_empty());
}

#[test]
fn test_single_neighbor() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    // Sequential construction
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();
    graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();

    // Verify forward link
    let node1 = graph.read_node_record(1).unwrap();
    assert_eq!(node1.get_neighbors(0), vec![0]);

    // Verify backward link
    let node0 = graph.read_node_record(0).unwrap();
    assert!(node0.get_neighbors(0).contains(&1));
}

#[test]
fn test_maximum_neighbors_no_pruning() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;

    // Build sequentially
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..=m0.min(50) as u64 {
        // Each node links to all previous nodes up to m0
        let neighbors: Vec<u64> = (0..i).take(m0).collect();
        graph.link_node_bidirectional(i, 1, &[neighbors]).unwrap();
    }

    // Check a node has correct count
    let node_mid = graph.read_node_record(m0.min(25) as u64).unwrap();
    assert!(node_mid.get_neighbors(0).len() <= m0);
}

#[test]
fn test_exceeds_maximum_triggers_pruning() {
    let (mut graph, _temp) = create_test_graph(200, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;

    // Build hub node sequentially
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    // Many nodes link to node 0
    for i in 1..=(m0 + 10) as u64 {
        graph.link_node_bidirectional(i, 1, &[vec![0]]).unwrap();
    }

    // Node 0 should have pruned its list
    let record = graph.read_node_record(0).unwrap();
    assert!(record.get_neighbors(0).len() <= m0);
}

#[test]
fn test_diversity_respects_capacity() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;

    // Build hub
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..=m0.min(50) {
        graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
    }

    let record = graph.read_node_record(0).unwrap();
    assert!(record.get_neighbors(0).len() <= m0);
}

#[test]
fn test_starvation_fallback_maintains_minimum_degree() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;
    let min_neighbors = m0 / 2;

    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..=m0.min(50) {
        graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
    }

    let record = graph.read_node_record(0).unwrap();
    let neighbor_count = record.neighbor_count(0);

    assert!(neighbor_count >= min_neighbors);
}

#[test]
fn test_layer_independence_preserved() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build multi-layer graph sequentially
    graph.link_node_bidirectional(0, 3, &[vec![], vec![], vec![]]).unwrap();
    graph.link_node_bidirectional(1, 3, &[vec![0], vec![0], vec![]]).unwrap();
    graph.link_node_bidirectional(2, 3, &[vec![0, 1], vec![], vec![]]).unwrap();
    graph.link_node_bidirectional(3, 3, &[vec![0, 1], vec![0], vec![]]).unwrap();

    // Verify forward links
    let record = graph.read_node_record(3).unwrap();
    assert_eq!(record.get_neighbors(0), vec![0, 1]);
    assert_eq!(record.get_neighbors(1), vec![0]);
    assert_eq!(record.get_neighbors(2), vec![]);

    // Verify layer-specific backward links
    let node0 = graph.read_node_record(0).unwrap();
    assert!(node0.get_neighbors(0).len() > 0); // Has backward links from layer 0
    assert!(node0.get_neighbors(1).len() > 0); // Has backward links from layer 1
}

#[test]
fn test_idempotency_retry_safety() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build small graph
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();
    graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();
    graph.link_node_bidirectional(2, 1, &[vec![0]]).unwrap();

    // Simulate retry
    graph.add_backward_link_with_pruning(1, 0, 0).unwrap();
    graph.add_backward_link_with_pruning(2, 0, 0).unwrap();

    // Should have exactly one back-link
    for neighbor_id in [1, 2] {
        let record = graph.read_node_record(neighbor_id).unwrap();
        let back_links = record.get_neighbors(0);
        let count = back_links.iter().filter(|&&id| id == 0).count();
        assert_eq!(count, 1);
    }
}

#[test]
fn test_self_links_filtered() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();
    // Try to link node 1 to itself and to node 0
    graph.link_node_bidirectional(1, 1, &[vec![0, 1]]).unwrap();

    // Self-link should be filtered
    let record = graph.read_node_record(1).unwrap();
    assert!(!record.get_neighbors(0).contains(&1));
    assert_eq!(record.get_neighbors(0), vec![0]);
}

#[test]
fn test_invalid_neighbor_ids_handled_gracefully() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    // Try to link to nodes that don't exist yet (Model A filters them)
    graph.link_node_bidirectional(0, 1, &[vec![1, 999, 1000]]).unwrap();

    // All non-existent neighbors filtered
    let record = graph.read_node_record(0).unwrap();
    assert_eq!(record.get_neighbors(0), vec![]);

    // Now add node 1
    graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();

    // Node 0 should have backward link from node 1
    let node0 = graph.read_node_record(0).unwrap();
    assert!(node0.get_neighbors(0).contains(&1));
}

#[test]
fn test_sequential_linking_maintains_consistency() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build ring topology
    for i in 0..50u64 {
        let neighbors = if i > 0 { vec![vec![i - 1]] } else { vec![vec![]] };
        graph.link_node_bidirectional(i, 1, &neighbors).unwrap();
    }

    assert_eq!(graph.node_count(), 50);

    // Spot check
    let node0 = graph.read_node_record(0).unwrap();
    assert!(node0.get_neighbors(0).contains(&1));

    let node25 = graph.read_node_record(25).unwrap();
    let neighbors = node25.get_neighbors(0);
    assert!(neighbors.contains(&24));
    assert!(!neighbors.is_empty());
}

#[test]
fn test_high_degree_hub_respects_limits() {
    let (mut graph, _temp) = create_test_graph(200, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;

    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..100u64 {
        graph.link_node_bidirectional(i, 1, &[vec![0]]).unwrap();
    }

    let hub = graph.read_node_record(0).unwrap();
    assert!(hub.get_neighbors(0).len() <= m0);
}

#[test]
fn test_identical_vectors_diversity() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut storage = Storage::open(temp_file.path(), 128).unwrap();

    // Insert identical vectors
    for _ in 0..50 {
        storage.insert(&vec![0.5; 128]).unwrap();
    }

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).unwrap();

    let m0 = graph.record_params().m0 as usize;

    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..=m0.min(30) {
        graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
    }

    let record = graph.read_node_record(0).unwrap();
    assert!(record.get_neighbors(0).len() <= m0);
}

#[test]
fn test_graph_header_updated_correctly() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    assert!(graph.entry_point.is_none());
    assert_eq!(graph.max_layer, 0);

    // Link 3-layer node
    graph.link_node_bidirectional(0, 3, &[vec![], vec![], vec![]]).unwrap();

    assert_eq!(graph.entry_point, Some(0));
    assert_eq!(graph.max_layer, 2);

    // Link higher-layer node
    graph.link_node_bidirectional(1, 5, &[vec![0], vec![0], vec![0], vec![0], vec![0]]).unwrap();

    assert_eq!(graph.entry_point, Some(1));
    assert_eq!(graph.max_layer, 4);
}

#[test]
fn test_commit_persists_links() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_owned();

    // Create and link
    {
        let mut storage = Storage::open(&path, 128).unwrap();
        for i in 0..20 {
            let mut vec = vec![0.0; 128];
            vec[0] = i as f32 / 20.0;
            storage.insert(&vec).unwrap();
        }

        let params = HnswParams::default();
        let mut graph = HnswGraph::open(storage, params).unwrap();

        // Sequential construction
        for i in 0..10u64 {
            let neighbors = if i > 0 { vec![vec![(i - 1)]] } else { vec![vec![]] };
            graph.link_node_bidirectional(i, 1, &neighbors).unwrap();
        }

        graph.commit().unwrap();
    }

    // Reopen and verify
    {
        let storage = Storage::open(&path, 128).unwrap();
        let params = HnswParams::default();
        let graph = HnswGraph::open(storage, params).unwrap();

        assert_eq!(graph.node_count(), 10);

        // Verify links
        let node1 = graph.read_node_record(1).unwrap();
        let neighbors = node1.get_neighbors(0);
        assert!(neighbors.contains(&0));
    }
}

#[test]
fn test_new_node_gets_backward_links() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build chain
    for i in 0..20u64 {
        let neighbors = if i > 0 { vec![vec![i - 1]] } else { vec![vec![]] };
        graph.link_node_bidirectional(i, 1, &neighbors).unwrap();
    }

    // Add new node
    graph.link_node_bidirectional(20, 1, &[vec![0, 1, 2]]).unwrap();

    // Verify it got backward links
    let new_record = graph.read_node_record(20).unwrap();
    assert_eq!(new_record.get_neighbors(0), vec![0, 1, 2]);

    // And nodes 0, 1, 2 got links back
    for i in [0, 1, 2] {
        let node = graph.read_node_record(i).unwrap();
        assert!(node.get_neighbors(0).len() > 0);
    }
}

#[test]
fn test_layer_count_mismatch_error() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    let neighbors = vec![vec![], vec![]];
    let result = graph.link_node_bidirectional(0, 3, &neighbors);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Layer count mismatch"));
}

#[test]
fn test_zero_dimensional_vectors_not_supported() {
    let temp_file = NamedTempFile::new().unwrap();

    let result = Storage::open(temp_file.path(), 0);
    assert!(result.is_err());
}

#[test]
fn test_neighbor_list_capped_at_capacity() {
    let (mut graph, _temp) = create_test_graph(500, 128);

    let params = graph.record_params();
    let m0 = params.m0 as usize;

    // Build large graph
    graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

    for i in 1..=m0.min(200) as u64 {
        graph.link_node_bidirectional(i, 1, &[vec![0]]).unwrap();
    }

    let record = graph.read_node_record(0).unwrap();
    assert!(record.get_neighbors(0).len() <= m0);
}

#[test]
fn test_node_id_invariant_returns_error() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    let result = graph.link_node_bidirectional(5, 1, &[vec![]]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Node ID invariant violated"));
}
