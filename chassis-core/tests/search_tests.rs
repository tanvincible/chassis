//! Integration tests for hardened HNSW search implementation.
//!
//! Tests cover:
//! - NaN safety (no panics)
//! - ef < k handling
//! - Visited filter correctness
//! - Cyclic graph handling
//! - Result invariants
//!
//! Updated to use two-phase protocol: write_node_and_backlinks + publish_node

use chassis_core::{HnswGraph, HnswParams, Storage};
use tempfile::NamedTempFile;

fn create_test_graph(num_vectors: usize, dims: u32) -> (HnswGraph, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let mut storage = Storage::open(temp_file.path(), dims).unwrap();

    for i in 0..num_vectors {
        let mut vec = vec![0.0; dims as usize];
        vec[0] = (i as f32) / (num_vectors as f32);
        vec[1] = ((i % 10) as f32) / 10.0;
        storage.insert(&vec).unwrap();
    }

    let params = HnswParams::default();
    let graph = HnswGraph::open(storage, params).unwrap();

    (graph, temp_file)
}

fn build_sequential_graph(graph: &mut HnswGraph, n: usize) {
    for i in 0..n as u64 {
        let neighbors = if i > 0 {
            let num_neighbors = i.min(8);
            let start = i.saturating_sub(num_neighbors);
            let neighs: Vec<u64> = (start..i).collect();
            vec![neighs]
        } else {
            vec![vec![]]
        };

        graph.write_node_and_backlinks(i, 1, &neighbors).unwrap();
        graph.publish_node(i, 1).unwrap();
    }
}

#[test]
fn test_search_empty_graph() {
    let (graph, _temp) = create_test_graph(10, 128);

    // No nodes linked yet
    let query = vec![0.5; 128];
    let results = graph.search(&query, 10, 50).unwrap();

    assert!(results.is_empty(), "Empty graph should return no results");
}

#[test]
fn test_search_single_node() {
    let (mut graph, _temp) = create_test_graph(10, 128);

    // Link just one node
    graph.write_node_and_backlinks(0, 1, &[vec![]]).unwrap();
    graph.publish_node(0, 1).unwrap();

    let query = vec![0.0; 128];
    let results = graph.search(&query, 5, 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_search_returns_k_results() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 50);

    let query = vec![0.5; 128];

    for k in [1, 5, 10, 20] {
        let results = graph.search(&query, k, 50).unwrap();
        assert!(results.len() <= k, "Should return at most k results");
    }
}

#[test]
fn test_ef_less_than_k_handling() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 30);

    let query = vec![0.5; 128];

    // ef < k should be silently corrected
    let results = graph.search(&query, 10, 3).unwrap();

    // Should not fail and should return up to 10 results
    assert!(results.len() <= 10);
    assert!(!results.is_empty(), "Should find some results even with small ef");
}

#[test]
fn test_results_sorted_by_distance() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 50);

    let query = vec![0.25; 128];
    let results = graph.search(&query, 20, 50).unwrap();

    // Verify sorted by ascending distance
    for i in 1..results.len() {
        assert!(
            results[i - 1].distance <= results[i].distance,
            "Results must be sorted by distance"
        );
    }
}

#[test]
fn test_no_duplicate_results() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 50);

    let query = vec![0.5; 128];
    let results = graph.search(&query, 20, 50).unwrap();

    // Check for duplicates
    let mut seen = std::collections::HashSet::new();
    for result in results {
        assert!(seen.insert(result.id), "Result set contains duplicate node ID: {}", result.id);
    }
}

#[test]
fn test_cyclic_graph_terminates() {
    let (mut graph, _temp) = create_test_graph(20, 128);

    // Build a ring topology (cyclic)
    for i in 0..10u64 {
        let neighbors = if i > 0 { vec![vec![i - 1]] } else { vec![vec![]] };
        graph.write_node_and_backlinks(i, 1, &neighbors).unwrap();
        graph.publish_node(i, 1).unwrap();
    }

    let query = vec![0.5; 128];
    let results = graph.search(&query, 5, 20).unwrap();

    // Should terminate without infinite loop
    assert!(results.len() <= 5);
}

#[test]
fn test_visited_filter_prevents_revisits() {
    let (mut graph, _temp) = create_test_graph(50, 128);

    // Build a diamond graph: 0 -> 1,2 -> 3
    graph.write_node_and_backlinks(0, 1, &[vec![]]).unwrap();
    graph.publish_node(0, 1).unwrap();

    graph.write_node_and_backlinks(1, 1, &[vec![0]]).unwrap();
    graph.publish_node(1, 1).unwrap();

    graph.write_node_and_backlinks(2, 1, &[vec![0]]).unwrap();
    graph.publish_node(2, 1).unwrap();

    graph.write_node_and_backlinks(3, 1, &[vec![1, 2]]).unwrap();
    graph.publish_node(3, 1).unwrap();

    let query = vec![0.0; 128];
    let results = graph.search(&query, 10, 20).unwrap();

    // Should visit each node at most once
    let mut seen = std::collections::HashSet::new();
    for result in results {
        assert!(seen.insert(result.id), "Node {} visited multiple times", result.id);
    }
}

#[test]
fn test_search_quality_improves_with_ef() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 50);

    let query = vec![0.25; 128];

    // Search with different ef values
    let results_ef10 = graph.search(&query, 5, 10).unwrap();
    let results_ef50 = graph.search(&query, 5, 50).unwrap();
    let results_ef100 = graph.search(&query, 5, 100).unwrap();

    // Higher ef should find better or equal results
    assert_eq!(results_ef10.len(), results_ef50.len());
    assert_eq!(results_ef50.len(), results_ef100.len());

    // The best result should improve or stay the same
    if !results_ef10.is_empty() && !results_ef50.is_empty() {
        assert!(
            results_ef50[0].distance <= results_ef10[0].distance,
            "Higher ef should not produce worse results"
        );
    }
}

#[test]
fn test_search_with_zero_k() {
    let (mut graph, _temp) = create_test_graph(50, 128);
    build_sequential_graph(&mut graph, 20);

    let query = vec![0.5; 128];
    let results = graph.search(&query, 0, 10).unwrap();

    assert!(results.is_empty(), "k=0 should return no results");
}

#[test]
fn test_search_with_large_ef() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 50);

    let query = vec![0.5; 128];

    // ef larger than graph size
    let results = graph.search(&query, 10, 1000).unwrap();

    // Should not fail and return valid results
    assert!(results.len() <= 10);
    assert!(results.len() <= 50); // Can't return more than graph size
}

#[test]
fn test_distance_values_are_valid() {
    let (mut graph, _temp) = create_test_graph(100, 128);
    build_sequential_graph(&mut graph, 30);

    let query = vec![0.5; 128];
    let results = graph.search(&query, 10, 50).unwrap();

    for result in results {
        assert!(
            result.distance.is_finite() || result.distance.is_nan(),
            "Distance should be finite or NaN, not infinite"
        );
        assert!(
            result.distance >= 0.0 || result.distance.is_nan(),
            "Euclidean distance should be non-negative"
        );
    }
}

#[test]
fn test_multi_layer_search() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build multi-layer graph
    for i in 0..30u64 {
        let layers = if i % 3 == 0 { 2 } else { 1 };
        let neighbors = if i > 0 {
            let mut neighs = vec![vec![i - 1]];
            if layers > 1 {
                neighs.push(vec![i.saturating_sub(2)]);
            }
            neighs
        } else {
            vec![vec![]; layers]
        };

        graph.write_node_and_backlinks(i, layers, &neighbors).unwrap();
        graph.publish_node(i, layers).unwrap();
    }

    let query = vec![0.5; 128];
    let results = graph.search(&query, 10, 50).unwrap();

    assert!(!results.is_empty(), "Multi-layer search should find results");
    assert!(results.len() <= 10);
}

#[test]
fn test_identical_vectors() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut storage = Storage::open(temp_file.path(), 128).unwrap();

    // Insert identical vectors
    for _ in 0..20 {
        storage.insert(&vec![0.5; 128]).unwrap();
    }

    let params = HnswParams::default();
    let mut graph = HnswGraph::open(storage, params).unwrap();

    build_sequential_graph(&mut graph, 20);

    let query = vec![0.5; 128];
    let results = graph.search(&query, 5, 20).unwrap();

    // Should handle identical vectors gracefully
    assert!(!results.is_empty());

    // All distances should be very small (or exactly 0)
    for result in results {
        assert!(result.distance < 0.001 || result.distance == 0.0);
    }
}

#[test]
fn test_greedy_layer_descent() {
    let (mut graph, _temp) = create_test_graph(100, 128);

    // Build a 3-layer graph
    for i in 0..20u64 {
        let layers = if i == 0 {
            3
        } else if i < 5 {
            2
        } else {
            1
        };
        let neighbors = if i > 0 { vec![vec![i - 1]; layers] } else { vec![vec![]; layers] };

        graph.write_node_and_backlinks(i, layers, &neighbors).unwrap();
        graph.publish_node(i, layers).unwrap();
    }

    let query = vec![0.5; 128];
    let results = graph.search(&query, 5, 20).unwrap();

    // Should successfully navigate through layers
    assert!(!results.is_empty());
}

#[test]
fn test_search_correctness_on_small_graph() {
    let (mut graph, _temp) = create_test_graph(10, 2);

    build_sequential_graph(&mut graph, 5);

    let query = vec![0.15, 0.0]; // Closest to vector 1 and 2
    let results = graph.search(&query, 3, 10).unwrap();

    // Should find the closest vectors
    assert_eq!(results.len(), 3);

    // Verify the results make sense (distances increase)
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}
