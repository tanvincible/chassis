//!  HNSW search implementation with zero-allocation optimizations.
//!
//! # Performance Optimizations
//!
//!  This module uses the optimized code paths from `HnswGraph`:
//!  - `neighbors_iter_from_mmap()` - Zero-allocation neighbor iteration (~100ns vs ~400ns)
//! - `compute_distance_zero_copy()` - Zero-allocation vector access

use crate::hnsw::graph::HnswGraph;
use crate::hnsw::node::NodeId;
use anyhow::Result;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Search result with distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: NodeId,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl HnswGraph {
    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (must match index dimensions)
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search quality parameter (higher = better quality, slower)

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.unwrap();
        let mut current_layer = self.max_layer;

        // Greedy search from top layer to layer 1
        let mut current = entry;
        while current_layer > 0 {
            current = self.search_layer_optimized(query, current, 1, current_layer)?[0].id;
            current_layer -= 1;
        }

        // Search base layer with ef candidates
        let mut candidates = self.search_layer_optimized(query, current, ef, 0)?;

        // Return top k
        candidates.truncate(k);
        Ok(candidates)
    }

    /// Search within a single layer using zero-allocation optimizations.
    ///
    /// # Optimizations Applied
    ///
    /// 1. **Zero-allocation neighbor iteration**:  Uses `neighbors_iter_from_mmap()`
    ///    instead of `get_neighbors()` which allocates a `Vec<NodeId>`.
    ///    Benchmark:  ~100ns vs ~400ns per node.
    ///
    /// 2. **Zero-copy distance computation**: Uses `compute_distance_zero_copy()`
    ///    instead of `get_vector()` which allocates a `Vec<f32>`.
    ///
    ///
    /// No `Node::from_record()` calls
    /// No `Vec<NodeId>` allocation for neighbors  
    /// No `Vec<f32>` allocation for vectors

    fn search_layer_optimized(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Zero-copy distance computation
        let entry_dist = self.compute_distance_zero_copy(query, entry)?;
        candidates.push(Reverse(SearchResult { id: entry, distance: entry_dist }));
        results.push(SearchResult { id: entry, distance: entry_dist });
        visited.insert(entry);

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef && current.distance > results.peek().unwrap().distance {
                break;
            }

            // Zero-allocation neighbor iteration
            // Uses mmap-based iteration (~100ns) instead of Vec allocation (~400ns)
            for neighbor_id in self.neighbors_iter_from_mmap(current.id, layer)? {
                if visited.insert(neighbor_id) {
                    // Zero-copy distance computation
                    // Reads directly from mmap instead of allocating Vec<f32>
                    let dist = self.compute_distance_zero_copy(query, neighbor_id)?;

                    if results.len() < ef || dist < results.peek().unwrap().distance {
                        candidates.push(Reverse(SearchResult { id: neighbor_id, distance: dist }));
                        results.push(SearchResult { id: neighbor_id, distance: dist });

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<_> = results.into_iter().collect();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(sorted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_ordering() {
        let r1 = SearchResult { id: 1, distance: 0.5 };
        let r2 = SearchResult { id: 2, distance: 1.0 };
        let r3 = SearchResult { id: 3, distance: 0.5 };

        assert!(r1 < r2);
        assert!(r1 == r3); // Same distance
        assert!(r2 > r1);
    }

    #[test]
    fn test_search_result_in_heap() {
        let mut heap = BinaryHeap::new();

        heap.push(SearchResult { id: 1, distance: 0.5 });
        heap.push(SearchResult { id: 2, distance: 1.0 });
        heap.push(SearchResult { id: 3, distance: 0.1 });

        // Max-heap:  largest distance first
        assert_eq!(heap.pop().unwrap().id, 2); // distance 1.0
        assert_eq!(heap.pop().unwrap().id, 1); // distance 0.5
        assert_eq!(heap.pop().unwrap().id, 3); // distance 0.1
    }
}
