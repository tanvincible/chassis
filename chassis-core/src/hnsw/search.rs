use crate::distance::euclidean_distance;
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
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }
        
        let entry = self.entry_point.unwrap();
        let mut current_layer = self.max_layer;
        
        // Greedy search from top layer to layer 1
        let mut current = entry;
        while current_layer > 0 {
            current = self.search_layer(query, current, 1, current_layer)?[0].id;
            current_layer -= 1;
        }
        
        // Search base layer with ef candidates
        let mut candidates = self.search_layer(query, current, ef, 0)?;
        
        // Return top k
        candidates.truncate(k);
        Ok(candidates)
    }
    
    /// Search within a single layer
    fn search_layer(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        
        let entry_dist = self.compute_distance(query, entry)?;
        candidates.push(Reverse(SearchResult {
            id: entry,
            distance: entry_dist,
        }));
        results.push(SearchResult {
            id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);
        
        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef && current.distance > results.peek().unwrap().distance {
                break;
            }
            
            // Get neighbors at this layer
            let neighbors = self.get_neighbors(current.id, layer)?;
            
            for neighbor_id in neighbors {
                if visited.insert(neighbor_id) {
                    let dist = self.compute_distance(query, neighbor_id)?;
                    
                    if results.len() < ef || dist < results.peek().unwrap().distance {
                        candidates.push(Reverse(SearchResult {
                            id: neighbor_id,
                            distance: dist,
                        }));
                        results.push(SearchResult {
                            id: neighbor_id,
                            distance: dist,
                        });
                        
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
    
    /// Compute distance between query and stored vector
    fn compute_distance(&self, query: &[f32], node_id: NodeId) -> Result<f32> {
        let vector = self.storage.get_vector(node_id)?;
        Ok(euclidean_distance(query, &vector))
    }
    
    /// Get neighbors of a node at specific layer
    fn get_neighbors(&self, node_id: NodeId, layer: usize) -> Result<Vec<NodeId>> {
        let offset = self.get_node_offset(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", node_id))?;
        
        let node = self.read_node(offset)?;
        
        if layer < node.layers.len() {
            Ok(node.layers[layer].clone())
        } else {
            Ok(Vec::new())
        }
    }
}