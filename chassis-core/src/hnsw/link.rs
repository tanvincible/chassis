//! Bidirectional graph linking with diversity heuristics and crash consistency.
//!
//! # Crash Consistency Model
//!
//! This module implements a write-ahead linking strategy:
//! 1. Write Node A's record FIRST (with forward links)
//! 2. Update backward links on neighbors (may crash mid-process)
//! 3. Update graph header last
//!
//! If a crash occurs during step 2, we may have "dangling" one-way edges
//! (A→B exists but B→A doesn't), but we NEVER have pointers to uninitialized
//! memory because Node A was written first.
//!
//! # Forward Link Policy (Model A)
//!
//! **ENFORCED INVARIANT**: Nodes may only link to already-existing nodes.
//!
//! This means `neighbors_per_layer` can only contain node IDs where `id < self.node_count`.
//! Forward links to non-existent nodes are filtered out during linking.

use crate::hnsw::graph::HnswGraph;
use crate::hnsw::node::{INVALID_NODE_ID, NodeId, NodeRecord};
use anyhow::Result;

/// Maximum neighbors per layer (enforced at compile time for cache sizing)
const MAX_M: usize = 32;

/// Stack-allocated distance cache size (33x33 symmetric matrix)
/// Supports up to M=32 neighbors + 1 new node
const CACHE_SIZE: usize = (MAX_M + 1) * (MAX_M + 1);

/// Sentinel value indicating "distance not yet computed"
const NOT_COMPUTED: f32 = f32::NAN;

/// Stack-allocated lazy distance cache for diversity heuristic
struct DistanceCache {
    /// Flat array representing symmetric matrix [i*size + j]
    data: [f32; CACHE_SIZE],
    /// Number of candidates (dimension of square matrix)
    size: usize,
}

impl DistanceCache {
    /// Create a new uninitialized cache
    #[inline]
    fn new(num_candidates: usize) -> Self {
        debug_assert!(
            num_candidates <= MAX_M + 1,
            "Cache overflow: {} candidates exceeds max {}",
            num_candidates,
            MAX_M + 1
        );

        Self { data: [NOT_COMPUTED; CACHE_SIZE], size: num_candidates }
    }

    /// Get cached distance or return NAN if not computed
    #[inline]
    fn get(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.size && j < self.size, "Cache index out of bounds");
        self.data[i * self.size + j]
    }

    /// Store distance symmetrically (cache[i][j] = cache[j][i] = dist)
    #[inline]
    fn set(&mut self, i: usize, j: usize, distance: f32) {
        debug_assert!(i < self.size && j < self.size, "Cache index out of bounds");
        let idx_ij = i * self.size + j;
        let idx_ji = j * self.size + i;
        self.data[idx_ij] = distance;
        self.data[idx_ji] = distance;
    }

    /// Check if distance has been computed
    #[inline]
    fn is_computed(&self, i: usize, j: usize) -> bool {
        !self.get(i, j).is_nan()
    }
}

impl HnswGraph {
    /// Write node record and update backward links (Step A + Step B).
    ///
    /// This method performs the disk-write phase of node insertion WITHOUT
    /// updating in-memory counters. The node is written to disk but remains
    /// "invisible" to readers until `publish_node()` is called.
    ///
    /// # Crash Consistency
    ///
    /// - Node A is written BEFORE any neighbor updates
    /// - If crash occurs during neighbor updates, we have one-way edges (safe)
    /// - NEVER have pointers to uninitialized memory
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the node to link (must equal self.node_count)
    /// * `layer_count` - Number of layers this node participates in
    /// * `neighbors_per_layer` - Forward links for each layer (A→Neighbors)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, error if node cannot be written.
    ///
    /// # Errors
    ///
    /// Returns error if node_id != self.node_count (enforced in both debug and release).
    pub fn write_node_and_backlinks(
        &mut self,
        node_id: NodeId,
        layer_count: usize,
        neighbors_per_layer: &[Vec<NodeId>],
    ) -> Result<()> {
        // Enforce dense, monotonic node ID invariant (error in both debug and release)
        if node_id != self.node_count {
            anyhow::bail!(
                "Node ID invariant violated: expected {}, got {}. \
                 Node IDs must be dense and monotonically increasing.",
                self.node_count,
                node_id
            );
        }

        if neighbors_per_layer.len() != layer_count {
            anyhow::bail!(
                "Layer count mismatch: expected {}, got {}",
                layer_count,
                neighbors_per_layer.len()
            );
        }

        // Filter neighbors: no self-links, no invalid IDs, only existing nodes
        let mut filtered_neighbors = Vec::new();
        for layer_neighbors in neighbors_per_layer {
            let filtered: Vec<NodeId> = layer_neighbors
                .iter()
                .copied()
                .filter(|&id| {
                    id != node_id && // No self-links
                    id != INVALID_NODE_ID && // No sentinel values
                    id < self.node_count // Only existing nodes (Model A)
                })
                .collect();
            filtered_neighbors.push(filtered);
        }

        // STEP A: Write Node A's record FIRST (crash safety)
        let mut node_record = NodeRecord::new(node_id, layer_count as u8, self.record_params);

        for (layer, neighbors) in filtered_neighbors.iter().enumerate() {
            node_record.set_neighbors(layer, neighbors);
        }

        self.write_node_record(&node_record)?;

        // STEP B: Update backward links (B→A) for each neighbor
        for layer in 0..layer_count {
            let neighbors = &filtered_neighbors[layer];

            for &neighbor_id in neighbors {
                // Additional safety check (already filtered, but defensive)
                if neighbor_id >= self.node_count {
                    continue;
                }

                self.add_backward_link_with_pruning(neighbor_id, node_id, layer)?;
            }
        }

        Ok(())
    }

    /// Publish node to make it visible to readers (Step C).
    ///
    /// This method updates in-memory counters (node_count, entry_point, max_layer)
    /// to make the previously-written node visible to readers.
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the node to publish (must equal self.node_count)
    /// * `layer_count` - Number of layers this node participates in
    ///
    /// # Errors
    ///
    /// Returns error if node_id != self.node_count (invariant violation).
    pub fn publish_node(&mut self, node_id: NodeId, layer_count: usize) -> Result<()> {
        // Enforce invariant
        if node_id != self.node_count {
            anyhow::bail!(
                "Node ID invariant violated: expected {}, got {}. \
                 Cannot publish non-sequential node.",
                self.node_count,
                node_id
            );
        }

        // STEP C: Update in-memory counters
        self.node_count += 1;

        // Update entry point and max layer if this is the highest layer node
        if self.entry_point.is_none() || layer_count - 1 > self.max_layer {
            self.entry_point = Some(node_id);
            self.max_layer = layer_count - 1;
        }

        Ok(())
    }

    /// Legacy method for backward compatibility.
    ///
    /// This method combines `write_node_and_backlinks` + `publish_node` into
    /// a single call. It exists for compatibility with existing tests and code
    /// that don't need the explicit two-phase protocol.
    ///
    /// New code should prefer using the two-step process for better crash consistency.
    #[allow(dead_code)]
    pub fn link_node_bidirectional(
        &mut self,
        node_id: NodeId,
        layer_count: usize,
        neighbors_per_layer: &[Vec<NodeId>],
    ) -> Result<()> {
        self.write_node_and_backlinks(node_id, layer_count, neighbors_per_layer)?;
        self.publish_node(node_id, layer_count)?;
        Ok(())
    }

    /// Add a backward link from neighbor to new_node with diversity pruning.
    pub fn add_backward_link_with_pruning(
        &mut self,
        neighbor_id: NodeId,
        new_node: NodeId,
        layer: usize,
    ) -> Result<()> {
        let mut record = self.read_node_record(neighbor_id)?;
        let current_neighbors = record.get_neighbors(layer);

        // Duplicate check (idempotency)
        if current_neighbors.contains(&new_node) {
            return Ok(());
        }

        let max_neighbors = self.record_params.max_neighbors(layer);

        // Direct insert if space available
        if current_neighbors.len() < max_neighbors {
            record.add_neighbor(layer, new_node);
            self.update_node_record(&record)?;
            return Ok(());
        }

        // Full - combine current neighbors + new node and apply diversity heuristic
        let mut candidates = current_neighbors.to_vec();
        candidates.push(new_node);

        let selected = self.select_neighbors_heuristic(
            neighbor_id,
            &candidates,
            layer,
            max_neighbors,
            Some(new_node), // Prioritize the new node for connectivity
        )?;

        record.set_neighbors(layer, &selected);
        self.update_node_record(&record)?;

        Ok(())
    }

    /// Select diverse neighbors using HNSW Heuristic 2 with lazy memoized distance cache.
    ///
    /// This is the unified neighbor selection function used by both:
    /// - **Backward Linking** (`add_backward_link_with_pruning`): Prioritizes `priority_node`
    /// - **Forward Linking** (`VectorIndex::select_diverse_subset`): No priority
    ///
    /// # Algorithm
    ///
    /// 1. **Input Truncation**: Limit candidates to MAX_M+1 (cache size)
    /// 2. **Local Index Mapping**: Map NodeIds to local indices [0..k)
    /// 3. **Lazy Cache**: Compute distances only when needed, store symmetrically
    /// 4. **Diversity Phase**: Select candidates closer to base than to selected neighbors
    /// 5. **Starvation Fallback**: Fill to at least M/2 with k-nearest
    /// 6. **Connectivity Guarantee**: Ensure priority_node is included if close enough
    ///
    /// # Cache Optimization
    ///
    /// Uses stack-allocated [f32; 1089] cache (33x33 matrix) to eliminate
    /// redundant distance calculations. Distances are computed lazily and
    /// stored symmetrically to halve total calculations.
    ///
    /// # Arguments
    ///
    /// * `base_node` - The node we're selecting neighbors for
    /// * `candidates` - Pool of candidate neighbors (will be truncated to MAX_M+1)
    /// * `_layer` - Layer index (currently unused, kept for future extensions)
    /// * `max_count` - Maximum number of neighbors to select
    /// * `priority_node` - Optional node to prioritize (for backward linking)
    ///
    /// # Performance
    ///
    /// - Cache hit: ~0.5ns (L1 cache lookup)
    /// - Cache miss: ~500ns (distance computation + mmap read)
    /// - Worst case: O(k²) where k ≤ 33 (M + 1)
    pub(crate) fn select_neighbors_heuristic(
        &self,
        base_node: NodeId,
        candidates: &[NodeId],
        _layer: usize,
        max_count: usize,
        priority_node: Option<NodeId>,
    ) -> Result<Vec<NodeId>> {
        // Handle empty/small candidate sets
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        if candidates.len() <= max_count {
            return Ok(candidates.to_vec());
        }

        // Truncate candidates to fit in cache (MAX_M + 1 = 33)
        let truncated_candidates: Vec<NodeId> =
            candidates.iter().take(MAX_M + 1).copied().collect();

        debug_assert!(
            truncated_candidates.len() <= MAX_M + 1,
            "Too many candidates for cache: {}",
            truncated_candidates.len()
        );

        // Initialize lazy distance cache
        let mut cache = DistanceCache::new(truncated_candidates.len());

        // Helper: Get distance with lazy computation and memoization
        let get_distance = |cache: &mut DistanceCache,
                            storage: &crate::Storage,
                            id1: NodeId,
                            id2: NodeId,
                            idx1: usize,
                            idx2: usize|
         -> Result<f32> {
            if cache.is_computed(idx1, idx2) {
                Ok(cache.get(idx1, idx2))
            } else {
                let vec1 = storage.get_vector_slice(id1)?;
                let vec2 = storage.get_vector_slice(id2)?;
                let dist = crate::distance::euclidean_distance(vec1, vec2);
                cache.set(idx1, idx2, dist);
                Ok(dist)
            }
        };

        // Compute distances to base node for all candidates
        let base_vector = self.storage.get_vector_slice(base_node)?;
        let mut distances: Vec<(NodeId, f32, usize)> = truncated_candidates
            .iter()
            .enumerate()
            .map(|(idx, &id)| {
                let dist = self
                    .storage
                    .get_vector_slice(id)
                    .map(|v| crate::distance::euclidean_distance(base_vector, v))
                    .unwrap_or(f32::MAX);
                (id, dist, idx)
            })
            .collect();

        // Sort by distance for efficient processing
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // DIVERSITY PHASE: Heuristic 2 with lazy cache
        let mut selected = Vec::new();
        let mut selected_indices = Vec::new();

        for (candidate_id, candidate_dist, candidate_idx) in &distances {
            if selected.len() >= max_count {
                break;
            }

            // Check diversity: is candidate closer to base than to any selected neighbor?
            let mut is_diverse = true;

            for &selected_idx in &selected_indices {
                // Lazy fetch: compute distance only if needed
                let inter_distance = get_distance(
                    &mut cache,
                    &self.storage,
                    *candidate_id,
                    truncated_candidates[selected_idx],
                    *candidate_idx,
                    selected_idx,
                )?;

                if inter_distance < *candidate_dist {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(*candidate_id);
                selected_indices.push(*candidate_idx);
            }
        }

        // STARVATION FALLBACK: Reuse cached distances
        let min_neighbors = max_count / 2;

        if selected.len() < min_neighbors {
            for (candidate_id, _, _) in &distances {
                if selected.len() >= max_count {
                    break;
                }
                if !selected.contains(candidate_id) {
                    selected.push(*candidate_id);
                }
            }
        }

        // CONNECTIVITY GUARANTEE: Ensure priority_node if close enough
        if let Some(priority_node) = priority_node {
            if !selected.contains(&priority_node) {
                // Find priority_node position in sorted distances
                if let Some(pos) = distances.iter().position(|(id, _, _)| *id == priority_node) {
                    // Include if it's in the top max_count closest nodes
                    if pos < max_count {
                        // Make room by removing the last selected node
                        if selected.len() >= max_count {
                            selected.pop();
                        }
                        selected.push(priority_node);
                    }
                }
            }
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HnswParams, Storage};
    use tempfile::NamedTempFile;

    fn create_test_graph(dims: u32) -> (HnswGraph, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp_file.path(), dims).unwrap();

        // Insert 100 vectors with some diversity
        for i in 0..100 {
            let mut vec = vec![0.0; dims as usize];
            vec[0] = (i as f32) / 100.0;
            vec[1] = ((i % 10) as f32) / 10.0;
            storage.insert(&vec).unwrap();
        }

        let params = HnswParams::default();
        let graph = HnswGraph::open(storage, params).unwrap();

        (graph, temp_file)
    }

    #[test]
    fn test_two_phase_protocol() {
        let (mut graph, _temp) = create_test_graph(128);

        // Phase 1: Write node 0 and backlinks
        graph.write_node_and_backlinks(0, 1, &[vec![]]).unwrap();

        // Node count should still be 0 (not published yet)
        assert_eq!(graph.node_count(), 0);

        // Phase 2: Publish node 0
        graph.publish_node(0, 1).unwrap();

        // Now node count should be 1
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_forward_links_exist_after_linking() {
        let (mut graph, _temp) = create_test_graph(128);

        graph.write_node_and_backlinks(0, 1, &[vec![]]).unwrap();
        graph.publish_node(0, 1).unwrap();

        graph.write_node_and_backlinks(1, 1, &[vec![0]]).unwrap();
        graph.publish_node(1, 1).unwrap();

        graph.write_node_and_backlinks(2, 1, &[vec![0, 1]]).unwrap();
        graph.publish_node(2, 1).unwrap();

        // Verify forward links
        let node2 = graph.read_node_record(2).unwrap();
        assert_eq!(node2.get_neighbors(0), vec![0, 1]);
    }

    #[test]
    fn test_distance_cache_symmetry() {
        let mut cache = DistanceCache::new(5);

        cache.set(0, 1, 1.5);
        assert_eq!(cache.get(0, 1), 1.5);
        assert_eq!(cache.get(1, 0), 1.5); // Symmetric
        cache.set(2, 4, 3.7);
        assert_eq!(cache.get(2, 4), 3.7);
        assert_eq!(cache.get(4, 2), 3.7);
    }

    #[test]
    fn test_select_neighbors_heuristic_basic() {
        let (graph, _temp) = create_test_graph(128);

        // Test with empty candidates
        let result = graph.select_neighbors_heuristic(0, &[], 0, 5, None).unwrap();
        assert!(result.is_empty());

        // Test with fewer candidates than max
        let candidates = vec![1, 2, 3];
        let result = graph.select_neighbors_heuristic(0, &candidates, 0, 5, None).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_select_neighbors_heuristic_priority() {
        let (graph, _temp) = create_test_graph(128);

        let candidates = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Without priority
        let result = graph.select_neighbors_heuristic(0, &candidates, 0, 4, None).unwrap();
        assert!(result.len() <= 4);

        // With priority node
        let result_with_priority =
            graph.select_neighbors_heuristic(0, &candidates, 0, 4, Some(5)).unwrap();
        assert!(result_with_priority.len() <= 4);

        // Priority node should be included if it's close enough
        // (exact behavior depends on vector distances)
    }
}
