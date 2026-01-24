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
//!
//! **Pros**:
//! - Strong safety guarantees
//! - No dangling forward edges
//! - Simpler search (no existence checks)
//! - Cleaner crash recovery
//!
//! **Cons**:
//! - Graph must be built sequentially
//! - Cannot pre-declare future neighbors

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

/// Result of a neighbor selection operation
#[derive(Debug)]
pub struct SelectionResult {
    /// Selected neighbors (respects M/M0 limits)
    selected: Vec<NodeId>,
    /// Whether Node A was included in the selection
    #[allow(dead_code)]
    includes_new_node: bool,
}

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
    /// Link a node bidirectionally to its neighbors with diversity heuristics.
    ///
    /// # Execution Sequence (MANDATORY ORDER)
    ///
    /// 1. **Step A (Forward Link)**: Write Node A's record with forward links
    /// 2. **Step B (Backward Links)**: For each layer, update each neighbor's back-link
    ///    - Duplicate check (idempotency)
    ///    - Diversity pruning if full
    ///    - Write neighbor's record
    /// 3. **Step C (Header Update)**: Update graph header
    ///
    /// # Forward Link Policy (Model A)
    ///
    /// Only existing nodes can be referenced in `neighbors_per_layer`.
    /// Any neighbor ID >= `self.node_count` will be filtered out.
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
    pub fn link_node_bidirectional(
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

        // Increment node count AFTER successful write (crash safety)
        self.node_count += 1;

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

        // STEP C: Update graph header (entry point, max layer)
        if self.entry_point.is_none() || layer_count - 1 > self.max_layer {
            self.entry_point = Some(node_id);
            self.max_layer = layer_count - 1;
        }

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

        // Full - apply diversity heuristic with cache
        let selected = self.select_diverse_neighbors_cached(
            neighbor_id,
            &current_neighbors,
            new_node,
            layer,
            max_neighbors,
        )?;

        record.set_neighbors(layer, &selected.selected);
        self.update_node_record(&record)?;

        Ok(())
    }

    /// Select diverse neighbors using Heuristic 2 with lazy memoized distance cache.
    ///
    /// # Algorithm
    ///
    /// 1. **Local Index Mapping**: Map NodeIds to local indices [0..k)
    /// 2. **Lazy Cache**: Compute distances only when needed, store symmetrically
    /// 3. **Diversity Phase**: Select candidates closer to base than to selected neighbors
    /// 4. **Starvation Fallback**: Fill to at least M/2 with k-nearest
    ///
    /// # Cache Optimization
    ///
    /// Uses stack-allocated [f32; 1089] cache (33x33 matrix) to eliminate
    /// redundant distance calculations. Distances are computed lazily and
    /// stored symmetrically to halve total calculations.
    ///
    /// # Performance
    ///
    /// - Cache hit: ~0.5ns (L1 cache lookup)
    /// - Cache miss: ~500ns (distance computation + mmap read)
    /// - Worst case: O(k²) where k ≤ 33 (M + 1)
    pub fn select_diverse_neighbors_cached(
        &self,
        base_node: NodeId,
        current_neighbors: &[NodeId],
        new_node: NodeId,
        _layer: usize,
        max_count: usize,
    ) -> Result<SelectionResult> {
        // Build candidate set and local index mapping
        let mut candidates: Vec<NodeId> = current_neighbors.to_vec();
        candidates.push(new_node);

        debug_assert!(
            candidates.len() <= MAX_M + 1,
            "Too many candidates for cache: {}",
            candidates.len()
        );

        // Initialize lazy distance cache
        let mut cache = DistanceCache::new(candidates.len());

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
        let mut distances: Vec<(NodeId, f32, usize)> = candidates
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
                    candidates[selected_idx],
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

        let includes_new = selected.contains(&new_node);

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

        // CONNECTIVITY GUARANTEE: Ensure new_node if close enough
        if !includes_new && selected.len() < max_count {
            if let Some(pos) = distances.iter().position(|(id, _, _)| *id == new_node) {
                if pos < max_count {
                    if !selected.contains(&new_node) {
                        if selected.len() >= max_count {
                            selected.pop();
                        }
                        selected.push(new_node);
                    }
                }
            }
        }

        let includes_new = selected.contains(&new_node);

        Ok(SelectionResult { selected, includes_new_node: includes_new })
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
    fn test_forward_links_exist_after_linking() {
        let (mut graph, _temp) = create_test_graph(128);

        // IMPORTANT: Must link in sequence due to Model A
        // Node 0 can only link to nodes that exist (none yet, so empty)
        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

        // Node 1 can link to node 0 (which now exists)
        graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();

        // Node 2 can link to nodes 0 and 1
        graph.link_node_bidirectional(2, 1, &[vec![0, 1]]).unwrap();

        // Verify forward links
        let node2 = graph.read_node_record(2).unwrap();
        assert_eq!(node2.get_neighbors(0), vec![0, 1]);

        // Verify backward links
        let node0 = graph.read_node_record(0).unwrap();
        assert!(node0.get_neighbors(0).contains(&1) || node0.get_neighbors(0).contains(&2));
    }

    #[test]
    fn test_self_links_are_filtered() {
        let (mut graph, _temp) = create_test_graph(128);

        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

        // Try to create self-link (will be filtered)
        graph.link_node_bidirectional(1, 1, &[vec![0, 1]]).unwrap();

        // Self-link should be filtered out
        let record = graph.read_node_record(1).unwrap();
        assert!(!record.get_neighbors(0).contains(&1), "Self-links should be filtered");
        assert_eq!(record.get_neighbors(0), vec![0]);
    }

    #[test]
    fn test_future_neighbors_filtered() {
        let (mut graph, _temp) = create_test_graph(128);

        // Try to link to nodes that don't exist yet (will be filtered by Model A)
        graph.link_node_bidirectional(0, 1, &[vec![1, 2, 999]]).unwrap();

        // All non-existent neighbors should be filtered
        let record = graph.read_node_record(0).unwrap();
        assert_eq!(record.get_neighbors(0), vec![], "Future neighbors should be filtered");
    }

    #[test]
    fn test_idempotency_no_duplicates() {
        let (mut graph, _temp) = create_test_graph(128);

        // Link in sequence
        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();
        graph.link_node_bidirectional(1, 1, &[vec![0]]).unwrap();

        // Manually retry adding backward link
        graph.add_backward_link_with_pruning(0, 1, 0).unwrap();

        // Should have exactly one back-link
        let record = graph.read_node_record(0).unwrap();
        let back_neighbors = record.get_neighbors(0);
        assert_eq!(back_neighbors.iter().filter(|&&id| id == 1).count(), 1);
    }

    #[test]
    fn test_respects_capacity_limits() {
        let (mut graph, _temp) = create_test_graph(128);

        let params = graph.record_params();
        let m0 = params.m0 as usize;

        // Build graph sequentially (Model A requirement)
        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

        for i in 1..=m0.min(20) as u64 {
            graph.link_node_bidirectional(i, 1, &[vec![0]]).unwrap();
        }

        // Node 0 should respect capacity
        let record = graph.read_node_record(0).unwrap();
        assert!(record.get_neighbors(0).len() <= m0);
    }

    #[test]
    fn test_maintains_minimum_degree() {
        let (mut graph, _temp) = create_test_graph(128);

        let params = graph.record_params();
        let m0 = params.m0 as usize;
        let min_neighbors = m0 / 2;

        // Build hub
        graph.link_node_bidirectional(0, 1, &[vec![]]).unwrap();

        for i in 1..=m0.min(30) as u64 {
            graph.link_node_bidirectional(i, 1, &[vec![0]]).unwrap();
        }

        // Add more to trigger pruning
        for i in (m0.min(30) + 1)..=(m0.min(30) + 10) {
            graph.link_node_bidirectional(i as u64, 1, &[vec![0]]).unwrap();
        }

        let record = graph.read_node_record(0).unwrap();
        let neighbor_count = record.neighbor_count(0);

        assert!(neighbor_count >= min_neighbors);
    }

    #[test]
    fn test_layer_independence() {
        let (mut graph, _temp) = create_test_graph(128);

        // Build sequentially
        graph.link_node_bidirectional(0, 3, &[vec![], vec![], vec![]]).unwrap();
        graph.link_node_bidirectional(1, 3, &[vec![0], vec![0], vec![]]).unwrap();
        graph.link_node_bidirectional(2, 3, &[vec![0, 1], vec![], vec![]]).unwrap();

        // Verify layer-specific links
        let record = graph.read_node_record(0).unwrap();
        assert!(record.get_neighbors(0).contains(&1) || record.get_neighbors(0).contains(&2));
        assert!(record.get_neighbors(1).contains(&1));
    }

    #[test]
    fn test_node_id_invariant_returns_error() {
        let (mut graph, _temp) = create_test_graph(128);

        let result = graph.link_node_bidirectional(5, 1, &[vec![]]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Node ID invariant violated"));
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
}
