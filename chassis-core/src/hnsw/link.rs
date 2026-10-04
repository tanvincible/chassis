//! Bidirectional graph linking with diversity heuristics and crash consistency.
//!
//! # Crash Consistency Model
//!
//! This module implements a write-ahead linking strategy:
//! 1. Write Node A's record FIRST (with forward links)
//! 2. Update backward links on neighbors (may crash mid-process)
//! 3. Update graph header last
//!
//! If a crash occurs during step 2, we may have one-way edges (A→B exists but B→A doesn't).
//! Backlinks to A can also outlive a crash that rolls A back: search skips those ids, and
//! pruning drops them, until a new node reuses the id.
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
        self.write_node_with_id_and_backlinks(node_id, node_id, layer_count, neighbors_per_layer)
    }

    /// Like `write_node_and_backlinks`, but stores the caller's `id` instead of the slot.
    pub(crate) fn write_node_with_id_and_backlinks(
        &mut self,
        node_id: NodeId,
        id: u64,
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
        node_record.header.id = id;

        for (layer, neighbors) in filtered_neighbors.iter().enumerate() {
            node_record.set_neighbors(layer, neighbors);
        }

        self.write_node_record(&node_record)?;

        // STEP B: Update backward links (B→A) for each neighbor
        for (layer, neighbors) in filtered_neighbors.iter().enumerate().take(layer_count) {
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
        // Ids at or past node_count point at nodes a crash rolled back; their vectors are gone.
        let mut candidates: Vec<NodeId> =
            record.get_neighbors(layer).into_iter().filter(|&id| id < self.node_count).collect();

        // Duplicate check (idempotency)
        if candidates.contains(&new_node) {
            return Ok(());
        }
        candidates.push(new_node);

        let selected = self.select_neighbors_heuristic(
            neighbor_id,
            &candidates,
            layer,
            self.record_params.max_neighbors(layer),
            Some(new_node), // Prioritize the new node for connectivity
        )?;

        record.set_neighbors(layer, &selected);
        self.update_node_record(&record)
    }

    /// Select diverse neighbors using HNSW Heuristic 2.
    ///
    /// Used for a new node's own neighbors (no priority) and when a backlink overfills a neighbor's
    /// list (`priority_node` is the new node). Candidates are taken nearest first, and one is kept
    /// only if it is closer to `base_node` than to every neighbor already kept. If that keeps fewer
    /// than `max_count / 2`, the nearest remaining candidates fill up to `max_count`. `priority_node`
    /// is kept if it ranks within the nearest `max_count`.
    pub(crate) fn select_neighbors_heuristic(
        &self,
        base_node: NodeId,
        candidates: &[NodeId],
        _layer: usize,
        max_count: usize,
        priority_node: Option<NodeId>,
    ) -> Result<Vec<NodeId>> {
        if candidates.len() <= max_count {
            return Ok(candidates.to_vec());
        }

        let base_vector = self.storage.get_vector_slice(base_node)?;
        let mut by_distance: Vec<(NodeId, f32)> = candidates
            .iter()
            .map(|&id| {
                let distance = self
                    .storage
                    .get_vector_slice(id)
                    .map(|v| crate::distance::euclidean_distance(base_vector, v))
                    .unwrap_or(f32::MAX);
                (id, distance)
            })
            .collect();
        by_distance.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Each (candidate, kept) pair is compared at most once, so there is nothing to cache.
        let mut selected: Vec<NodeId> = Vec::with_capacity(max_count);
        for &(candidate, distance) in &by_distance {
            if selected.len() >= max_count {
                break;
            }
            let vector = self.storage.get_vector_slice(candidate)?;
            let mut is_diverse = true;
            for &kept in &selected {
                let kept_vector = self.storage.get_vector_slice(kept)?;
                if crate::distance::euclidean_distance(vector, kept_vector) < distance {
                    is_diverse = false;
                    break;
                }
            }
            if is_diverse {
                selected.push(candidate);
            }
        }

        if selected.len() < max_count / 2 {
            for &(candidate, _) in &by_distance {
                if selected.len() >= max_count {
                    break;
                }
                if !selected.contains(&candidate) {
                    selected.push(candidate);
                }
            }
        }

        if let Some(priority_node) = priority_node
            && !selected.contains(&priority_node)
            && let Some(pos) = by_distance.iter().position(|&(id, _)| id == priority_node)
            && pos < max_count
        {
            if selected.len() >= max_count {
                selected.pop();
            }
            selected.push(priority_node);
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
