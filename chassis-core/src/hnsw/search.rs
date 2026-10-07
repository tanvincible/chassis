//! HNSW search over the memory-mapped graph.
//!
//! # Performance Optimizations
//!
//! - Dense visited filter (no HashSet in hot path)
//! - Zero-allocation neighbor iteration via `neighbors_iter_from_mmap()`
//! - Zero-copy distance computation via `compute_distance_zero_copy()`
//! - NaN-safe ordering with `f32::total_cmp`
//!
//! # Safety Guarantees
//!
//! - No panics on NaN distances
//! - No hash table overhead in search loop
//! - Concurrent searches from many threads (`&self`)
//! - Deterministic performance

use crate::hnsw::graph::HnswGraph;
use crate::hnsw::node::NodeId;
use anyhow::Result;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Slots a filtered search samples to estimate how many match.
const FILTER_SAMPLE: u64 = 1024;

/// Graph visits a filtered search may spend before checking every slot instead, with `matches` of
/// `slots` passing the filter. Half the matches: a graph visit costs more than a scanned distance.
/// Twice the 16 × ef / s visits a random filter needs: matches far from the query cost far more
/// (ADR-0009).
fn filter_budget(matches: u64, slots: u64, ef: usize) -> u64 {
    let far = 32u64.saturating_mul(ef as u64).saturating_mul(slots) / matches.max(1);
    (matches / 2).min(far)
}

/// Search result with distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: NodeId,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == std::cmp::Ordering::Equal
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    /// Total ordering using f32::total_cmp (NaN-safe)
    ///
    /// This ensures:
    /// - No panics on NaN values
    /// - Deterministic ordering
    /// - Correct heap behavior
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

/// Dense visited filter using a BitSet for maximum cache locality.
///
/// # Design
///
/// Uses a packed bit array (1 bit per node) to fit more status flags into
/// CPU cache lines.
///
/// # Performance
///
/// - Memory: 125KB for 1M nodes (vs 1MB for Vec<bool>)
/// - Cache Density: 512 nodes per cache line (vs 64)
/// - Init Cost: ~8x faster allocation/zeroing
pub struct VisitedFilter {
    /// Dense bit array: 1 bit per node. Stores 64 nodes per u64.
    data: Vec<u64>,
    /// Capacity track to avoid checking len() bounds repeatedly
    capacity: usize,
}

impl VisitedFilter {
    /// Create a new visited filter for a graph with `node_count` nodes
    #[inline]
    pub fn new(node_count: usize) -> Self {
        // Calculate number of u64s needed: ceil(N / 64)
        let num_u64s = node_count.div_ceil(64);
        Self { data: vec![0; num_u64s], capacity: node_count }
    }

    /// Mark a node as visited.
    ///
    /// Returns:
    /// - `true` if the node was NOT visited before (we just marked it).
    /// - `false` if it was ALREADY visited.
    #[inline]
    pub fn visit(&mut self, node_id: u64) -> bool {
        let idx = node_id as usize;

        if idx >= self.capacity {
            return false;
        }

        let word_idx = idx >> 6;
        let bit_idx = idx & 63;
        let mask = 1u64 << bit_idx;

        let word = unsafe { self.data.get_unchecked_mut(word_idx) };

        if *word & mask != 0 {
            false // Already visited (Return false to match old API)
        } else {
            *word |= mask; // Mark as visited
            true // Newly visited (Success)
        }
    }
    /// Check if a node is visited without modifying state.
    #[inline]
    #[allow(dead_code)]
    /// Tested by test_visited_filter
    fn is_visited(&self, node_id: u64) -> bool {
        let idx = node_id as usize;
        if idx >= self.capacity {
            return false;
        }

        let word_idx = idx >> 6;
        let bit_idx = idx & 63;
        let mask = 1u64 << bit_idx;

        let word = unsafe { self.data.get_unchecked(word_idx) };
        (*word & mask) != 0
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
    ///
    /// # Guarantees
    ///
    /// - Returns ≤ k results (never more)
    /// - If ef < k, silently corrects to ef = k
    /// - No panics on NaN distances
    /// - Deterministic ordering
    ///
    /// # Performance
    ///
    /// - O(ef × log(ef)) time complexity
    /// - O(node_count) space for visited filter
    /// - Allocates a visited set per layer, plus two heaps on layer 0
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        // Enforce ef >= k silently (no error, no surprise)
        let ef = ef.max(k);

        let entry = self.entry_point.unwrap();
        let mut current_layer = self.max_layer;

        // Greedy search from top layer to layer 1
        let mut current = entry;
        while current_layer > 0 {
            current = self.search_layer_greedy(query, current, current_layer)?;
            current_layer -= 1;
        }

        // Search base layer with ef candidates
        let mut candidates = self.search_layer_filtered(query, current, ef, 0, self.skips())?;

        // Return top k
        candidates.truncate(k);
        Ok(candidates)
    }

    /// The `k` nearest live nodes that `allow` accepts (ADR-0009). HNSW visits about `1 / s` times
    /// more nodes when a fraction `s` matches, so past a budget it gives up and checks every slot.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(NodeId) -> Result<bool>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self.search_filtered_reporting(query, k, ef, allow)?.0)
    }

    /// `search_filtered`, and whether it fell back to checking every slot.
    pub(crate) fn search_filtered_reporting(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(NodeId) -> Result<bool>,
    ) -> Result<(Vec<SearchResult>, bool)> {
        let Some(mut current) = self.entry_point else { return Ok((Vec::new(), false)) };
        let ef = ef.max(k);
        let budget = filter_budget(self.estimated_matches(allow)?, self.node_count, ef);

        for layer in (1..=self.max_layer).rev() {
            current = self.search_layer_greedy(query, current, layer)?;
        }
        let filter = Some((allow, budget as usize));
        if let Some(mut found) =
            self.search_layer::<true>(query, current, ef, 0, self.skips(), filter)?
        {
            found.truncate(k);
            return Ok((found, false));
        }

        let mut nearest = BinaryHeap::new();
        for slot in 0..self.node_count {
            if self.is_live(slot)? && allow(slot)? {
                let distance = self.compute_distance_zero_copy(query, slot)?;
                let nearer = |w: &SearchResult| distance.total_cmp(&w.distance).is_lt();
                if nearest.len() < k || nearest.peek().is_some_and(nearer) {
                    nearest.push(SearchResult { id: slot, distance });
                    if nearest.len() > k {
                        nearest.pop();
                    }
                }
            }
        }
        Ok((nearest.into_sorted_vec(), true))
    }

    /// Live slots `allow` accepts, estimated from about `FILTER_SAMPLE` of them, one per step,
    /// jittered within the step so a filter periodic in the slot can't line up with the sample.
    pub(crate) fn estimated_matches(&self, allow: &dyn Fn(NodeId) -> Result<bool>) -> Result<u64> {
        let step = (self.node_count / FILTER_SAMPLE).max(1);
        let sampled = self.node_count / step;
        let mut matched = 0;
        for i in 0..sampled {
            let slot = i * step + (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) % step;
            matched += u64::from(self.is_live(slot)? && allow(slot)?);
        }
        Ok(matched * self.node_count / sampled.max(1))
    }

    /// Whether some slot below `node_count` may be deleted or outside a reader's snapshot.
    fn skips(&self) -> bool {
        self.deleted_count > 0 || self.view.is_some_and(|v| self.node_count > v.committed)
    }

    fn is_live(&self, slot: NodeId) -> Result<bool> {
        Ok(!(self.skips() && self.is_deleted(slot)?))
    }

    /// Greedy search for a single best node (used for layer descent).
    ///
    /// This avoids allocating a Vec just to read the single best candidate during upper-layer descent.
    ///
    /// # Returns
    ///
    /// The closest node to the query at this layer.
    pub fn search_layer_greedy(
        &self,
        query: &[f32],
        entry: NodeId,
        layer: usize,
    ) -> Result<NodeId> {
        let mut best_id = entry;
        let mut best_dist = self.compute_distance_zero_copy(query, entry)?;

        let mut visited = VisitedFilter::new(self.node_count as usize);
        visited.visit(entry);

        let mut changed = true;
        while changed {
            changed = false;

            for neighbor_id in self.neighbors_iter_from_mmap(best_id, layer)? {
                if visited.visit(neighbor_id) {
                    let dist = self.compute_distance_zero_copy(query, neighbor_id)?;

                    if dist.total_cmp(&best_dist) == std::cmp::Ordering::Less {
                        best_id = neighbor_id;
                        best_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        Ok(best_id)
    }

    /// Search within a single layer.
    ///
    /// # Optimizations Applied
    ///
    /// 1. **Dense visited filter**: O(1) array access instead of HashSet hashing
    ///    - Check: ~1ns vs ~100ns
    ///    - No cache misses from pointer chasing
    ///
    /// 2. **Zero-allocation neighbor iteration**: `neighbors_iter_from_mmap()`
    ///    - No `Vec<NodeId>` allocation per node
    ///    - ~100ns vs ~400ns per node
    ///
    /// 3. **Zero-copy distance computation**: `compute_distance_zero_copy()`
    ///    - No `Vec<f32>` allocation per distance calculation
    ///    - Direct mmap reads
    ///
    /// 4. **NaN-safe ordering**: `f32::total_cmp`
    ///    - No panics on NaN
    ///    - Deterministic behavior
    ///
    /// # Hot Path Analysis
    ///
    /// No allocations:
    /// - ✓ No `Node::from_record()` calls
    /// - ✓ No `Vec<NodeId>` for neighbors
    /// - ✓ No `Vec<f32>` for vectors
    /// - ✓ No HashSet operations
    pub fn search_layer_optimized(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_layer_filtered(query, entry, ef, layer, false)
    }

    /// Like `search_layer_optimized`; with `skip_deleted`, deleted nodes are still traversed
    /// (they keep the graph connected) but never returned.
    pub(crate) fn search_layer_filtered(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
        skip_deleted: bool,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .search_layer::<false>(query, entry, ef, layer, skip_deleted, None)?
            .unwrap_or_default())
    }

    /// The search loop. A filter's nodes are traversed like deleted ones but never returned; it
    /// returns `None` once it has computed the filter's budget of distances. `FILTERED` compiles
    /// the filter out of unfiltered searches.
    #[allow(clippy::type_complexity)]
    fn search_layer<const FILTERED: bool>(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
        skip_deleted: bool,
        filter: Option<(&dyn Fn(NodeId) -> Result<bool>, usize)>,
    ) -> Result<Option<Vec<SearchResult>>> {
        let excluded = |id| -> Result<bool> {
            if skip_deleted && self.is_deleted(id)? {
                return Ok(true);
            }
            Ok(match filter {
                Some((allow, _)) if FILTERED => !allow(id)?,
                _ => false,
            })
        };
        let mut budget = filter.map_or(usize::MAX, |(_, budget)| budget);

        // Dense visited filter: O(n) space, O(1) time per check
        let mut visited = VisitedFilter::new(self.node_count as usize);

        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        let mut fresh = Vec::with_capacity(self.record_params.max_neighbors(layer));

        // Zero-copy distance computation
        let entry_dist = self.compute_distance_zero_copy(query, entry)?;
        candidates.push(Reverse(SearchResult { id: entry, distance: entry_dist }));
        if !excluded(entry)? {
            results.push(SearchResult { id: entry, distance: entry_dist });
        }
        visited.visit(entry);

        while let Some(Reverse(current)) = candidates.pop() {
            // Early termination: current is further than worst result
            if results.len() >= ef
                && let Some(worst) = results.peek()
                && current.distance.total_cmp(&worst.distance) == std::cmp::Ordering::Greater
            {
                break;
            }

            // Start loading every unvisited neighbor's vector before computing any distance, so the
            // cache misses overlap instead of each distance waiting on its own.
            fresh.clear();
            for neighbor_id in self.neighbors_iter_from_mmap(current.id, layer)? {
                if visited.visit(neighbor_id) {
                    self.storage.prefetch_vector(neighbor_id);
                    fresh.push(neighbor_id);
                }
            }
            for &neighbor_id in &fresh {
                if FILTERED {
                    budget = match budget.checked_sub(1) {
                        Some(left) => left,
                        None => return Ok(None),
                    };
                }
                // Zero-copy distance computation
                // Reads directly from mmap instead of allocating Vec<f32>
                let dist = self.compute_distance_zero_copy(query, neighbor_id)?;

                let should_add = if results.len() < ef {
                    true
                } else if let Some(worst) = results.peek() {
                    dist.total_cmp(&worst.distance) == std::cmp::Ordering::Less
                } else {
                    false
                };

                if should_add {
                    candidates.push(Reverse(SearchResult { id: neighbor_id, distance: dist }));
                    if excluded(neighbor_id)? {
                        continue;
                    }
                    results.push(SearchResult { id: neighbor_id, distance: dist });

                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut sorted: Vec<_> = results.into_iter().collect();
        sorted.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(Some(sorted))
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

        // Max-heap: largest distance first
        assert_eq!(heap.pop().unwrap().id, 2); // distance 1.0
        assert_eq!(heap.pop().unwrap().id, 1); // distance 0.5
        assert_eq!(heap.pop().unwrap().id, 3); // distance 0.1
    }

    #[test]
    fn test_nan_safe_ordering() {
        // Test that NaN doesn't cause panics
        let r1 = SearchResult { id: 1, distance: 0.5 };
        let r2 = SearchResult { id: 2, distance: f32::NAN };
        let r3 = SearchResult { id: 3, distance: 1.0 };

        // Should not panic
        let mut results = [r1.clone(), r2.clone(), r3.clone()];
        results.sort();

        // NaN should be ordered deterministically (typically at the end)
        assert!(!results[0].distance.is_nan());
        assert!(!results[1].distance.is_nan());
    }

    #[test]
    fn test_visited_filter() {
        let mut filter = VisitedFilter::new(10);

        // First visit returns true (was not visited)
        assert!(filter.visit(0));
        assert!(filter.visit(5));
        assert!(filter.visit(9));

        // Second visit returns false (already visited)
        assert!(!filter.visit(0));
        assert!(!filter.visit(5));
        assert!(!filter.visit(9));

        // Check visited status
        assert!(filter.is_visited(0));
        assert!(filter.is_visited(5));
        assert!(filter.is_visited(9));
        assert!(!filter.is_visited(1));
        assert!(!filter.is_visited(7));
    }

    #[test]
    fn test_visited_filter_out_of_bounds() {
        let mut filter = VisitedFilter::new(10);

        // Out of bounds returns false (not visited, can't visit)
        assert!(!filter.visit(100));
        assert!(!filter.visit(100));
    }

    #[test]
    fn test_ef_less_than_k_correction() {
        // This is a behavioral test - we can't easily test the internal
        // correction without a full graph, but we document the expectation

        // If ef < k, search should silently correct to ef = k
        // This prevents returning fewer than k results due to misconfiguration

        // Actual test would require a built graph:
        // let results = graph.search(query, k=10, ef=3);
        // assert!(results.len() <= 10);
    }

    #[test]
    fn test_total_cmp_properties() {
        // Verify total_cmp provides total ordering
        let values = vec![0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];

        // All comparisons should be deterministic and not panic
        for &a in &values {
            for &b in &values {
                let _cmp = a.total_cmp(&b); // Should not panic
            }
        }

        // NaN should have consistent ordering
        assert_eq!(f32::NAN.total_cmp(&f32::NAN), std::cmp::Ordering::Equal);
        assert_eq!(f32::NAN.total_cmp(&0.0), std::cmp::Ordering::Greater);
        assert_eq!(f32::NAN.total_cmp(&f32::INFINITY), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_filter_budget() {
        // Selective: half the matches.
        assert_eq!(filter_budget(1_000, 1_000_000, 64), 500);
        // Broad: twice what a random filter needs, 32 × 64 / 0.3.
        assert_eq!(filter_budget(300_000, 1_000_000, 64), 6_826);
        assert_eq!(filter_budget(0, 1_000_000, 64), 0);
        // Saturates instead of overflowing.
        assert_eq!(filter_budget(u64::MAX, u64::MAX, usize::MAX), 1);
    }
}
