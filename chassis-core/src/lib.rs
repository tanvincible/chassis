//! Chassis - Embeddable on-disk vector storage engine
//!
//! Chassis is a local-first vector storage engine designed for embedding-based
//! search in edge devices, mobile apps, and local-first software. It's built
//! in Rust and runs anywhere from a Raspberry Pi to a data center.
//!
//! # Features
//!
//! - On-disk storage using memory-mapped I/O
//! - Page-aligned file format (4KB boundaries)
//! - Single-writer, multi-reader concurrency (SWMR)
//! - Explicit durability control via flush()
//! - Zero external dependencies (no daemons or services)
//!
//! # Example
//!
//! ```no_run
//! use chassis_core::{VectorIndex, IndexOptions};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Open or create an index
//! let mut index = VectorIndex::open("embeddings.chassis", 768, IndexOptions::default())?;
//!
//! // Insert vectors
//! let embedding = vec![0.1; 768];
//! let id = index.add(&embedding)?;
//!
//! // Flush to disk for durability
//! index.flush()?;
//!
//! // Search for nearest neighbors
//! let query = vec![0.1; 768];
//! let results = index.search(&query, 10)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Design Philosophy
//!
//! Chassis is intentionally simple and focused. It does not aim to be:
//! - A database server
//! - A cloud service
//! - A distributed system
//! - A query engine
//!
//! These concerns are left to the application layer. Chassis is a storage
//! primitive, like SQLite for relational data.

pub mod distance;
mod header;
#[cfg(feature = "internals")]
pub mod hnsw;
#[cfg(not(feature = "internals"))]
pub(crate) mod hnsw;

#[cfg(feature = "internals")]
pub mod storage;
#[cfg(not(feature = "internals"))]
pub(crate) mod storage;

pub use distance::{DistanceMetric, cosine_distance, euclidean_distance};
pub use header::{HEADER_SIZE, Header, MAGIC, VERSION};
pub use hnsw::{HnswBuilder, HnswGraph, HnswParams, SearchResult};
pub use storage::Storage;

use anyhow::Result;
use std::path::Path;

/// Maximum candidates to pass to diversity heuristic (cache limit)
const MAX_CANDIDATES_FOR_HEURISTIC: usize = 33;

/// Configuration options for VectorIndex
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Maximum connections per node (M parameter)
    pub max_connections: u16,

    /// Construction quality parameter (efConstruction)
    pub ef_construction: usize,

    /// Search quality parameter (efSearch)
    pub ef_search: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self { max_connections: 16, ef_construction: 200, ef_search: 50 }
    }
}

/// Public facade for Chassis vector index
///
/// This struct provides the main API for interacting with a Chassis index.
/// It handles crash consistency, ghost node recovery, and provides a clean
/// interface for vector insertion and search.
#[derive(Debug)]
pub struct VectorIndex {
    /// Internal HNSW graph (owns the storage)
    graph: HnswGraph,

    /// Configuration options
    options: IndexOptions,

    /// Layer multiplier cache: 1.0 / ln(M)
    ml: f32,
}

impl VectorIndex {
    /// Open or create a vector index
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    /// * `dims` - Number of dimensions per vector
    /// * `options` - Index configuration options
    ///
    /// # Crash Consistency
    ///
    /// This method handles ghost nodes (vectors written but not indexed due to crash):
    /// - If `storage.count() < graph.node_count()`: Returns error (corruption)
    /// - If `storage.count() >= graph.node_count()`: Success (ghost nodes ignored)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or created
    /// - The file is corrupted
    /// - Dimension mismatch with existing index
    /// - Graph references non-existent vectors
    pub fn open<P: AsRef<Path>>(path: P, dims: u32, options: IndexOptions) -> Result<Self> {
        // Open storage
        let storage = Storage::open(path, dims)?;

        // Compute layer multiplier
        let ml = 1.0 / (options.max_connections as f32).ln();

        // Create HNSW params from options
        let params = HnswParams {
            max_connections: options.max_connections,
            ef_construction: options.ef_construction,
            ef_search: options.ef_search,
            ml,
            max_layers: 16, // Fixed for now
        };

        // Open graph
        let mut graph = HnswGraph::open(storage, params)?;

        // Consistency check: Ghost node handling
        let storage_count = graph.storage.count();
        let graph_node_count = graph.node_count();

        if storage_count < graph_node_count {
            // Graph references vectors that don't exist = corruption
            anyhow::bail!(
                "Index corruption detected: graph has {} nodes but storage has only {} vectors",
                graph_node_count,
                storage_count
            );
        } else if storage_count >= graph_node_count {
            // GHOST NODE RECOVERY
            // Storage is ahead of Graph (crash during write).
            // We must rollback Storage to match Graph so the next insert
            // reclaims the 'ghost' ID instead of appending after it.
            graph.storage.truncate_logical(graph_node_count);
        }

        // If storage_count >= graph_node_count, we're good
        // Extra vectors are "ghost nodes" from a previous crash
        // The next add() will reclaim that space

        Ok(Self { graph, options, ml })
    }

    /// Add a vector to the index
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector to add (must match index dimensions)
    ///
    /// # Returns
    ///
    /// Returns the ID of the inserted vector
    ///
    /// # Crash Consistency Protocol (ADR-005)
    ///
    /// This method follows a strict 3-phase protocol:
    /// 1. **Persist Vector**: Write vector data to storage
    /// 2. **Write Node**: Write HNSW node with links (invisible to readers)
    /// 3. **Publish**: Update in-memory counters (visible to readers)
    ///
    /// If a crash occurs during step 2, we have a "ghost node" (vector without
    /// graph entry). The next add() will reclaim this space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector dimensions don't match index dimensions
    /// - Storage write fails
    /// - Graph write fails
    pub fn add(&mut self, vector: &[f32]) -> Result<u64> {
        // Validate dimensions
        let dims = self.graph.storage.dimensions() as usize;
        if vector.len() != dims {
            anyhow::bail!("Vector dimension mismatch: expected {}, got {}", dims, vector.len());
        }

        // STEP 1: Persist vector (reclaims ghost node space if any)
        let new_id = self.graph.storage.insert(vector)?;

        // STEP 2: Determine layer for new node
        let layer = self.select_layer();
        let layer_count = layer + 1;

        // STEP 3: Handle empty graph case
        if self.graph.node_count() == 0 {
            // Empty graph - just publish the node
            self.graph.write_node_and_backlinks(new_id, layer_count, &vec![vec![]; layer_count])?;
            self.graph.publish_node(new_id, layer_count)?;
            return Ok(new_id);
        }

        // STEP 4: Neighbor selection (in-memory phase)
        let neighbors = self.select_neighbors(vector, new_id, layer)?;

        // STEP 5: Atomic write (disk phase)
        // Node is written but invisible (node_count not incremented)
        self.graph.write_node_and_backlinks(new_id, layer_count, &neighbors)?;

        // STEP 6: Publish (commit phase)
        // Node becomes visible to readers
        self.graph.publish_node(new_id, layer_count)?;

        Ok(new_id)
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (must match index dimensions)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// Returns a vector of search results, sorted by distance (ascending)
    ///
    /// # Errors
    ///
    /// Returns an error if query dimensions don't match index dimensions
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Validate dimensions
        let dims = self.graph.storage.dimensions() as usize;
        if query.len() != dims {
            anyhow::bail!("Query dimension mismatch: expected {}, got {}", dims, query.len());
        }

        // Delegate to graph search with configured ef_search
        self.graph.search(query, k, self.options.ef_search)
    }

    /// Flush all changes to disk
    ///
    /// This method ensures durability by:
    /// 1. Flushing vector data to disk
    /// 2. Flushing graph metadata to disk
    ///
    /// # Performance Warning
    ///
    /// This operation is expensive (1-50ms depending on storage device).
    /// Batch multiple add() calls and flush() once at the end.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails
    pub fn flush(&mut self) -> Result<()> {
        // Flush vector storage first
        self.graph.storage.commit()?;

        // Then flush graph metadata
        self.graph.commit()?;

        Ok(())
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> u64 {
        self.graph.node_count()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Get the dimensionality of vectors in this index
    pub fn dimensions(&self) -> u32 {
        self.graph.storage.dimensions()
    }

    // Private helper methods

    /// Select layer for a new node using exponential decay
    fn select_layer(&self) -> usize {
        let uniform: f32 = rand::random();
        let layer = (-uniform.ln() * self.ml).floor() as usize;

        // Cap at max_layers - 1
        layer.min(15) // max_layers is 16, so max layer index is 15
    }

    /// Select neighbors for a new node at each layer
    ///
    /// This implements the HNSW neighbor selection algorithm:
    /// - Phase 1 (Zoom): Greedy descent from entry point to target layer
    /// - Phase 2 (Construction): Select diverse neighbors at each layer
    fn select_neighbors(
        &mut self,
        vector: &[f32],
        new_id: u64,
        target_layer: usize,
    ) -> Result<Vec<Vec<u64>>> {
        let mut neighbors = vec![Vec::new(); target_layer + 1];

        let entry_point = self.graph.entry_point.expect("Graph should have entry point");
        let max_layer = self.graph.max_layer;

        // Phase 1: Zoom down from max_layer to target_layer + 1
        let mut curr = entry_point;
        for layer in (target_layer + 1..=max_layer).rev() {
            curr = self.graph.search_layer_greedy(vector, curr, layer)?;
        }

        // Phase 2: Construction - select neighbors at each layer
        for layer in (0..=target_layer.min(max_layer)).rev() {
            // Search for candidates
            let candidates = self.graph.search_layer_optimized(
                vector,
                curr,
                self.options.ef_construction,
                layer,
            )?;

            // Extract candidate IDs
            let candidate_ids: Vec<u64> = candidates.iter().map(|r| r.id).collect();

            // Determine max neighbors for this layer
            let max_neighbors = if layer == 0 {
                self.options.max_connections as usize * 2
            } else {
                self.options.max_connections as usize
            };

            // Select diverse neighbors using unified heuristic
            let selected =
                self.select_diverse_subset(new_id, &candidate_ids, layer, max_neighbors)?;

            neighbors[layer] = selected;

            // Update curr to closest candidate for next layer
            if !candidates.is_empty() {
                curr = candidates[0].id;
            }
        }

        Ok(neighbors)
    }

    /// Select a diverse subset of neighbors using the diversity heuristic
    ///
    /// This method delegates to the unified `select_neighbors_heuristic` in link.rs,
    /// which implements HNSW Heuristic 2 with a stack-allocated distance cache.
    ///
    /// # Truncation for Cache Safety
    ///
    /// The diversity heuristic uses a stack-allocated cache sized for MAX_M+1 (33) nodes.
    /// Since search results from `ef_construction` can be much larger (e.g., 200 nodes),
    /// we truncate to the top 33 closest candidates. This is safe because:
    /// 1. Search results are already sorted by distance (ascending)
    /// 2. The best diverse neighbors are likely among the closest candidates
    /// 3. We maintain O(1) stack allocation for the distance cache
    fn select_diverse_subset(
        &self,
        base_node: u64,
        candidates: &[u64],
        layer: usize,
        max_count: usize,
    ) -> Result<Vec<u64>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Truncate candidates to fit in cache (33 = MAX_M + 1)
        // Search results are already sorted by distance, so we keep the closest
        let truncated_candidates: Vec<u64> =
            candidates.iter().take(MAX_CANDIDATES_FOR_HEURISTIC).copied().collect();

        // Delegate to unified heuristic (no priority for forward linking)
        self.graph.select_neighbors_heuristic(
            base_node,
            &truncated_candidates,
            layer,
            max_count,
            None, // No priority node for forward linking
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_vector_index_create_and_add() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.dimensions(), 128);

        // Add a vector
        let vector = vec![0.1; 128];
        let id = index.add(&vector).unwrap();

        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_vector_index_dimension_mismatch() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Try to add vector with wrong dimensions
        let vector = vec![0.1; 64];
        let result = index.add(&vector);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_vector_index_search() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Add some vectors
        for i in 0..10 {
            let mut vector = vec![0.0; 128];
            vector[0] = i as f32 / 10.0;
            index.add(&vector).unwrap();
        }

        // Search
        let query = vec![0.5; 128];
        let results = index.search(&query, 5).unwrap();

        assert!(results.len() <= 5);
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_vector_index_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_owned();

        // Create and populate index
        {
            let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

            for i in 0..5 {
                let mut vector = vec![0.0; 128];
                vector[0] = i as f32;
                index.add(&vector).unwrap();
            }

            index.flush().unwrap();
        }

        // Reopen and verify
        {
            let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
            assert_eq!(index.len(), 5);

            let mut query = vec![0.0; 128];
            query[0] = 2.0;

            let results = index.search(&query, 3).unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_ghost_node_recovery() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_owned();

        // Simulate ghost node: insert vector but don't publish node
        {
            let storage = Storage::open(&path, 128).unwrap();
            let params = HnswParams::default();
            let mut graph = HnswGraph::open(storage, params).unwrap();

            // Insert vector into storage (creates ghost node)
            let vector = vec![1.0; 128];
            graph.storage.insert(&vector).unwrap();
            graph.storage.commit().unwrap();

            // Don't add to graph - this creates a ghost node
            assert_eq!(graph.storage.count(), 1);
            assert_eq!(graph.node_count(), 0);
        }

        // Reopen with VectorIndex - should handle ghost node
        {
            let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

            // Ghost node should be ignored (not counted)
            assert_eq!(index.len(), 0);

            // Next add should reclaim the ghost node space
            let vector = vec![2.0; 128];
            let id = index.add(&vector).unwrap();

            // Should reuse the ghost node's ID (0)
            assert_eq!(id, 0);
            assert_eq!(index.len(), 1);
        }
    }

    #[test]
    fn test_diverse_neighbor_selection() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Add vectors in a line (should create diverse neighborhoods)
        for i in 0..20 {
            let mut vector = vec![0.0; 128];
            vector[0] = i as f32;
            index.add(&vector).unwrap();
        }

        // Verify graph was built
        assert_eq!(index.len(), 20);

        // Search should work
        let query = vec![10.0; 128];
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
    }
}
