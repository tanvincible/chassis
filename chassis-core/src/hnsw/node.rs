//! Fixed-width HNSW node record layout for O(1) addressing. 
//!
//! # Design Principles
//!
//! Each node record has a **fixed size** determined at index creation time.
//! This allows direct addressing:  `node_offset = graph_start + (node_id * NODE_RECORD_SIZE)`.
//!
//! The record reserves space for the **maximum possible** neighbors across all layers.
//!  Unused neighbor slots are filled with `INVALID_NODE_ID`.
//!
//! # Layout Invariants
//!
//! - Record size is immutable once an index is created
//! - Changing `M` or `max_layers` requires rebuilding the index
//! - Record size is always a multiple of 8 bytes

use std::mem;

/// File offset type (u64 for 64-bit addressing)
pub type Offset = u64;

/// Node ID (index into vector storage)
pub type NodeId = u64;

/// Sentinel value indicating an empty neighbor slot
pub const INVALID_NODE_ID: NodeId = NodeId::MAX;

/// Default maximum connections per layer (M parameter)
pub const DEFAULT_M:  u16 = 16;

/// Default maximum layers in the graph
pub const DEFAULT_MAX_LAYERS: u8 = 16;

/// Maximum connections at layer 0 (typically 2*M)
pub const DEFAULT_M0:  u16 = DEFAULT_M * 2;

/// Fixed-size on-disk node header. 
///
/// # Layout (16 bytes, 8-byte aligned)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       8     node_id:  NodeId
/// 8       1     layer_count: u8 (highest layer this node belongs to + 1)
/// 9       1     flags: u8 (reserved for future use)
/// 10      6     _padding: [u8; 6]
/// ```
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct NodeHeader {
    /// Vector ID in storage (also serves as the node's index)
    pub node_id: NodeId,

    /// Number of layers this node participates in (1 = layer 0 only)
    pub layer_count:  u8,

    /// Flags for future extensions (deleted flag, etc.)
    pub flags: u8,

    /// Reserved for alignment
    _padding: [u8; 6],
}

impl NodeHeader {
        /// Size of the header in bytes
    pub const SIZE: usize = std::mem::size_of:: <Self>();

    /// Create a new node header
    #[must_use]
    pub const fn new(node_id: NodeId, layer_count: u8) -> Self {
        Self {
            node_id,
            layer_count,
            flags:  0,
            _padding:  [0; 6],
        }
    }

    /// Read header from bytes with validation. 
    ///
    /// # Safety Guarantees
    ///
    /// This method is safe because it: 
    /// - Checks slice length before reading
    /// - Uses `read_unaligned` to handle arbitrary alignment
    /// - Validates header fields for sanity
    ///
    /// # Errors
    ///
    /// Returns an error if: 
    /// - Slice is too small
    /// - `layer_count` is 0 (invalid)
    /// - `layer_count` exceeds reasonable maximum (255)
    /// - `node_id` is `INVALID_NODE_ID` (reserved sentinel)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < Self::SIZE {
            return Err("Buffer too small for NodeHeader");
        }

        // Use read_unaligned for safety - don't assume alignment
        let header = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>())
        };

        // Validate fields
        if header.layer_count == 0 {
            return Err("Invalid NodeHeader:  layer_count cannot be 0");
        }

        // node_id == INVALID_NODE_ID is reserved for empty slots
        // A valid header should never have this value
        if header.node_id == INVALID_NODE_ID {
            return Err("Invalid NodeHeader: node_id is INVALID_NODE_ID sentinel");
        }

        Ok(header)
    }

    /// Read header from bytes WITHOUT validation (for performance-critical paths).
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `bytes.len() >= NodeHeader::SIZE`
    /// - The bytes represent a valid, previously-written header
    ///
    /// Use this only when reading from known-good locations (e.g., after
    /// successful write or when iterating over validated node records).
    #[inline]
    pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() >= Self::SIZE, "Buffer too small for NodeHeader");
        unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) }
    }

    /// Check if the node is marked as deleted
    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Mark the node as deleted
    pub fn set_deleted(&mut self) {
        self.flags |= 0x01;
    }
}

/// Parameters that determine the fixed record size. 
///
/// These parameters are set at index creation and cannot be changed
/// without rebuilding the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRecordParams {
    /// Maximum connections per layer for layers > 0 (M)
    pub m: u16,

    /// Maximum connections at layer 0 (typically 2*M)
    pub m0: u16,

    /// Maximum number of layers
    pub max_layers: u8,
}

impl Default for NodeRecordParams {
    fn default() -> Self {
        Self {
            m: DEFAULT_M,
            m0: DEFAULT_M0,
            max_layers: DEFAULT_MAX_LAYERS,
        }
    }
}

impl NodeRecordParams {
    /// Create new node record parameters
    #[must_use]
    pub const fn new(m: u16, m0: u16, max_layers: u8) -> Self {
        Self { m, m0, max_layers }
    }

    /// Calculate the fixed record size for these parameters. 
    ///
    /// # Layout
    ///
    /// ```text
    /// [NodeHeader:  16 bytes]
    /// [Layer 0 neighbors: m0 * 8 bytes]
    /// [Layer 1 neighbors: m * 8 bytes]
    /// [Layer 2 neighbors: m * 8 bytes]
    /// ... 
    /// [Layer (max_layers-1) neighbors: m * 8 bytes]
    /// ```
    ///
    /// Total = 16 + (m0 * 8) + ((max_layers - 1) * m * 8), rounded up to 8-byte boundary
    #[must_use]
    pub const fn record_size(&self) -> usize {
        let header_size = NodeHeader::SIZE;

        // Layer 0 has m0 connections
        let layer0_size = self.m0 as usize * mem::size_of::<NodeId>();

        // Layers 1 to max_layers-1 each have m connections
        let upper_layers_size = if self.max_layers > 1 {
            (self.max_layers as usize - 1) * self.m as usize * mem:: size_of::<NodeId>()
        } else {
            0
        };

        let total = header_size + layer0_size + upper_layers_size;

        // Round up to 8-byte boundary
        (total + 7) & !7
    }

    /// Calculate the offset of a specific layer's neighbor array within a record.
    ///
    /// Returns `None` if the layer is out of bounds.
    #[must_use]
    pub const fn layer_offset(&self, layer: usize) -> Option<usize> {
        if layer >= self.max_layers as usize {
            return None;
        }

        let mut offset = NodeHeader::SIZE;

        if layer == 0 {
            return Some(offset);
        }

        // Skip layer 0
        offset += self.m0 as usize * mem::size_of::<NodeId>();

        // Skip layers 1 to layer-1
        offset += (layer - 1) * self.m as usize * mem::size_of::<NodeId>();

        Some(offset)
    }

    /// Get the maximum number of neighbors for a given layer. 
    #[must_use]
    pub const fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 {
            self. m0 as usize
        } else {
            self.m as usize
        }
    }

    /// Calculate the total maximum neighbors across all layers.
    #[must_use]
    pub const fn total_max_neighbors(&self) -> usize {
        let layer0 = self.m0 as usize;
        let upper = if self.max_layers > 1 {
            (self. max_layers as usize - 1) * self.m as usize
        } else {
            0
        };
        layer0 + upper
    }
}

/// Fixed-size node record that can be directly mapped to/from disk.
///
/// This struct represents the complete on-disk format of a node. 
/// The size is determined by `NodeRecordParams` at index creation. 
///
/// # Addressing Formula
///
/// ```text
/// node_offset = graph_start + (node_id * NODE_RECORD_SIZE)
/// ```
///
/// No hash maps or indirection required. 
#[derive(Debug)]
pub struct NodeRecord {
    /// Node header
    pub header: NodeHeader,

    /// Neighbor IDs for all layers (flattened).
    /// Layout: [layer0_neighbors... ][layer1_neighbors...][layer2_neighbors...]... 
    /// Unused slots contain `INVALID_NODE_ID`.
    pub neighbors: Vec<NodeId>,

    /// Parameters used to interpret the neighbor array
    params: NodeRecordParams,
}

impl NodeRecord {
    /// Create a new empty node record. 
    #[must_use]
    pub fn new(node_id: NodeId, layer_count: u8, params:  NodeRecordParams) -> Self {
        let total_slots = params.total_max_neighbors();
        let neighbors = vec![INVALID_NODE_ID; total_slots];

        Self {
            header: NodeHeader:: new(node_id, layer_count),
            neighbors,
            params,
        }
    }

    /// Get the fixed record size.
    #[must_use]
    pub const fn record_size(&self) -> usize {
        self.params.record_size()
    }

    /// Get neighbors at a specific layer.
    ///
    /// Returns only valid neighbors (excludes `INVALID_NODE_ID`).
    #[must_use]
    pub fn get_neighbors(&self, layer:  usize) -> Vec<NodeId> {
        if layer >= self.header.layer_count as usize {
            return Vec::new();
        }

        let (start, count) = self.layer_slice_bounds(layer);
        self.neighbors[start..start + count]
            .iter()
            .copied()
            .filter(|&id| id != INVALID_NODE_ID)
            .collect()
    }

    // In chassis-core/src/hnsw/node.rs, inside impl NodeRecord, after get_neighbors()
    
    /// Iterator over valid neighbors at a specific layer (zero-allocation).
    ///
    /// This is more efficient than `get_neighbors()` in hot paths like HNSW search
    /// because it avoids allocating a Vec.
    ///
    /// Returns an empty iterator if the layer is out of bounds.
    pub fn neighbors_iter(&self, layer: usize) -> impl Iterator<Item = NodeId> + '_ {
        let (start, count) = if layer >= self.header.layer_count as usize {
            (0, 0) // Empty iterator for out-of-bounds layer
        } else {
            self. layer_slice_bounds(layer)
        };
    
        self.neighbors[start..start + count]
            .iter()
            .copied()
            .filter(|&id| id != INVALID_NODE_ID)
    }
    /// Set neighbors for a specific layer.
    ///
    /// # Panics
    ///
    /// Panics if the layer is out of bounds or too many neighbors are provided.
    pub fn set_neighbors(&mut self, layer: usize, neighbors: &[NodeId]) {
        assert!(
            layer < self.params.max_layers as usize,
            "Layer {} exceeds max_layers {}",
            layer,
            self.params.max_layers
        );

        let max = self.params.max_neighbors(layer);
        assert!(
            neighbors.len() <= max,
            "Too many neighbors:  {} > {}",
            neighbors.len(),
            max
        );

        let (start, count) = self.layer_slice_bounds(layer);

        // Clear existing neighbors
        for slot in &mut self.neighbors[start..start + count] {
            *slot = INVALID_NODE_ID;
        }

        // Write new neighbors
        for (i, &neighbor) in neighbors.iter().enumerate() {
            self.neighbors[start + i] = neighbor;
        }
    }

    /// Add a neighbor to a specific layer if there's space.
    ///
    /// Returns `true` if the neighbor was added, `false` if the layer is full.
    pub fn add_neighbor(&mut self, layer: usize, neighbor: NodeId) -> bool {
        if layer >= self.params.max_layers as usize {
            return false;
        }

        let (start, count) = self.layer_slice_bounds(layer);

        // Find first empty slot
        for slot in &mut self.neighbors[start..start + count] {
            if *slot == INVALID_NODE_ID {
                *slot = neighbor;
                return true;
            }
        }

        false // Layer is full
    }

    /// Get the number of valid neighbors at a specific layer.
    #[must_use]
    pub fn neighbor_count(&self, layer: usize) -> usize {
        if layer >= self.header.layer_count as usize {
            return 0;
        }

        let (start, count) = self.layer_slice_bounds(layer);
        self.neighbors[start..start + count]
            .iter()
            .filter(|&&id| id != INVALID_NODE_ID)
            .count()
    }

    /// Get the slice bounds for a layer's neighbors.
    fn layer_slice_bounds(&self, layer: usize) -> (usize, usize) {
        let start = if layer == 0 {
            0
        } else {
            self.params.m0 as usize + (layer - 1) * self.params.m as usize
        };

        let count = self.params.max_neighbors(layer);

        (start, count)
    }

    /// Serialize the node record to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = self.params.record_size();
        let mut bytes = vec![0u8; size];

        // Write header
        let header_bytes:  &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&self.header as *const NodeHeader).cast::<u8>(),
                NodeHeader::SIZE,
            )
        };
        bytes[..NodeHeader::SIZE].copy_from_slice(header_bytes);

        // Write neighbors
        let mut offset = NodeHeader::SIZE;
        for &neighbor in &self.neighbors {
            bytes[offset..offset + 8].copy_from_slice(&neighbor.to_le_bytes());
            offset += 8;
        }

        bytes
    }

    /// Deserialize a node record from bytes. 
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Byte slice is too small
    /// - Header validation fails
    pub fn from_bytes(bytes: &[u8], params: NodeRecordParams) -> Result<Self, &'static str> {
        let expected_size = params.record_size();
        if bytes.len() < expected_size {
            return Err("Byte slice too small for node record");
        }

        let header = NodeHeader::from_bytes(bytes)?;

        // Read neighbors
        let total_slots = params.total_max_neighbors();
        let mut neighbors = Vec::with_capacity(total_slots);

        let mut offset = NodeHeader::SIZE;
        for _ in 0..total_slots {
            if offset + 8 > bytes.len() {
                return Err("Byte slice too small for neighbor data");
            }
            let neighbor = u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .map_err(|_| "Invalid neighbor bytes")?,
            );
            neighbors. push(neighbor);
            offset += 8;
        }

        Ok(Self {
            header,
            neighbors,
            params,
        })
    }
}

/// In-memory representation of a node (for building/manipulation).
///
/// This is a more ergonomic representation that doesn't require
/// knowing the fixed size ahead of time.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID (same as vector storage index)
    pub id: NodeId,

    /// File offset where this node is stored (for caching)
    pub offset: Offset,

    /// Neighbors per layer (layer 0, layer 1, ...)
    pub layers: Vec<Vec<NodeId>>,
}

impl Node {
    /// Create a new empty node.
    #[must_use]
    pub fn new(id: NodeId, layer_count: usize) -> Self {
        Self {
            id,
            offset: 0,
            layers:  vec![Vec::new(); layer_count],
        }
    }

    /// Returns the highest layer this node belongs to.
    #[must_use]
    pub fn top_layer(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Convert to a fixed-size `NodeRecord` for disk storage.
    #[must_use]
    pub fn to_record(&self, params: NodeRecordParams) -> NodeRecord {
        let mut record = NodeRecord::new(self.id, self.layers.len() as u8, params);

        for (layer, neighbors) in self.layers.iter().enumerate() {
            record.set_neighbors(layer, neighbors);
        }

        record
    }

    /// Create from a `NodeRecord`.
    #[must_use]
    pub fn from_record(record: &NodeRecord) -> Self {
        let layer_count = record.header.layer_count as usize;
        let mut layers = Vec::with_capacity(layer_count);

        for layer in 0..layer_count {
            layers.push(record. get_neighbors(layer));
        }

        Self {
            id: record.header.node_id,
            offset: 0,
            layers,
        }
    }
}

/// Compute the offset of a node given its ID. 
///
/// # Formula
///
/// ```text
/// node_offset = graph_start + (node_id * record_size)
/// ```
#[inline]
#[must_use]
pub const fn compute_node_offset(graph_start: Offset, node_id: NodeId, record_size: usize) -> Offset {
    graph_start + (node_id * record_size as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_header_size() {
        assert_eq!(NodeHeader::SIZE, 16, "NodeHeader must be exactly 16 bytes");
    }

    #[test]
    fn test_node_header_alignment() {
        assert_eq!(
            std::mem::align_of:: <NodeHeader>(),
            8,
            "NodeHeader must be 8-byte aligned"
        );
    }

    #[test]
    fn test_default_record_params() {
        let params = NodeRecordParams::default();
        assert_eq!(params.m, 16);
        assert_eq!(params.m0, 32);
        assert_eq!(params.max_layers, 16);
    }

    #[test]
    fn test_record_size_calculation() {
        let params = NodeRecordParams::new(16, 32, 4);

        // Header:  16 bytes
        // Layer 0: 32 * 8 = 256 bytes
        // Layers 1-3: 3 * 16 * 8 = 384 bytes
        // Total: 16 + 256 + 384 = 656 bytes (already 8-byte aligned)
        assert_eq!(params.record_size(), 656);
    }

    #[test]
    fn test_record_size_is_8_byte_aligned() {
        for m in [8, 12, 16, 32, 64] {
            for max_layers in [1, 4, 8, 16] {
                let params = NodeRecordParams::new(m, m * 2, max_layers);
                let size = params.record_size();
                assert_eq!(
                    size % 8,
                    0,
                    "Record size {} is not 8-byte aligned for m={}, max_layers={}",
                    size,
                    m,
                    max_layers
                );
            }
        }
    }

    #[test]
    fn test_layer_offset_calculation() {
        let params = NodeRecordParams::new(16, 32, 4);

        // Layer 0 starts after header
        assert_eq!(params.layer_offset(0), Some(16));

        // Layer 1 starts after header + layer0
        assert_eq!(params. layer_offset(1), Some(16 + 32 * 8));

        // Layer 2 starts after header + layer0 + layer1
        assert_eq!(params.layer_offset(2), Some(16 + 32 * 8 + 16 * 8));

        // Layer 3 starts after header + layer0 + layer1 + layer2
        assert_eq!(params.layer_offset(3), Some(16 + 32 * 8 + 16 * 8 + 16 * 8));

        // Layer 4 is out of bounds
        assert_eq!(params.layer_offset(4), None);
    }

    #[test]
    fn test_max_neighbors_per_layer() {
        let params = NodeRecordParams::new(16, 32, 4);

        assert_eq!(params.max_neighbors(0), 32); // m0
        assert_eq!(params.max_neighbors(1), 16); // m
        assert_eq!(params.max_neighbors(2), 16); // m
        assert_eq!(params.max_neighbors(3), 16); // m
    }

    #[test]
    fn test_total_max_neighbors() {
        let params = NodeRecordParams::new(16, 32, 4);

        // 32 (layer 0) + 3 * 16 (layers 1-3) = 80
        assert_eq!(params. total_max_neighbors(), 80);
    }

    #[test]
    fn test_node_record_creation() {
        let params = NodeRecordParams::new(16, 32, 4);
        let record = NodeRecord::new(42, 3, params);

        assert_eq!(record.header.node_id, 42);
        assert_eq!(record.header.layer_count, 3);
        assert_eq!(record.neighbors.len(), params.total_max_neighbors());

        // All neighbors should be invalid initially
        assert!(record.neighbors.iter().all(|&id| id == INVALID_NODE_ID));
    }

    #[test]
    fn test_set_and_get_neighbors() {
        let params = NodeRecordParams::new(16, 32, 4);
        let mut record = NodeRecord::new(0, 3, params);

        // Set layer 0 neighbors
        record. set_neighbors(0, &[1, 2, 3, 4, 5]);
        assert_eq!(record.get_neighbors(0), vec![1, 2, 3, 4, 5]);

        // Set layer 1 neighbors
        record.set_neighbors(1, &[10, 20]);
        assert_eq!(record. get_neighbors(1), vec![10, 20]);

        // Layer 2 should still be empty
        assert!(record. get_neighbors(2).is_empty());
    }

    #[test]
    fn test_add_neighbor() {
        let params = NodeRecordParams::new(2, 4, 2); // Small for testing
        let mut record = NodeRecord::new(0, 2, params);

        // Add neighbors to layer 0
        assert!(record.add_neighbor(0, 1));
        assert!(record.add_neighbor(0, 2));
        assert!(record.add_neighbor(0, 3));
        assert!(record. add_neighbor(0, 4));
        assert! (!record.add_neighbor(0, 5)); // Layer full

        assert_eq!(record.get_neighbors(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_neighbor_count() {
        let params = NodeRecordParams::new(16, 32, 4);
        let mut record = NodeRecord::new(0, 2, params);

        assert_eq!(record.neighbor_count(0), 0);

        record.set_neighbors(0, &[1, 2, 3]);
        assert_eq!(record. neighbor_count(0), 3);

        record.set_neighbors(1, &[10, 20, 30, 40, 50]);
        assert_eq!(record.neighbor_count(1), 5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let params = NodeRecordParams::new(16, 32, 4);
        let mut record = NodeRecord::new(123, 3, params);

        record.set_neighbors(0, &[1, 2, 3, 4, 5]);
        record.set_neighbors(1, &[10, 20]);
        record.set_neighbors(2, &[100]);

        let bytes = record.to_bytes();
        assert_eq!(bytes.len(), params.record_size());

        let restored = NodeRecord::from_bytes(&bytes, params).unwrap();

        assert_eq!(restored.header.node_id, 123);
        assert_eq!(restored. header.layer_count, 3);
        assert_eq!(restored.get_neighbors(0), vec![1, 2, 3, 4, 5]);
        assert_eq!(restored.get_neighbors(1), vec![10, 20]);
        assert_eq!(restored.get_neighbors(2), vec![100]);
    }

    #[test]
    fn test_node_to_record_conversion() {
        let params = NodeRecordParams::new(16, 32, 4);

        let mut node = Node::new(42, 3);
        node.layers[0] = vec![1, 2, 3];
        node.layers[1] = vec![10, 20];
        node.layers[2] = vec![100];

        let record = node.to_record(params);

        assert_eq!(record.header.node_id, 42);
        assert_eq!(record.get_neighbors(0), vec![1, 2, 3]);
        assert_eq!(record. get_neighbors(1), vec![10, 20]);
        assert_eq!(record.get_neighbors(2), vec![100]);

        let restored = Node::from_record(&record);
        assert_eq!(restored.id, 42);
        assert_eq!(restored.layers[0], vec![1, 2, 3]);
        assert_eq!(restored.layers[1], vec![10, 20]);
        assert_eq!(restored.layers[2], vec![100]);
    }

    #[test]
    fn test_compute_node_offset() {
        let graph_start:  Offset = 8192; // 2 pages
        let record_size = 656; // From earlier test

        assert_eq!(compute_node_offset(graph_start, 0, record_size), 8192);
        assert_eq!(compute_node_offset(graph_start, 1, record_size), 8192 + 656);
        assert_eq!(compute_node_offset(graph_start, 2, record_size), 8192 + 1312);
        assert_eq!(
            compute_node_offset(graph_start, 100, record_size),
            8192 + 100 * 656
        );
    }

    #[test]
    fn test_node_header_deleted_flag() {
        let mut header = NodeHeader::new(0, 1);

        assert! (!header.is_deleted());

        header.set_deleted();
        assert!(header.is_deleted());
    }

    #[test]
    fn test_addressing_formula_consistency() {
        // Verify that the addressing formula works for various node IDs
        let params = NodeRecordParams::default();
        let graph_start:  Offset = 4096;
        let record_size = params.record_size();

        for node_id in [0, 1, 10, 100, 1000, 10000] {
            let offset = compute_node_offset(graph_start, node_id, record_size);

            // Verify the offset is what we expect
            assert_eq!(offset, graph_start + (node_id * record_size as u64));

            // Verify we can compute the node_id back from the offset
            let computed_id = (offset - graph_start) / record_size as u64;
            assert_eq!(computed_id, node_id);
        }
    }

    #[test]
    fn test_single_layer_record() {
        let params = NodeRecordParams::new(16, 32, 1);

        // Only layer 0 with m0 connections
        assert_eq!(params.record_size(), 16 + 32 * 8); // 272 bytes
        assert_eq!(params.total_max_neighbors(), 32);
        assert_eq!(params.layer_offset(0), Some(16));
        assert_eq!(params.layer_offset(1), None);
    }

    #[test]
    fn test_record_size_stability() {
        // Once params are set, record size must be stable
        let params = NodeRecordParams::new(16, 32, 8);
        let size = params.record_size();

        // Create records with different layer counts - size should be the same
        for layer_count in 1..=8 {
            let record = NodeRecord::new(0, layer_count, params);
            assert_eq!(
                record.record_size(),
                size,
                "Record size should be constant regardless of layer_count"
            );
        }
    }
}
