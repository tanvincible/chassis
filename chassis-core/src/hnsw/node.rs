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
#![cfg_attr(not(any(test, feature = "internals")), allow(dead_code, unused_imports))]

use std::mem;

/// File offset type (u64 for 64-bit addressing)
pub type Offset = u64;

/// Node ID (index into vector storage)
pub type NodeId = u64;

/// Sentinel value indicating an empty neighbor slot
pub const INVALID_NODE_ID: NodeId = NodeId::MAX;

/// Default maximum connections per layer (M parameter)
pub const DEFAULT_M: u16 = 16;

/// Default maximum layers in the graph
pub const DEFAULT_MAX_LAYERS: u8 = 16;

/// Maximum connections at layer 0 (typically 2*M)
pub const DEFAULT_M0: u16 = DEFAULT_M * 2;

/// A node's header in a v1 or v2 file.
///
/// # Layout (16 bytes, 8-byte aligned)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       8     id: u64 (the caller's id; equals the node's slot in files without custom ids)
/// 8       1     layer_count: u8 (highest layer this node belongs to + 1)
/// 9       1     flags: u8 (reserved)
/// 10      2     _padding
/// 12      4     deleted_epoch: u32 (flush epoch that deleted the node, 0 while live)
/// ```
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct NodeHeader {
    /// The caller's id for this node
    pub id: u64,

    /// Number of layers this node participates in (1 = layer 0 only)
    pub layer_count: u8,

    /// Reserved for future use
    pub flags: u8,

    _padding: [u8; 2],

    /// Flush epoch that deleted this node, or 0 while it is live (ADR-0007)
    pub deleted_epoch: u32,
}

impl NodeHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    #[must_use]
    pub const fn new(id: u64, layer_count: u8) -> Self {
        Self { id, layer_count, flags: 0, _padding: [0; 2], deleted_epoch: 0 }
    }

    /// Reads a header, refusing a short slice, a `layer_count` of 0 and an `INVALID_NODE_ID` id.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < Self::SIZE {
            return Err("Buffer too small for NodeHeader");
        }

        // SAFETY: `bytes` holds at least `SIZE` bytes, any bytes make a header of plain integers,
        // and `read_unaligned` needs no alignment.
        let header = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) };

        if header.layer_count == 0 {
            return Err("Invalid NodeHeader:  layer_count cannot be 0");
        }

        if header.id == INVALID_NODE_ID {
            return Err("Invalid NodeHeader: id is the INVALID_NODE_ID sentinel");
        }

        Ok(header)
    }

    /// `from_bytes` without its checks.
    ///
    /// # Safety
    ///
    /// `bytes` must hold at least `SIZE` bytes, of a header written before.
    #[inline]
    pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() >= Self::SIZE, "Buffer too small for NodeHeader");
        // SAFETY: as in `from_bytes`; the caller guarantees the length.
        unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) }
    }

    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.deleted_epoch != 0
    }
}

/// Neighbors per layer and the most layers, fixed when the index is created.
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
        Self { m: DEFAULT_M, m0: DEFAULT_M0, max_layers: DEFAULT_MAX_LAYERS }
    }
}

impl NodeRecordParams {
    #[must_use]
    pub const fn new(m: u16, m0: u16, max_layers: u8) -> Self {
        Self { m, m0, max_layers }
    }

    /// Bytes of a record in a v1 or v2 file: the header, then `m0` neighbors on layer 0 and `m` on
    /// each layer above, 8 bytes each, rounded up to a multiple of 8.
    #[must_use]
    pub const fn record_size(&self) -> usize {
        let header_size = NodeHeader::SIZE;

        let layer0_size = self.m0 as usize * mem::size_of::<NodeId>();

        let upper_layers_size = if self.max_layers > 1 {
            (self.max_layers as usize - 1) * self.m as usize * mem::size_of::<NodeId>()
        } else {
            0
        };

        let total = header_size + layer0_size + upper_layers_size;

        (total + 7) & !7
    }

    /// Where `layer`'s neighbors start in a record; `None` past the last layer.
    #[must_use]
    pub const fn layer_offset(&self, layer: usize) -> Option<usize> {
        if layer >= self.max_layers as usize {
            return None;
        }

        let mut offset = NodeHeader::SIZE;

        if layer == 0 {
            return Some(offset);
        }

        offset += self.m0 as usize * mem::size_of::<NodeId>();

        offset += (layer - 1) * self.m as usize * mem::size_of::<NodeId>();

        Some(offset)
    }

    #[must_use]
    pub const fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 as usize } else { self.m as usize }
    }

    #[must_use]
    pub const fn total_max_neighbors(&self) -> usize {
        let layer0 = self.m0 as usize;
        let upper =
            if self.max_layers > 1 { (self.max_layers as usize - 1) * self.m as usize } else { 0 };
        layer0 + upper
    }
}

/// A node's record, which a v1 or v2 file keeps `slot × record_size` bytes into its graph.
#[derive(Debug)]
pub struct NodeRecord {
    /// The slot this record belongs to. Not stored on disk; reads and writes are addressed by it.
    pub slot: NodeId,

    pub header: NodeHeader,

    /// Neighbor IDs for all layers (flattened): layer 0 slots, then layer 1, then layer 2, and so on.
    /// Unused slots contain `INVALID_NODE_ID`.
    pub neighbors: Vec<NodeId>,

    params: NodeRecordParams,
}

impl NodeRecord {
    /// Create a new empty node record at `slot`, with the slot as its id.
    #[must_use]
    pub fn new(slot: NodeId, layer_count: u8, params: NodeRecordParams) -> Self {
        let total_slots = params.total_max_neighbors();
        let neighbors = vec![INVALID_NODE_ID; total_slots];

        Self { slot, header: NodeHeader::new(slot, layer_count), neighbors, params }
    }

    #[must_use]
    pub const fn record_size(&self) -> usize {
        self.params.record_size()
    }

    /// The neighbors at `layer`, without the empty entries.
    #[must_use]
    pub fn get_neighbors(&self, layer: usize) -> Vec<NodeId> {
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

    /// `get_neighbors` without allocating.
    pub fn neighbors_iter(&self, layer: usize) -> impl Iterator<Item = NodeId> + '_ {
        let (start, count) = if layer >= self.header.layer_count as usize {
            (0, 0)
        } else {
            self.layer_slice_bounds(layer)
        };

        self.neighbors[start..start + count].iter().copied().filter(|&id| id != INVALID_NODE_ID)
    }
    /// Replaces the neighbors at `layer`. Panics past the last layer, or with more than fit.
    pub fn set_neighbors(&mut self, layer: usize, neighbors: &[NodeId]) {
        assert!(
            layer < self.params.max_layers as usize,
            "Layer {} exceeds max_layers {}",
            layer,
            self.params.max_layers
        );

        let max = self.params.max_neighbors(layer);
        assert!(neighbors.len() <= max, "Too many neighbors:  {} > {}", neighbors.len(), max);

        let (start, count) = self.layer_slice_bounds(layer);

        for slot in &mut self.neighbors[start..start + count] {
            *slot = INVALID_NODE_ID;
        }

        for (i, &neighbor) in neighbors.iter().enumerate() {
            self.neighbors[start + i] = neighbor;
        }
    }

    /// Adds a neighbor at `layer` if there is room, and says whether there was.
    pub fn add_neighbor(&mut self, layer: usize, neighbor: NodeId) -> bool {
        if layer >= self.params.max_layers as usize {
            return false;
        }

        let (start, count) = self.layer_slice_bounds(layer);

        for slot in &mut self.neighbors[start..start + count] {
            if *slot == INVALID_NODE_ID {
                *slot = neighbor;
                return true;
            }
        }

        false
    }

    #[must_use]
    pub fn neighbor_count(&self, layer: usize) -> usize {
        if layer >= self.header.layer_count as usize {
            return 0;
        }

        let (start, count) = self.layer_slice_bounds(layer);
        self.neighbors[start..start + count].iter().filter(|&&id| id != INVALID_NODE_ID).count()
    }

    /// Where `layer`'s entries start in `neighbors`, and how many it has.
    fn layer_slice_bounds(&self, layer: usize) -> (usize, usize) {
        let start = if layer == 0 {
            0
        } else {
            self.params.m0 as usize + (layer - 1) * self.params.m as usize
        };

        let count = self.params.max_neighbors(layer);

        (start, count)
    }

    /// The record as a v1 or v2 file keeps it.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = self.params.record_size();
        let mut bytes = vec![0u8; size];

        // SAFETY: the header is `repr(C)` with its padding as a field, so all of its `SIZE` bytes
        // are initialized.
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&self.header as *const NodeHeader).cast::<u8>(),
                NodeHeader::SIZE,
            )
        };
        bytes[..NodeHeader::SIZE].copy_from_slice(header_bytes);

        let mut offset = NodeHeader::SIZE;
        for &neighbor in &self.neighbors {
            bytes[offset..offset + 8].copy_from_slice(&neighbor.to_le_bytes());
            offset += 8;
        }

        bytes
    }

    /// Reads a record from a v1 or v2 file, refusing a short slice, a header `NodeHeader` refuses
    /// and more layers than `params` allows.
    pub fn from_bytes(bytes: &[u8], params: NodeRecordParams) -> Result<Self, &'static str> {
        let expected_size = params.record_size();
        if bytes.len() < expected_size {
            return Err("Byte slice too small for node record");
        }

        let header = NodeHeader::from_bytes(bytes)?;
        if header.layer_count > params.max_layers {
            return Err("Invalid NodeRecord: layer_count exceeds max_layers");
        }

        let total_slots = params.total_max_neighbors();
        let mut neighbors = Vec::with_capacity(total_slots);

        let mut offset = NodeHeader::SIZE;
        for _ in 0..total_slots {
            if offset + 8 > bytes.len() {
                return Err("Byte slice too small for neighbor data");
            }
            let neighbor = u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().map_err(|_| "Invalid neighbor bytes")?,
            );
            neighbors.push(neighbor);
            offset += 8;
        }

        Ok(Self { slot: header.id, header, neighbors, params })
    }
}

/// A node's lists, one `Vec` a layer, for building a record.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID (same as vector storage index)
    pub id: NodeId,

    /// Unused: always 0.
    pub offset: Offset,

    /// Neighbors per layer (layer 0, layer 1, ...)
    pub layers: Vec<Vec<NodeId>>,
}

impl Node {
    #[must_use]
    pub fn new(id: NodeId, layer_count: usize) -> Self {
        Self { id, offset: 0, layers: vec![Vec::new(); layer_count] }
    }

    #[must_use]
    pub fn top_layer(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    #[must_use]
    pub fn to_record(&self, params: NodeRecordParams) -> NodeRecord {
        let mut record = NodeRecord::new(self.id, self.layers.len() as u8, params);

        for (layer, neighbors) in self.layers.iter().enumerate() {
            record.set_neighbors(layer, neighbors);
        }

        record
    }

    #[must_use]
    pub fn from_record(record: &NodeRecord) -> Self {
        let layer_count = record.header.layer_count as usize;
        let mut layers = Vec::with_capacity(layer_count);

        for layer in 0..layer_count {
            layers.push(record.get_neighbors(layer));
        }

        Self { id: record.slot, offset: 0, layers }
    }
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
        assert_eq!(std::mem::align_of::<NodeHeader>(), 8, "NodeHeader must be 8-byte aligned");
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
        assert_eq!(params.layer_offset(1), Some(16 + 32 * 8));

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
        assert_eq!(params.total_max_neighbors(), 80);
    }

    #[test]
    fn test_node_record_creation() {
        let params = NodeRecordParams::new(16, 32, 4);
        let record = NodeRecord::new(42, 3, params);

        assert_eq!(record.header.id, 42);
        assert_eq!(record.header.layer_count, 3);
        assert_eq!(record.neighbors.len(), params.total_max_neighbors());

        assert!(record.neighbors.iter().all(|&id| id == INVALID_NODE_ID));
    }

    #[test]
    fn test_set_and_get_neighbors() {
        let params = NodeRecordParams::new(16, 32, 4);
        let mut record = NodeRecord::new(0, 3, params);

        record.set_neighbors(0, &[1, 2, 3, 4, 5]);
        assert_eq!(record.get_neighbors(0), vec![1, 2, 3, 4, 5]);

        record.set_neighbors(1, &[10, 20]);
        assert_eq!(record.get_neighbors(1), vec![10, 20]);

        assert!(record.get_neighbors(2).is_empty());
    }

    #[test]
    fn test_add_neighbor() {
        let params = NodeRecordParams::new(2, 4, 2); // Small for testing
        let mut record = NodeRecord::new(0, 2, params);

        assert!(record.add_neighbor(0, 1));
        assert!(record.add_neighbor(0, 2));
        assert!(record.add_neighbor(0, 3));
        assert!(record.add_neighbor(0, 4));
        assert!(!record.add_neighbor(0, 5));

        assert_eq!(record.get_neighbors(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_neighbor_count() {
        let params = NodeRecordParams::new(16, 32, 4);
        let mut record = NodeRecord::new(0, 2, params);

        assert_eq!(record.neighbor_count(0), 0);

        record.set_neighbors(0, &[1, 2, 3]);
        assert_eq!(record.neighbor_count(0), 3);

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

        assert_eq!(restored.header.id, 123);
        assert_eq!(restored.header.layer_count, 3);
        assert_eq!(restored.get_neighbors(0), vec![1, 2, 3, 4, 5]);
        assert_eq!(restored.get_neighbors(1), vec![10, 20]);
        assert_eq!(restored.get_neighbors(2), vec![100]);
    }

    #[test]
    fn test_node_record_rejects_layer_count_above_params() {
        let params = NodeRecordParams::new(16, 32, 4);
        let record = NodeRecord::new(123, 4, params);
        let mut bytes = record.to_bytes();

        bytes[8] = 5;

        let result = NodeRecord::from_bytes(&bytes, params);
        assert_eq!(result.unwrap_err(), "Invalid NodeRecord: layer_count exceeds max_layers");
    }

    #[test]
    fn test_node_to_record_conversion() {
        let params = NodeRecordParams::new(16, 32, 4);

        let mut node = Node::new(42, 3);
        node.layers[0] = vec![1, 2, 3];
        node.layers[1] = vec![10, 20];
        node.layers[2] = vec![100];

        let record = node.to_record(params);

        assert_eq!(record.header.id, 42);
        assert_eq!(record.get_neighbors(0), vec![1, 2, 3]);
        assert_eq!(record.get_neighbors(1), vec![10, 20]);
        assert_eq!(record.get_neighbors(2), vec![100]);

        let restored = Node::from_record(&record);
        assert_eq!(restored.id, 42);
        assert_eq!(restored.layers[0], vec![1, 2, 3]);
        assert_eq!(restored.layers[1], vec![10, 20]);
        assert_eq!(restored.layers[2], vec![100]);
    }

    #[test]
    fn test_node_header_deleted_flag() {
        let mut header = NodeHeader::new(0, 1);
        assert!(!header.is_deleted());

        header.deleted_epoch = 3;
        assert!(header.is_deleted());
        assert_eq!(std::mem::offset_of!(NodeHeader, deleted_epoch), 12);
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
