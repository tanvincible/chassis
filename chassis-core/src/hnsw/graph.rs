//!  HNSW Graph with persistent storage and zero-allocation neighbor iteration.
//!
//! # Design Principles
//!
//! - **O(1) node addressing**: `offset = node_records_start + (node_id * record_size)`
//! - **No HashMap for offsets**: Deterministic formula eliminates memory overhead
//! - **Zero-allocation iteration**: `neighbors_iter_from_mmap` reads directly from mmap
//! - **Persistent header**: Entry point and max layer survive restarts

use crate::Storage;
use crate::hnsw::HnswParams;
use crate::hnsw::node::{
    INVALID_NODE_ID, Node, NodeHeader, NodeId, NodeRecord, NodeRecordParams, Offset,
};
use anyhow::Result;

/// Size of the graph header in bytes
const GRAPH_HEADER_SIZE: usize = 64;

/// Persistent graph header stored at the beginning of the graph zone.
///
/// # Layout (64 bytes, 8-byte aligned)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       4     magic:  [u8; 4] (b"HNSW")
/// 4       4     version: u32
/// 8       8     entry_point: NodeId (u64)
/// 16      8     node_count: u64
/// 24      4     max_layer: u32
/// 28      2     m: u16
/// 30      2     m0: u16
/// 32      1     max_layers: u8
/// 33      31    _reserved: [u8; 31]
/// Total:  64 bytes
/// ```
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct GraphHeader {
    /// Magic bytes for validation (b"HNSW")
    magic: [u8; 4],

    /// Format version
    version: u32,

    /// Entry point node ID (INVALID_NODE_ID if empty)
    pub entry_point: NodeId, // u64 - naturally aligned at offset 8

    /// Number of nodes written  
    pub node_count: u64, // u64 - naturally aligned at offset 16

    /// Highest layer in graph
    pub max_layer: u32, // u32 at offset 24

    /// Max connections per layer
    pub m: u16, // u16 at offset 28

    /// Max connections at layer 0
    pub m0: u16, // u16 at offset 30

    /// Maximum layers
    pub max_layers: u8, // u8 at offset 32

    /// Padding to 64 bytes
    _reserved: [u8; 31], // 31 bytes:  offset 33-63
}

impl GraphHeader {
    /// Magic bytes for graph header validation
    const MAGIC: &'static [u8; 4] = b"HNSW";

    /// Current format version
    const VERSION: u32 = 1;

    /// Create a new empty graph header
    #[must_use]
    pub fn new(params: NodeRecordParams) -> Self {
        Self {
            magic: *Self::MAGIC,
            version: Self::VERSION,
            entry_point: INVALID_NODE_ID,
            node_count: 0,
            max_layer: 0,
            m: params.m,
            m0: params.m0,
            max_layers: params.max_layers,
            _reserved: [0; 31],
        }
    }

    /// Validate magic bytes and version
    pub fn is_valid(&self) -> bool {
        self.magic == *Self::MAGIC && self.version == Self::VERSION
    }

    /// Convert to bytes for writing
    #[must_use]
    pub fn to_bytes(&self) -> [u8; GRAPH_HEADER_SIZE] {
        let mut bytes = [0u8; GRAPH_HEADER_SIZE];

        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.entry_point.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.node_count.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.max_layer.to_le_bytes());
        bytes[28..30].copy_from_slice(&self.m.to_le_bytes());
        bytes[30..32].copy_from_slice(&self.m0.to_le_bytes());
        bytes[32] = self.max_layers;
        bytes[33..64].copy_from_slice(&self._reserved);

        bytes
    }

    /// Read from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < GRAPH_HEADER_SIZE {
            anyhow::bail!("Buffer too small for graph header");
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        let version = u32::from_le_bytes(bytes[4..8].try_into()?);
        let entry_point = u64::from_le_bytes(bytes[8..16].try_into()?);
        let node_count = u64::from_le_bytes(bytes[16..24].try_into()?);
        let max_layer = u32::from_le_bytes(bytes[24..28].try_into()?);
        let m = u16::from_le_bytes(bytes[28..30].try_into()?);
        let m0 = u16::from_le_bytes(bytes[30..32].try_into()?);
        let max_layers = bytes[32];

        let mut reserved = [0u8; 31];
        reserved.copy_from_slice(&bytes[33..64]);

        Ok(Self {
            magic,
            version,
            entry_point,
            node_count,
            max_layer,
            m,
            m0,
            max_layers,
            _reserved: reserved,
        })
    }

    /// Get record params from header
    #[must_use]
    pub const fn to_record_params(&self) -> NodeRecordParams {
        NodeRecordParams::new(self.m, self.m0, self.max_layers)
    }
}

/// HNSW graph stored in segmented layout:
/// [Vector Storage] | [Graph Header] | [Node Data]
///
/// # Key Design Decisions
///
/// - **No HashMap for node offsets**: Uses O(1) formula `compute_node_offset()`
/// - **Persistent header**: Entry point, max layer, and node count survive restarts
/// - **Zero-allocation neighbor iteration**: `neighbors_iter_from_mmap()` for hot paths
#[derive(Debug)]
pub struct HnswGraph {
    pub(crate) storage: Storage,

    #[allow(dead_code)]
    params: HnswParams,

    /// Cached record parameters for O(1) lookup
    pub record_params: NodeRecordParams,

    /// Offset where graph section begins (includes header)
    graph_start: Offset,

    /// Entry point node ID (highest layer node)
    pub entry_point: Option<NodeId>,

    /// Maximum layer in the graph
    pub max_layer: usize,

    /// Number of nodes in the graph (tracked for header persistence)
    pub node_count: u64,
}

impl HnswGraph {
    /// Opens existing graph or creates new one
    pub fn open(mut storage: Storage, params: HnswParams) -> Result<Self> {
        let graph_start = Self::find_or_create_graph_start(&storage)?;
        let record_params = params.to_record_params();

        // Ensure graph zone has space for header
        let header_end = graph_start as usize + GRAPH_HEADER_SIZE;
        storage.ensure_graph_capacity(header_end)?;

        // Try to read existing header
        let (entry_point, max_layer, node_count) =
            match Self::try_read_graph_header(&storage, graph_start, record_params) {
                Ok(header) => {
                    // Existing graph found
                    let entry_point = if header.entry_point == INVALID_NODE_ID {
                        None
                    } else {
                        Some(header.entry_point)
                    };
                    (entry_point, header.max_layer as usize, header.node_count)
                }
                Err(_) => {
                    // New graph - initialize header
                    let header = GraphHeader::new(record_params);
                    let bytes = header.to_bytes();
                    let zone = storage.graph_zone_mut(graph_start as usize, GRAPH_HEADER_SIZE)?;
                    zone.copy_from_slice(&bytes);
                    (None, 0, 0)
                }
            };

        Ok(Self { storage, params, record_params, graph_start, entry_point, max_layer, node_count })
    }

    /// Try to read graph header if it exists
    fn try_read_graph_header(
        storage: &Storage,
        graph_start: Offset,
        expected_params: NodeRecordParams,
    ) -> Result<GraphHeader> {
        let zone = storage.graph_zone(graph_start as usize, GRAPH_HEADER_SIZE)?;
        let header = GraphHeader::from_bytes(zone)?;

        if !header.is_valid() {
            anyhow::bail!("Invalid graph header magic or version");
        }

        // Verify params match
        let header_params = header.to_record_params();
        if header_params != expected_params {
            anyhow::bail!(
                "Graph header params mismatch: expected {:?}, got {:?}",
                expected_params,
                header_params
            );
        }

        Ok(header)
    }

    /// Read graph header from mmap
    pub fn read_graph_header(&self) -> Result<GraphHeader> {
        let zone = self.storage.graph_zone(self.graph_start as usize, GRAPH_HEADER_SIZE)?;
        let header = GraphHeader::from_bytes(zone)?;

        if !header.is_valid() {
            anyhow::bail!("Invalid graph header");
        }

        Ok(header)
    }

    /// Write graph header to mmap
    pub fn write_graph_header(&mut self) -> Result<()> {
        let mut header = GraphHeader::new(self.record_params);
        header.entry_point = self.entry_point.unwrap_or(INVALID_NODE_ID);
        header.max_layer = self.max_layer as u32;
        header.node_count = self.node_count;

        let bytes = header.to_bytes();
        let zone = self.storage.graph_zone_mut(self.graph_start as usize, GRAPH_HEADER_SIZE)?;
        zone.copy_from_slice(&bytes);

        Ok(())
    }

    /// Compute the file offset for a node record.
    ///
    /// # Centralized Offset Computation
    ///
    /// **ALL** node read/write operations MUST use this method.
    /// This ensures:
    /// - Consistent addressing formula across all code paths
    /// - Single point of change for layout modifications
    /// - No off-by-one bugs from manual offset arithmetic
    ///
    /// # Formula
    ///
    /// ```text
    /// offset = graph_start + GRAPH_HEADER_SIZE + (node_id * record_size)
    /// ```
    #[inline]
    pub(crate) fn node_offset(&self, node_id: NodeId) -> Offset {
        let base = self.graph_start + GRAPH_HEADER_SIZE as u64;
        base + (node_id * self.record_params.record_size() as u64)
    }

    /// Read a node record directly from mmap.
    pub fn read_node_record(&self, node_id: NodeId) -> Result<NodeRecord> {
        let record_size = self.record_params.record_size();
        let offset = self.node_offset(node_id);

        let zone = self.storage.graph_zone(offset as usize, record_size)?;
        NodeRecord::from_bytes(zone, self.record_params)
            .map_err(|e| anyhow::anyhow!("Failed to read node record: {}", e))
    }

    /// Write a node record directly to mmap.
    pub fn write_node_record(&mut self, record: &NodeRecord) -> Result<()> {
        let record_size = self.record_params.record_size();
        let offset = self.node_offset(record.header.node_id);

        let required_size = offset as usize + record_size;
        self.storage.ensure_graph_capacity(required_size)?;

        let bytes = record.to_bytes();
        let zone = self.storage.graph_zone_mut(offset as usize, record_size)?;
        zone.copy_from_slice(&bytes);

        Ok(())
    }

    /// Get raw bytes slice for a node (zero-copy).
    pub fn get_node_bytes(&self, node_id: NodeId) -> Result<&[u8]> {
        let record_size = self.record_params.record_size();
        let offset = self.node_offset(node_id);

        self.storage.graph_zone(offset as usize, record_size)
    }

    /// Iterate neighbors directly from mmap bytes (zero-allocation).
    ///
    /// This is the **preferred method** for search hot paths because it:
    /// - Does NOT allocate any `Vec`
    /// - Reads directly from memory-mapped bytes
    /// - Filters out `INVALID_NODE_ID` entries lazily
    ///
    /// # Example
    ///
    /// ```ignore
    /// for neighbor_id in graph.neighbors_iter_from_mmap(node_id, layer)? {
    ///     // Process neighbor without allocation
    /// }
    /// ```
    /// Iterate neighbors directly from mmap bytes (zero-allocation).
    ///
    /// This is the **preferred method** for search hot paths because it:
    /// - Does NOT allocate any `Vec`
    /// - Reads directly from memory-mapped bytes
    /// - Filters out `INVALID_NODE_ID` entries lazily
    pub fn neighbors_iter_from_mmap(
        &self,
        node_id: NodeId,
        layer: usize,
    ) -> Result<impl Iterator<Item = NodeId> + '_> {
        let bytes = self.get_node_bytes(node_id)?;

        // Validated header read with alignment handling
        let header = NodeHeader::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Invalid node header for node {}: {}", node_id, e))?;

        // Check if layer is valid
        if layer >= header.layer_count as usize {
            return Ok(NeighborIterator { bytes: &[], start_offset: 0, count: 0, pos: 0 });
        }

        let layer_offset = self
            .record_params
            .layer_offset(layer)
            .ok_or_else(|| anyhow::anyhow!("Invalid layer"))?;
        let neighbor_count = self.record_params.max_neighbors(layer);

        Ok(NeighborIterator { bytes, start_offset: layer_offset, count: neighbor_count, pos: 0 })
    }

    /// Compute distance using zero-copy vector slice access.
    ///
    /// This is the **preferred method** for search because it:
    /// - Does NOT allocate a `Vec<f32>` for the vector
    /// - Reads directly from memory-mapped storage
    #[inline]
    pub fn compute_distance_zero_copy(&self, query: &[f32], node_id: NodeId) -> Result<f32> {
        let vector_slice = self.storage.get_vector_slice(node_id)?;
        Ok(crate::distance::euclidean_distance(query, vector_slice))
    }

    /// Commit graph state (write header and flush to disk).
    ///
    /// # Performance Warning
    ///
    /// This method performs a full fsync which can take 1-50ms depending on
    /// the storage device. **Do NOT call this after every node insertion**.
    /// Instead, batch insertions and call `commit()` once at the end.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // BAD:  1000x slower due to fsync per insert
    /// for vector in vectors {
    ///     graph.insert(vector)?;
    ///     graph. commit()?;  // DON'T DO THIS
    /// }
    ///
    /// // GOOD: Single fsync at the end
    /// for vector in vectors {
    ///     graph.insert(vector)?;
    /// }
    /// graph.commit()?;  // Once at the end
    /// ```
    pub fn commit(&mut self) -> Result<()> {
        self.write_graph_header()?;
        self.storage.commit()
    }

    /// Finds where graph data starts in file
    fn find_or_create_graph_start(storage: &Storage) -> Result<Offset> {
        // FIXED: Use a Sparse Layout.
        // Reserve the first 1GB for vectors.
        // The OS will not allocate physical disk space for the empty gap (sparse file).
        const GRAPH_ZONE_START: u64 = 1024 * 1024 * 1024; // 1 GiB

        // Safety check: Ensure we haven't already overflowed the vector zone
        let vector_count = storage.count();
        let vector_size = storage.dimensions() as usize * std::mem::size_of::<f32>();
        let vector_zone_current_end =
            crate::header::HEADER_SIZE + (vector_count as usize * vector_size);

        if vector_zone_current_end as u64 > GRAPH_ZONE_START {
            anyhow::bail!(
                "Vector storage exceeded 1GB limit. Migration required. (Current: {} bytes)",
                vector_zone_current_end
            );
        }

        Ok(GRAPH_ZONE_START)
    }

    /// Inserts a new node into the graph.
    ///
    /// # Node ID Invariant
    ///
    /// **Node IDs MUST be dense and monotonically increasing.**
    ///
    /// This means:
    /// - First node must have `vector_id = 0`
    /// - Each subsequent node must have `vector_id = previous + 1`
    /// - No gaps, no out-of-order insertions, no overwrites
    ///
    /// This invariant is required because:
    /// - O(1) addressing formula assumes `offset = base + (node_id * record_size)`
    /// - `node_count` is used to track graph size for persistence
    /// - Sparse or out-of-order IDs would corrupt the offset calculation
    ///
    /// # Panics (Debug Mode)
    ///
    /// In debug builds, panics if `vector_id != self.node_count` (invariant violation).
    ///
    /// # Errors
    ///
    /// Returns an error if the node cannot be written to storage.
    ///
    /// # Note
    ///
    /// This method does NOT call `commit()`. For durability, call `commit()`
    /// after inserting a batch of nodes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // CORRECT: Dense, monotonic insertion
    /// graph.insert(0, layer_0)?;
    /// graph.insert(1, layer_1)?;
    /// graph.insert(2, layer_2)?;
    ///
    /// // WRONG: Out of order (will panic in debug)
    /// graph.insert(0, layer_0)?;
    /// graph.insert(5, layer_1)?;  // Gap!  Should be 1
    ///
    /// // WRONG: Overwrite (will panic in debug)
    /// graph.insert(0, layer_0)?;
    /// graph.insert(0, layer_1)?;  // Duplicate! Should be 1
    /// ```
    pub fn insert(&mut self, vector_id: NodeId, layer: usize) -> Result<()> {
        // Enforce dense, monotonic node ID invariant
        debug_assert!(
            vector_id == self.node_count,
            "Node ID invariant violated: expected {}, got {}.  \
             Node IDs must be dense and monotonically increasing (0, 1, 2, ...).",
            self.node_count,
            vector_id
        );

        // In release mode, check and return error instead of UB
        if vector_id != self.node_count {
            anyhow::bail!(
                "Node ID invariant violated: expected {}, got {}. \
                 Node IDs must be dense and monotonically increasing.",
                self.node_count,
                vector_id
            );
        }

        let node = Node { id: vector_id, offset: 0, layers: vec![Vec::new(); layer + 1] };

        // TODO: Connect to neighbors using HNSW algorithm

        self.write_node(&node)?;

        if self.entry_point.is_none() || layer > self.max_layer {
            self.entry_point = Some(vector_id);
            self.max_layer = layer;
        }

        Ok(())
    }

    /// Writes node to disk using O(1) addressing.
    ///
    /// # Invariant
    ///
    /// Caller MUST ensure `node.id == self.node_count` before calling.
    /// This is enforced by `insert()`.
    fn write_node(&mut self, node: &Node) -> Result<Offset> {
        // Double-check invariant (belt and suspenders)
        debug_assert!(node.id == self.node_count, "write_node called with non-sequential node ID");

        // Convert to fixed-size record and write directly to mmap
        let record = node.to_record(self.record_params);
        self.write_node_record(&record)?;

        // Increment count AFTER successful write (crash safety)
        self.node_count += 1;

        Ok(self.node_offset(node.id))
    }

    /// Update an existing node's neighbors (does NOT increment node_count).
    ///
    /// Use this for:
    /// - Updating neighbor connections during HNSW construction
    /// - Modifying an existing node's links
    ///
    /// # Panics (Debug Mode)
    ///
    /// Panics if `node_id >= self.node_count` (node doesn't exist).
    pub fn update_node_record(&mut self, record: &NodeRecord) -> Result<()> {
        let node_id = record.header.node_id;

        debug_assert!(
            node_id < self.node_count,
            "Cannot update non-existent node:  {} (node_count = {})",
            node_id,
            self.node_count
        );

        if node_id >= self.node_count {
            anyhow::bail!(
                "Cannot update non-existent node: {} (node_count = {})",
                node_id,
                self.node_count
            );
        }

        // Write without incrementing count
        let record_size = self.record_params.record_size();
        let offset = self.node_offset(node_id);

        let bytes = record.to_bytes();
        let zone = self.storage.graph_zone_mut(offset as usize, record_size)?;
        zone.copy_from_slice(&bytes);

        Ok(())
    }

    /// Returns total size of graph data written so far
    #[allow(dead_code)]
    fn total_graph_size(&self) -> u64 {
        GRAPH_HEADER_SIZE as u64 + (self.node_count * self.record_params.record_size() as u64)
    }

    /// Returns the record params for this graph
    #[inline]
    pub fn record_params(&self) -> NodeRecordParams {
        self.record_params
    }

    /// Returns the current node count
    #[inline]
    pub fn node_count(&self) -> u64 {
        self.node_count
    }
}

/// Zero-allocation iterator over neighbors in a layer.
///
/// Reads NodeId values directly from mmap bytes, filtering out INVALID_NODE_ID.
pub struct NeighborIterator<'a> {
    bytes: &'a [u8],
    start_offset: usize,
    count: usize,
    pos: usize,
}

impl<'a> Iterator for NeighborIterator<'a> {
    type Item = NodeId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.count {
            let offset = self.start_offset + self.pos * 8;
            self.pos += 1;

            if offset + 8 <= self.bytes.len() {
                // SAFETY: We've bounds-checked above
                let neighbor = u64::from_le_bytes(self.bytes[offset..offset + 8].try_into().ok()?);

                if neighbor != INVALID_NODE_ID {
                    return Some(neighbor);
                }
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Upper bound is remaining slots, lower bound is 0 (all could be INVALID)
        (0, Some(self.count - self.pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_storage(dims: u32) -> (Storage, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let storage = Storage::open(temp_file.path(), dims).unwrap();
        (storage, temp_file)
    }

    #[test]
    fn test_graph_header_size() {
        assert_eq!(
            std::mem::size_of::<GraphHeader>(),
            GRAPH_HEADER_SIZE,
            "GraphHeader must be exactly 64 bytes"
        );
    }

    #[test]
    fn test_graph_header_alignment() {
        assert_eq!(std::mem::align_of::<GraphHeader>(), 8, "GraphHeader must be 8-byte aligned");
    }

    #[test]
    fn test_graph_header_roundtrip() {
        let params = NodeRecordParams::new(16, 32, 8);
        let mut header = GraphHeader::new(params);
        header.entry_point = 42;
        header.max_layer = 3;
        header.node_count = 1000;

        let bytes = header.to_bytes();
        let restored = GraphHeader::from_bytes(&bytes).unwrap();

        assert!(restored.is_valid());
        assert_eq!(restored.entry_point, 42);
        assert_eq!(restored.max_layer, 3);
        assert_eq!(restored.node_count, 1000);
        assert_eq!(restored.m, 16);
        assert_eq!(restored.m0, 32);
        assert_eq!(restored.max_layers, 8);
    }

    #[test]
    fn test_o1_addressing_no_hashmap() {
        let (mut storage, _temp) = create_test_storage(128);

        // Insert some vectors
        for _ in 0..100 {
            storage.insert(&vec![1.0; 128]).unwrap();
        }

        let params = HnswParams::default();
        let graph = HnswGraph::open(storage, params).unwrap();

        let record_size = graph.record_params().record_size() as u64;

        // Verify O(1) offset computation:  offsets should be evenly spaced
        let offset_0 = graph.node_offset(0);
        let offset_1 = graph.node_offset(1);
        let offset_2 = graph.node_offset(2);
        let offset_99 = graph.node_offset(99);

        // Check linear spacing (O(1) formula property)
        assert_eq!(offset_1 - offset_0, record_size, "Offset spacing should equal record_size");
        assert_eq!(offset_2 - offset_1, record_size, "Offset spacing should equal record_size");
        assert_eq!(
            offset_99 - offset_0,
            99 * record_size,
            "Offset for node 99 should be 99 * record_size from node 0"
        );

        // Verify formula:  offset_n = offset_0 + n * record_size
        for n in [0u64, 1, 10, 50, 99] {
            let expected = offset_0 + n * record_size;
            let actual = graph.node_offset(n);
            assert_eq!(actual, expected, "O(1) formula failed for node {}", n);
        }
    }

    #[test]
    fn test_graph_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create and populate graph
        {
            let mut storage = Storage::open(path, 128).unwrap();
            for _ in 0..10 {
                storage.insert(&vec![1.0; 128]).unwrap();
            }

            let params = HnswParams::default();
            let mut graph = HnswGraph::open(storage, params).unwrap();

            for i in 0..10u64 {
                graph.insert(i, (i % 3) as usize).unwrap();
            }

            graph.commit().unwrap();
        }

        // Reopen and verify
        {
            let storage = Storage::open(path, 128).unwrap();
            let params = HnswParams::default();
            let graph = HnswGraph::open(storage, params).unwrap();

            assert!(graph.entry_point.is_some());
            assert!(graph.node_count() > 0);
        }
    }

    #[test]
    #[should_panic(expected = "Node ID invariant violated")]
    fn test_insert_out_of_order_panics() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut storage = Storage::open(path, 128).unwrap();
        for _ in 0..10 {
            storage.insert(&vec![1.0; 128]).unwrap();
        }

        let params = HnswParams::default();
        let mut graph = HnswGraph::open(storage, params).unwrap();

        graph.insert(0, 0).unwrap(); // OK:  node_count was 0
        graph.insert(5, 0).unwrap(); // PANIC: expected 1, got 5
    }

    #[test]
    #[should_panic(expected = "Node ID invariant violated")]
    fn test_insert_duplicate_panics() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut storage = Storage::open(path, 128).unwrap();
        for _ in 0..10 {
            storage.insert(&vec![1.0; 128]).unwrap();
        }

        let params = HnswParams::default();
        let mut graph = HnswGraph::open(storage, params).unwrap();

        graph.insert(0, 0).unwrap(); // OK
        graph.insert(0, 0).unwrap(); // PANIC: expected 1, got 0
    }

    #[test]
    fn test_update_existing_node() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut storage = Storage::open(path, 128).unwrap();
        for _ in 0..10 {
            storage.insert(&vec![1.0; 128]).unwrap();
        }

        let params = HnswParams::default();
        let mut graph = HnswGraph::open(storage, params).unwrap();

        // Insert node
        graph.insert(0, 1).unwrap();
        assert_eq!(graph.node_count(), 1);

        // Update neighbors (doesn't increment count)
        let record_params = params.to_record_params();
        let mut record = NodeRecord::new(0, 2, record_params);
        record.set_neighbors(0, &[1, 2, 3]);
        record.set_neighbors(1, &[10, 20]);

        graph.update_node_record(&record).unwrap();
        assert_eq!(graph.node_count(), 1); // Still 1, not 2

        // Verify update
        let read_back = graph.read_node_record(0).unwrap();
        assert_eq!(read_back.get_neighbors(0), vec![1, 2, 3]);
    }
}
