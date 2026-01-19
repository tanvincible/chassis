use crate::hnsw::node::{Node, NodeId, NodeRecord, NodeRecordParams, Offset, INVALID_NODE_ID};
use crate::hnsw::{HnswParams, compute_node_offset};
use crate::Storage;
use anyhow::Result;
use std::collections::HashMap;

/// File offset where graph data begins (after vector data)
#[allow(dead_code)]
const GRAPH_OFFSET_MARKER: u64 = u64::MAX - 1;

/// Size of the graph header in bytes
const GRAPH_HEADER_SIZE: usize = 64;

/// Persistent graph header stored at the beginning of the graph zone.
///
/// # Layout (64 bytes, 8-byte aligned)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       4     magic: [u8; 4] (b"HNSW")
/// 4       4     version: u32
/// 8       8     entry_point: NodeId
/// 16      4     max_layer: u32
/// 20      8     node_count: u64
/// 28      2     m: u16
/// 30      2     m0: u16
/// 32      1     max_layers: u8
/// 33      31    _reserved: [u8; 31]
/// ```
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct GraphHeader {
    /// Magic bytes for validation (b"HNSW")
    magic: [u8; 4],
    
    /// Format version
    version: u32,
    
    /// Entry point node ID (INVALID_NODE_ID if empty)
    entry_point: NodeId,
    
    /// Highest layer in graph
    max_layer: u32,
    
    /// Number of nodes written
    node_count: u64,
    
    /// Max connections per layer
    m: u16,
    
    /// Max connections at layer 0
    m0: u16,
    
    /// Maximum layers
    max_layers: u8,
    
    /// Padding to 64 bytes
    _reserved: [u8; 31],
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
            max_layer: 0,
            node_count: 0,
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
        bytes[16..20].copy_from_slice(&self.max_layer.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.node_count.to_le_bytes());
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
        let max_layer = u32::from_le_bytes(bytes[16..20].try_into()?);
        let node_count = u64::from_le_bytes(bytes[20..28].try_into()?);
        let m = u16::from_le_bytes(bytes[28..30].try_into()?);
        let m0 = u16::from_le_bytes(bytes[30..32].try_into()?);
        let max_layers = bytes[32];
        
        let mut reserved = [0u8; 31];
        reserved.copy_from_slice(&bytes[33..64]);
        
        Ok(Self {
            magic,
            version,
            entry_point,
            max_layer,
            node_count,
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
pub struct HnswGraph {
    pub(crate) storage: Storage,
    #[allow(dead_code)]
    params: HnswParams,
    
    /// Cached record parameters for O(1) lookup
    record_params: NodeRecordParams,
    
    /// Offset where graph section begins (includes header)
    graph_start: Offset,
    
    /// Entry point node ID (highest layer node)
    pub(crate) entry_point: Option<NodeId>,
    
    /// Maximum layer in the graph
    pub max_layer: usize,
    
    /// Node offset cache (node_id -> file_offset)
    node_offsets: HashMap<NodeId, Offset>,
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
        let (entry_point, max_layer) = match Self::try_read_graph_header(&storage, graph_start, record_params) {
            Ok(header) => {
                // Existing graph found
                let entry_point = if header.entry_point == INVALID_NODE_ID {
                    None
                } else {
                    Some(header.entry_point)
                };
                (entry_point, header.max_layer as usize)
            }
            Err(_) => {
                // New graph - initialize header
                let header = GraphHeader::new(record_params);
                let bytes = header.to_bytes();
                let zone = storage.graph_zone_mut(graph_start as usize, GRAPH_HEADER_SIZE)?;
                zone.copy_from_slice(&bytes);
                (None, 0)
            }
        };
        
        Ok(Self {
            storage,
            params,
            record_params,
            graph_start,
            entry_point,
            max_layer,
            node_offsets: HashMap::new(),
        })
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
    
    /// Read graph header from mmap (for internal use or debugging)
    #[allow(dead_code)]
    fn read_graph_header(&self) -> Result<GraphHeader> {
        let zone = self.storage.graph_zone(self.graph_start as usize, GRAPH_HEADER_SIZE)?;
        let header = GraphHeader::from_bytes(zone)?;
        
        if !header.is_valid() {
            anyhow::bail!("Invalid graph header");
        }
        
        Ok(header)
    }
    
    /// Write graph header to mmap
    fn write_graph_header(&mut self) -> Result<()> {
        let mut header = GraphHeader::new(self.record_params);
        header.entry_point = self.entry_point.unwrap_or(INVALID_NODE_ID);
        header.max_layer = self.max_layer as u32;
        header.node_count = self.node_offsets.len() as u64;
        
        let bytes = header.to_bytes();
        let zone = self.storage.graph_zone_mut(self.graph_start as usize, GRAPH_HEADER_SIZE)?;
        zone.copy_from_slice(&bytes);
        
        Ok(())
    }
    
    /// Calculate the offset where node records begin (after header)
    #[inline]
    const fn node_records_start(&self) -> Offset {
        self.graph_start + GRAPH_HEADER_SIZE as u64
    }
    
    /// Read a node record directly from mmap (zero-copy header, minimal allocation for neighbors)
    pub fn read_node_record(&self, node_id: NodeId) -> Result<NodeRecord> {
        let record_size = self.record_params.record_size();
        let offset = compute_node_offset(self.node_records_start(), node_id, record_size);
        
        let zone = self.storage.graph_zone(offset as usize, record_size)?;
        NodeRecord::from_bytes(zone, self.record_params)
            .map_err(|e| anyhow::anyhow!("Failed to read node record: {}", e))
    }
    
    /// Write a node record directly to mmap
    pub fn write_node_record(&mut self, record: &NodeRecord) -> Result<()> {
        let record_size = self.record_params.record_size();
        let node_id = record.header.node_id;
        let offset = compute_node_offset(self.node_records_start(), node_id, record_size);
        
        // Ensure capacity for this node
        let required_size = offset as usize + record_size;
        self.storage.ensure_graph_capacity(required_size)?;
        
        // Write record bytes
        let bytes = record.to_bytes();
        let zone = self.storage.graph_zone_mut(offset as usize, record_size)?;
        zone.copy_from_slice(&bytes);
        
        // Update cache
        self.node_offsets.insert(node_id, offset);
        
        Ok(())
    }
    
    /// Get raw bytes slice for a node (for truly zero-copy iteration)
    pub fn get_node_bytes(&self, node_id: NodeId) -> Result<&[u8]> {
        let record_size = self.record_params.record_size();
        let offset = compute_node_offset(self.node_records_start(), node_id, record_size);
        
        self.storage.graph_zone(offset as usize, record_size)
    }
    
    /// Iterate neighbors directly from mmap bytes (zero-allocation)
    /// 
    /// This returns an iterator that directly reads from the memory-mapped bytes.
    /// The iterator filters out INVALID_NODE_ID entries.
    pub fn neighbors_iter_from_mmap(
        &self, 
        node_id: NodeId, 
        layer: usize
    ) -> Result<impl Iterator<Item = NodeId> + '_> {
        use crate::hnsw::node::NodeHeader;
        
        let bytes = self.get_node_bytes(node_id)?;
        
        // Parse just the header to get layer_count
        let header = unsafe { *(bytes.as_ptr().cast::<NodeHeader>()) };
        
        // Check if layer is valid
        if layer >= header.layer_count as usize {
            // Return empty iterator for out of bounds
            return Ok(NeighborIterator {
                bytes: &[],
                start_offset: 0,
                count: 0,
                pos: 0,
            });
        }
        
        // Calculate offset and count for this layer
        let layer_offset = self.record_params.layer_offset(layer)
            .ok_or_else(|| anyhow::anyhow!("Invalid layer"))?;
        let neighbor_count = self.record_params.max_neighbors(layer);
        
        Ok(NeighborIterator {
            bytes,
            start_offset: layer_offset,
            count: neighbor_count,
            pos: 0,
        })
    }
    
    /// Commit graph state (write header and flush to disk)
    pub fn commit(&mut self) -> Result<()> {
        self.write_graph_header()?;
        self.storage.commit()
    }
    
    /// Finds where graph data starts in file
    fn find_or_create_graph_start(storage: &Storage) -> Result<Offset> {
        // Graph starts after all vector data
        let vector_count = storage.count();
        let vector_size = storage.dimensions() as usize * std::mem::size_of::<f32>();
        let vector_zone_end = crate::header::HEADER_SIZE + (vector_count as usize * vector_size);
        
        // Align to page boundary
        let graph_start = ((vector_zone_end + 4095) & !4095) as u64;
        
        Ok(graph_start)
    }
    
    /// Inserts a new node into the graph
    pub fn insert(&mut self, vector_id: NodeId, layer: usize) -> Result<()> {
        let node = Node {
            id: vector_id,
            offset: 0, // Will be set when written
            layers: vec![Vec::new(); layer + 1],
        };
        
        // TODO: Connect to neighbors using HNSW algorithm
        
        self.write_node(&node)?;
        
        if self.entry_point.is_none() || layer > self.max_layer {
            self.entry_point = Some(vector_id);
            self.max_layer = layer;
        }
        
        Ok(())
    }
    
    /// Writes node to disk at next available offset
    fn write_node(&mut self, node: &Node) -> Result<Offset> {
        let record_size = self.record_params.record_size();
        
        // O(1) addressing: offset = node_records_start + (node_id * record_size)
        let offset = compute_node_offset(self.node_records_start(), node.id, record_size);
        
        // Convert to fixed-size record and write directly to mmap
        let record = node.to_record(self.record_params);
        self.write_node_record(&record)?;
        
        Ok(offset)
    }

    /// Allocates space for a node in the graph zone
    #[allow(dead_code)]
    fn allocate_node_space(&self, _size: usize) -> Result<Offset> {
        // Find next available offset
        let current_end = self.graph_start + self.total_graph_size();
        Ok(current_end)
    }
    
    /// Returns total size of graph data written so far
    #[allow(dead_code)]
    fn total_graph_size(&self) -> u64 {
        // TODO: Track actual graph size
        0
    }
    
    /// Reads a node from disk
    pub fn read_node(&self, _offset: Offset) -> Result<Node> {
        // TODO: Read node header and neighbor data from mmap
        unimplemented!("read_node")
    }
    
    /// Gets node offset by ID
    pub fn get_node_offset(&self, node_id: NodeId) -> Option<Offset> {
        self.node_offsets.get(&node_id).copied()
    }
}

/// Zero-allocation iterator over neighbors in a layer.
/// 
/// Reads NodeId values directly from mmap bytes, filtering out INVALID_NODE_ID.
struct NeighborIterator<'a> {
    bytes: &'a [u8],
    start_offset: usize,
    count: usize,
    pos: usize,
}

impl<'a> Iterator for NeighborIterator<'a> {
    type Item = NodeId;
    
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.count {
            let offset = self.start_offset + self.pos * 8;
            self.pos += 1;
            
            if offset + 8 <= self.bytes.len() {
                let neighbor = u64::from_le_bytes(
                    self.bytes[offset..offset + 8].try_into().ok()?
                );
                
                if neighbor != INVALID_NODE_ID {
                    return Some(neighbor);
                }
            }
        }
        
        None
    }
}
