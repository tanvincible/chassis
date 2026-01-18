/// File offset type (u64 for 64-bit addressing)
pub type Offset = u64;

/// Node ID (index into vector storage)
pub type NodeId = u64;

/// On-disk node structure using relative offsets
/// 
/// Layout:
/// ```text
/// [node_id: u64]
/// [layer_count: u8]
/// [neighbors_per_layer: [u16; layer_count]]
/// [neighbor_offsets: [Offset; sum(neighbors_per_layer)]]
/// ```
#[repr(C)]
pub struct NodeHeader {
    /// Vector ID in storage
    pub node_id: NodeId,
    
    /// Number of layers this node participates in
    pub layer_count: u8,
    
    /// Reserved for alignment
    _padding: [u8; 7],
}

impl NodeHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    
    pub fn new(node_id: NodeId, layer_count: u8) -> Self {
        Self {
            node_id,
            layer_count,
            _padding: [0; 7],
        }
    }
}

/// In-memory representation of a node
pub struct Node {
    pub id: NodeId,
    pub offset: Offset,
    pub layers: Vec<Vec<NodeId>>,
}

impl Node {
    /// Calculate size needed to store this node on disk
    pub fn disk_size(&self) -> usize {
        let mut size = NodeHeader::SIZE;
        
        // Space for neighbor counts per layer
        size += self.layers.len() * std::mem::size_of::<u16>();
        
        // Space for all neighbor offsets
        let total_neighbors: usize = self.layers.iter().map(|l| l.len()).sum();
        size += total_neighbors * std::mem::size_of::<Offset>();
        
        // Align to 8 bytes
        (size + 7) & !7
    }
    
    /// Returns the layer this node belongs to (highest layer)
    pub fn top_layer(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }
}
