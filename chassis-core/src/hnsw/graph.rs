use crate::hnsw::node::{Node, NodeId, NodeRecord, NodeRecordParams, Offset};
use crate::hnsw::{HnswParams, compute_node_offset};
use crate::Storage;
use anyhow::Result;
use std::collections::HashMap;

/// File offset where graph data begins (after vector data)
#[allow(dead_code)]
const GRAPH_OFFSET_MARKER: u64 = u64::MAX - 1;

/// HNSW graph stored in segmented layout: 
/// [Vector Storage] | [Graph Header] | [Node Data]
pub struct HnswGraph {
    pub(crate) storage: Storage,
    #[allow(dead_code)]
    params: HnswParams,
    
    /// Offset where graph section begins
    graph_start:  Offset,
    
    /// Entry point node ID (highest layer node)
    pub(crate) entry_point: Option<NodeId>,
    
    /// Maximum layer in the graph
    pub max_layer: usize,
    
    /// Node offset cache (node_id -> file_offset)
    node_offsets: HashMap<NodeId, Offset>,
    
    /// Cached record parameters for O(1) record size lookup
    record_params: NodeRecordParams,
}

impl HnswGraph {
    /// Opens existing graph or creates new one
    pub fn open(storage: Storage, params: HnswParams) -> Result<Self> {
        let graph_start = Self::find_or_create_graph_start(&storage)?;
        let record_params = params.to_record_params();
        
        Ok(Self {
            storage,
            params,
            graph_start,
            entry_point: None,
            max_layer: 0,
            node_offsets: HashMap::new(),
            record_params,
        })
    }
    
    /// Finds where graph data starts in file
    fn find_or_create_graph_start(storage: &Storage) -> Result<Offset> {
        // Graph starts after all vector data
        let vector_count = storage.count();
        let vector_size = storage.dimensions() as usize * std::mem::size_of:: <f32>();
        let vector_zone_end = crate::header::HEADER_SIZE + (vector_count as usize * vector_size);
        
        // Align to page boundary
        let graph_start = ((vector_zone_end + 4095) & !4095) as u64;
        
        Ok(graph_start)
    }
    
    /// Returns the record parameters for this graph
    #[inline]
    pub fn record_params(&self) -> NodeRecordParams {
        self. record_params
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
        
        // O(1) addressing:  offset = graph_start + (node_id * record_size)
        let offset = compute_node_offset(self.graph_start, node.id, record_size);
        
        // Convert to fixed-size record and serialize
        let record = node.to_record(self.record_params);
        let _bytes = record.to_bytes();
        
        // TODO: Write bytes to mmap at offset
        
        self.node_offsets.insert(node. id, offset);
        
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
    
    /// Reads a node from disk (allocates full Node struct)
    /// 
    /// ⚠️ PERFORMANCE WARNING:  This method allocates vectors for each layer. 
    /// For search hot paths, use `read_node_record()` + `neighbors_iter()` instead.
    pub fn read_node(&self, _offset: Offset) -> Result<Node> {
        // TODO: Read node header and neighbor data from mmap
        unimplemented!("read_node")
    }
    
    /// Reads a node record from disk (zero-allocation on the record itself)
    /// 
    /// This is the preferred method for search operations as it avoids
    /// allocating vectors for layer neighbor lists.
    /// 
    /// # Arguments
    /// 
    /// * `node_id` - The node ID to read
    /// 
    /// # Returns
    /// 
    /// Returns the NodeRecord with fixed-size neighbor arrays
    pub fn read_node_record(&self, node_id: NodeId) -> Result<NodeRecord> {
        let record_size = self.record_params.record_size();
        let _offset = compute_node_offset(self.graph_start, node_id, record_size);
        
        // TODO: Read bytes from mmap at offset
        // For now, return a placeholder
        // In production, this would be: 
        // let bytes = &self.mmap[offset as usize..offset as usize + record_size];
        // NodeRecord::from_bytes(bytes, self.record_params)
        //     .map_err(|e| anyhow::anyhow!("Failed to read node record: {}", e))
        
        unimplemented!("read_node_record - needs mmap integration")
    }
    
    /// Gets node offset by ID
    pub fn get_node_offset(&self, node_id: NodeId) -> Option<Offset> {
        self.node_offsets. get(&node_id).copied()
    }
    
    /// Computes node offset using O(1) addressing formula
    /// 
    /// This is the preferred method over `get_node_offset` when the node
    /// is known to exist, as it doesn't require HashMap lookup.
    #[inline]
    pub fn compute_node_offset(&self, node_id: NodeId) -> Offset {
        compute_node_offset(self.graph_start, node_id, self. record_params.record_size())
    }
}
