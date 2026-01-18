use crate::hnsw::graph::HnswGraph;
use crate::hnsw::node::NodeId;
use crate::hnsw::HnswParams;
use crate::Storage;
use anyhow::Result;

/// Builder for constructing HNSW index
pub struct HnswBuilder {
    params: HnswParams,
}

impl HnswBuilder {
    pub fn new(params: HnswParams) -> Self {
        Self { params }
    }
    
    /// Build index from existing storage
    pub fn build(self, storage: Storage) -> Result<HnswGraph> {
        let mut graph = HnswGraph::open(storage, self.params)?;
        
        let count = graph.storage.count();
        
        for node_id in 0..count {
            let layer = self.select_layer();
            graph.insert(node_id, layer)?;
        }
        
        Ok(graph)
    }
    
    /// Select layer for new node using exponential decay
    fn select_layer(&self) -> usize {
        let uniform: f32 = rand::random();
        let level = (-uniform.ln() * self.params.ml).floor() as usize;
        level
    }
}
