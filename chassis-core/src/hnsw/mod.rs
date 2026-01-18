mod builder;
mod graph;
mod layer;
mod node;
mod search;

pub use builder::HnswBuilder;
pub use graph::HnswGraph;
pub use search::SearchResult;

/// HNSW construction parameters
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Maximum connections per node (M)
    pub max_connections: u16,
    
    /// Construction quality (efConstruction)
    pub ef_construction: usize,
    
    /// Search quality (efSearch)
    pub ef_search: usize,
    
    /// Layer selection multiplier
    pub ml: f32,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16.0_f32).ln(),
        }
    }
}
