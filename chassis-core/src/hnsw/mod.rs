mod builder;
mod graph;
mod link;
pub mod node;
mod search;

pub use builder::HnswBuilder;
pub use graph::{GraphHeader, HnswGraph};
pub use node::{
    INVALID_NODE_ID, Node, NodeHeader, NodeId, NodeRecord, NodeRecordParams, Offset,
    compute_node_offset,
};
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

    /// Maximum layers (determines fixed record size)
    pub max_layers: u8,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16.0_f32).ln(),
            max_layers: 16,
        }
    }
}

impl HnswParams {
    /// Convert to NodeRecordParams for fixed-size record allocation
    #[must_use]
    pub const fn to_record_params(&self) -> NodeRecordParams {
        NodeRecordParams::new(
            self.max_connections,
            self.max_connections * 2, // m0 = 2 * M
            self.max_layers,
        )
    }
}
