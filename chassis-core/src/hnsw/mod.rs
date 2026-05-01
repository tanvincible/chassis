mod builder;
mod graph;
mod link;
pub mod node;
mod search;

pub use builder::HnswBuilder;
pub use graph::HnswGraph;

#[cfg(any(test, feature = "internals"))]
pub use graph::GraphHeader;
pub use node::NodeRecordParams;

#[cfg(any(test, feature = "internals"))]
pub use node::{
    INVALID_NODE_ID, Node, NodeHeader, NodeId, NodeRecord, Offset, compute_node_offset,
};
pub use search::SearchResult;

/// Select an HNSW layer from a uniform random sample using exponential decay.
#[inline]
pub(crate) fn layer_from_uniform(uniform: f32, ml: f32, max_layers: u8) -> usize {
    let layer = (-uniform.ln() * ml).floor() as usize;
    layer.min(max_layers.saturating_sub(1) as usize)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_from_uniform_caps_at_max_layer() {
        let layer = layer_from_uniform(f32::MIN_POSITIVE, 1.0, 16);
        assert_eq!(layer, 15);
    }

    #[test]
    fn layer_from_uniform_respects_single_layer_graphs() {
        let layer = layer_from_uniform(f32::MIN_POSITIVE, 1.0, 1);
        assert_eq!(layer, 0);
    }

    #[test]
    fn layer_from_uniform_maps_one_to_layer_zero() {
        let layer = layer_from_uniform(1.0, 1.0, 16);
        assert_eq!(layer, 0);
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
