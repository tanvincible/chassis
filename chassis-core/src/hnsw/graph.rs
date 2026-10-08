//! HNSW graph over the v3 file: each node's level-0 record sits in its slot next to its vector,
//! and its upper layers in the heap (see `storage.rs`). Neighbor ids are slot numbers.

use crate::Storage;
use crate::header::FLAG_CUSTOM_IDS;
use crate::hnsw::HnswParams;
use crate::hnsw::node::{INVALID_NODE_ID, Node, NodeId, NodeRecord, NodeRecordParams};
use crate::storage::EMPTY;
use anyhow::{Result, bail};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// HNSW graph stored in an index file.
#[derive(Debug)]
pub struct HnswGraph {
    pub(crate) storage: Storage,

    #[allow(dead_code)]
    params: HnswParams,

    /// Cached record parameters for O(1) lookup
    pub record_params: NodeRecordParams,

    /// Entry point node ID (highest layer node)
    pub entry_point: Option<NodeId>,

    /// Maximum layer in the graph
    pub max_layer: usize,

    /// Number of nodes in the graph
    pub node_count: u64,

    /// Deleted nodes, committed and pending
    pub(crate) deleted_count: u64,

    /// Deletes since the last flush; they reach the file only in `commit`
    pending_deletes: HashSet<NodeId>,

    /// Some node's id differs from its slot
    pub(crate) custom_ids: bool,

    /// In a reader, the snapshot searches return results from
    pub(crate) view: Option<View>,

    /// The batch being linked, if one is
    pub(crate) linking: Option<Linking>,
}

/// A batch being linked: its first slot, and for each of its slots whether the node has written
/// its own lists. Until then no search may reach it, though links that a crash left in older
/// nodes can already name its slot (ADR-0010).
#[derive(Debug)]
pub(crate) struct Linking {
    pub first: NodeId,
    pub linked: Vec<AtomicBool>,
}

/// A reader's snapshot: slots below `committed` that no flush up to `epoch` deleted.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View {
    pub committed: u64,
    pub epoch: u64,
}

impl HnswGraph {
    /// Opens the graph in `storage`. Stored vectors without a graph become slots to link again.
    ///
    /// # Errors
    ///
    /// Returns an error if the file was created with different graph parameters.
    pub fn open(storage: Storage, params: HnswParams) -> Result<Self> {
        Self::open_inner(storage, params, false)
    }

    /// Like `open`, but stored vectors without a graph are corruption, not a new index.
    pub(crate) fn open_no_reset(storage: Storage, params: HnswParams) -> Result<Self> {
        Self::open_inner(storage, params, true)
    }

    fn open_inner(storage: Storage, params: HnswParams, no_reset: bool) -> Result<Self> {
        let record_params = params.to_record_params();
        if storage.params() != record_params {
            bail!(
                "Index was created with {:?} but opened with {record_params:?}; open it with the \
                 original max_connections",
                storage.params()
            );
        }
        let state = &storage.state;
        let entry_point = (state.entry_point != INVALID_NODE_ID).then_some(state.entry_point);
        if entry_point.is_none() && no_reset && storage.count() > 0 {
            bail!(
                "File has {} vectors but no graph: it was written without VectorIndex",
                storage.count()
            );
        }
        Ok(Self {
            params,
            record_params,
            entry_point,
            max_layer: state.max_layer as usize,
            node_count: if entry_point.is_some() { state.count } else { 0 },
            deleted_count: state.deleted_count,
            pending_deletes: HashSet::new(),
            custom_ids: state.flags & FLAG_CUSTOM_IDS != 0,
            view: None,
            linking: None,
            storage,
        })
    }

    /// A graph over a read-only `storage`; `refresh` moves it to the newest snapshot.
    pub(crate) fn reader(storage: Storage, params: HnswParams) -> Result<Self> {
        let mut graph = Self::open_inner(storage, params, false)?;
        graph.refresh()?;
        Ok(graph)
    }

    /// Takes a new snapshot (ADR-0008, decision 5): searches route through every slot the writer
    /// has linked, but return only committed ones.
    pub(crate) fn refresh(&mut self) -> Result<()> {
        self.storage.refresh()?;
        let state = &self.storage.state;
        self.entry_point = (state.entry_point != INVALID_NODE_ID).then_some(state.entry_point);
        self.max_layer = state.max_layer as usize;
        self.node_count = if self.entry_point.is_some() { self.storage.count() } else { 0 };
        self.deleted_count = state.deleted_count;
        self.custom_ids = state.flags & FLAG_CUSTOM_IDS != 0;
        self.view = Some(View { committed: state.count, epoch: state.epoch });
        Ok(())
    }

    /// Whether the node at `slot` is deleted, including deletes not yet flushed. In a reader,
    /// whether its snapshot leaves it out: not committed yet, or deleted by a committed flush.
    pub(crate) fn is_deleted(&self, slot: NodeId) -> Result<bool> {
        let epoch = self.storage.slot_epoch(slot)?;
        Ok(match self.view {
            Some(view) => slot >= view.committed || (epoch != 0 && epoch <= view.epoch),
            None => self.pending_deletes.contains(&slot) || epoch != 0,
        })
    }

    /// Marks the node at `slot` deleted from the next flush on; false if it already was.
    pub(crate) fn mark_deleted(&mut self, slot: NodeId) -> Result<bool> {
        if self.is_deleted(slot)? {
            return Ok(false);
        }
        self.pending_deletes.insert(slot);
        self.deleted_count += 1;
        Ok(true)
    }

    /// The caller's id stored for the node at `slot`.
    pub(crate) fn id_of(&self, slot: NodeId) -> Result<u64> {
        self.storage.slot_id(slot)
    }

    /// Reads a node's record: its id, layers and neighbor lists.
    ///
    /// # Errors
    ///
    /// Returns an error if `node_id` has no written record.
    pub fn read_node_record(&self, node_id: NodeId) -> Result<NodeRecord> {
        let layer_count = self.storage.layer_count(node_id)?;
        let mut record = NodeRecord::new(node_id, layer_count as u8, self.record_params);
        record.header.id = self.storage.slot_id(node_id)?;
        for layer in 0..layer_count {
            let ids: Vec<NodeId> = self.neighbors_iter_from_mmap(node_id, layer)?.collect();
            record.set_neighbors(layer, &ids);
        }
        Ok(record)
    }

    /// Writes a new node's record into its slot, whose vector must already be stored.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot has no vector or the file cannot grow.
    pub fn write_node_record(&mut self, record: &NodeRecord) -> Result<()> {
        let layers: Vec<Vec<NodeId>> =
            (0..usize::from(record.header.layer_count)).map(|l| record.get_neighbors(l)).collect();
        self.storage.write_record(record.slot, &layers)
    }

    /// Iterate a node's neighbors at `layer` straight from the mapped file, without allocating.
    ///
    /// # Errors
    ///
    /// Returns an error if `node_id` has no valid record.
    #[inline]
    pub fn neighbors_iter_from_mmap(
        &self,
        node_id: NodeId,
        layer: usize,
    ) -> Result<impl Iterator<Item = NodeId> + '_> {
        let list = self.storage.neighbors(node_id, layer)?;
        let ids = list.iter().map(|id| id.load(Ordering::Relaxed));
        Ok(ids.filter(|&id| id != EMPTY).map(NodeId::from).filter(|&id| self.is_linked(id)))
    }

    /// False only for a node of the batch being linked that hasn't written its own lists yet.
    #[inline]
    fn is_linked(&self, slot: NodeId) -> bool {
        match &self.linking {
            Some(batch) if slot >= batch.first => batch
                .linked
                .get((slot - batch.first) as usize)
                .is_none_or(|linked| linked.load(Ordering::Acquire)),
            _ => true,
        }
    }

    /// Distance from `query` to a stored vector, read in place.
    ///
    /// # Errors
    ///
    /// Returns an error if `node_id` has no stored vector.
    #[inline]
    pub fn compute_distance_zero_copy(&self, query: &[f32], node_id: NodeId) -> Result<f32> {
        #[cfg(lab_count)]
        crate::lab::DISTANCES.fetch_add(1, Ordering::Relaxed);
        let vector_slice = self.storage.get_vector_slice(node_id)?;
        Ok(crate::distance::euclidean_distance(query, vector_slice))
    }

    /// Makes every node and delete so far durable: the flush commit point (ADR-0008, decision 6).
    ///
    /// Costs a few fsyncs, so call it once per batch, not once per insert.
    ///
    /// # Errors
    ///
    /// Returns an error if writing or syncing the file fails.
    pub fn commit(&mut self) -> Result<()> {
        let deletes: Vec<NodeId> = self.pending_deletes.iter().copied().collect();
        let state = &mut self.storage.state;
        state.count = self.node_count;
        state.entry_point = self.entry_point.unwrap_or(INVALID_NODE_ID);
        state.max_layer = self.max_layer as u32;
        state.deleted_count = self.deleted_count;
        state.flags = if self.custom_ids { FLAG_CUSTOM_IDS } else { 0 };
        self.storage.commit_deleting(&deletes)?;
        self.pending_deletes.clear();
        Ok(())
    }

    /// Inserts a node with no neighbors into the next slot, whose vector must already be stored.
    ///
    /// # Errors
    ///
    /// Returns an error unless `vector_id` is the next slot (node ids are dense and
    /// monotonically increasing), or if the record cannot be written.
    pub fn insert(&mut self, vector_id: NodeId, layer: usize) -> Result<()> {
        // Enforce dense, monotonic node ID invariant
        debug_assert!(
            vector_id == self.node_count,
            "Node ID invariant violated: expected {}, got {}.  \
             Node IDs must be dense and monotonically increasing (0, 1, 2, ...).",
            self.node_count,
            vector_id
        );
        if vector_id != self.node_count {
            bail!(
                "Node ID invariant violated: expected {}, got {}. \
                 Node IDs must be dense and monotonically increasing.",
                self.node_count,
                vector_id
            );
        }

        let node = Node { id: vector_id, offset: 0, layers: vec![Vec::new(); layer + 1] };
        self.write_node_record(&node.to_record(self.record_params))?;
        self.node_count += 1;

        if self.entry_point.is_none() || layer > self.max_layer {
            self.entry_point = Some(vector_id);
            self.max_layer = layer;
        }
        Ok(())
    }

    /// Replaces an existing node's neighbor lists (does NOT increment `node_count`).
    ///
    /// # Errors
    ///
    /// Returns an error if the node doesn't exist or `record` has a different layer count.
    pub fn update_node_record(&mut self, record: &NodeRecord) -> Result<()> {
        let node_id = record.slot;
        if node_id >= self.node_count {
            bail!("Cannot update non-existent node: {node_id} (node_count = {})", self.node_count);
        }
        let layer_count = self.storage.layer_count(node_id)?;
        if usize::from(record.header.layer_count) != layer_count {
            bail!("Node {node_id} has {layer_count} layers, the update has a different count");
        }
        for layer in 0..layer_count {
            self.storage.write_neighbors(node_id, layer, &record.get_neighbors(layer))?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn graph_with_vectors(n: u64) -> (HnswGraph, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp.path(), 128).unwrap();
        for _ in 0..n {
            storage.insert(&[1.0; 128]).unwrap();
        }
        (HnswGraph::open(storage, HnswParams::default()).unwrap(), temp)
    }

    #[test]
    fn test_corrupted_headers_are_an_error_not_a_new_graph() {
        let (mut graph, temp) = graph_with_vectors(1);
        graph.insert(0, 0).unwrap();
        graph.commit().unwrap();
        drop(graph);

        let mut bytes = std::fs::read(temp.path()).unwrap();
        bytes[100] ^= 1;
        bytes[64 * 1024 + 100] ^= 1;
        std::fs::write(temp.path(), bytes).unwrap();
        assert!(Storage::open(temp.path(), 128).is_err());
    }

    #[test]
    fn test_graph_persistence() {
        let (mut graph, temp) = graph_with_vectors(10);
        for i in 0..10u64 {
            graph.insert(i, (i % 3) as usize).unwrap();
        }
        graph.commit().unwrap();
        drop(graph);

        let graph =
            HnswGraph::open(Storage::open(temp.path(), 128).unwrap(), HnswParams::default())
                .unwrap();
        assert_eq!((graph.entry_point, graph.max_layer, graph.node_count()), (Some(2), 2, 10));
        assert_eq!(graph.read_node_record(5).unwrap().header.layer_count, 3);
    }

    #[test]
    #[should_panic(expected = "Node ID invariant violated")]
    fn test_insert_out_of_order_panics() {
        let (mut graph, _temp) = graph_with_vectors(10);
        graph.insert(0, 0).unwrap(); // OK:  node_count was 0
        graph.insert(5, 0).unwrap(); // PANIC: expected 1, got 5
    }

    #[test]
    #[should_panic(expected = "Node ID invariant violated")]
    fn test_insert_duplicate_panics() {
        let (mut graph, _temp) = graph_with_vectors(10);
        graph.insert(0, 0).unwrap(); // OK
        graph.insert(0, 0).unwrap(); // PANIC: expected 1, got 0
    }

    #[test]
    fn test_update_existing_node() {
        let (mut graph, _temp) = graph_with_vectors(10);
        graph.insert(0, 1).unwrap();
        assert_eq!(graph.node_count(), 1);

        let mut record = NodeRecord::new(0, 2, graph.record_params());
        record.set_neighbors(0, &[1, 2, 3]);
        record.set_neighbors(1, &[10, 20]);
        graph.update_node_record(&record).unwrap();
        assert_eq!(graph.node_count(), 1); // Still 1, not 2

        let read_back = graph.read_node_record(0).unwrap();
        assert_eq!(read_back.get_neighbors(0), vec![1, 2, 3]);
        assert_eq!(read_back.get_neighbors(1), vec![10, 20]);

        let wrong_layers = NodeRecord::new(0, 1, graph.record_params());
        assert!(graph.update_node_record(&wrong_layers).is_err());
    }

    #[test]
    fn test_nodes_need_their_vector_first() {
        let (mut graph, _temp) = graph_with_vectors(1);
        graph.insert(0, 0).unwrap();
        assert!(graph.insert(1, 0).is_err());
    }
}
