//! Read-only access to format v1 and v2 files, for migration to v3 (ADR-0008, decision 8).
//!
//! v1/v2 layout: a 4 KiB header, vectors from byte 4096, and a graph zone holding a 64-byte graph
//! header and fixed-size node records (`NodeRecord`'s byte format).

use crate::header::MAGIC;
use crate::hnsw::node::{NodeId, NodeRecord, NodeRecordParams};
use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;

const HEADER_SIZE: usize = 4096;
const GRAPH_HEADER_SIZE: usize = 64;
/// Where the original sparse layout put the graph, before the header recorded its offset.
const LEGACY_GRAPH_ZONE_START: usize = 1 << 30;
const FLAG_CUSTOM_IDS: u8 = 2;

/// Whether `prefix`, the first bytes of a file, starts a v1 or v2 Chassis file.
pub(crate) fn is_legacy(prefix: &[u8]) -> bool {
    prefix.len() >= 12
        && prefix[0..8] == *MAGIC
        && matches!(u32::from_le_bytes(prefix[8..12].try_into().expect("4 bytes")), 1 | 2)
}

/// The committed state of a v1 or v2 index.
pub(crate) struct LegacyIndex {
    map: Mmap,
    dims: usize,
    graph_start: usize,
    pub params: NodeRecordParams,
    /// Committed nodes; vectors past this are leftovers of a crashed insert.
    pub count: u64,
    pub entry_point: Option<NodeId>,
    pub max_layer: u32,
    pub epoch: u32,
    pub custom_ids: bool,
}

impl LegacyIndex {
    pub fn open(file: &File, dims: u32) -> Result<Self> {
        let map = unsafe { Mmap::map(file)? };
        let u32_at = |at: usize| u32::from_le_bytes(map[at..at + 4].try_into().expect("4 bytes"));
        let u64_at = |at: usize| u64::from_le_bytes(map[at..at + 8].try_into().expect("8 bytes"));
        if map.len() < HEADER_SIZE || !is_legacy(&map) {
            crate::error::fail!(
                NotAnIndex,
                "The file is not a Chassis index: it doesn't begin with a Chassis header\nhelp: \
                 check the path; to create a new index, give a path where no file exists yet"
            );
        }
        if u32_at(12) != dims {
            crate::error::fail!(
                DimensionMismatch,
                "The index holds vectors of {} dimensions, not {dims}\nhelp: open it with {} \
                 dimensions, or create a new index at another path",
                u32_at(12),
                u32_at(12)
            );
        }
        let stored = u64_at(16);

        // The graph offset lives in the header's reserved bytes, behind a layout tag.
        let recorded = (map[24..32] == *b"CHLAYOUT" && u32_at(32) == 1)
            .then(|| u64_at(40) as usize)
            .filter(|&offset| offset != 0);
        let at_legacy_start = map.get(LEGACY_GRAPH_ZONE_START..LEGACY_GRAPH_ZONE_START + 4)
            == Some(b"HNSW".as_slice());
        let graph_start = recorded.or(at_legacy_start.then_some(LEGACY_GRAPH_ZONE_START));
        let graph = graph_start.and_then(|start| map.get(start..start + GRAPH_HEADER_SIZE));

        let Some(graph) = graph.filter(|g| g.iter().any(|&b| b != 0)) else {
            if stored > 0 {
                crate::error::fail!(
                    Corrupt,
                    "The file has {stored} vectors but no graph: it was cut short or damaged\nhelp: \
                     restore it from a backup, or rebuild it from its vectors"
                );
            }
            let params = NodeRecordParams::default();
            return Ok(Self {
                map,
                dims: dims as usize,
                graph_start: 0,
                params,
                count: 0,
                entry_point: None,
                max_layer: 0,
                epoch: 0,
                custom_ids: false,
            });
        };
        if graph[0..4] != *b"HNSW" || u32::from_le_bytes(graph[4..8].try_into()?) != 1 {
            crate::error::fail!(
                Corrupt,
                "Corrupted graph header\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }
        let g64 = |at: usize| u64::from_le_bytes(graph[at..at + 8].try_into().expect("8 bytes"));
        let g16 = |at: usize| u16::from_le_bytes(graph[at..at + 2].try_into().expect("2 bytes"));
        let params = NodeRecordParams::new(g16(28), g16(30), graph[32]);
        let count = g64(16);
        let graph_start = graph_start.expect("graph header found above");

        if stored < count {
            crate::error::fail!(
                Corrupt,
                "The graph has {count} nodes but only {stored} vectors are stored: the file is \
                 damaged\nhelp: restore it from a backup, or rebuild it from its vectors"
            );
        }
        let graph_end = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(params.record_size()))
            .and_then(|n| n.checked_add(graph_start + GRAPH_HEADER_SIZE))
            .context("Graph size overflow")?;
        if graph_end > map.len() || HEADER_SIZE + count as usize * dims as usize * 4 > map.len() {
            crate::error::fail!(
                Corrupt,
                "File truncated: it ends inside the graph\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }

        let entry = g64(8);
        Ok(Self {
            dims: dims as usize,
            graph_start,
            params,
            count,
            entry_point: (count > 0 && entry < count).then_some(entry),
            max_layer: u32::from_le_bytes(graph[24..28].try_into()?),
            epoch: u32::from_le_bytes(graph[36..40].try_into()?),
            custom_ids: graph[33] & FLAG_CUSTOM_IDS != 0,
            map,
        })
    }

    pub fn vector(&self, slot: u64) -> Vec<f32> {
        let start = HEADER_SIZE + slot as usize * self.dims * 4;
        self.map[start..start + self.dims * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect()
    }

    pub fn record(&self, slot: u64) -> Result<NodeRecord> {
        let size = self.params.record_size();
        let start = self.graph_start + GRAPH_HEADER_SIZE + slot as usize * size;
        NodeRecord::from_bytes(&self.map[start..start + size], self.params)
            .map_err(|e| anyhow::anyhow!("Invalid node record {slot}: {e}"))
    }

    /// Whether a committed flush deleted this node; marks above the header's epoch were written
    /// by a flush that never committed.
    pub fn is_deleted(&self, record: &NodeRecord) -> bool {
        (1..=self.epoch).contains(&record.header.deleted_epoch)
    }
}
