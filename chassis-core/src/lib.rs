//! Chassis - Embeddable on-disk vector storage engine
//!
//! Chassis is a local-first vector storage engine designed for embedding-based
//! search in edge devices, mobile apps, and local-first software. It's built
//! in Rust and runs anywhere from a Raspberry Pi to a data center.
//!
//! # Features
//!
//! - On-disk storage using memory-mapped I/O
//! - One file: two checksummed header copies, then append-only regions aligned to 64 KiB
//! - One writer per file; concurrent searches from its threads and from `IndexReader`s in other
//!   processes
//! - Explicit durability control via flush()
//! - Zero external dependencies (no daemons or services)
//!
//! # Example
//!
//! ```no_run
//! use chassis_core::{VectorIndex, IndexOptions};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Open or create an index
//! let mut index = VectorIndex::open("embeddings.chassis", 768, IndexOptions::default())?;
//!
//! // Insert vectors
//! let embedding = vec![0.1; 768];
//! let id = index.add(&embedding)?;
//!
//! // Flush to disk for durability
//! index.flush()?;
//!
//! // Search for nearest neighbors
//! let query = vec![0.1; 768];
//! let results = index.search(&query, 10)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Design Philosophy
//!
//! Chassis is intentionally simple and focused. It does not aim to be:
//! - A database server
//! - A cloud service
//! - A distributed system
//! - A query engine
//!
//! These concerns are left to the application layer. Chassis is a storage
//! primitive, like SQLite for relational data.

pub mod distance;
mod header;
mod hnsw;
mod legacy;
mod storage;

#[cfg(test)]
mod power_loss;

/// Experiment switches for the x86 lab; compiled only with `--cfg lab`.
#[cfg(lab)]
pub mod lab {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static DISTANCES: AtomicU64 = AtomicU64::new(0);

    /// Distance computations so far (counted only with `--cfg lab_count`).
    pub fn distances() -> u64 {
        DISTANCES.load(Ordering::Relaxed)
    }

    /// `LAB_FAST=1` runs unfiltered layer searches through `search_layer_lean`.
    pub fn fast() -> bool {
        static FAST: OnceLock<bool> = OnceLock::new();
        *FAST.get_or_init(|| std::env::var("LAB_FAST").is_ok_and(|v| v == "1"))
    }

    /// `LAB_BUILD=own,fill`: `own` caps the neighbors a new node selects on layer 0 (the list
    /// still holds M0; hnswlib and the paper select M), `fill` 0 turns off filling a selection
    /// that kept fewer than half.
    pub fn build() -> (usize, bool) {
        static BUILD: OnceLock<(usize, bool)> = OnceLock::new();
        *BUILD.get_or_init(|| {
            let spec = std::env::var("LAB_BUILD").unwrap_or_else(|_| "9999,1".into());
            let p: Vec<usize> = spec.split(',').map(|x| x.parse().expect("LAB_BUILD")).collect();
            (p[0], p[1] != 0)
        })
    }

    /// `LAB_PF=mode,lines,ahead,list`: mode 0 prefetches every unvisited neighbor before any
    /// distance, 1 prefetches the neighbor `ahead` places on while computing one, 2 none, 3 like
    /// 0 but reads the first line instead of prefetching it (a read can't be dropped on a TLB
    /// miss), 4 like 0 but reads every line; `lines` cache lines per vector; `list` 1 prefetches
    /// the next candidate's neighbor list, 2 the list of every candidate as it is queued.
    pub fn pf() -> (u8, usize, usize, u8) {
        static PF: OnceLock<(u8, usize, usize, u8)> = OnceLock::new();
        *PF.get_or_init(|| {
            let spec = std::env::var("LAB_PF").unwrap_or_else(|_| "0,8,0,0".into());
            let p: Vec<usize> = spec.split(',').map(|x| x.parse().expect("LAB_PF")).collect();
            (p[0] as u8, p[1], p[2], p[3] as u8)
        })
    }
}

#[cfg(feature = "internals")]
pub use hnsw::*;

pub use distance::{DistanceMetric, cosine_distance, euclidean_distance};
pub use header::{MAGIC, VERSION};
pub use hnsw::{HnswBuilder, HnswGraph, HnswParams, NodeRecordParams, SearchResult};
pub use storage::Storage;

use anyhow::Result;
use hnsw::layer_from_uniform;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Configuration options for VectorIndex
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Maximum connections per node (M parameter)
    pub max_connections: u16,

    /// Construction quality parameter (efConstruction)
    pub ef_construction: usize,

    /// Search quality parameter (efSearch)
    pub ef_search: usize,

    /// How vectors are compared. Set when the index is created; reopening needs the same one.
    pub metric: DistanceMetric,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// Public facade for Chassis vector index
///
/// This struct provides the main API for interacting with a Chassis index.
/// It handles crash consistency, ghost node recovery, and provides a clean
/// interface for vector insertion and search.
#[derive(Debug)]
pub struct VectorIndex {
    /// Internal HNSW graph (owns the storage)
    graph: HnswGraph,

    /// Configuration options
    options: IndexOptions,

    /// Layer multiplier cache: 1.0 / ln(M)
    ml: f32,

    /// id → slot, built on first use once some id differs from its slot (ADR-0007)
    ids: Option<IdMap>,

    /// One past the largest id ever stored, valid once `ids` is built
    next_id: u64,

    /// Where the index file is, for `compact` to replace it
    path: PathBuf,
}

impl VectorIndex {
    /// Open or create a vector index
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    /// * `dims` - Number of dimensions per vector
    /// * `options` - Index configuration options
    ///
    /// A file in format v1 or v2 is migrated to v3 first (ADR-0008). After a crash, slots written
    /// after the last flush are simply reused.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or created
    /// - The file is corrupted
    /// - Dimension mismatch with existing index
    /// - The file was created with a different `max_connections`
    pub fn open<P: AsRef<Path>>(path: P, dims: u32, options: IndexOptions) -> Result<Self> {
        let params = hnsw_params(&options);
        // Compaction and migration rename a new file over this one: through a symlink that would
        // replace the link and leave the file it points to behind.
        let path = std::fs::canonicalize(&path).unwrap_or_else(|_| path.as_ref().to_path_buf());
        let storage = Storage::open_with(&path, dims, params.to_record_params(), options.metric)?;
        if storage.metric() != options.metric {
            anyhow::bail!(
                "Index was created with {:?} distance but opened with {:?}",
                storage.metric(),
                options.metric
            );
        }
        let graph = HnswGraph::open_no_reset(storage, params)?;
        // This is the one writer, so no compaction is running: a copy here is a crashed one's.
        let _ = std::fs::remove_file(storage::sibling(&path, "compacting"));
        Ok(Self { graph, ml: params.ml, options, ids: None, next_id: 0, path })
    }

    /// Add a vector and return the id assigned to it: one past the largest id used so far.
    ///
    /// Ids are never reused by `add`, even after a delete. Use `add_with_id` to choose the id.
    ///
    /// The vector, its id and its graph record go into the next slot; the slot counts only once
    /// linking finishes, so a failed or interrupted add leaves a slot the next add reuses.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector dimensions don't match index dimensions
    /// - Storage write fails
    /// - Graph write fails
    pub fn add(&mut self, vector: &[f32]) -> Result<u64> {
        let id = self.next_ids(1)?.start;
        self.insert(id, vector)?;
        Ok(id)
    }

    /// The ids `add` gives the next `count` vectors. `u64::MAX` is reserved, so they can run out.
    fn next_ids(&mut self, count: u64) -> Result<std::ops::Range<u64>> {
        let first = if self.graph.custom_ids {
            self.ids()?;
            self.next_id
        } else {
            self.graph.node_count()
        };
        match first.checked_add(count) {
            Some(end) => Ok(first..end),
            None => anyhow::bail!("No ids are left; choose free ones with add_with_id"),
        }
    }

    /// Add a vector under the caller's `id`. Search results report this id.
    ///
    /// # Errors
    ///
    /// Returns an error if a live vector already has `id` (delete it first to replace it), if
    /// `id` is `u64::MAX` (reserved), or for the same reasons as `add`.
    pub fn add_with_id(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if id == u64::MAX {
            anyhow::bail!("Id u64::MAX is reserved");
        }
        if self.slot_of(id)?.is_some() {
            anyhow::bail!("Id {id} already exists; delete it first to replace it");
        }
        if !self.graph.custom_ids && id != self.graph.node_count() {
            self.ids()?;
            self.graph.custom_ids = true;
        }
        self.insert(id, vector)
    }

    /// Add many vectors at once, given back to back, `dims` floats each, and return their ids,
    /// assigned as `add` assigns them. Linking runs on every core, so a large batch builds many
    /// times faster than adding one vector at a time; the graph then depends on thread timing
    /// (ADR-0010). All or nothing: after an error, none of the batch is added.
    ///
    /// # Errors
    ///
    /// Returns an error if `vectors` isn't a whole number of vectors of the index's dimensions,
    /// or if storing or linking them fails.
    pub fn add_batch(&mut self, vectors: &[f32]) -> Result<Vec<u64>> {
        let count = self.batch_len(vectors)? as u64;
        let ids: Vec<u64> = self.next_ids(count)?.collect();
        self.insert_batch(&ids, vectors)?;
        Ok(ids)
    }

    /// `add_with_id` for many vectors at once: `ids[i]` is the id of the `i`-th vector in
    /// `vectors`. Linking runs on every core, as for `add_batch`. All or nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the counts differ, an id is `u64::MAX`, repeats, or already exists, or
    /// storing or linking fails.
    pub fn add_batch_with_ids(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if self.batch_len(vectors)? != ids.len() {
            anyhow::bail!("{} ids for {} vectors", ids.len(), vectors.len() / self.dims());
        }
        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if id == u64::MAX {
                anyhow::bail!("Id u64::MAX is reserved");
            }
            if !seen.insert(id) {
                anyhow::bail!("Id {id} appears twice in the batch");
            }
            if self.slot_of(id)?.is_some() {
                anyhow::bail!("Id {id} already exists; delete it first to replace it");
            }
        }
        let first = self.graph.node_count();
        if !self.graph.custom_ids && ids.iter().zip(first..).any(|(&id, slot)| id != slot) {
            self.ids()?;
            self.graph.custom_ids = true;
        }
        self.insert_batch(ids, vectors)
    }

    fn dims(&self) -> usize {
        self.graph.storage.dimensions() as usize
    }

    /// Vectors in a batch, which must be a whole number of them.
    fn batch_len(&self, vectors: &[f32]) -> Result<usize> {
        if !vectors.len().is_multiple_of(self.dims()) {
            anyhow::bail!(
                "Vector dimension mismatch: {} floats is not a whole number of {}-dimensional \
                 vectors",
                vectors.len(),
                self.dims()
            );
        }
        Ok(vectors.len() / self.dims())
    }

    /// Writes every vector and an empty record for it, then links them all on every core.
    fn insert_batch(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        match self.graph.storage.metric() {
            DistanceMetric::Euclidean => self.insert_stored(ids, vectors),
            DistanceMetric::Cosine => {
                let rows = vectors.chunks_exact(self.dims()).map(distance::unit);
                self.insert_stored(ids, &rows.collect::<Result<Vec<_>>>()?.concat())
            }
        }
    }

    /// `insert_batch` for vectors already in their stored form: unit length in a cosine index.
    fn insert_stored(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let layers: Vec<usize> = ids.iter().map(|_| self.select_layer()).collect();
        let first = self.graph.node_count();
        let saved = (self.graph.entry_point, self.graph.max_layer);
        let linked = self.write_and_link(first, ids, vectors, &layers);
        if linked.is_err() {
            self.graph.node_count = first;
            (self.graph.entry_point, self.graph.max_layer) = saved;
            self.graph.storage.truncate_logical(first);
            return linked;
        }
        self.graph.storage.publish_routing(self.graph.node_count);
        if let Some(map) = &mut self.ids {
            map.extend(ids.iter().copied().zip(first..));
        }
        self.next_id = self.next_id.max(ids.iter().max().map_or(0, |&id| id + 1));
        Ok(())
    }

    fn write_and_link(
        &mut self,
        first: u64,
        ids: &[u64],
        vectors: &[f32],
        layers: &[usize],
    ) -> Result<()> {
        let dims = self.dims();
        let storage = &mut self.graph.storage;
        storage.truncate_logical(first);
        for ((&id, vector), &layer) in ids.iter().zip(vectors.chunks_exact(dims)).zip(layers) {
            let slot = storage.append(id, vector)?;
            storage.write_record(slot, &vec![Vec::new(); layer + 1])?;
        }
        // An empty graph's first node is its entry point, with nothing to link to.
        let start = if self.graph.entry_point.is_none() {
            (self.graph.entry_point, self.graph.max_layer) = (Some(first), layers[0]);
            1
        } else {
            0
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let ef = self.options.ef_construction;
        self.graph.link_batch(first + start as u64, &layers[start..], ef, threads.min(ids.len()))
    }

    /// Delete the vector with `id`. Returns false if no live vector has that id.
    ///
    /// Search stops returning it immediately. Like an add, the delete is durable after the next
    /// `flush()`; a crash before then rolls it back.
    ///
    /// # Errors
    ///
    /// Returns an error if the index file cannot be read.
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        let Some(slot) = self.slot_of(id)? else {
            return Ok(false);
        };
        self.graph.mark_deleted(slot)?;
        if let Some(ids) = &mut self.ids {
            ids.remove(&id);
        }
        Ok(true)
    }

    /// Slot of the live vector with `id`.
    fn slot_of(&mut self, id: u64) -> Result<Option<u64>> {
        if !self.graph.custom_ids {
            let live = id < self.graph.node_count() && !self.graph.is_deleted(id)?;
            return Ok(live.then_some(id));
        }
        Ok(self.ids()?.get(&id).copied())
    }

    /// The id → slot table of live vectors, built by one scan of the graph on first use.
    // ponytail: O(n) scan once per process when custom ids are in use; persist the table if
    // open-to-first-write latency on large indexes matters.
    fn ids(&mut self) -> Result<&mut IdMap> {
        if self.ids.is_none() {
            let live = self.graph.node_count() - self.graph.deleted_count;
            let mut pairs = Vec::with_capacity(live as usize);
            // A compaction removed the deleted slots that showed which ids were used (ADR-0011).
            let mut next_id = self.graph.storage.next_id();
            for slot in 0..self.graph.node_count() {
                let id = self.graph.id_of(slot)?;
                next_id = next_id.max(id + 1);
                if !self.graph.is_deleted(slot)? {
                    pairs.push((id, slot));
                }
            }
            // Filling from a Vec keeps the loop small enough to overlap the table's cache misses.
            let mut ids = IdMap::with_capacity_and_hasher(pairs.len(), IdHasher::default());
            ids.extend(pairs);
            self.next_id = next_id;
            self.ids = Some(ids);
        }
        Ok(self.ids.as_mut().expect("ids built above"))
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        // Validate dimensions
        let dims = self.graph.storage.dimensions() as usize;
        if vector.len() != dims {
            anyhow::bail!("Vector dimension mismatch: expected {}, got {}", dims, vector.len());
        }

        let unit;
        let vector = match self.graph.storage.metric() {
            DistanceMetric::Euclidean => vector,
            DistanceMetric::Cosine => {
                unit = distance::unit(vector)?;
                &unit[..]
            }
        };

        // STEP 1: Persist vector and id, reusing the slot of an add that failed part way
        let slot = self.graph.node_count();
        self.graph.storage.truncate_logical(slot);
        let new_id = self.graph.storage.append(id, vector)?;

        // STEP 2: Determine layer for new node
        let layer = self.select_layer();
        let layer_count = layer + 1;

        // STEP 3: Neighbor selection (in-memory phase); an empty graph has none
        let neighbors = match self.graph.entry_point {
            Some(entry) if self.graph.node_count() > 0 => {
                let start = (entry, self.graph.max_layer);
                let ef = self.options.ef_construction;
                self.graph.find_neighbors(vector, new_id, layer, start, ef)?
            }
            _ => vec![vec![]; layer_count],
        };

        // STEP 4: Atomic write (disk phase)
        // Node is written but invisible (node_count not incremented)
        self.graph.write_node_with_id_and_backlinks(new_id, id, layer_count, &neighbors)?;

        // STEP 5: Publish (commit phase)
        // Node becomes visible to readers
        self.graph.publish_node(new_id, layer_count)?;

        if let Some(ids) = &mut self.ids {
            ids.insert(id, new_id);
        }
        self.next_id = self.next_id.max(id + 1);
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (must match index dimensions)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// Returns a vector of search results, sorted by distance (ascending), identified by the
    /// ids given to `add`/`add_with_id`. Deleted vectors are never returned.
    ///
    /// # Errors
    ///
    /// Returns an error if query dimensions don't match index dimensions
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        search_graph(&self.graph, query, k, self.options.ef_search, None)
    }

    /// Like `search`, among only the vectors whose id `filter` accepts. When walking the graph
    /// would cost more than checking every vector, as when few match, it checks every vector and
    /// returns the exact nearest.
    ///
    /// # Errors
    ///
    /// Returns an error if query dimensions don't match index dimensions
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u64) -> bool,
    ) -> Result<Vec<SearchResult>> {
        search_graph(&self.graph, query, k, self.options.ef_search, Some(&filter))
    }

    /// Flush all changes to disk
    ///
    /// This method ensures durability by:
    /// 1. Flushing vector data to disk
    /// 2. Flushing graph metadata to disk
    ///
    /// # Performance Warning
    ///
    /// This operation is expensive (1-50ms depending on storage device).
    /// Batch multiple add() calls and flush() once at the end.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails
    pub fn flush(&mut self) -> Result<()> {
        self.graph.commit()
    }

    /// Rewrites the index without its deleted vectors and with a newly built graph, then replaces
    /// the file with the copy (ADR-0011). Ids don't change. Like `flush`, it makes every add and
    /// delete so far durable. It takes as long as building the index, on every core, and needs
    /// free disk for a second copy of the live vectors. Readers in other processes keep searching
    /// and move to the new file by themselves.
    ///
    /// # Errors
    ///
    /// Returns an error, leaving the index as it was, if the copy can't be written or, on
    /// Windows, another process has the index open.
    pub fn compact(&mut self) -> Result<()> {
        // Windows has to close this file before the rename and reopen it if the rename fails,
        // which shows only what was flushed.
        self.flush()?;
        let temp = storage::sibling(&self.path, "compacting");
        let swapped = self.compacted_copy(&temp).and_then(|fresh| {
            self.graph.storage.set_superseded(true)?;
            self.swap_in(fresh, &temp)
        });
        if swapped.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        swapped
    }

    /// A flushed index at `temp` holding this one's live vectors under their ids.
    fn compacted_copy(&mut self, temp: &Path) -> Result<Self> {
        match std::fs::remove_file(temp) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
        self.ids()?;
        let dims = self.dims();
        let mut fresh = Self::open(temp, dims as u32, self.options.clone())?;
        fresh.ids()?;
        let count = self.graph.node_count();
        // About 32 MiB of vectors at a time.
        let chunk = ((32 << 20) / (dims * 4)).max(1);
        let (mut ids, mut vectors) = (Vec::new(), Vec::new());
        for start in (0..count).step_by(chunk) {
            ids.clear();
            vectors.clear();
            for slot in start..(start + chunk as u64).min(count) {
                if !self.graph.is_deleted(slot)? {
                    ids.push(self.graph.id_of(slot)?);
                    vectors.extend_from_slice(self.graph.storage.get_vector_slice(slot)?);
                }
            }
            let first = fresh.graph.node_count();
            fresh.graph.custom_ids |= ids.iter().zip(first..).any(|(&id, slot)| id != slot);
            fresh.insert_stored(&ids, &vectors)?;
        }
        // The deleted slots showed which ids were used; without them the header has to, and
        // `add` must stop taking the slot number for the next id.
        fresh.graph.custom_ids |= self.next_id != fresh.graph.node_count();
        fresh.next_id = self.next_id;
        fresh.graph.storage.carry_over(self.next_id, self.graph.storage.epoch() + 1);
        fresh.flush()?;
        Ok(fresh)
    }

    /// Renames the compacted copy at `temp` over this index's file and continues on it. A failed
    /// rename leaves this index on its old file, with the superseded flag cleared.
    #[cfg(unix)]
    fn swap_in(&mut self, mut fresh: Self, temp: &Path) -> Result<()> {
        if let Err(e) = std::fs::rename(temp, &self.path) {
            self.graph.storage.set_superseded(false)?;
            return Err(e.into());
        }
        fresh.path = std::mem::take(&mut self.path);
        *self = fresh;
        // Either file may be at the path after a power loss; they hold the same vectors.
        storage::sync_dir(&self.path)
    }

    /// Windows can't rename over a file with open handles, so this index moves to the copy before
    /// the rename, and reopens the old file, which clears its superseded flag, if that fails. If
    /// the old file can't be reopened either, this index is left on the copy, which the next open
    /// of the path removes, so it refuses to flush anything more.
    #[cfg(windows)]
    fn swap_in(&mut self, fresh: Self, temp: &Path) -> Result<()> {
        let (path, dims, options) = (self.path.clone(), self.dims() as u32, self.options.clone());
        drop(std::mem::replace(self, fresh));
        if let Err(e) = storage::rename_over(temp, &path) {
            match Self::open(&path, dims, options) {
                Ok(old) => *self = old,
                Err(reopen) => {
                    self.graph.storage.poison();
                    return Err(e.context(format!("and the index couldn't be reopened: {reopen}")));
                }
            }
            return Err(e);
        }
        self.path = path;
        Ok(())
    }

    /// Lab: `MADV_HUGEPAGE` on every region.
    #[cfg(all(lab, target_os = "linux"))]
    pub fn lab_huge(&self) -> usize {
        self.graph.storage.lab_huge()
    }

    /// Get the number of live (not deleted) vectors in the index
    pub fn len(&self) -> u64 {
        self.graph.node_count() - self.graph.deleted_count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimensionality of vectors in this index
    pub fn dimensions(&self) -> u32 {
        self.graph.storage.dimensions()
    }

    /// The distance metric the index was created with
    pub fn metric(&self) -> DistanceMetric {
        self.graph.storage.metric()
    }

    // Private helper methods

    /// Select layer for a new node using exponential decay
    fn select_layer(&self) -> usize {
        let uniform: f32 = rand::random();
        layer_from_uniform(uniform, self.ml, self.graph.record_params.max_layers)
    }
}

type IdMap = HashMap<u64, u64, IdHasher>;

/// Hashes an id with one 64×64→128-bit multiply folded to 64 bits, keyed per table: ids are
/// integers, so SipHash's byte-stream work buys nothing here.
#[derive(Clone, Copy)]
struct IdHasher(u64);

impl Default for IdHasher {
    fn default() -> Self {
        Self(rand::random())
    }
}

impl std::hash::BuildHasher for IdHasher {
    type Hasher = Self;
    fn build_hasher(&self) -> Self {
        *self
    }
}

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
    fn write_u64(&mut self, id: u64) {
        let product = u128::from(id ^ self.0) * 0x9e37_79b9_7f4a_7c15;
        self.0 = (product as u64) ^ ((product >> 64) as u64);
    }
}

/// Searches as the file's metric defines, reporting the caller's ids. A cosine index stores unit
/// vectors, so the query is scaled too, and for unit vectors 1 − cos = L2² / 2.
fn search_graph(
    graph: &HnswGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<&dyn Fn(u64) -> bool>,
) -> Result<Vec<SearchResult>> {
    let dims = graph.storage.dimensions() as usize;
    if query.len() != dims {
        anyhow::bail!("Query dimension mismatch: expected {}, got {}", dims, query.len());
    }
    let search = |query: &[f32]| match filter {
        None => graph.search(query, k, ef),
        Some(filter) => graph.search_filtered(query, k, ef, &|slot| {
            Ok(filter(if graph.custom_ids { graph.id_of(slot)? } else { slot }))
        }),
    };
    let mut results = match graph.storage.metric() {
        DistanceMetric::Euclidean => search(query)?,
        DistanceMetric::Cosine => {
            let query = distance::unit(query)?;
            let mut results = search(&query)?;
            for result in &mut results {
                // Rounding can take antipodal vectors a hair past 2.
                result.distance = (result.distance * result.distance / 2.0).min(2.0);
            }
            results
        }
    };
    if graph.custom_ids {
        for result in &mut results {
            result.id = graph.id_of(result.id)?;
        }
    }
    Ok(results)
}

fn hnsw_params(options: &IndexOptions) -> HnswParams {
    HnswParams {
        max_connections: options.max_connections,
        ef_construction: options.ef_construction,
        ef_search: options.ef_search,
        ml: 1.0 / f32::from(options.max_connections).ln(),
        max_layers: 16, // Fixed for now
    }
}

/// Searches an index while a `VectorIndex` in another process may be writing it (ADR-0008,
/// decision 5). Any number of readers can open the same file; they take no lock.
///
/// Every search first takes a new snapshot: it returns vectors committed by the writer's last
/// `flush()`, and none deleted by it, while routing through everything the writer has added since,
/// so a long unflushed batch costs readers little recall.
#[derive(Debug)]
pub struct IndexReader {
    graph: HnswGraph,
    ef_search: usize,
    /// What it was opened with, to open the file that replaces this one after a compaction
    path: PathBuf,
    dims: u32,
    options: IndexOptions,
}

impl IndexReader {
    /// Opens `path` for searching.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing, uses file format 1 or 2 (open it once with
    /// `VectorIndex::open` to migrate it), has other dimensions, was created with another
    /// `max_connections`, or is corrupt.
    pub fn open<P: AsRef<Path>>(path: P, dims: u32, options: IndexOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let storage = Storage::open_read_only(&path, dims)?;
        let graph = HnswGraph::reader(storage, hnsw_params(&options))?;
        Ok(Self { graph, ef_search: options.ef_search, path, dims, options })
    }

    /// Takes a new snapshot, moving to the file a compaction put at the path (ADR-0011). Until
    /// that file is there, and if it never arrives, this file's snapshot stands.
    fn sync(&mut self) -> Result<()> {
        self.graph.refresh()?;
        if self.graph.storage.superseded() && !self.graph.storage.is_at(&self.path)? {
            *self = Self::open(&self.path, self.dims, self.options.clone())?;
        }
        Ok(())
    }

    /// Search for the `k` nearest committed vectors, by the ids given to `add`/`add_with_id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query has other dimensions or the file is corrupt.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.sync()?;
        search_graph(&self.graph, query, k, self.ef_search, None)
    }

    /// Like `search`, among only the vectors whose id `filter` accepts; see
    /// `VectorIndex::search_filtered`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query has other dimensions or the file is corrupt.
    pub fn search_filtered(
        &mut self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u64) -> bool,
    ) -> Result<Vec<SearchResult>> {
        self.sync()?;
        search_graph(&self.graph, query, k, self.ef_search, Some(&filter))
    }

    /// Get the dimensionality of vectors in this index
    pub fn dimensions(&self) -> u32 {
        self.graph.storage.dimensions()
    }

    /// The distance metric the index was created with
    pub fn metric(&self) -> DistanceMetric {
        self.graph.storage.metric()
    }

    /// Take a new snapshot without searching, so `len` reflects the writer's latest flush.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupt.
    pub fn refresh(&mut self) -> Result<()> {
        self.sync()
    }

    /// Identifies the last snapshot: committed slots and the last delete flush. Two snapshots with
    /// the same value have the same contents, so it can key a cache of search results, unless a
    /// power loss discarded a flush readers had already seen.
    pub fn snapshot(&self) -> (u64, u64) {
        self.graph.view.map_or((0, 0), |view| (view.committed, view.epoch))
    }

    /// Live vectors as of the last snapshot, taken by `search` or `refresh`.
    pub fn len(&self) -> u64 {
        self.graph.view.map_or(0, |view| view.committed) - self.graph.deleted_count
    }

    /// Whether the last snapshot had no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn random_vector(id: u64) -> Vec<f32> {
        let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..16)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    /// 2,500 vectors: sampled every second slot, so a filter on odd slots meets an unjittered
    /// sample nowhere.
    fn filter_index(file: &NamedTempFile) -> VectorIndex {
        let options = IndexOptions { ef_construction: 64, ..IndexOptions::default() };
        let mut index = VectorIndex::open(file.path(), 16, options).unwrap();
        for id in 0..2500 {
            index.add(&random_vector(id)).unwrap();
        }
        index
    }

    #[test]
    fn test_filtered_search_estimates_matches_and_picks_its_path() {
        let file = NamedTempFile::new().unwrap();
        let mut index = filter_index(&file);
        let graph = &index.graph;
        let estimate = |allow: &dyn Fn(u64) -> bool| graph.estimated_matches(&|s| Ok(allow(s)));
        assert_eq!(estimate(&|_| true).unwrap(), 2500);
        assert_eq!(estimate(&|_| false).unwrap(), 0);
        let odd = estimate(&|slot| slot % 2 == 1).unwrap();
        assert!((1000..=1500).contains(&odd), "odd slots estimated at {odd}");

        let query = random_vector(99_999);
        let nearest = index.search(&query, 1).unwrap()[0].id;
        let path = |allow: &dyn Fn(u64) -> bool| {
            index.graph.search_filtered_reporting(&query, 10, 50, &|s| Ok(allow(s))).unwrap()
        };
        let (found, scanned) = path(&|slot| slot != nearest);
        assert!(!scanned && found.len() == 10, "a broad filter walks the graph");
        let (found, scanned) = path(&|slot| slot % 250 == 7);
        assert!(scanned && found.len() == 10, "a selective filter scans");

        for slot in (0..2500).step_by(2) {
            index.delete(slot).unwrap();
        }
        let live = index.graph.estimated_matches(&|_| Ok(true)).unwrap();
        assert!((1000..=1500).contains(&live), "deleted slots counted as matches: {live}");
    }

    /// A rename that fails after the copy is built: the path is a directory with a file in it.
    #[cfg(unix)]
    #[test]
    fn test_a_failed_swap_clears_the_flag_and_removes_the_copy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut index = VectorIndex::open(&path, 16, IndexOptions::default()).unwrap();
        for id in 0..50 {
            index.add(&random_vector(id)).unwrap();
        }
        index.flush().unwrap();
        let blocked = dir.path().join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("file"), b"").unwrap();
        index.path = blocked.clone();

        assert!(index.compact().is_err());
        assert!(!dir.path().join("blocked.compacting").exists(), "the copy was left behind");
        assert!(!index.graph.storage.superseded(), "readers would wait for a file that won't come");
        index.path = path.clone();
        assert_eq!(index.search(&random_vector(7), 1).unwrap()[0].id, 7);
        let mut reader = IndexReader::open(&path, 16, IndexOptions::default()).unwrap();
        assert!(!reader.graph.storage.superseded());
        assert_eq!(reader.search(&random_vector(7), 1).unwrap()[0].id, 7);
        index.compact().unwrap();
        assert_eq!(index.len(), 50);
    }

    #[test]
    fn test_a_superseded_flag_left_by_a_crash_is_cleared() {
        let file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(file.path(), 16, IndexOptions::default()).unwrap();
        for id in 0..50 {
            index.add(&random_vector(id)).unwrap();
        }
        index.flush().unwrap();
        // A compaction that died after flagging the file and before renaming its copy over it.
        index.graph.storage.set_superseded(true).unwrap();
        drop(index);

        // The file is still the one at the path, so a reader's snapshot of it stands.
        let mut reader = IndexReader::open(file.path(), 16, IndexOptions::default()).unwrap();
        assert!(reader.graph.storage.superseded());
        assert_eq!(reader.search(&random_vector(7), 1).unwrap()[0].id, 7);

        let mut index = VectorIndex::open(file.path(), 16, IndexOptions::default()).unwrap();
        assert!(!index.graph.storage.superseded(), "the next writer clears the flag");
        index.add(&random_vector(50)).unwrap();
        index.flush().unwrap();
        assert_eq!(reader.search(&random_vector(50), 1).unwrap()[0].id, 50);
        assert!(!reader.graph.storage.superseded());
    }

    #[test]
    fn test_batch_entry_point_is_on_the_top_layer() {
        for existing in [0, 50] {
            let file = NamedTempFile::new().unwrap();
            let mut index = VectorIndex::open(file.path(), 16, IndexOptions::default()).unwrap();
            for id in 0..existing {
                index.add(&random_vector(id)).unwrap();
            }
            let vectors: Vec<f32> = (existing..existing + 3000).flat_map(random_vector).collect();
            index.add_batch(&vectors).unwrap();
            let graph = &index.graph;
            let layers = |slot| graph.storage.layer_count(slot).unwrap() - 1;
            let top = (0..graph.node_count()).map(layers).max().unwrap();
            assert!(top > 0, "3,050 draws should reach layer 1");
            assert_eq!((graph.max_layer, layers(graph.entry_point.unwrap())), (top, top));
        }
    }

    #[test]
    fn test_filtered_scan_is_exact_with_a_nan_vector() {
        let file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(file.path(), 16, IndexOptions::default()).unwrap();
        index.add(&[f32::NAN; 16]).unwrap();
        for id in 1..300 {
            index.add(&random_vector(id)).unwrap();
        }
        let allow = |slot: u64| slot.is_multiple_of(10);
        let (found, scanned) = index
            .graph
            .search_filtered_reporting(&random_vector(150), 3, 50, &|s| Ok(allow(s)))
            .unwrap();
        assert!(scanned);
        assert_eq!(found[0].id, 150, "{found:?}");
    }

    #[test]
    fn test_vector_index_create_and_add() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.dimensions(), 128);

        // Add a vector
        let vector = vec![0.1; 128];
        let id = index.add(&vector).unwrap();

        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_vector_index_dimension_mismatch() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Try to add vector with wrong dimensions
        let vector = vec![0.1; 64];
        let result = index.add(&vector);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_vector_index_search() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Add some vectors
        for i in 0..10 {
            let mut vector = vec![0.0; 128];
            vector[0] = i as f32 / 10.0;
            index.add(&vector).unwrap();
        }

        // Search
        let query = vec![0.5; 128];
        let results = index.search(&query, 5).unwrap();

        assert!(results.len() <= 5);
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_vector_index_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_owned();

        // Create and populate index
        {
            let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

            for i in 0..5 {
                let mut vector = vec![0.0; 128];
                vector[0] = i as f32;
                index.add(&vector).unwrap();
            }

            index.flush().unwrap();
        }

        // Reopen and verify
        {
            let index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();
            assert_eq!(index.len(), 5);

            let mut query = vec![0.0; 128];
            query[0] = 2.0;

            let results = index.search(&query, 3).unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_ghost_node_recovery() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_owned();

        // Simulate ghost node: insert vector but don't publish node
        {
            let storage = Storage::open(&path, 128).unwrap();
            let params = HnswParams::default();
            let mut graph = HnswGraph::open(storage, params).unwrap();

            // A vector written but never linked when the graph flushes (creates ghost node)
            let vector = vec![1.0; 128];
            graph.storage.insert(&vector).unwrap();
            graph.commit().unwrap();

            // Don't add to graph - this creates a ghost node
            assert_eq!(graph.storage.count(), 1);
            assert_eq!(graph.node_count(), 0);
        }

        // Reopen with VectorIndex - should handle ghost node
        {
            let mut index = VectorIndex::open(&path, 128, IndexOptions::default()).unwrap();

            // Ghost node should be ignored (not counted)
            assert_eq!(index.len(), 0);

            // Next add should reclaim the ghost node space
            let vector = vec![2.0; 128];
            let id = index.add(&vector).unwrap();

            // Should reuse the ghost node's ID (0)
            assert_eq!(id, 0);
            assert_eq!(index.len(), 1);
        }
    }

    #[test]
    fn test_diverse_neighbor_selection() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut index = VectorIndex::open(temp_file.path(), 128, IndexOptions::default()).unwrap();

        // Add vectors in a line (should create diverse neighborhoods)
        for i in 0..20 {
            let mut vector = vec![0.0; 128];
            vector[0] = i as f32;
            index.add(&vector).unwrap();
        }

        // Verify graph was built
        assert_eq!(index.len(), 20);

        // Search should work
        let query = vec![10.0; 128];
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
    }
}
