//! File storage, format v3 (ADR-0008).
//!
//! Two header copies, then append-only regions; a committed region is mapped once and never moves:
//! - **Segments** hold slots as three arrays: 24-byte slot headers (id, metadata reference, deleted
//!   epoch), vectors, and level-0 graph records (`8 + 4·M0` bytes: a head word, then neighbors).
//! - **Heap chunks** hold upper-layer neighbor lists for the few nodes above layer 0.
//! - **Table pages** list where each segment and heap chunk starts.
//!
//! A flush makes data durable, then writes the header copy that does not hold the newest header.
//! Readers in other processes map the file read-only and take a snapshot per query; a writer
//! publishes what it adds since its last commit in the live page (ADR-0008, decision 5).
//!
//! Adds change the lists of committed nodes in place before they are committed themselves. The
//! lists as committed are saved in `<name>.undo` first, and a writer opening the file after a
//! crash writes them back (ADR-0012).

use crate::distance::{DistanceMetric, Element};
use crate::error::{ErrorKind, fail, raised};
use crate::half::{self, Precision};
use crate::header::{FLAG_SUPERSEDED, FileHeader, HEADER_STRIDE, MAX_TABLE_PAGES, REGIONS_START};
use crate::hnsw::node::NodeRecordParams;
use crate::legacy::{LegacyIndex, is_legacy};
use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions, MmapRaw};
use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, Mutex, PoisonError};
use xxhash_rust::xxh3::xxh3_64_with_seed;

#[cfg(target_endian = "big")]
compile_error!("Chassis files are little-endian and read in place");

/// Every region starts at a multiple of this: the Windows mapping granularity, and a multiple of
/// every supported page size.
const REGION_ALIGN: u64 = 64 * 1024;
const SLOT_HEADER: usize = 24;
const MAX_DIMENSIONS: u32 = 4096;
/// An empty neighbor entry.
pub(crate) const EMPTY: u32 = u32::MAX;

#[cfg(not(test))]
mod sizes {
    pub const SEGMENT_BASE_LOG2: u8 = 10;
    pub const MAX_SEGMENT_BYTES: usize = 256 << 20;
    pub const HEAP_BASE_MIN_LOG2: u8 = 16;
    pub const MAX_CHUNK_LOG2: u8 = 28;
    pub const TABLE_PAGE_LOG2: u8 = 13;
}
/// Small in unit tests, so short workloads cross segment, heap chunk and table page boundaries.
#[cfg(test)]
mod sizes {
    pub const SEGMENT_BASE_LOG2: u8 = 4;
    pub const MAX_SEGMENT_BYTES: usize = 96 << 10;
    pub const HEAP_BASE_MIN_LOG2: u8 = 10;
    pub const MAX_CHUNK_LOG2: u8 = 13;
    pub const TABLE_PAGE_LOG2: u8 = 2;
}
use sizes::*;

const fn align(n: u64, to: u64) -> u64 {
    n.div_ceil(to) * to
}

/// Where things are, derived from the header.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    dims: usize,
    precision: Precision,
    params: NodeRecordParams,
    segment_base_log2: u32,
    doubling_segments: u64,
    /// Slots in all doubling segments together.
    doubled_slots: u64,
    /// log2 of the slots in every later segment.
    full_log2: u32,
    heap_base_log2: u32,
    doubling_chunks: u64,
    table_page_log2: u32,
}

/// Byte offsets of a segment's arrays, and its size.
struct SegmentLayout {
    vectors: usize,
    level0: usize,
    bytes: usize,
}

impl Geometry {
    fn new(header: &FileHeader) -> Result<Self> {
        let params = NodeRecordParams::new(header.m, header.m0, header.max_layers);
        let base = u32::from(header.segment_base_log2);
        let full_log2 = base + u32::from(header.doubling_segments);
        let heap_base = u32::from(header.heap_base_log2);
        if !(1..=MAX_DIMENSIONS).contains(&header.dims)
            || header.m < 2
            || header.m0 < header.m
            || header.max_layers == 0
            || full_log2 > 32
            || header.table_page_log2 > 16
            || heap_base + u32::from(header.doubling_chunks) > 32
        {
            fail!(
                Corrupt,
                "Corrupt file header: impossible dimensions, graph parameters or geometry\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }
        let geometry = Self {
            dims: header.dims as usize,
            precision: if header.precision == 1 { Precision::Half } else { Precision::Full },
            params,
            segment_base_log2: base,
            doubling_segments: u64::from(header.doubling_segments),
            doubled_slots: (1u64 << full_log2) - (1u64 << base),
            full_log2,
            heap_base_log2: heap_base,
            doubling_chunks: u64::from(header.doubling_chunks),
            table_page_log2: u32::from(header.table_page_log2),
        };
        if geometry.upper_bytes(usize::from(header.max_layers)) > geometry.chunk_bytes(0) {
            fail!(
                Corrupt,
                "Corrupt file header: heap chunks smaller than one entry\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }
        Ok(geometry)
    }

    /// The header of a new, empty file.
    fn initial_header(
        dims: u32,
        params: NodeRecordParams,
        metric: DistanceMetric,
        precision: Precision,
    ) -> Result<FileHeader> {
        if !(1..=MAX_DIMENSIONS).contains(&dims) {
            fail!(
                InvalidArgument,
                "dimensions is {dims}, but has to be between 1 and {MAX_DIMENSIONS}\nhelp: give the \
                 length of the vectors the index will hold, such as 384, 768 or 1536"
            );
        }
        if params.m < 2 || params.m > 32_767 || params.max_layers == 0 {
            fail!(
                InvalidArgument,
                "max_connections is {}, but has to be between 2 and 32,767\nhelp: 16 suits most \
                 indexes; 32 gives higher recall for more memory",
                params.m
            );
        }
        let mut header = FileHeader {
            write_version: crate::header::version_of(u8::from(precision == Precision::Half)),
            sequence: 0,
            dims,
            m: params.m,
            m0: params.m0,
            max_layers: params.max_layers,
            metric: u8::from(metric == DistanceMetric::Cosine),
            precision: u8::from(precision == Precision::Half),
            flags: 0,
            segment_base_log2: SEGMENT_BASE_LOG2,
            doubling_segments: 0,
            heap_base_log2: 0,
            doubling_chunks: 0,
            table_page_log2: TABLE_PAGE_LOG2,
            count: 0,
            entry_point: u64::MAX,
            max_layer: 0,
            epoch: 0,
            pending_epoch: 0,
            deleted_count: 0,
            file_end: REGIONS_START,
            segments: 0,
            heap_chunks: 0,
            heap_used: 0,
            segment_table: Vec::new(),
            heap_table: Vec::new(),
            next_id: 0,
        };
        let entry = (usize::from(params.max_layers) - 1) * 4 * usize::from(params.m);
        header.heap_base_log2 =
            (entry.next_power_of_two().trailing_zeros() as u8).max(HEAP_BASE_MIN_LOG2);
        header.doubling_chunks = MAX_CHUNK_LOG2.saturating_sub(header.heap_base_log2);
        // The largest doubling count whose last segment still fits the size cap.
        while u32::from(SEGMENT_BASE_LOG2 + header.doubling_segments) < 32 {
            let next =
                FileHeader { doubling_segments: header.doubling_segments + 1, ..header.clone() };
            let geometry = Self::new(&next)?;
            if geometry.segment_layout(1 << geometry.full_log2).bytes > MAX_SEGMENT_BYTES {
                break;
            }
            header = next;
        }
        Self::new(&header)?;
        Ok(header)
    }

    /// Bytes of one vector in the file.
    fn vector_bytes(&self) -> usize {
        self.dims * self.precision.bytes()
    }

    fn level0_bytes(&self) -> usize {
        8 + 4 * usize::from(self.params.m0)
    }

    /// Bytes of a heap entry: one list of `M` neighbors per layer above 0.
    fn upper_bytes(&self, layer_count: usize) -> usize {
        layer_count.saturating_sub(1) * 4 * usize::from(self.params.m)
    }

    fn segment_slots(&self, k: u64) -> u64 {
        1 << (u64::from(self.segment_base_log2) + k.min(self.doubling_segments))
    }

    /// Slots in the first `segments` segments.
    fn capacity(&self, segments: u64) -> u64 {
        if segments <= self.doubling_segments {
            ((1 << segments) - 1) << self.segment_base_log2
        } else {
            self.doubled_slots + ((segments - self.doubling_segments) << self.full_log2)
        }
    }

    fn segment_layout(&self, slots: u64) -> SegmentLayout {
        let slots = slots as usize;
        let vectors = align((SLOT_HEADER * slots) as u64, 64) as usize;
        let level0 = align((vectors + self.vector_bytes() * slots) as u64, 64) as usize;
        SegmentLayout { vectors, level0, bytes: level0 + self.level0_bytes() * slots }
    }

    /// Segment and index within it of `slot`.
    #[inline]
    fn locate(&self, slot: u64) -> (usize, usize) {
        if slot < self.doubled_slots {
            let x = slot + (1 << self.segment_base_log2);
            let top = 63 - x.leading_zeros();
            ((top - self.segment_base_log2) as usize, (x - (1 << top)) as usize)
        } else {
            let rest = slot - self.doubled_slots;
            let k = self.doubling_segments + (rest >> self.full_log2);
            (k as usize, (rest & ((1 << self.full_log2) - 1)) as usize)
        }
    }

    fn chunk_bytes(&self, chunk: u64) -> usize {
        1 << (u64::from(self.heap_base_log2) + chunk.min(self.doubling_chunks))
    }
}

/// A mapped region of the file; read-only in a reader. Another process may write it at any time:
/// everything but vectors is read and written as atomic words. Vectors are written before the
/// routing count that publishes them and are plain bytes; only a writer that reopens the file after
/// an unflushed exit rewrites some, in slots past the committed count that readers only route
/// through.
#[derive(Debug)]
struct Region {
    map: MmapRaw,
    offset: u64,
}

impl Region {
    fn map(file: &File, offset: u64, len: usize, writable: bool) -> Result<Self> {
        let options = MmapOptions::new().offset(offset).len(len).clone();
        // SAFETY: one writer at a time holds the lock, and committed regions never shrink.
        let map = if writable { options.map_raw(file) } else { options.map_raw_read_only(file) };
        let map = map.with_context(|| format!("Failed to map {len} bytes at offset {offset}"))?;
        Ok(Self { map, offset })
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    /// Asks for huge pages under the file offsets `from..to`, which lie in this region. Only a
    /// request: a kernel or filesystem that doesn't keep files on huge pages ignores it.
    fn ask_for_huge_pages(&self, from: u64, to: u64) {
        #[cfg(target_os = "linux")]
        {
            let (at, len) = ((from - self.offset) as usize, (to - from) as usize);
            // A miss here reads its huge page and, unless told the reads are random, the next one
            // too, which may be one `whole_huge_pages` left out.
            if self.map.advise_range(memmap2::Advice::HugePage, at, len).is_ok() {
                let _ = self.map.advise_range(memmap2::Advice::Random, at, len);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (from, to);
    }

    #[inline]
    fn bytes(&self, at: usize, len: usize) -> &[u8] {
        assert!(at + len <= self.len(), "read past the end of a region");
        // SAFETY: in bounds, and the map lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(at), len) }
    }

    fn bytes_mut(&mut self, at: usize, len: usize) -> &mut [u8] {
        assert!(at + len <= self.len(), "write past the end of a region");
        // SAFETY: in bounds; only a writer, which maps regions writable, calls this.
        unsafe { std::slice::from_raw_parts_mut(self.map.as_mut_ptr().add(at), len) }
    }

    #[inline]
    fn u32s(&self, at: usize, len: usize) -> &[AtomicU32] {
        assert!(at + 4 * len <= self.len() && at.is_multiple_of(4), "bad neighbor list");
        // SAFETY: in bounds and aligned (maps are page aligned); atomics allow concurrent writes.
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(at).cast::<AtomicU32>(), len) }
    }

    #[inline]
    fn u64_at(&self, at: usize) -> &AtomicU64 {
        assert!(at + 8 <= self.len() && at.is_multiple_of(8), "bad word");
        // SAFETY: as for `u32s`.
        unsafe { &*self.map.as_ptr().add(at).cast::<AtomicU64>() }
    }

    /// Stores `bytes`, a multiple of 8 long, at `at` one word at a time.
    fn store_words(&self, at: usize, bytes: &[u8]) {
        for (i, word) in bytes.as_chunks::<8>().0.iter().enumerate() {
            self.u64_at(at + 8 * i).store(u64::from_le_bytes(*word), Ordering::Relaxed);
        }
    }
}

/// How much the warming thread asks the system for at a time, so that it can be stopped between.
const WARM_STEP: usize = 8 << 20;

/// A thread reading the index into memory (`IndexOptions::warm`, ADR-0019). It hands the system
/// address ranges and never reads them itself. It has to be stopped before any of them is
/// unmapped: the system would take the same advice about whatever is mapped there next.
#[derive(Debug)]
struct Warming {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The process that started the thread. A forked child has the handle and no thread.
    process: u32,
}

// A thread's handle isn't unwind safe, and an index was before it held one. A panic elsewhere
// leaves nothing here half done.
impl std::panic::UnwindSafe for Warming {}
impl std::panic::RefUnwindSafe for Warming {}

impl Warming {
    /// Starts reading `ranges` of (address, length) in, in order, holding `view` mapped until it
    /// is done; `None` if no thread could be started.
    fn start(ranges: Vec<(usize, usize)>, view: Option<Mmap>) -> Option<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let work = move || {
            let _view = view;
            for (at, len) in steps(ranges) {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }
                read_in(at, len);
            }
        };
        let thread = std::thread::Builder::new().name("chassis-warm".into()).spawn(work).ok()?;
        Some(Self { stop, thread: Some(thread), process: std::process::id() })
    }

    fn running(&self) -> bool {
        self.process == std::process::id()
            && self.thread.as_ref().is_some_and(|thread| !thread.is_finished())
    }
}

impl Drop for Warming {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let Some(thread) = self.thread.take() else { return };
        if self.process == std::process::id() {
            let _ = thread.join();
        } else {
            // Joining a thread this process never had panics, or worse.
            std::mem::forget(thread);
        }
    }
}

/// `ranges` in pieces of at most `WARM_STEP` bytes.
fn steps(ranges: Vec<(usize, usize)>) -> impl Iterator<Item = (usize, usize)> {
    ranges.into_iter().flat_map(|(at, len)| {
        (0..len).step_by(WARM_STEP).map(move |offset| (at + offset, WARM_STEP.min(len - offset)))
    })
}

/// Asks the system to read the pages of the `len` bytes at address `at` into memory.
#[cfg(unix)]
fn read_in(at: usize, len: usize) {
    let at = at as *mut libc::c_void;
    // SAFETY (both calls): advice about an address range, which nothing here reads or writes.
    // `Warming` is stopped before a range it was given is unmapped.
    #[cfg(target_os = "linux")]
    // Reads the pages in and maps them, as touching each would, and leaves them clean. Linux 5.14
    // and later; before that, the request below, which the kernel honors only in part.
    if unsafe { libc::madvise(at, len, libc::MADV_POPULATE_READ) } == 0 {
        return;
    }
    unsafe { libc::madvise(at, len, libc::MADV_WILLNEED) };
}

#[cfg(not(unix))]
fn read_in(_: usize, _: usize) {}

/// The `len` bytes at `at` from the start of their first page of `page` bytes: the system takes
/// advice about whole pages.
fn from_page_start(page: usize, (at, len): (usize, usize)) -> (usize, usize) {
    (at / page * page, len + at % page)
}

fn page_size() -> usize {
    #[cfg(unix)]
    // SAFETY: sysconf only reads.
    return unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    #[cfg(not(unix))]
    4096
}

#[derive(Debug)]
struct Segment {
    region: Region,
    vectors: usize,
    level0: usize,
    /// The file offset up to which huge pages were asked for under this segment's vectors.
    huge: u64,
}

/// A huge page. The page cache keeps a file in pieces of this size at offsets that are multiples
/// of it, where it keeps it on huge pages at all.
const HUGE_PAGE: u64 = 2 << 20;

/// The whole huge pages inside the file offsets `start..end`, if there are any. Bytes that share
/// a huge page with bytes outside the range are left out: a page written through the mapping is
/// written back whole, so what is still being written must not share one with what is asked for
/// (ADR-0016).
fn whole_huge_pages(start: u64, end: u64) -> Option<(u64, u64)> {
    let (from, to) = (start.next_multiple_of(HUGE_PAGE), end / HUGE_PAGE * HUGE_PAGE);
    (from < to).then_some((from, to))
}

/// Which table lists a region.
#[derive(Clone, Copy)]
enum Table {
    Segments,
    Chunks,
}

/// The live page: what the writer has added since its last commit, for readers in other
/// processes. Not durable: a writer resets it to its committed state when it opens the file.
const LIVE: usize = 2 * HEADER_STRIDE;
/// Slots readers may traverse (ADR-0008's routing count).
const LIVE_ROUTING: usize = LIVE;
/// Segments, heap chunks, segment table pages and heap table pages, one u64 each.
const LIVE_COUNTS: usize = LIVE + 8;
/// Segment table page offsets, then heap table page offsets.
const LIVE_PAGES: usize = LIVE + 64;

const UNDO_MAGIC: &[u8; 8] = b"CHSUNDO1";

/// The undo file, `<name>.undo` (ADR-0012): the lists of each committed node as the last commit
/// left them, saved before they first change. After `UNDO_MAGIC`, each entry is a slot (u32), its
/// layer count (u8), per layer a count (u16) and that many neighbors (u32), then an xxh3-64 of the
/// entry seeded with the hash of the graph it was saved from, so no other commit's entries pass.
#[derive(Debug)]
struct Undo {
    path: PathBuf,
    /// `FileHeader::graph_hash` of the commit the saved lists belong to.
    graph: u64,
    /// Created by the first save after a commit and removed by the next commit, with its length.
    file: Mutex<Option<(File, u64)>>,
    /// One bit per committed slot, set once its lists are saved.
    saved: Vec<AtomicU64>,
}

impl Undo {
    /// An empty undo file at `path`, not yet created, for changes to the graph `committed` names.
    fn new(path: PathBuf, committed: &FileHeader) -> Self {
        let saved = (0..committed.count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Self { path, graph: committed.graph_hash(), file: Mutex::new(None), saved }
    }

    fn append(&self, entry: &[u8]) -> Result<()> {
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        let mut bytes = Vec::with_capacity(16 + entry.len());
        if file.is_none() {
            let created = File::create(&self.path)
                .with_context(|| format!("Failed to create {}", self.path.display()))?;
            *file = Some((created, 0));
            bytes.extend_from_slice(UNDO_MAGIC);
        }
        bytes.extend_from_slice(entry);
        let (file, len) = file.as_mut().expect("created above");
        if let Err(e) = file.write_all(&bytes) {
            // Whatever part was written would hide every entry after it.
            let _ = file.set_len(*len);
            let _ = file.seek(SeekFrom::Start(*len));
            return Err(e).with_context(|| format!("Failed to write {}", self.path.display()));
        }
        *len += bytes.len() as u64;
        Ok(())
    }
}

/// Storage engine for one index file.
#[derive(Debug)]
pub struct Storage {
    /// The thread reading the index into memory, if `warm` started one. First, so that a drop
    /// stops it before the regions it was told about are unmapped.
    warming: Option<Warming>,
    /// File handle (owns the file lock)
    file: File,
    /// A reader maps the file read-only and takes no lock.
    writable: bool,
    /// Both header copies and the live page.
    headers: Region,
    segments: Vec<Segment>,
    chunks: Vec<Region>,
    segment_pages: Vec<Region>,
    chunk_pages: Vec<Region>,
    geometry: Geometry,
    /// What the next commit writes; in a reader, the snapshot it last took.
    pub(crate) state: FileHeader,
    /// The newest durable header.
    committed: FileHeader,
    /// The copy holding the newest header, and its sequence number.
    current_copy: usize,
    sequence: u64,
    /// In a reader, the checksum and sequence words of both header copies when it last found
    /// both valid.
    header_words: Option<[u64; 4]>,
    /// Slots with data written; in a reader, the slots it may traverse.
    count: u64,
    /// A failed fsync may have dropped dirty pages, so no later commit can be trusted.
    poisoned: bool,
    /// The file a migration replaced, kept locked while this one is open (Unix).
    _replaced: Option<File>,
    /// In a writer, where committed nodes' lists are saved before they change.
    undo: Option<Undo>,
    /// With `IndexOptions::huge_pages`, the slots whose vectors may be on huge pages: those
    /// written for good, and those a batch is about to write.
    huge_slots: Option<u64>,
}

impl Storage {
    /// Opens or creates a Chassis index file with the default graph parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or created
    /// - The file is already locked by another process
    /// - The file exists but has different dimensions
    /// - The file is corrupted
    pub fn open<P: AsRef<Path>>(path: P, dimensions: u32) -> Result<Self> {
        Self::open_with(path, dimensions, NodeRecordParams::default(), DistanceMetric::Euclidean)
    }

    /// Opens a Chassis index file, creating it with `params` and `metric` if it is new. A v1 or v2
    /// file is migrated to v3 first.
    ///
    /// # Errors
    ///
    /// As for `open`.
    pub fn open_with<P: AsRef<Path>>(
        path: P,
        dimensions: u32,
        params: NodeRecordParams,
        metric: DistanceMetric,
    ) -> Result<Self> {
        Self::open_with_precision(path, dimensions, params, metric, Precision::Full)
    }

    /// `open_with`, creating a new file with vectors in `precision` (ADR-0018). An existing file
    /// keeps the precision it was created with, which `precision()` reports.
    ///
    /// # Errors
    ///
    /// As for `open`.
    pub fn open_with_precision<P: AsRef<Path>>(
        path: P,
        dimensions: u32,
        params: NodeRecordParams,
        metric: DistanceMetric,
        precision: Precision,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = open_locked(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Self::initialize(file, path, dimensions, params, metric, precision);
        }

        let mut prefix = [0u8; 12];
        read_prefix(&file, &mut prefix)?;
        if is_legacy(&prefix) {
            return Self::migrate(path, file, dimensions);
        }
        if len < REGIONS_START {
            return Err(not_an_index(path));
        }
        let headers = Region::map(&file, 0, REGIONS_START as usize, true)?;
        // Creation sets the length, then writes both header copies, so a crash leaves both zero or
        // torn: only then are both invalid at once. Its live page is zero; a foreign file's isn't.
        let invalid =
            |at| matches!(FileHeader::from_bytes(headers.bytes(at, HEADER_STRIDE)), Ok(None));
        if len == REGIONS_START
            && invalid(0)
            && invalid(HEADER_STRIDE)
            && headers.bytes(LIVE, HEADER_STRIDE).iter().all(|&b| b == 0)
        {
            drop(headers);
            return Self::initialize(file, path, dimensions, params, metric, precision);
        }
        Self::load(file, headers, dimensions, path)
    }

    /// Opens a file for reading while a `VectorIndex` in another process may be writing it: no
    /// lock and no recovery. `refresh` takes a new snapshot (ADR-0008, decision 5).
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing, in format 1 or 2 (open it once with write
    /// access to migrate it), has different dimensions, or is corrupt.
    pub fn open_read_only<P: AsRef<Path>>(path: P, dimensions: u32) -> Result<Self> {
        let path = path.as_ref();
        // Windows refuses to open a directory with "access denied", which would read as permissions.
        if path.is_dir() {
            return Err(open_failed(path, std::io::ErrorKind::IsADirectory.into(), false));
        }
        let file = File::open(path).map_err(|e| open_failed(path, e, false))?;
        let mut prefix = [0u8; 12];
        read_prefix(&file, &mut prefix)?;
        if is_legacy(&prefix) {
            bail!(
                "{} uses an older file format; open it once with write access to migrate it",
                path.display()
            );
        }
        if file.metadata()?.len() < REGIONS_START {
            return Err(not_an_index(path));
        }
        let headers = Region::map(&file, 0, REGIONS_START as usize, false)?;
        let (header, _) = newest_header(&headers)?;
        if header.dims != dimensions {
            return Err(dimension_mismatch(path, header.dims, dimensions));
        }
        let mut storage = Self::new(file, false, headers, header, 0)?;
        storage.refresh()?;
        Ok(storage)
    }

    /// Writes the headers of a new, empty file.
    fn initialize(
        file: File,
        path: &Path,
        dims: u32,
        params: NodeRecordParams,
        metric: DistanceMetric,
        precision: Precision,
    ) -> Result<Self> {
        let header = Geometry::initial_header(dims, params, metric, precision)?;
        file.set_len(REGIONS_START)?;
        let headers = Region::map(&file, 0, REGIONS_START as usize, true)?;
        headers.store_words(0, &FileHeader { sequence: 1, ..header.clone() }.to_bytes());
        headers.store_words(HEADER_STRIDE, &header.to_bytes());
        let mut storage = Self::new(file, true, headers, FileHeader { sequence: 1, ..header }, 0)?;
        storage.undo = Some(Undo::new(sibling(path, "undo"), &storage.committed));
        storage.publish_regions();
        storage.publish_routing(0);
        storage.sync()?;
        // The directory entry too: otherwise power loss can take the whole file with it.
        sync_dir(path)?;
        Ok(storage)
    }

    fn new(
        file: File,
        writable: bool,
        headers: Region,
        header: FileHeader,
        copy: usize,
    ) -> Result<Self> {
        Ok(Self {
            warming: None,
            file,
            writable,
            headers,
            segments: Vec::new(),
            chunks: Vec::new(),
            segment_pages: Vec::new(),
            chunk_pages: Vec::new(),
            geometry: Geometry::new(&header)?,
            count: header.count,
            sequence: header.sequence,
            header_words: None,
            current_copy: copy,
            state: header.clone(),
            committed: header,
            poisoned: false,
            _replaced: None,
            undo: None,
            huge_slots: None,
        })
    }

    /// Starts another thread reading what the index holds into memory (`IndexOptions::warm`,
    /// ADR-0019), unless one is at it already, and returns at once.
    pub(crate) fn warm(&mut self) {
        if self.warming.as_ref().is_some_and(Warming::running) {
            return;
        }
        if !cfg!(unix) || self.written(None).is_empty() {
            return;
        }
        // Asked to read pages in through a writable mapping, macOS marks them all as written to,
        // and writes them back. A writer there asks through a read-only view of the file instead,
        // whose pages its own mappings then fault in.
        let view = match self.writable && !cfg!(target_os = "linux") {
            // SAFETY: nothing reads through the view; it is only named in advice.
            true => match unsafe { Mmap::map(&self.file) } {
                Ok(view) => Some(view),
                Err(_) => return,
            },
            false => None,
        };
        let page = page_size();
        let ranges = self.written(view.as_ref()).into_iter().map(|r| from_page_start(page, r));
        self.warming = Warming::start(ranges.collect(), view);
    }

    /// The address ranges, in the index's mappings or else in `view` of the whole file, of what
    /// has been written: each segment's three arrays up to the last slot counted, the heap chunks
    /// before the one in use, and that one up to its last entry. What lies past them was never
    /// written, and in a sparse file is a hole. A reader counts slots its writer hasn't committed,
    /// and not their heap entries.
    fn written(&self, view: Option<&Mmap>) -> Vec<(usize, usize)> {
        let start = |region: &Region| match view {
            Some(view) => view.as_ptr() as usize + region.offset as usize,
            None => region.map.as_ptr() as usize,
        };
        let g = self.geometry;
        let mut ranges = Vec::new();
        for (k, segment) in self.segments.iter().enumerate() {
            let k = k as u64;
            let slots = self.count.saturating_sub(g.capacity(k)).min(g.segment_slots(k)) as usize;
            let base = start(&segment.region);
            let arrays = [
                (0, SLOT_HEADER),
                (segment.vectors, g.vector_bytes()),
                (segment.level0, g.level0_bytes()),
            ];
            ranges.extend(arrays.map(|(at, per_slot)| (base + at, per_slot * slots)));
        }
        let (last, used) =
            ((self.state.heap_used >> 32) as usize, (self.state.heap_used & 0xffff_ffff) as usize);
        for (c, chunk) in self.chunks.iter().enumerate() {
            let len = match c.cmp(&last) {
                std::cmp::Ordering::Less => chunk.len(),
                std::cmp::Ordering::Equal => used,
                std::cmp::Ordering::Greater => 0,
            };
            ranges.push((start(chunk), len));
        }
        ranges.retain(|&(_, len)| len > 0);
        ranges
    }

    /// Asks for huge pages under the committed vectors, and from here on under vectors as they
    /// are committed or written by a batch (`IndexOptions::huge_pages`, ADR-0016).
    pub(crate) fn use_huge_pages(&mut self) {
        self.huge_slots = Some(self.committed.count);
        self.ask_for_huge_pages();
    }

    /// The vectors in slots below `slots` are written for good, or a batch is about to write
    /// them: with huge pages in use, asks for them there.
    pub(crate) fn huge_pages_up_to(&mut self, slots: u64) {
        if self.huge_slots.is_some_and(|asked| asked < slots) {
            self.huge_slots = Some(slots);
            self.ask_for_huge_pages();
        }
    }

    /// Asks for whatever `huge_slots` allows and hasn't been asked for yet.
    fn ask_for_huge_pages(&mut self) {
        let Some(slots) = self.huge_slots else { return };
        let g = self.geometry;
        for (k, segment) in self.segments.iter_mut().enumerate() {
            let k = k as u64;
            let slots = slots.saturating_sub(g.capacity(k)).min(g.segment_slots(k));
            let start = segment.region.offset + segment.vectors as u64;
            let end = start + slots * g.vector_bytes() as u64;
            if let Some((from, to)) = whole_huge_pages(start, end)
                && to > segment.huge
            {
                segment.region.ask_for_huge_pages(from.max(segment.huge), to);
                segment.huge = to;
            }
        }
    }

    /// Opens an existing v3 file for writing.
    fn load(file: File, headers: Region, dims: u32, path: &Path) -> Result<Self> {
        let a = FileHeader::from_bytes(headers.bytes(0, HEADER_STRIDE))?;
        let b = FileHeader::from_bytes(headers.bytes(HEADER_STRIDE, HEADER_STRIDE))?;
        let (copy, header) = FileHeader::newest(a, b)?;
        if header.write_version > crate::header::VERSION {
            bail!(
                "File was written by a newer Chassis (format {}); this release can only read it, \
                 with a reader (IndexReader, chassis_open_reader, or read_only=True in Python)",
                header.write_version
            );
        }
        if header.dims != dims {
            return Err(dimension_mismatch(path, header.dims, dims));
        }
        if header.file_end > file.metadata()?.len() {
            fail!(
                Corrupt,
                "File truncated: it ends before its last region\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }
        let mut storage = Self::new(file, true, headers, header, copy)?;
        storage.map_regions(None)?;
        storage.undo = Some(Undo::new(sibling(path, "undo"), &storage.committed));
        storage.roll_back()?;
        if storage.state.pending_epoch != 0 {
            storage.recover()?;
        }
        if storage.superseded() {
            // A compaction stopped before its rename, so this file is still the index.
            storage.set_superseded(false)?;
        }
        // Readers must not route through slots a crashed writer left: the next adds reuse them.
        storage.publish_regions();
        storage.publish_routing(storage.state.count);
        Ok(storage)
    }

    /// Maps every region the header counts, plus, in a reader, those the live page shows.
    fn map_regions(&mut self, live: Option<[u64; 4]>) -> Result<()> {
        let g = self.geometry;
        let s = &self.state;
        let per_page = 1u64 << g.table_page_log2;
        if (s.segment_table.len() as u64) < u64::from(s.segments).div_ceil(per_page)
            || (s.heap_table.len() as u64) < u64::from(s.heap_chunks).div_ceil(per_page)
            || g.capacity(u64::from(s.segments)) < s.count
            || (s.entry_point != u64::MAX && s.entry_point >= s.count)
            || s.heap_used >> 32 > u64::from(s.heap_chunks)
            || s.heap_used & 0xffff_ffff > g.chunk_bytes(s.heap_used >> 32) as u64
        {
            fail!(
                Corrupt,
                "Corrupt file header: counts don't match its tables\nhelp: restore the index from a backup, or rebuild it from its vectors"
            );
        }
        let [segments, chunks, segment_pages, chunk_pages] = live.unwrap_or_default();
        let mut file_len = None;
        let page_bytes = 8 << g.table_page_log2;

        for table in [Table::Segments, Table::Chunks] {
            let (committed_pages, live_pages, base) = match table {
                Table::Segments => (self.state.segment_table.len(), segment_pages, 0),
                Table::Chunks => (self.state.heap_table.len(), chunk_pages, MAX_TABLE_PAGES),
            };
            for i in 0..committed_pages.max(live_pages as usize).min(MAX_TABLE_PAGES) {
                let offset = match table {
                    Table::Segments => self.state.segment_table.get(i),
                    Table::Chunks => self.state.heap_table.get(i),
                }
                .copied()
                .unwrap_or_else(|| {
                    self.headers.u64_at(LIVE_PAGES + 8 * (base + i)).load(Ordering::Acquire)
                });
                let pages = match table {
                    Table::Segments => &mut self.segment_pages,
                    Table::Chunks => &mut self.chunk_pages,
                };
                // A restarted writer may reuse an uncommitted region's offset for another one.
                if pages.get(i).is_some_and(|page| page.offset != offset) {
                    pages.truncate(i);
                }
                if pages.len() == i {
                    let Some(page) =
                        self.map_checked(offset, page_bytes, i < committed_pages, &mut file_len)?
                    else {
                        break;
                    };
                    match table {
                        Table::Segments => self.segment_pages.push(page),
                        Table::Chunks => self.chunk_pages.push(page),
                    }
                }
            }
        }

        let wanted = u64::from(self.state.segments).max(segments);
        for k in 0..wanted {
            let committed = k < u64::from(self.state.segments);
            let Some(offset) = self.table_entry(Table::Segments, k) else { break };
            if self.segments.get(k as usize).is_some_and(|s| s.region.offset != offset) {
                // What this unmaps, the warming thread may still have to ask for.
                self.warming = None;
                self.segments.truncate(k as usize);
            }
            if self.segments.len() as u64 == k {
                let layout = g.segment_layout(g.segment_slots(k));
                let Some(region) =
                    self.map_checked(offset, layout.bytes, committed, &mut file_len)?
                else {
                    break;
                };
                self.segments.push(Segment {
                    region,
                    vectors: layout.vectors,
                    level0: layout.level0,
                    huge: 0,
                });
            }
        }
        let wanted = u64::from(self.state.heap_chunks).max(chunks);
        for c in 0..wanted {
            let committed = c < u64::from(self.state.heap_chunks);
            let Some(offset) = self.table_entry(Table::Chunks, c) else { break };
            if self.chunks.get(c as usize).is_some_and(|chunk| chunk.offset != offset) {
                self.warming = None;
                self.chunks.truncate(c as usize);
            }
            if self.chunks.len() as u64 == c {
                let Some(region) =
                    self.map_checked(offset, g.chunk_bytes(c), committed, &mut file_len)?
                else {
                    break;
                };
                self.chunks.push(region);
            }
        }
        Ok(())
    }

    /// Maps a region a table points at. A committed one must lie inside the committed file; one
    /// the writer hasn't committed yet is skipped (`None`) until it lies inside the file, whose
    /// length is read into `file_len` the first time it is needed.
    fn map_checked(
        &self,
        offset: u64,
        len: usize,
        committed: bool,
        file_len: &mut Option<u64>,
    ) -> Result<Option<Region>> {
        let limit = match (committed, *file_len) {
            (true, _) => self.state.file_end,
            (false, Some(len)) => len,
            (false, None) => *file_len.insert(self.file.metadata()?.len()),
        };
        let fits = offset >= REGIONS_START
            && offset.is_multiple_of(REGION_ALIGN)
            && offset.checked_add(len as u64).is_some_and(|end| end <= limit);
        match (fits, committed) {
            (true, _) => Ok(Some(Region::map(&self.file, offset, len, self.writable)?)),
            (false, true) => {
                fail!(
                    Corrupt,
                    "Corrupt file: a region at offset {offset} lies outside the file\nhelp: restore the index from a backup, or rebuild it from its vectors"
                )
            }
            (false, false) => Ok(None),
        }
    }

    fn table_entry(&self, table: Table, index: u64) -> Option<u64> {
        let pages = match table {
            Table::Segments => &self.segment_pages,
            Table::Chunks => &self.chunk_pages,
        };
        let page = pages.get((index >> self.geometry.table_page_log2) as usize)?;
        let at = ((index & ((1 << self.geometry.table_page_log2) - 1)) * 8) as usize;
        Some(page.u64_at(at).load(Ordering::Acquire))
    }

    /// Takes a reader's snapshot: the newest committed header, never older than the last one,
    /// plus the regions and routing count the writer has published since.
    pub(crate) fn refresh(&mut self) -> Result<()> {
        debug_assert!(!self.writable, "only readers take snapshots");
        // A commit changes the sequence and checksum words of the copy it writes. While neither
        // copy's changed since both were last valid, no commit has finished: the header stands.
        let words = HEADER_WORDS.map(|at| self.headers.u64_at(at).load(Ordering::Relaxed));
        if self.header_words != Some(words) {
            let (header, both_valid) = newest_header(&self.headers)?;
            self.header_words = both_valid;
            if header.sequence >= self.sequence {
                self.sequence = header.sequence;
                self.state = header.clone();
                self.committed = header;
            }
        }
        let h = &self.headers;
        let routing = h.u64_at(LIVE_ROUTING).load(Ordering::Acquire);
        let live = std::array::from_fn(|i| h.u64_at(LIVE_COUNTS + 8 * i).load(Ordering::Acquire));
        self.map_regions(Some(live))?;
        let mapped = self.geometry.capacity(self.segments.len() as u64);
        self.count = routing.max(self.state.count).min(mapped);
        self.huge_pages_up_to(self.committed.count);
        Ok(())
    }

    /// Shows readers the writer's regions; called after every allocation and on open.
    fn publish_regions(&self) {
        let (h, s) = (&self.headers, &self.state);
        let pages = s.segment_table.iter().enumerate();
        for (i, &offset) in
            pages.chain(s.heap_table.iter().enumerate().map(|(i, o)| (i + MAX_TABLE_PAGES, o)))
        {
            h.u64_at(LIVE_PAGES + 8 * i).store(offset, Ordering::Relaxed);
        }
        let counts = [
            u64::from(s.segments),
            u64::from(s.heap_chunks),
            s.segment_table.len() as u64,
            s.heap_table.len() as u64,
        ];
        for (i, count) in counts.into_iter().enumerate() {
            h.u64_at(LIVE_COUNTS + 8 * i).store(count, Ordering::Release);
        }
    }

    /// Whether the newest header says a compacted copy replaces this file (ADR-0011).
    pub(crate) fn superseded(&self) -> bool {
        self.state.flags & FLAG_SUPERSEDED != 0
    }

    /// Commits whether a compacted copy is replacing this file at its path.
    pub(crate) fn set_superseded(&mut self, superseded: bool) -> Result<()> {
        self.state.flags =
            (self.state.flags & !FLAG_SUPERSEDED) | if superseded { FLAG_SUPERSEDED } else { 0 };
        self.commit_deleting(&[])
    }

    /// Whether this file is still the one at `path`.
    pub(crate) fn is_at(&self, path: &Path) -> Result<bool> {
        is_at(&self.file, path)
    }

    /// One past the largest id used before the last compaction; 0 if never compacted.
    pub(crate) fn next_id(&self) -> u64 {
        self.state.next_id
    }

    /// The last flush that committed deletes.
    pub(crate) fn epoch(&self) -> u64 {
        self.state.epoch
    }

    /// In a compacted copy, what its source showed only through its deleted slots: the id
    /// high-water mark, and a delete epoch past every snapshot of the source.
    pub(crate) fn carry_over(&mut self, next_id: u64, epoch: u64) {
        self.state.next_id = next_id;
        self.state.epoch = epoch;
    }

    /// Lets readers route through slots below `count`, once their nodes and backlinks are written.
    pub(crate) fn publish_routing(&self, count: u64) {
        self.headers.u64_at(LIVE_ROUTING).store(count, Ordering::Release);
    }

    /// Converts a v1 or v2 file to v3, then swaps it in (ADR-0008, decision 8).
    fn migrate(path: &Path, original: File, dims: u32) -> Result<Self> {
        let legacy = LegacyIndex::open(&original, dims)?;
        let temp = sibling(path, "migrating");
        match std::fs::remove_file(&temp) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
        let (euclidean, full) = (DistanceMetric::Euclidean, Precision::Full);
        let mut storage =
            Self::initialize(open_locked(&temp)?, &temp, dims, legacy.params, euclidean, full)?;

        let mut deleted = 0;
        let mut custom_ids = legacy.custom_ids;
        for slot in 0..legacy.count {
            let record = legacy.record(slot)?;
            storage.append(record.header.id, &legacy.vector(slot))?;
            custom_ids |= record.header.id != slot;
            if legacy.is_deleted(&record) {
                storage.set_slot_epoch(slot, u64::from(record.header.deleted_epoch))?;
                deleted += 1;
            }
            // Ids at or past the node count are backlinks to nodes a crash rolled back.
            let layers: Vec<Vec<u64>> = (0..usize::from(record.header.layer_count))
                .map(|l| {
                    record.get_neighbors(l).into_iter().filter(|&n| n < legacy.count).collect()
                })
                .collect();
            storage.write_record(slot, &layers)?;
        }
        storage.state.count = legacy.count;
        storage.state.entry_point = legacy.entry_point.unwrap_or(u64::MAX);
        storage.state.max_layer = legacy.max_layer;
        storage.state.epoch = u64::from(legacy.epoch);
        storage.state.deleted_count = deleted;
        storage.state.flags = if custom_ids { crate::header::FLAG_CUSTOM_IDS } else { 0 };
        storage.commit_deleting(&[])?;
        storage.publish_routing(legacy.count);
        drop(legacy);

        replace(&temp, path, original, &mut storage)?;
        storage.moved_to(path);
        Ok(storage)
    }

    /// Inserts a vector into the next slot, with the slot number as its id.
    ///
    /// # Returns
    ///
    /// Returns the slot of the inserted vector
    ///
    /// # Errors
    ///
    /// Returns an error if the vector has the wrong dimensions, the file cannot grow, or the
    /// storage was opened read-only.
    ///
    /// # Note
    ///
    /// This method does NOT guarantee durability. Call `commit()` to ensure
    /// data is written to disk.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u64> {
        match self.metric() {
            DistanceMetric::Cosine => self.append(self.count, &crate::distance::unit(vector)?),
            _ => self.append(self.count, vector),
        }
    }

    /// Writes `id` and `vector` into the next slot; its graph record is written separately.
    pub(crate) fn append(&mut self, id: u64, vector: &[f32]) -> Result<u64> {
        if !self.writable {
            fail!(
                ReadOnly,
                "This index was opened as a reader, which only searches\nhelp: to add, delete, flush or compact, open it as a writer; an index has one writer at a time"
            );
        }
        let g = self.geometry;
        if vector.len() != g.dims {
            bail!("Vector dimension mismatch: expected {}, got {}", g.dims, vector.len());
        }
        if g.precision == Precision::Half {
            half::check(vector, g.dims)?;
        }
        let slot = self.count;
        if slot >= u64::from(EMPTY) {
            fail!(
                Full,
                "The index is full: it holds at most {} vectors\nhelp: split the vectors across indexes",
                EMPTY
            );
        }
        while g.capacity(self.segments.len() as u64) <= slot {
            let layout = g.segment_layout(g.segment_slots(self.segments.len() as u64));
            let region = self.push_region(Table::Segments, layout.bytes)?;
            let (vectors, level0) = (layout.vectors, layout.level0);
            self.segments.push(Segment { region, vectors, level0, huge: 0 });
            self.state.segments += 1;
            self.publish_regions();
            self.ask_for_huge_pages();
        }

        let (k, i) = g.locate(slot);
        let segment = &mut self.segments[k];
        // The whole slot header, so a crashed flush's delete mark never survives in a reused slot.
        let mut header = [0u8; SLOT_HEADER];
        header[..8].copy_from_slice(&id.to_le_bytes());
        segment.region.store_words(i * SLOT_HEADER, &header);
        let bytes = g.vector_bytes();
        let stored = segment.region.bytes_mut(segment.vectors + i * bytes, bytes);
        match g.precision {
            Precision::Full => {
                for (dst, x) in stored.as_chunks_mut::<4>().0.iter_mut().zip(vector) {
                    dst.copy_from_slice(&x.to_le_bytes());
                }
            }
            Precision::Half => half::write(vector, stored),
        }
        self.count += 1;
        Ok(slot)
    }

    /// Allocates a region, recorded in `table`, and maps it.
    fn push_region(&mut self, table: Table, bytes: usize) -> Result<Region> {
        let log2 = self.geometry.table_page_log2;
        let index = match table {
            Table::Segments => self.segments.len(),
            Table::Chunks => self.chunks.len(),
        } as u64;
        let page = (index >> log2) as usize;
        let pages = match table {
            Table::Segments => self.segment_pages.len(),
            Table::Chunks => self.chunk_pages.len(),
        };
        if page == pages {
            if page == MAX_TABLE_PAGES {
                fail!(
                    Full,
                    "The index is full: no room for another table page\nhelp: compact it to drop deleted vectors, or split the vectors across indexes"
                );
            }
            let region = self.allocate(8 << log2)?;
            let (pages, offsets) = match table {
                Table::Segments => (&mut self.segment_pages, &mut self.state.segment_table),
                Table::Chunks => (&mut self.chunk_pages, &mut self.state.heap_table),
            };
            offsets.push(region.offset);
            pages.push(region);
        }
        let region = self.allocate(bytes)?;
        let page = match table {
            Table::Segments => &self.segment_pages[page],
            Table::Chunks => &self.chunk_pages[page],
        };
        let at = ((index & ((1 << log2) - 1)) * 8) as usize;
        page.u64_at(at).store(region.offset, Ordering::Release);
        Ok(region)
    }

    /// Appends a region of `bytes` at the end of the committed regions and maps it.
    fn allocate(&mut self, bytes: usize) -> Result<Region> {
        let offset = self.state.file_end;
        let end =
            offset.checked_add(align(bytes as u64, REGION_ALIGN)).context("File too large")?;
        // Never shrink: bytes past `end` were left by a crashed flush and are simply reused.
        if self.file.metadata()?.len() < end {
            self.file.set_len(end)?;
        }
        let region = Region::map(&self.file, offset, bytes, true)?;
        self.state.file_end = end;
        Ok(region)
    }

    /// Reserves heap space for a node's upper-layer lists; returns its `chunk << 32 | offset`.
    fn allocate_upper(&mut self, bytes: usize) -> Result<u64> {
        let (mut chunk, mut offset) =
            (self.state.heap_used >> 32, self.state.heap_used & 0xffff_ffff);
        if offset as usize + bytes > self.geometry.chunk_bytes(chunk) {
            chunk += 1;
            offset = 0;
        }
        while self.chunks.len() as u64 <= chunk {
            if self.chunks.len() >= 1 << 24 {
                fail!(
                    Full,
                    "The index is full: no room for another heap chunk\nhelp: compact it to drop deleted vectors, or split the vectors across indexes"
                );
            }
            let region = self.push_region(Table::Chunks, self.geometry.chunk_bytes(chunk))?;
            self.chunks.push(region);
            self.state.heap_chunks += 1;
            self.publish_regions();
        }
        self.state.heap_used = chunk << 32 | (offset + bytes as u64);
        Ok(chunk << 32 | offset)
    }

    /// Segment and index of a written slot.
    #[inline]
    fn segment(&self, slot: u64) -> Result<(&Segment, usize)> {
        if slot >= self.count {
            return Err(self.out_of_bounds(slot));
        }
        let (k, i) = self.geometry.locate(slot);
        Ok((&self.segments[k], i))
    }

    /// Built out of line, so that the lookups a search makes per neighbor stay small enough to
    /// inline into its loop.
    #[cold]
    fn out_of_bounds(&self, slot: u64) -> anyhow::Error {
        anyhow::anyhow!("Index out of bounds: {slot} (count is {})", self.count)
    }

    /// Likewise out of line, for `record`.
    #[cold]
    fn invalid_record(slot: u64, layers: usize) -> anyhow::Error {
        anyhow::anyhow!("Invalid graph record for slot {slot}: {layers} layers")
    }

    /// Retrieves a zero-copy view of the vector in `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is past the last written slot, or if the file keeps its
    /// vectors in half precision, which `get_vector` widens.
    #[inline]
    pub fn get_vector_slice(&self, index: u64) -> Result<&[f32]> {
        if self.geometry.precision != Precision::Full {
            bail!("This index keeps its vectors in half precision: read them with get_vector");
        }
        self.stored(index)
    }

    /// The vector in `index` as the file keeps it. `E` must be the file's precision: callers
    /// choose it from `precision()`, once for many vectors.
    #[inline]
    pub(crate) fn stored<E: Element>(&self, index: u64) -> Result<&[E]> {
        debug_assert_eq!(E::PRECISION, self.geometry.precision);
        let (segment, i) = self.segment(index)?;
        let dims = self.geometry.dims;
        let bytes = segment
            .region
            .bytes(segment.vectors + i * dims * size_of::<E>(), dims * size_of::<E>());
        // SAFETY: maps are page aligned and a vector starts at a multiple of its components' size;
        // a published vector is never rewritten.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<E>(), dims) })
    }

    /// The vector in `index` as f32s: in place, or widened from half precision.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is past the last written slot.
    pub(crate) fn vector(&self, index: u64) -> Result<Cow<'_, [f32]>> {
        Ok(match self.geometry.precision {
            Precision::Full => Cow::Borrowed(self.stored(index)?),
            Precision::Half => {
                let mut widened = Vec::new();
                half::widen(self.stored(index)?, &mut widened);
                Cow::Owned(widened)
            }
        })
    }

    /// Asks the CPU to start loading the level-0 neighbor list of `slot`, which a search is about
    /// to queue for expanding. Only a hint: a slot that isn't mapped is skipped.
    #[inline(always)]
    pub(crate) fn prefetch_record(&self, slot: u64) {
        if let Ok((segment, i)) = self.segment(slot) {
            let bytes = self.geometry.level0_bytes();
            let record = segment.region.bytes(segment.level0 + i * bytes, bytes);
            crate::prefetch::list(record.as_ptr(), bytes);
        }
    }

    /// Retrieves an owned copy of the vector in `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is past the last written slot.
    pub fn get_vector(&self, index: u64) -> Result<Vec<f32>> {
        Ok(self.vector(index)?.into_owned())
    }

    /// The caller's id stored in `slot`.
    pub(crate) fn slot_id(&self, slot: u64) -> Result<u64> {
        let (segment, i) = self.segment(slot)?;
        Ok(segment.region.u64_at(i * SLOT_HEADER).load(Ordering::Relaxed))
    }

    /// The flush epoch that deleted `slot`, or 0 while it is live.
    pub(crate) fn slot_epoch(&self, slot: u64) -> Result<u64> {
        let (segment, i) = self.segment(slot)?;
        Ok(segment.region.u64_at(i * SLOT_HEADER + 16).load(Ordering::Relaxed))
    }

    fn set_slot_epoch(&mut self, slot: u64, epoch: u64) -> Result<()> {
        let (segment, i) = self.segment(slot)?;
        segment.region.u64_at(i * SLOT_HEADER + 16).store(epoch, Ordering::Relaxed);
        Ok(())
    }

    /// The level-0 record of `slot`, and its head word.
    #[inline]
    fn record(&self, slot: u64) -> Result<(&Region, usize, u64)> {
        let (segment, i) = self.segment(slot)?;
        let at = segment.level0 + i * self.geometry.level0_bytes();
        let head = segment.region.u64_at(at).load(Ordering::Acquire);
        let layers = (head & 0xff) as usize;
        if layers == 0 || layers > usize::from(self.geometry.params.max_layers) {
            return Err(Self::invalid_record(slot, layers));
        }
        Ok((&segment.region, at, head))
    }

    /// Layers the node in `slot` belongs to.
    pub(crate) fn layer_count(&self, slot: u64) -> Result<usize> {
        Ok((self.record(slot)?.2 & 0xff) as usize)
    }

    /// `layer`'s list from a record head; `None` in a reader that hasn't mapped its chunk yet.
    fn upper_list(&self, head: u64, layer: usize) -> Result<Option<&[AtomicU32]>> {
        let at = head >> 8;
        let m = usize::from(self.geometry.params.m);
        let start = (at & 0xffff_ffff) as usize + (layer - 1) * 4 * m;
        match self.chunks.get((at >> 32) as usize) {
            Some(chunk) if start.is_multiple_of(4) && start + 4 * m <= chunk.len() => {
                Ok(Some(chunk.u32s(start, m)))
            }
            None if !self.writable => Ok(None),
            _ => fail!(
                Corrupt,
                "Corrupt graph record: invalid upper-layer reference\nhelp: restore the index from a backup, or rebuild it from its vectors"
            ),
        }
    }

    /// The neighbor list of the node in `slot` at `layer`, `EMPTY` entries included; empty if
    /// the node is not on that layer.
    #[inline]
    pub(crate) fn neighbors(&self, slot: u64, layer: usize) -> Result<&[AtomicU32]> {
        // A reader also routes through slots past its snapshot; after a writer restart they may be
        // reused, or lie in regions its mappings no longer describe. They are only hints.
        let ghost = !self.writable && slot >= self.state.count;
        let (region, at, head) = match self.record(slot) {
            Err(_) if ghost => return Ok(&[]),
            record => record?,
        };
        if layer >= (head & 0xff) as usize {
            return Ok(&[]);
        }
        if layer == 0 {
            return Ok(region.u32s(at + 8, usize::from(self.geometry.params.m0)));
        }
        match self.upper_list(head, layer) {
            Err(_) if ghost => Ok(&[]),
            list => Ok(list?.unwrap_or(&[])),
        }
    }

    /// Writes a new graph record for a written slot: `layers[l]` is its list at layer `l`.
    pub(crate) fn write_record(&mut self, slot: u64, layers: &[Vec<u64>]) -> Result<()> {
        if layers.is_empty() || layers.len() > usize::from(self.geometry.params.max_layers) {
            bail!("A node needs between 1 and {} layers", self.geometry.params.max_layers);
        }
        self.segment(slot)?;
        let upper = match layers.len() {
            1 => 0,
            n => self.allocate_upper(self.geometry.upper_bytes(n))?,
        };
        let head = layers.len() as u64 | upper << 8;
        let (segment, i) = self.segment(slot)?;
        let at = segment.level0 + i * self.geometry.level0_bytes();
        store_list(segment.region.u32s(at + 8, usize::from(self.geometry.params.m0)), &layers[0]);
        for (layer, ids) in layers.iter().enumerate().skip(1) {
            store_list(self.upper_list(head, layer)?.context("Heap chunk not mapped")?, ids);
        }
        // The head last: a reader that sees it finds the lists already written.
        segment.region.u64_at(at).store(head, Ordering::Release);
        Ok(())
    }

    /// Replaces the list of an existing record at one of its layers. Two threads must not call
    /// this for the same slot at once.
    pub(crate) fn write_neighbors(&self, slot: u64, layer: usize, ids: &[u64]) -> Result<()> {
        let list = self.neighbors(slot, layer)?;
        if list.is_empty() {
            bail!("Node {slot} is not on layer {layer}");
        }
        self.save_lists(&[slot])?;
        store_list(list, ids);
        Ok(())
    }

    /// Saves the lists of those of `slots` that are committed and not saved since the commit, in
    /// one write to the undo file, before any of them changes. Callers pass every node they are
    /// about to relink at once, as a write per node costs many times more.
    ///
    /// No lock is needed: a list only changes once an entry for it is written, so the first entry
    /// written for a slot always holds its committed lists, and `roll_back` uses that one.
    pub(crate) fn save_lists(&self, slots: &[u64]) -> Result<()> {
        let committed = self.committed.count;
        let Some(undo) = &self.undo else { return Ok(()) };
        let saved = |slot: u64| (&undo.saved[(slot / 64) as usize], 1u64 << (slot % 64));
        let (mut entries, mut saving) = (Vec::new(), Vec::new());
        for &slot in slots {
            if slot >= committed || saving.contains(&slot) {
                continue;
            }
            let (word, bit) = saved(slot);
            if word.load(Ordering::Acquire) & bit != 0 {
                continue;
            }
            let start = entries.len();
            let layers = self.layer_count(slot)?;
            entries.extend_from_slice(&(slot as u32).to_le_bytes());
            entries.push(layers as u8);
            for layer in 0..layers {
                // Links to slots past the commit, which an older crash left, aren't kept.
                let ids = self.neighbors(slot, layer)?.iter().map(|id| id.load(Ordering::Relaxed));
                let ids: Vec<u32> = ids.filter(|&id| u64::from(id) < committed).collect();
                entries.extend_from_slice(&(ids.len() as u16).to_le_bytes());
                entries.extend(ids.iter().flat_map(|id| id.to_le_bytes()));
            }
            let sum = xxh3_64_with_seed(&entries[start..], undo.graph);
            entries.extend_from_slice(&sum.to_le_bytes());
            saving.push(slot);
        }
        if saving.is_empty() {
            return Ok(());
        }
        undo.append(&entries)?;
        for slot in saving {
            let (word, bit) = saved(slot);
            word.fetch_or(bit, Ordering::Release);
        }
        Ok(())
    }

    /// Writes back the lists the undo file saved, if it saved them from the commit this file is
    /// at: the adds made since are gone, and now their links in committed nodes are too. Then
    /// removes the file.
    fn roll_back(&mut self) -> Result<()> {
        let Some(undo) = &self.undo else { return Ok(()) };
        let path = undo.path.clone();
        // ponytail: reads the whole file, at most about 150 bytes per committed vector; read it
        // in pieces if indexes outgrow that.
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).with_context(|| format!("Failed to read {}", path.display())),
        };
        // Two threads can each save a slot; only the first entry is sure to predate any change.
        let mut restored = std::collections::HashSet::new();
        for (slot, lists) in self.undo_entries(&bytes, undo.graph) {
            if restored.insert(slot) {
                for (layer, ids) in lists.iter().enumerate() {
                    store_list(self.neighbors(slot, layer)?, ids);
                }
            }
        }
        if !restored.is_empty() {
            // Before the file goes: a power loss must not take the saved lists and keep the links.
            self.sync()?;
        }
        // If it can't be removed it is applied again, to no effect, or no longer matches.
        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    /// The entries of an undo file saved from the commit `graph` names, up to the first that is
    /// torn or doesn't fit this file: a slot and its list on each layer.
    fn undo_entries<'a>(
        &'a self,
        bytes: &'a [u8],
        graph: u64,
    ) -> impl Iterator<Item = (u64, Vec<Vec<u64>>)> + 'a {
        fn take<'b>(rest: &mut &'b [u8], n: usize) -> Option<&'b [u8]> {
            let (head, tail) = rest.split_at_checked(n)?;
            *rest = tail;
            Some(head)
        }
        let mut rest = bytes.strip_prefix(UNDO_MAGIC.as_slice()).unwrap_or_default();
        std::iter::from_fn(move || {
            let start = rest;
            let entry = (|| {
                let slot = u64::from(u32::from_le_bytes(take(&mut rest, 4)?.try_into().ok()?));
                let layers = usize::from(take(&mut rest, 1)?[0]);
                if slot >= self.committed.count || self.layer_count(slot).ok()? != layers {
                    return None;
                }
                let mut lists = Vec::with_capacity(layers);
                for layer in 0..layers {
                    let count = u16::from_le_bytes(take(&mut rest, 2)?.try_into().ok()?);
                    if usize::from(count) > self.neighbors(slot, layer).ok()?.len() {
                        return None;
                    }
                    let ids = take(&mut rest, 4 * usize::from(count))?.as_chunks::<4>().0;
                    let ids: Vec<u64> =
                        ids.iter().map(|&id| u32::from_le_bytes(id).into()).collect();
                    if ids.iter().any(|&id| id >= self.committed.count) {
                        return None;
                    }
                    lists.push(ids);
                }
                let body = &start[..start.len() - rest.len()];
                let sum = take(&mut rest, 8)?;
                (sum == xxh3_64_with_seed(body, graph).to_le_bytes()).then_some((slot, lists))
            })();
            if entry.is_none() {
                rest = &[];
            }
            entry
        })
    }

    /// Before a commit forces changed lists to disk, makes the lists saved for them durable.
    fn sync_undo(&mut self) -> Result<()> {
        #[cfg(test)]
        if crate::power_loss::simulating() {
            return Ok(());
        }
        let Some(undo) = &mut self.undo else { return Ok(()) };
        match undo.file.get_mut().unwrap_or_else(PoisonError::into_inner) {
            Some((file, _)) => file.sync_data().context("Failed to sync the undo file"),
            None => Ok(()),
        }
    }

    /// Once a commit has changed the graph, the lists saved from the one before are obsolete:
    /// removes the undo file and starts over.
    fn restart_undo(&mut self) {
        let Some(undo) = &mut self.undo else { return };
        if self.committed.graph_hash() == undo.graph {
            return;
        }
        if undo.file.get_mut().unwrap_or_else(PoisonError::into_inner).take().is_some() {
            let _ = std::fs::remove_file(&undo.path);
        }
        *undo = Undo::new(std::mem::take(&mut undo.path), &self.committed);
    }

    /// This file has been renamed to `path`.
    pub(crate) fn moved_to(&mut self, path: &Path) {
        if let Some(undo) = &mut self.undo {
            undo.path = sibling(path, "undo");
        }
    }

    /// Commits all written slots (the graph's own commit counts only the slots it linked).
    ///
    /// # Errors
    ///
    /// Returns an error if writing or syncing fails; after that, every later commit fails too.
    pub fn commit(&mut self) -> Result<()> {
        self.state.count = self.count;
        self.commit_deleting(&[])
    }

    /// Makes every later commit fail: this file is no longer the index.
    #[cfg(windows)]
    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Commits `state`, deleting `deletes` in the same commit (ADR-0008, decision 6).
    pub(crate) fn commit_deleting(&mut self, deletes: &[u64]) -> Result<()> {
        if !self.writable {
            fail!(
                ReadOnly,
                "This index was opened as a reader, which only searches\nhelp: to add, delete, flush or compact, open it as a writer; an index has one writer at a time"
            );
        }
        if self.poisoned {
            fail!(Io, "An earlier flush failed, so later ones can't be trusted; reopen the index");
        }
        let result = self.commit_inner(deletes);
        self.poisoned = result.is_err();
        result
    }

    fn commit_inner(&mut self, deletes: &[u64]) -> Result<()> {
        self.sync_undo()?;
        if !deletes.is_empty() {
            // An intent header first: the previous commit, plus the epoch recovery rolls back.
            let epoch = self.state.epoch.checked_add(1).context("Delete epoch overflow")?;
            self.write_header(&FileHeader { pending_epoch: epoch, ..self.committed.clone() });
            self.sync()?;
            for &slot in deletes {
                self.set_slot_epoch(slot, epoch)?;
            }
            self.state.epoch = epoch;
        }
        self.sync()?;
        self.state.pending_epoch = 0;
        let state = self.state.clone();
        self.write_header(&state);
        self.sync()?;
        self.committed = state;
        self.restart_undo();
        self.huge_pages_up_to(self.committed.count);
        Ok(())
    }

    /// Writes `header` into the copy that does not hold the newest one.
    fn write_header(&mut self, header: &FileHeader) {
        let copy = 1 - self.current_copy;
        let bytes = FileHeader { sequence: self.sequence + 1, ..header.clone() }.to_bytes();
        // Readers that see this header must see the data it commits.
        fence(Ordering::Release);
        self.headers.store_words(copy * HEADER_STRIDE, &bytes);
        self.current_copy = copy;
        self.sequence += 1;
    }

    /// Clears delete marks a crashed flush wrote but never committed.
    fn recover(&mut self) -> Result<()> {
        for slot in 0..self.count {
            if self.slot_epoch(slot)? > self.state.epoch {
                self.set_slot_epoch(slot, 0)?;
            }
        }
        self.commit_deleting(&[])
    }

    fn sync(&mut self) -> Result<()> {
        // The simulation models durability itself; real fsyncs would only slow it down.
        #[cfg(test)]
        if crate::power_loss::simulating() {
            crate::power_loss::before_fsync(&self.file_bytes());
            crate::power_loss::after_fsync(&self.file_bytes());
            return Ok(());
        }
        self.headers.map.flush()?;
        let regions = self.segments.iter().map(|s| &s.region);
        for region in
            regions.chain(&self.chunks).chain(&self.segment_pages).chain(&self.chunk_pages)
        {
            region.map.flush()?;
        }
        // sync_all, not sync_data: the file length must be durable too.
        self.file.sync_all()?;
        Ok(())
    }

    /// The whole file as the page cache holds it, for the power-loss simulation.
    #[cfg(test)]
    pub(crate) fn file_bytes(&self) -> Vec<u8> {
        use std::io::{Read, Seek};
        let mut bytes = Vec::new();
        let mut file = &self.file;
        file.seek(std::io::SeekFrom::Start(0)).expect("seek");
        file.read_to_end(&mut bytes).expect("read");
        bytes
    }

    /// Slots with data written
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the vector dimensions
    pub fn dimensions(&self) -> u32 {
        self.geometry.dims as u32
    }

    /// The graph parameters the file was created with.
    pub fn params(&self) -> NodeRecordParams {
        self.geometry.params
    }

    /// The distance metric the file was created with.
    pub fn metric(&self) -> DistanceMetric {
        if self.state.metric == 1 { DistanceMetric::Cosine } else { DistanceMetric::Euclidean }
    }

    /// What the file keeps of each component of a vector.
    pub fn precision(&self) -> Precision {
        self.geometry.precision
    }

    /// Forgets slots written past `count`, so the next insert reuses them.
    pub(crate) fn truncate_logical(&mut self, count: u64) {
        self.count = self.count.min(count);
    }
}

/// Writes `ids` into a list readers may be traversing. An id in both the old and the new list
/// keeps its entry, so a reader never misses it; entries left empty are cleared last.
fn store_list(list: &[AtomicU32], ids: &[u64]) {
    let new: Vec<u32> = ids
        .iter()
        .filter_map(|&id| u32::try_from(id).ok().filter(|&id| id != EMPTY))
        .take(list.len())
        .collect();
    let mut placed = vec![false; new.len()];
    let mut free = Vec::new();
    for entry in list {
        let old = entry.load(Ordering::Relaxed);
        match new.iter().zip(&mut placed).find(|(id, placed)| **id == old && !**placed) {
            Some((_, placed)) => *placed = true,
            None => free.push((entry, old)),
        }
    }
    let mut missing = new.iter().zip(&placed).filter(|(_, placed)| !**placed).map(|(id, _)| *id);
    let mut emptied = Vec::new();
    for (entry, old) in free {
        match missing.next() {
            Some(id) => entry.store(id, Ordering::Relaxed),
            None if old != EMPTY => emptied.push(entry),
            None => {}
        }
    }
    for entry in emptied {
        entry.store(EMPTY, Ordering::Relaxed);
    }
}

/// Offsets of the checksum and sequence words of both header copies.
const HEADER_WORDS: [usize; 4] = [16, 24, HEADER_STRIDE + 16, HEADER_STRIDE + 24];

/// The newest valid header copy, plus `HEADER_WORDS` if both copies were valid. A copy the writer
/// is rewriting fails its checksum, so it is read again while it keeps changing, a bounded number
/// of times; one that reads the same twice is torn.
fn newest_header(headers: &Region) -> Result<(FileHeader, Option<[u64; 4]>)> {
    let copy = |at: usize| -> Vec<u8> {
        let len = (headers.u64_at(at + 32).load(Ordering::Relaxed) as u32 as usize)
            .clamp(136, HEADER_STRIDE)
            .next_multiple_of(8);
        let words = (0..len / 8).map(|i| headers.u64_at(at + 8 * i).load(Ordering::Relaxed));
        words.flat_map(u64::to_le_bytes).collect()
    };
    let mut previous = None;
    for attempt in 0..=100 {
        let bytes = (copy(0), copy(HEADER_STRIDE));
        let (a, b) = (FileHeader::from_bytes(&bytes.0)?, FileHeader::from_bytes(&bytes.1)?);
        if (a.is_some() && b.is_some()) || previous.as_ref() == Some(&bytes) || attempt == 100 {
            fence(Ordering::Acquire);
            let both_valid = (a.is_some() && b.is_some()).then(|| {
                let word = |at: usize| {
                    let copy = if at < HEADER_STRIDE { &bytes.0 } else { &bytes.1 };
                    let at = at % HEADER_STRIDE;
                    u64::from_le_bytes(copy[at..at + 8].try_into().expect("8 bytes"))
                };
                HEADER_WORDS.map(word)
            });
            return FileHeader::newest(a, b).map(|(_, header)| (header, both_valid));
        }
        previous = Some(bytes);
        std::thread::yield_now();
    }
    unreachable!("the last attempt returns")
}

fn read_prefix(file: &File, prefix: &mut [u8]) -> Result<()> {
    use std::io::{Read, Seek};
    let mut file = file;
    file.seek(std::io::SeekFrom::Start(0))?;
    let n = file.read(prefix)?;
    prefix[n..].fill(0);
    Ok(())
}

/// The error for an index file that couldn't be opened, `for_writing` or not, saying what to do
/// about it, which depends on why.
fn open_failed(path: &Path, e: std::io::Error, for_writing: bool) -> anyhow::Error {
    use std::io::ErrorKind as Os;
    let at = path.display();
    let (kind, message) = match e.kind() {
        Os::NotFound if !for_writing => (
            ErrorKind::NotFound,
            format!(
                "There is no index at {at}\nhelp: a reader opens an index that exists; create it \
                 by opening it as a writer first, or check the path"
            ),
        ),
        Os::NotFound => (
            ErrorKind::NotFound,
            format!(
                "Can't create an index at {at}: its directory doesn't exist\nhelp: create the \
                 directory first, or give a path in one that exists"
            ),
        ),
        Os::IsADirectory => (
            ErrorKind::InvalidArgument,
            format!(
                "{at} is a directory, not an index file\nhelp: give the path of a file in it, such \
                 as {}",
                path.join("index.chassis").display()
            ),
        ),
        Os::PermissionDenied => (
            ErrorKind::Io,
            format!(
                "No permission to {} {at} ({e})\nhelp: check the file's and its directory's \
                 permissions, or give a path you can write",
                if for_writing { "write" } else { "read" }
            ),
        ),
        _ => (
            ErrorKind::Io,
            format!(
                "Can't open {at}: {e}\nhelp: check that its disk is mounted, writable and has \
                 room, or give another path"
            ),
        ),
    };
    raised(kind, message)
}

fn not_an_index(path: &Path) -> anyhow::Error {
    raised(
        ErrorKind::NotAnIndex,
        format!(
            "{} is not a Chassis index: it doesn't begin with a Chassis header\nhelp: check the \
             path; to create a new index, give a path where no file exists yet",
            path.display()
        ),
    )
}

fn dimension_mismatch(path: &Path, held: u32, asked: u32) -> anyhow::Error {
    raised(
        ErrorKind::DimensionMismatch,
        format!(
            "The index at {} holds vectors of {held} dimensions, not {asked}\nhelp: open it with \
             {held} dimensions, or create a new index at another path",
            path.display()
        ),
    )
}

/// Opens (creating if missing) and locks `path`, making sure the lock is on the file now at it.
fn open_locked(path: &Path) -> Result<File> {
    // Windows refuses to open a directory with "access denied", which would read as permissions.
    if path.is_dir() {
        return Err(open_failed(path, std::io::ErrorKind::IsADirectory.into(), true));
    }
    for _ in 0..8 {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| open_failed(path, e, true))?;

        // CRITICAL: Exclusive file locking prevents concurrent access corruption
        if lock_writer(&file).is_err() {
            fail!(
                Locked,
                "{} is already open as a writer, in this process or another, and an index has \
                 one writer at a time\nhelp: close the other writer first, or open a reader to \
                 search while it writes",
                path.display()
            );
        }
        // A migration may have replaced the file between our open and our lock.
        if is_at(&file, path)? {
            return Ok(file);
        }
    }
    bail!("{} kept being replaced while opening it", path.display())
}

/// Takes the one writer's lock; closing the file releases it.
#[cfg(not(windows))]
fn lock_writer(file: &File) -> std::io::Result<()> {
    fs2::FileExt::try_lock_exclusive(file)
}

/// On Windows a lock on the whole file would make readers' `ReadFile` fail, so the writer locks one
/// byte at 2^62, which still overlaps the whole-range lock releases up to 0.6.3 take.
#[cfg(windows)]
fn lock_writer(file: &File) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;
    // SAFETY: OVERLAPPED is plain data; zero is a valid value.
    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
    overlapped.Anonymous.Anonymous.OffsetHigh = 1 << 30; // byte 2^62
    let flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
    // SAFETY: a valid handle, and the call completes before `overlapped` goes out of scope.
    if unsafe { LockFileEx(file.as_raw_handle() as _, flags, 0, 1, 0, &mut overlapped) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(unix)]
fn is_at(file: &File, path: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let ours = file.metadata()?;
    match std::fs::metadata(path) {
        Ok(theirs) => Ok(ours.dev() == theirs.dev() && ours.ino() == theirs.ino()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Windows refuses to replace a file that has open handles, so the file at a path is ours.
#[cfg(not(unix))]
fn is_at(_file: &File, _path: &Path) -> Result<bool> {
    Ok(true)
}

/// `path` with `.suffix` appended to its file name.
pub(crate) fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{suffix}"));
    path.with_file_name(name)
}

pub(crate) fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        File::open(dir)?.sync_all().context("Failed to sync the index's directory")?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Renames `temp` over `path` and makes the rename durable. On Windows nothing may have `path`
/// open, this process included.
#[cfg(unix)]
pub(crate) fn rename_over(temp: &Path, path: &Path) -> Result<()> {
    std::fs::rename(temp, path)?;
    sync_dir(path)
}

/// `MoveFileExW` directly: `std::fs::rename` retries with POSIX semantics and would replace the
/// file under other processes' handles.
#[cfg(windows)]
pub(crate) fn rename_over(temp: &Path, path: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let wide = |p: &Path| p.as_os_str().encode_wide().chain(Some(0)).collect::<Vec<u16>>();
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    // SAFETY: both paths are NUL-terminated wide strings that outlive the call.
    if unsafe { MoveFileExW(wide(temp).as_ptr(), wide(path).as_ptr(), flags) } == 0 {
        return Err(std::io::Error::last_os_error())
            .context("Another process has the index open, so its file can't be replaced");
    }
    Ok(())
}

/// Renames the migrated file over the original and makes the rename durable.
#[cfg(unix)]
fn replace(temp: &Path, path: &Path, original: File, storage: &mut Storage) -> Result<()> {
    rename_over(temp, path)?;
    // Releases before v3 don't re-check which file their lock is on, so hold the old one.
    storage._replaced = Some(original);
    Ok(())
}

/// Windows refuses to rename over a file with any open handle, so the original is closed first.
#[cfg(windows)]
fn replace(temp: &Path, path: &Path, original: File, _storage: &mut Storage) -> Result<()> {
    drop(original);
    rename_over(temp, path).context("Migration will retry when the index is next opened")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_locate_matches_capacity() {
        let params = NodeRecordParams::default();
        let header =
            Geometry::initial_header(32, params, DistanceMetric::Euclidean, Precision::Full)
                .unwrap();
        let g = Geometry::new(&header).unwrap();
        let mut slot = 0;
        for k in 0..g.doubling_segments + 3 {
            assert_eq!(g.capacity(k), slot);
            for i in 0..g.segment_slots(k) {
                assert_eq!(g.locate(slot), (k as usize, i as usize), "slot {slot}");
                slot += 1;
            }
        }
    }

    #[test]
    fn test_segments_heap_and_tables_survive_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let n = 300u64;
        {
            let mut storage = Storage::open(&path, 8).unwrap();
            for slot in 0..n {
                assert_eq!(storage.append(slot * 3, &[slot as f32; 8]).unwrap(), slot);
                let layers = vec![vec![slot.saturating_sub(1)]; 1 + (slot % 3) as usize];
                storage.write_record(slot, &layers).unwrap();
            }
            storage.commit().unwrap();
            assert!(storage.segments.len() > 4 && storage.chunks.len() > 1);
            assert!(storage.segment_pages.len() > 1 && storage.chunk_pages.len() > 1);
        }
        let storage = Storage::open(&path, 8).unwrap();
        assert_eq!(storage.count(), n);
        for slot in 0..n {
            assert_eq!(storage.get_vector_slice(slot).unwrap(), &[slot as f32; 8]);
            assert_eq!(storage.slot_id(slot).unwrap(), slot * 3);
            assert_eq!(storage.layer_count(slot).unwrap(), 1 + (slot % 3) as usize);
            for layer in 0..storage.layer_count(slot).unwrap() {
                let list: Vec<u32> = storage
                    .neighbors(slot, layer)
                    .unwrap()
                    .iter()
                    .map(|id| id.load(Ordering::Relaxed))
                    .collect();
                assert_eq!(list[0], slot.saturating_sub(1) as u32);
                assert!(list[1..].iter().all(|&id| id == EMPTY));
            }
        }
    }

    #[test]
    fn test_uncommitted_slots_and_regions_are_reused() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let end = {
            let mut storage = Storage::open(&path, 4).unwrap();
            storage.insert(&[1.0; 4]).unwrap();
            storage.commit().unwrap();
            let end = storage.state.file_end;
            for _ in 0..100 {
                storage.insert(&[2.0; 4]).unwrap();
            }
            end
        };
        let mut storage = Storage::open(&path, 4).unwrap();
        assert_eq!(storage.count(), 1);
        assert_eq!(storage.state.file_end, end);
        for _ in 0..100 {
            storage.insert(&[3.0; 4]).unwrap();
        }
        storage.commit().unwrap();
        assert_eq!(storage.get_vector(100).unwrap(), vec![3.0; 4]);
    }

    #[test]
    fn test_only_whole_huge_pages_are_asked_for() {
        const MB: u64 = 1 << 20;
        assert_eq!(whole_huge_pages(0, 2 * MB), Some((0, 2 * MB)));
        assert_eq!(whole_huge_pages(4 * MB, 8 * MB), Some((4 * MB, 8 * MB)));
        // A page shared with bytes before the range, or after it, is left out.
        assert_eq!(whole_huge_pages(1, 4 * MB), Some((2 * MB, 4 * MB)));
        assert_eq!(whole_huge_pages(3 * MB, 9 * MB - 1), Some((4 * MB, 8 * MB)));
        assert_eq!(whole_huge_pages(1, 2 * MB), None);
        assert_eq!(whole_huge_pages(5 * MB, 6 * MB), None);
        assert_eq!(whole_huge_pages(4 * MB, 4 * MB), None);
    }

    #[test]
    fn test_warming_asks_for_what_was_written_and_nothing_past_it() {
        let dir = tempdir().unwrap();
        let mut storage = Storage::open(dir.path().join("index.chassis"), 8).unwrap();
        assert!(storage.written(None).is_empty());
        // Segments of 16, 32 and 64 slots here: the third is left with 40 written.
        let written = 16 + 32 + 40;
        for slot in 0..written {
            storage.insert(&[slot as f32; 8]).unwrap();
            // Every eighth node has lists above layer 0, in the heap.
            let lists = vec![vec![]; if slot % 8 == 0 { 3 } else { 1 }];
            storage.write_record(slot, &lists).unwrap();
        }
        storage.commit().unwrap();

        let g = storage.geometry;
        let ranges = storage.written(None);
        let per_slot = SLOT_HEADER + g.vector_bytes() + g.level0_bytes();
        // Heap chunks before the one in use count whole; that one up to its last entry.
        let (chunk, used) = (storage.state.heap_used >> 32, storage.state.heap_used & 0xffff_ffff);
        let heap = (0..chunk).map(|c| g.chunk_bytes(c)).sum::<usize>() + used as usize;
        assert!(heap > 0);
        assert_eq!(ranges.iter().map(|r| r.1).sum::<usize>(), written as usize * per_slot + heap);
        // Each range lies inside one region, and the last segment's arrays end at its last slot.
        let regions: Vec<(usize, usize)> = (storage.segments.iter().map(|s| &s.region))
            .chain(&storage.chunks)
            .map(|r| (r.map.as_ptr() as usize, r.len()))
            .collect();
        for &(at, len) in &ranges {
            assert!(regions.iter().any(|&(base, size)| base <= at && at + len <= base + size));
        }
        let last = &storage.segments[2];
        let at = last.region.map.as_ptr() as usize;
        for (start, per_slot) in [
            (at, SLOT_HEADER),
            (at + last.vectors, g.vector_bytes()),
            (at + last.level0, g.level0_bytes()),
        ] {
            assert!(ranges.contains(&(start, per_slot * 40)));
        }

        // Reading it in changes nothing. Where there is a way to ask (not on Windows yet), asking
        // again once that is done starts another.
        storage.warm();
        if cfg!(unix) {
            storage.warming.as_mut().unwrap().thread.take().unwrap().join().unwrap();
            let done = Arc::clone(&storage.warming.as_ref().unwrap().stop);
            storage.warm();
            assert!(!Arc::ptr_eq(&storage.warming.as_ref().unwrap().stop, &done));
        }
        assert_eq!(storage.get_vector(87).unwrap(), vec![87.0; 8]);
        // While one is under way, asking again starts none.
        let (busy, _) = stand_in();
        let stop = Arc::clone(&busy.stop);
        storage.warming = Some(busy);
        storage.warm();
        assert!(Arc::ptr_eq(&storage.warming.as_ref().unwrap().stop, &stop));

        // In pieces the thread can be stopped between.
        let pieces: Vec<_> = steps(vec![(5, 2 * WARM_STEP + 1), (9, 3)]).collect();
        let step = WARM_STEP;
        assert_eq!(pieces, [(5, step), (5 + step, step), (5 + 2 * step, 1), (9, 3)]);
        // From the start of a range's first page to where it ended.
        assert_eq!(from_page_start(4096, (3 * 4096 + 24, 100)), (3 * 4096, 124));
        assert_eq!(from_page_start(16384, (16384, 5)), (16384, 5));
        // Through a view of the file, the same ranges at the same offsets in it.
        let view = unsafe { Mmap::map(&storage.file) }.unwrap();
        let base = view.as_ptr() as usize;
        let offsets = (storage.segments.iter().map(|s| &s.region))
            .chain(&storage.chunks)
            .map(|r| (r.map.as_ptr() as usize, r.offset as usize));
        let to_view = |at: usize| {
            let (mapped, offset) =
                offsets.clone().filter(|&(mapped, _)| mapped <= at).max().unwrap();
            base + offset + (at - mapped)
        };
        let through_view: Vec<_> = ranges.iter().map(|&(at, len)| (to_view(at), len)).collect();
        assert_eq!(storage.written(Some(&view)), through_view);
    }

    /// A stand-in for a warming thread: it works until it is told to stop, or for ten seconds,
    /// then a moment more, and records whether it was told.
    fn stand_in() -> (Warming, Arc<AtomicBool>) {
        let (stop, told) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (stopped, record) = (Arc::clone(&stop), Arc::clone(&told));
        let thread = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            while !stopped.load(Ordering::Relaxed) && start.elapsed().as_secs() < 10 {
                std::thread::yield_now();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            record.store(stopped.load(Ordering::Relaxed), Ordering::Relaxed);
        });
        (Warming { stop, thread: Some(thread), process: std::process::id() }, told)
    }

    #[test]
    fn test_dropping_a_warming_stops_its_thread_and_waits_for_it() {
        let (warming, told) = stand_in();
        drop(warming);
        assert!(told.load(Ordering::Relaxed));
    }

    #[cfg(unix)]
    #[test]
    fn test_a_forked_child_drops_a_warming_it_has_no_thread_for() {
        let (warming, _) = stand_in();
        // SAFETY: the child drops one value and exits.
        let child = unsafe { libc::fork() };
        if child == 0 {
            let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(warming)));
            unsafe { libc::_exit(i32::from(dropped.is_err())) };
        }
        let mut status = -1;
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert_eq!(status, 0);
    }

    #[test]
    fn test_a_heap_that_ends_past_its_chunk_is_a_corrupt_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut storage = Storage::open(&path, 4).unwrap();
        storage.insert(&[0.0; 4]).unwrap();
        storage.write_record(0, &[vec![], vec![]]).unwrap();
        storage.state.heap_used += storage.geometry.chunk_bytes(0) as u64;
        storage.commit().unwrap();
        drop(storage);
        let error = Storage::open(&path, 4).unwrap_err().to_string();
        assert!(error.contains("Corrupt file header"), "{error}");
    }

    #[test]
    fn test_roll_back_keeps_the_first_entry_saved_for_a_slot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let list = |storage: &Storage| -> Vec<u32> {
            let list = storage.neighbors(0, 0).unwrap();
            list.iter().map(|id| id.load(Ordering::Relaxed)).filter(|&id| id != EMPTY).collect()
        };
        let mut storage = Storage::open(&path, 4).unwrap();
        for slot in 0..4 {
            storage.insert(&[slot as f32; 4]).unwrap();
            storage.write_record(slot, &[vec![(slot + 1) % 4]]).unwrap();
        }
        storage.commit().unwrap();
        // Changing a committed list saves it first, whoever calls.
        storage.write_neighbors(0, 0, &[2]).unwrap();
        // A second thread that read the list after that change, having missed that it was saved.
        storage.undo.as_ref().unwrap().saved[0].store(0, Ordering::Relaxed);
        storage.save_lists(&[0]).unwrap();
        assert_eq!(list(&storage), [2]);
        drop(storage);

        assert_eq!(list(&Storage::open(&path, 4).unwrap()), [1]);
    }

    #[test]
    fn test_torn_newest_header_falls_back_to_the_other_copy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let (copy, sequence) = {
            let mut storage = Storage::open(&path, 4).unwrap();
            storage.insert(&[1.0; 4]).unwrap();
            storage.commit().unwrap();
            storage.insert(&[2.0; 4]).unwrap();
            storage.commit().unwrap();
            (storage.current_copy, storage.sequence)
        };
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[copy * HEADER_STRIDE + 60] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        let storage = Storage::open(&path, 4).unwrap();
        assert_eq!((storage.count(), storage.sequence), (1, sequence - 1));
    }

    #[test]
    fn test_reader_never_takes_an_older_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        for _ in 0..2 {
            writer.insert(&[1.0; 4]).unwrap();
            writer.commit().unwrap();
        }
        let mut reader = Storage::open_read_only(&path, 4).unwrap();
        assert_eq!(reader.state.count, 2);
        // The newest copy fails its checksum (as a copy mid-rewrite does); the other one is older.
        let at = writer.current_copy * HEADER_STRIDE + 16;
        writer.headers.bytes_mut(at, 1)[0] ^= 1;
        reader.refresh().unwrap();
        assert_eq!((reader.state.count, reader.sequence), (2, writer.sequence));
    }

    #[test]
    fn test_refresh_sees_every_header_written_before_it() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[0.0; 4]).unwrap();
        writer.commit().unwrap();
        let mut reader = Storage::open_read_only(&path, 4).unwrap();
        let done = Arc::new(AtomicU64::new(writer.sequence));
        let stop = Arc::new(AtomicBool::new(false));
        let (d, s) = (done.clone(), stop.clone());
        // Header writes back to back, so refreshes often find a copy mid-rewrite.
        let thread = std::thread::spawn(move || {
            let header = writer.committed.clone();
            while !s.load(Ordering::Relaxed) {
                writer.write_header(&header);
                d.store(writer.sequence, Ordering::Release);
            }
        });
        let start = std::time::Instant::now();
        let (mut stale, mut errors, mut refreshes) = (0u64, 0u64, 0u64);
        while start.elapsed() < std::time::Duration::from_secs(2) {
            let before = done.load(Ordering::Acquire);
            match reader.refresh() {
                Ok(()) => stale += u64::from(reader.sequence < before),
                Err(_) => errors += 1,
            }
            refreshes += 1;
        }
        stop.store(true, Ordering::Relaxed);
        thread.join().unwrap();
        assert_eq!((stale, errors), (0, 0), "of {refreshes} refreshes");
    }

    #[test]
    fn test_reader_remaps_a_region_a_restarted_writer_reused() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[0.0; 4]).unwrap();
        writer.commit().unwrap();
        let mut reader = Storage::open_read_only(&path, 4).unwrap();
        // A writer adds a second segment, the reader maps it, and the writer dies unflushed.
        for _ in 1..20 {
            writer.insert(&[1.0; 4]).unwrap();
        }
        reader.refresh().unwrap();
        assert_eq!(reader.segments.len(), 2);
        reader.warm();
        drop(writer);
        // The next writer puts a heap chunk's table page where that segment was, and the segment
        // after it.
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[2.0; 4]).unwrap();
        writer.write_record(1, &[vec![0], vec![0]]).unwrap();
        for _ in 2..20 {
            writer.insert(&[3.0; 4]).unwrap();
        }
        writer.publish_routing(20);
        reader.refresh().unwrap();
        assert_eq!(reader.get_vector(19).unwrap(), vec![3.0; 4]);
        // Its warming thread was told about the segment that went, and was stopped for that.
        assert!(reader.warming.is_none());
    }

    #[test]
    fn test_reader_maps_regions_added_since_the_last_commit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[0.0; 4]).unwrap();
        writer.write_record(0, &[vec![]]).unwrap();
        writer.commit().unwrap();
        let mut reader = Storage::open_read_only(&path, 4).unwrap();
        for slot in 1..100u64 {
            writer.insert(&[slot as f32; 4]).unwrap();
            writer.write_record(slot, &[vec![slot - 1], vec![slot - 1]]).unwrap();
        }
        writer.publish_routing(100);
        reader.refresh().unwrap();
        assert_eq!(reader.count(), 100);
        assert_eq!(reader.get_vector(99).unwrap(), vec![99.0; 4]);
        assert_eq!(reader.neighbors(99, 1).unwrap()[0].load(Ordering::Relaxed), 98);
    }

    #[test]
    fn test_restarted_writer_withdraws_a_crashed_writers_routing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[0.0; 4]).unwrap();
        writer.commit().unwrap();
        for _ in 1..20 {
            writer.insert(&[1.0; 4]).unwrap();
        }
        writer.publish_routing(20);
        drop(writer);
        let _writer = Storage::open(&path, 4).unwrap();
        let reader = Storage::open_read_only(&path, 4).unwrap();
        assert_eq!(reader.count(), 1, "readers route through the crashed writer's slots");
    }

    #[test]
    fn test_store_list_keeps_the_entry_of_every_id_it_keeps() {
        let list: Vec<AtomicU32> = [5, 7, 9, EMPTY].map(AtomicU32::new).into();
        store_list(&list, &[9, 11, 5]);
        let ids: Vec<u32> = list.iter().map(|e| e.load(Ordering::Relaxed)).collect();
        assert_eq!(ids, [5, 11, 9, EMPTY]);
    }

    #[test]
    fn test_reader_ignores_marks_of_an_uncommitted_delete() {
        use crate::hnsw::{HnswGraph, HnswParams};
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut storage = Storage::open(&path, 4).unwrap();
        storage.insert(&[1.0; 4]).unwrap();
        storage.commit().unwrap();
        // A flush that has written its intent header and marks, not yet its commit header.
        let intent = FileHeader { pending_epoch: 1, ..storage.committed.clone() };
        storage.write_header(&intent);
        storage.set_slot_epoch(0, 1).unwrap();
        let reader = Storage::open_read_only(&path, 4).unwrap();
        let reader = HnswGraph::reader(reader, HnswParams::default()).unwrap();
        assert!(!reader.is_deleted(0).unwrap(), "an uncommitted delete applied");
    }

    #[test]
    fn test_reader_mid_search_survives_a_writer_restart() {
        use crate::hnsw::{HnswGraph, HnswParams};
        use std::io::{Seek, Write};
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        let mut writer = Storage::open(&path, 4).unwrap();
        writer.insert(&[0.0; 4]).unwrap();
        writer.write_record(0, &[vec![]]).unwrap();
        writer.state.entry_point = 0;
        writer.commit().unwrap();
        // Unflushed: a second segment, which committed node 0 now links into.
        for slot in 1..20u64 {
            writer.insert(&[slot as f32; 4]).unwrap();
            writer.write_record(slot, &[vec![0]]).unwrap();
        }
        writer.write_neighbors(0, 0, &[17, 18, 19]).unwrap();
        writer.publish_routing(20);
        let storage = Storage::open_read_only(&path, 4).unwrap();
        let reused = storage.state.file_end;
        let reader = HnswGraph::reader(storage, HnswParams::default()).unwrap();
        drop(writer);

        // The next writer reuses the space past the committed end for other data.
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(std::io::SeekFrom::Start(reused)).unwrap();
        file.write_all(&[0xff; 64 * 1024]).unwrap();
        // A search still on the old snapshot routes into those bytes, and returns committed
        // slots only.
        let hits = reader.search(&[18.0; 4], 5, 16).unwrap();
        assert!(hits.iter().all(|hit| hit.id == 0), "{hits:?}");
    }

    #[test]
    fn test_recovery_clears_marks_of_an_uncommitted_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.chassis");
        {
            let mut storage = Storage::open(&path, 4).unwrap();
            storage.insert(&[1.0; 4]).unwrap();
            storage.commit().unwrap();
            // A flush that wrote its intent header and marks, then died before committing.
            let intent = FileHeader { pending_epoch: 1, ..storage.committed.clone() };
            storage.write_header(&intent);
            storage.set_slot_epoch(0, 1).unwrap();
            storage.sync().unwrap();
        }
        let storage = Storage::open(&path, 4).unwrap();
        assert_eq!(storage.slot_epoch(0).unwrap(), 0);
        assert_eq!((storage.state.pending_epoch, storage.committed.pending_epoch), (0, 0));
    }
}
