//! File header, format v3 (ADR-0008).
//!
//! Two copies sit 64 KiB apart, so they never share a page. Each carries a sequence number and an
//! xxh3-64 checksum; the valid copy with the higher sequence number is current. A commit writes
//! the other copy, so a torn header write always leaves the previous commit readable.

use anyhow::{Context, Result, bail};
use xxhash_rust::xxh3::xxh3_64;

/// Magic bytes at the start of every Chassis file, in every format version.
pub const MAGIC: &[u8; 8] = b"CHASSIS\0";

/// The file format this release writes. It reads this version and migrates versions 1 and 2.
pub const VERSION: u32 = 3;

/// Distance between the two header copies.
pub(crate) const HEADER_STRIDE: usize = 64 * 1024;

/// Header A, header B, then the live page for readers in other processes; regions start after it.
pub(crate) const REGIONS_START: u64 = 3 * HEADER_STRIDE as u64;

/// Table pages one header can reference, per table: enough for 2^32 slots in 1,024-slot segments.
pub(crate) const MAX_TABLE_PAGES: usize = 512;

/// Some slot's id differs from the slot number.
pub(crate) const FLAG_CUSTOM_IDS: u8 = 1;

const CHECKSUM: std::ops::Range<usize> = 16..24;
const FIXED_LEN: usize = 136;

/// One header copy. Offsets are bytes, little-endian; see `docs/src/architecture/file-format.md`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileHeader {
    /// As read; `to_bytes` always writes `VERSION`. Only a writer refuses a newer one.
    pub write_version: u32,
    pub sequence: u64,
    pub dims: u32,
    pub m: u16,
    pub m0: u16,
    pub max_layers: u8,
    /// 0: Euclidean; 1: cosine, over vectors stored at unit length.
    pub metric: u8,
    pub flags: u8,
    /// log2 of the first segment's slot count.
    pub segment_base_log2: u8,
    /// Segments that double in size before every later one has the same size.
    pub doubling_segments: u8,
    /// log2 of the first upper-heap chunk's byte size.
    pub heap_base_log2: u8,
    /// Heap chunks that double in size before every later one has the same size.
    pub doubling_chunks: u8,
    /// log2 of the entries in one table page.
    pub table_page_log2: u8,
    /// Committed slots: slots at or past this are leftovers of a crashed flush.
    pub count: u64,
    /// Entry point slot, `u64::MAX` when the graph is empty.
    pub entry_point: u64,
    pub max_layer: u32,
    /// Last flush that committed deletes.
    pub epoch: u64,
    /// Nonzero while a flush with deletes is in progress: recovery clears marks above `epoch`.
    pub pending_epoch: u64,
    pub deleted_count: u64,
    /// End of the last committed region; the next region is allocated here.
    pub file_end: u64,
    pub segments: u32,
    pub heap_chunks: u32,
    /// Next free upper-heap position, as `chunk << 32 | byte offset`.
    pub heap_used: u64,
    /// File offsets of the table pages listing segment offsets.
    pub segment_table: Vec<u64>,
    /// File offsets of the table pages listing heap chunk offsets.
    pub heap_table: Vec<u64>,
}

impl FileHeader {
    /// Encodes this copy, checksum included.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; FIXED_LEN + 8 * (self.segment_table.len() + self.heap_table.len())];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes()); // read version
        b[12..16].copy_from_slice(&VERSION.to_le_bytes()); // write version
        b[24..32].copy_from_slice(&self.sequence.to_le_bytes());
        let len = b.len() as u32;
        b[32..36].copy_from_slice(&len.to_le_bytes());
        b[36..40].copy_from_slice(&self.dims.to_le_bytes());
        b[40..42].copy_from_slice(&self.m.to_le_bytes());
        b[42..44].copy_from_slice(&self.m0.to_le_bytes());
        b[44] = self.max_layers;
        b[45] = self.metric;
        b[46] = self.flags;
        b[47] = self.table_page_log2;
        b[48] = self.segment_base_log2;
        b[49] = self.doubling_segments;
        b[50] = self.heap_base_log2;
        b[51] = self.doubling_chunks;
        b[56..64].copy_from_slice(&self.count.to_le_bytes());
        b[64..72].copy_from_slice(&self.entry_point.to_le_bytes());
        b[72..76].copy_from_slice(&self.max_layer.to_le_bytes());
        b[80..88].copy_from_slice(&self.epoch.to_le_bytes());
        b[88..96].copy_from_slice(&self.pending_epoch.to_le_bytes());
        b[96..104].copy_from_slice(&self.deleted_count.to_le_bytes());
        b[104..112].copy_from_slice(&self.file_end.to_le_bytes());
        b[112..116].copy_from_slice(&self.segments.to_le_bytes());
        b[116..120].copy_from_slice(&self.heap_chunks.to_le_bytes());
        b[120..128].copy_from_slice(&self.heap_used.to_le_bytes());
        b[128..132].copy_from_slice(&(self.segment_table.len() as u32).to_le_bytes());
        b[132..136].copy_from_slice(&(self.heap_table.len() as u32).to_le_bytes());
        for (i, offset) in self.segment_table.iter().chain(&self.heap_table).enumerate() {
            b[FIXED_LEN + 8 * i..FIXED_LEN + 8 * (i + 1)].copy_from_slice(&offset.to_le_bytes());
        }
        let checksum = xxh3_64(&b);
        b[CHECKSUM].copy_from_slice(&checksum.to_le_bytes());
        b
    }

    /// Decodes one copy: `Ok(None)` if it is torn or was never written, an error if it needs a
    /// newer release to read. The read version is checked before the checksum, so a newer layout
    /// is reported as too new, not as corrupt. A newer write version is only recorded.
    pub fn from_bytes(copy: &[u8]) -> Result<Option<Self>> {
        if copy.len() < FIXED_LEN || copy[0..8] != *MAGIC {
            return Ok(None);
        }
        let u16_at = |at: usize| u16::from_le_bytes([copy[at], copy[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(copy[at..at + 4].try_into().expect("4 bytes"));
        let u64_at = |at: usize| u64::from_le_bytes(copy[at..at + 8].try_into().expect("8 bytes"));
        let (read_version, write_version) = (u32_at(8), u32_at(12));
        if read_version > VERSION {
            bail!(
                "File needs a newer Chassis: format {read_version}, this release reads {VERSION}"
            );
        }
        // Versions 1 and 2 keep the dimensions where v3 keeps the write version.
        if read_version < 3 {
            return Ok(None);
        }
        let len = u32_at(32) as usize;
        if !(FIXED_LEN..=copy.len()).contains(&len) {
            return Ok(None);
        }
        let mut scratch = copy[..len].to_vec();
        scratch[CHECKSUM].fill(0);
        if xxh3_64(&scratch) != u64_at(16) {
            return Ok(None);
        }

        let (segment_pages, heap_pages) = (u32_at(128) as usize, u32_at(132) as usize);
        // Bytes after the tables belong to a later write version; this release ignores them.
        if segment_pages > MAX_TABLE_PAGES
            || heap_pages > MAX_TABLE_PAGES
            || len < FIXED_LEN + 8 * (segment_pages + heap_pages)
        {
            bail!("Corrupt file header: table page counts don't match its length");
        }
        let table = |start: usize, n: usize| (0..n).map(|i| u64_at(start + 8 * i)).collect();
        if copy[45] > 1 {
            bail!("Unknown distance metric {} in file header", copy[45]);
        }
        Ok(Some(Self {
            write_version,
            sequence: u64_at(24),
            dims: u32_at(36),
            m: u16_at(40),
            m0: u16_at(42),
            max_layers: copy[44],
            metric: copy[45],
            flags: copy[46],
            table_page_log2: copy[47],
            segment_base_log2: copy[48],
            doubling_segments: copy[49],
            heap_base_log2: copy[50],
            doubling_chunks: copy[51],
            count: u64_at(56),
            entry_point: u64_at(64),
            max_layer: u32_at(72),
            epoch: u64_at(80),
            pending_epoch: u64_at(88),
            deleted_count: u64_at(96),
            file_end: u64_at(104),
            segments: u32_at(112),
            heap_chunks: u32_at(116),
            heap_used: u64_at(120),
            segment_table: table(FIXED_LEN, segment_pages),
            heap_table: table(FIXED_LEN + 8 * segment_pages, heap_pages),
        }))
    }

    /// The current header and which copy holds it.
    pub fn newest(a: Option<Self>, b: Option<Self>) -> Result<(usize, Self)> {
        match (a, b) {
            (Some(a), Some(b)) if a.sequence == b.sequence => {
                bail!("Corrupt file: both header copies have sequence number {}", a.sequence)
            }
            (Some(a), Some(b)) => Ok(if a.sequence > b.sequence { (0, a) } else { (1, b) }),
            (Some(a), None) => Ok((0, a)),
            (None, Some(b)) => Ok((1, b)),
            (None, None) => None.context("File is not a valid Chassis index: no valid header"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> FileHeader {
        FileHeader {
            write_version: VERSION,
            sequence: 7,
            dims: 768,
            m: 16,
            m0: 32,
            max_layers: 16,
            metric: 1,
            flags: FLAG_CUSTOM_IDS,
            segment_base_log2: 10,
            doubling_segments: 6,
            heap_base_log2: 16,
            doubling_chunks: 12,
            table_page_log2: 13,
            count: 5000,
            entry_point: 42,
            max_layer: 3,
            epoch: 2,
            pending_epoch: 0,
            deleted_count: 9,
            file_end: 1 << 30,
            segments: 3,
            heap_chunks: 1,
            heap_used: 960,
            segment_table: vec![196_608],
            heap_table: vec![262_144],
        }
    }

    #[test]
    fn test_roundtrip() {
        let header = sample();
        assert_eq!(FileHeader::from_bytes(&header.to_bytes()).unwrap(), Some(header));
    }

    #[test]
    fn test_any_flipped_bit_invalidates_the_copy() {
        let bytes = sample().to_bytes();
        for i in 16..bytes.len() {
            let mut torn = bytes.clone();
            torn[i] ^= 0x10;
            assert_eq!(FileHeader::from_bytes(&torn).unwrap(), None, "byte {i}");
        }
    }

    #[test]
    fn test_unknown_metric_is_an_error() {
        let mut bytes = sample().to_bytes();
        bytes[45] = 2;
        bytes[CHECKSUM].fill(0);
        let checksum = xxh3_64(&bytes);
        bytes[CHECKSUM].copy_from_slice(&checksum.to_le_bytes());
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_newer_versions_are_errors_and_old_ones_are_not_v3() {
        let mut bytes = sample().to_bytes();
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(FileHeader::from_bytes(&bytes).unwrap(), None);
        bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_a_newer_write_version_with_more_fields_still_reads() {
        let mut bytes = sample().to_bytes();
        bytes[12..16].copy_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0xab; 16]);
        let len = bytes.len() as u32;
        bytes[32..36].copy_from_slice(&len.to_le_bytes());
        bytes[CHECKSUM].fill(0);
        let checksum = xxh3_64(&bytes);
        bytes[CHECKSUM].copy_from_slice(&checksum.to_le_bytes());
        let header = FileHeader::from_bytes(&bytes).unwrap().unwrap();
        assert_eq!(header, FileHeader { write_version: 4, ..sample() });
    }

    #[test]
    fn test_newest_copy() {
        let a = sample();
        let b = FileHeader { sequence: 8, ..sample() };
        assert_eq!(FileHeader::newest(Some(a.clone()), Some(b.clone())).unwrap().0, 1);
        assert_eq!(FileHeader::newest(Some(a.clone()), None).unwrap().0, 0);
        assert!(FileHeader::newest(Some(a.clone()), Some(a)).is_err());
        assert!(FileHeader::newest(None, None).is_err());
    }
}
