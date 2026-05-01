use crate::header::{HEADER_SIZE, Header, MAGIC};
use anyhow::{Context, Result};
use fs2::FileExt;
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Page size for file alignment (4KB)
const PAGE_SIZE: usize = 4096;

/// Storage engine for on-disk vector data
#[derive(Debug)]
pub struct Storage {
    /// File handle (owns the file lock)
    #[allow(dead_code)]
    file: File,

    /// Memory-mapped view of the file
    mmap: MmapMut,
}

impl Storage {
    /// Opens or creates a Chassis index file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    /// * `dimensions` - Number of dimensions per vector
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or created
    /// - The file is already locked by another process
    /// - The file exists but has different dimensions
    /// - The file is corrupted
    pub fn open<P: AsRef<Path>>(path: P, dimensions: u32) -> Result<Self> {
        let path = path.as_ref();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("Failed to open chassis file: {}", path.display()))?;

        // CRITICAL: Exclusive file locking prevents concurrent access corruption
        file.try_lock_exclusive().context("Chassis file is already open by another process")?;

        let needs_init = file.metadata().map(|m| m.len() < HEADER_SIZE as u64).unwrap_or(true);

        if needs_init {
            // Initialize new file with header
            let header = Header::new(dimensions);
            file.set_len(HEADER_SIZE as u64)?;

            unsafe {
                let mut mmap = MmapMut::map_mut(&file)?;
                mmap[..HEADER_SIZE].copy_from_slice(header.as_bytes());
                mmap.flush()?;
            }
        }

        // Create persistent mapping
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Validate file header
        if mmap.len() < MAGIC.len() || &mmap[..MAGIC.len()] != MAGIC {
            anyhow::bail!("File is not a valid Chassis index");
        }

        let header = unsafe { &*(mmap.as_ptr() as *const Header) };

        if !header.is_valid() {
            anyhow::bail!("Corrupted or incompatible Chassis file at {}", path.display());
        }

        if header.dimensions != dimensions {
            anyhow::bail!(
                "Dimension mismatch: file has {}, requested {}",
                header.dimensions,
                dimensions
            );
        }

        Ok(Self { file, mmap })
    }

    /// Inserts a vector into the storage
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector data to insert
    ///
    /// # Returns
    ///
    /// Returns the index of the inserted vector
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector dimensions don't match the index
    /// - File cannot be expanded
    ///
    /// # Note
    ///
    /// This method does NOT guarantee durability. Call `commit()` to ensure
    /// data is written to disk.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u64> {
        let dims = self.header().dimensions as usize;

        if vector.len() != dims {
            anyhow::bail!("Vector dimension mismatch: expected {}, got {}", dims, vector.len());
        }

        let current_count = self.header().count;
        let vector_bytes = dims * std::mem::size_of::<f32>();
        let offset = HEADER_SIZE + (current_count as usize * vector_bytes);
        let required_size = offset + vector_bytes;

        // Ensure file has enough capacity (may remap)
        self.ensure_capacity(required_size)?;

        // Write vector data first (data-before-header invariant)
        unsafe {
            let dst = self.mmap.as_mut_ptr().add(offset) as *mut f32;
            std::ptr::copy_nonoverlapping(vector.as_ptr(), dst, dims);
        }

        // Update header count only after data is written
        self.header_mut().count = current_count + 1;

        Ok(current_count)
    }

    /// Commits all pending changes to disk
    ///
    /// This method flushes the memory map to the kernel page cache and then
    /// forces a physical write to disk via fsync. This guarantees durability
    /// even in the event of power loss.
    ///
    /// # Performance
    ///
    /// This operation is expensive (1-50ms depending on storage device).
    /// For batch inserts, insert many vectors and call commit() once.
    pub fn commit(&mut self) -> Result<()> {
        // Flush mmap to kernel page cache
        self.mmap.flush()?;

        // Force kernel to flush to physical device
        // On Linux: fdatasync() - flushes data but not metadata
        self.file.sync_data()?;

        // Additional barrier: sync_all() flushes metadata too
        // This is slower but guarantees file size is durable
        self.file.sync_all()?;

        Ok(())
    }

    /// Retrieves a zero-copy slice view of a vector by index
    ///
    /// This method returns a slice that points directly into the memory-mapped
    /// file, avoiding heap allocation. The slice lifetime is tied to `&self`,
    /// which prevents remapping operations while the slice is alive.
    ///
    /// # Arguments
    ///
    /// * `index` - Vector index to retrieve
    ///
    /// # Returns
    ///
    /// Returns a slice backed directly by the mmap (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The index is out of bounds (index >= count)
    /// - The calculated mmap offset is invalid
    ///
    /// # Safety Notes
    ///
    /// This method uses `unsafe` internally but maintains safety through:
    /// - Explicit bounds checking on both index and mmap offset
    /// - Verification that HEADER_SIZE and vector stride are f32-aligned (4 bytes)
    /// - Lifetime binding to `&self` prevents use-after-remap bugs
    ///
    /// The returned slice is guaranteed valid as long as:
    /// - No `&mut self` methods are called (enforced by Rust borrow checker)
    /// - The Storage instance remains alive
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use chassis_core::Storage;
    /// # fn main() -> anyhow::Result<()> {
    /// let storage = Storage::open("vectors.chassis", 128)?;
    /// let slice = storage.get_vector_slice(0)?;
    ///
    /// // Use slice for distance calculations without allocation
    /// let sum: f32 = slice.iter().sum();
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_vector_slice(&self, index: u64) -> Result<&[f32]> {
        let count = self.header().count;

        // Bounds check: Ensure index is within valid range
        if index >= count {
            anyhow::bail!("Index out of bounds: {} (count is {})", index, count);
        }

        let dims = self.header().dimensions as usize;
        let vector_bytes = dims * std::mem::size_of::<f32>();

        // Use checked arithmetic to prevent overflow
        let index_usize = usize::try_from(index).context("Index too large for this platform")?;

        let byte_offset =
            index_usize.checked_mul(vector_bytes).context("Vector offset calculation overflow")?;

        let offset = HEADER_SIZE.checked_add(byte_offset).context("Offset calculation overflow")?;

        // Bounds check: Ensure the calculated offset + vector data fits within mmap
        let end_offset =
            offset.checked_add(vector_bytes).context("End offset calculation overflow")?;

        if end_offset > self.mmap.len() {
            anyhow::bail!(
                "Vector at index {} extends beyond mmap bounds (offset: {}, size: {}, mmap len: {})",
                index,
                offset,
                vector_bytes,
                self.mmap.len()
            );
        }

        // SAFETY:
        // - offset is bounds-checked above with overflow protection
        // - HEADER_SIZE (4096) is 4-byte aligned
        // - vector_bytes is dims * 4, so 4-byte aligned
        // - Therefore offset is 4-byte aligned (required for f32)
        // - dims is the correct length for the slice
        // - Lifetime is tied to &self, preventing use after remap
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset) as *const f32;
            Ok(std::slice::from_raw_parts(ptr, dims))
        }
    }

    /// Retrieves a vector by index
    ///
    /// Returns an owned copy of the vector data. For zero-copy access,
    /// use `get_vector_slice()` instead.
    ///
    /// # Arguments
    ///
    /// * `index` - Vector index to retrieve
    ///
    /// # Returns
    ///
    /// Returns an owned copy of the vector data
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds
    pub fn get_vector(&self, index: u64) -> Result<Vec<f32>> {
        let slice = self.get_vector_slice(index)?;
        Ok(slice.to_vec())
    }

    /// Returns the current vector count
    pub fn count(&self) -> u64 {
        self.header().count
    }

    /// Returns the vector dimensions
    pub fn dimensions(&self) -> u32 {
        self.header().dimensions
    }

    /// Returns the byte offset immediately after the current logical vector data.
    pub(crate) fn vector_end(&self) -> Result<usize> {
        self.vector_end_for_count(self.header().count)
    }

    /// Returns the byte offset immediately after `count` vectors.
    pub(crate) fn vector_end_for_count(&self, count: u64) -> Result<usize> {
        let dims = self.header().dimensions as usize;
        let vector_bytes = dims
            .checked_mul(std::mem::size_of::<f32>())
            .context("Vector byte size calculation overflow")?;
        let count = usize::try_from(count).context("Vector count too large for this platform")?;
        let vector_data_bytes =
            count.checked_mul(vector_bytes).context("Vector zone size calculation overflow")?;

        HEADER_SIZE.checked_add(vector_data_bytes).context("Vector end calculation overflow")
    }

    /// Returns the persisted graph zone offset, if present.
    pub(crate) fn graph_offset(&self) -> Option<u64> {
        self.header().graph_offset()
    }

    /// Persists the graph zone offset in the file header.
    pub(crate) fn set_graph_offset(&mut self, offset: u64) {
        self.header_mut().set_graph_offset(offset);
    }

    /// Align a byte count to the next page boundary.
    #[inline]
    pub(crate) const fn page_align(size: usize) -> usize {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }

    /// Ensures file has enough capacity, growing if necessary
    ///
    /// File growth is page-aligned (4KB boundaries) to optimize for:
    /// - SSD write amplification
    /// - Kernel page cache efficiency
    /// - Hardware block alignment
    ///
    /// # Warning
    ///
    /// This method invalidates all existing pointers into the mmap.
    /// Do not hold references across calls to this method.
    fn ensure_capacity(&mut self, required_size: usize) -> Result<()> {
        if self.mmap.len() >= required_size {
            return Ok(());
        }

        // Round up to next page boundary (4KB)
        let new_size = Self::page_align(required_size);

        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    /// Returns a reference to the header
    fn header(&self) -> &Header {
        unsafe { &*(self.mmap.as_ptr() as *const Header) }
    }

    /// Returns a mutable reference to the header
    fn header_mut(&mut self) -> &mut Header {
        unsafe { &mut *(self.mmap.as_mut_ptr() as *mut Header) }
    }

    /// Get immutable mmap slice for graph zone
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from the start of the file
    /// * `len` - Number of bytes to return
    ///
    /// # Returns
    ///
    /// Returns a slice backed directly by the mmap (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns an error if the requested range is out of bounds
    pub fn graph_zone(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let end = offset.checked_add(len).context("Graph zone end offset overflow")?;

        if end > self.mmap.len() {
            anyhow::bail!(
                "Graph zone access out of bounds: offset={}, len={}, mmap_len={}",
                offset,
                len,
                self.mmap.len()
            );
        }

        Ok(&self.mmap[offset..end])
    }

    /// Get mutable mmap slice for graph zone
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from the start of the file
    /// * `len` - Number of bytes to return
    ///
    /// # Returns
    ///
    /// Returns a mutable slice backed directly by the mmap (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns an error if the requested range is out of bounds
    pub fn graph_zone_mut(&mut self, offset: usize, len: usize) -> Result<&mut [u8]> {
        let end = offset.checked_add(len).context("Graph zone end offset overflow")?;

        if end > self.mmap.len() {
            anyhow::bail!(
                "Graph zone access out of bounds: offset={}, len={}, mmap_len={}",
                offset,
                len,
                self.mmap.len()
            );
        }

        Ok(&mut self.mmap[offset..end])
    }

    /// Ensure graph zone has enough capacity
    ///
    /// This method grows the file and remaps it if necessary to accommodate
    /// the required size.
    ///
    /// # Arguments
    ///
    /// * `required_size` - Minimum required file size in bytes
    ///
    /// # Warning
    ///
    /// This method invalidates all existing pointers into the mmap.
    /// Do not hold references across calls to this method.
    pub fn ensure_graph_capacity(&mut self, required_size: usize) -> Result<()> {
        self.ensure_capacity(required_size)
    }

    /// Move the graph zone to a new offset and update the persisted offset.
    ///
    /// The copy uses memmove semantics so overlapping source and destination ranges are safe.
    /// The file is then resized to the end of the moved graph zone, rounded to a page boundary.
    pub(crate) fn move_graph_zone(
        &mut self,
        old_offset: usize,
        new_offset: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            self.set_graph_offset(new_offset as u64);
            return Ok(());
        }

        let old_end = old_offset.checked_add(len).context("Old graph zone end overflow")?;
        let new_end = new_offset.checked_add(len).context("New graph zone end overflow")?;

        self.ensure_capacity(old_end.max(new_end))?;
        self.mmap.copy_within(old_offset..old_end, new_offset);
        self.set_graph_offset(new_offset as u64);

        let new_file_len = Self::page_align(new_end);
        self.file.set_len(new_file_len as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    /// Truncate the logical count of vectors to handle ghost node recovery.
    ///
    /// This method is used during index opening to recover from crashes where
    /// vectors were written to storage but not indexed in the graph. It updates
    /// the in-memory header count without physically truncating the file.
    ///
    /// # Ghost Node Recovery
    ///
    /// When we detect `storage.count() > graph.node_count()`, we have "ghost nodes"
    /// (vectors written but not indexed). We truncate the logical count so the next
    /// `insert()` will reuse the ghost node's space.
    ///
    /// # Arguments
    ///
    /// * `new_count` - The new logical vector count (must be ≤ current count)
    ///
    /// # Safety
    ///
    /// This does NOT physically truncate the file or free space. Ghost vectors
    /// remain on disk but are logically "invisible" until overwritten by the
    /// next insert operations.
    ///
    /// # Example
    ///
    /// ```text
    /// Before crash:
    ///   storage.count() = 3 (vectors 0, 1, 2)
    ///   graph.node_count() = 3 (nodes 0, 1, 2)
    ///
    /// After crash (vector 3 written but not indexed):
    ///   storage.count() = 4 (vectors 0, 1, 2, 3)
    ///   graph.node_count() = 3 (nodes 0, 1, 2)
    ///
    /// After truncate_logical(3):
    ///   storage.count() = 3 (vector 3 is now a ghost)
    ///   Next insert() will reuse ID 3
    /// ```
    pub(crate) fn truncate_logical(&mut self, new_count: u64) {
        let current_count = self.header().count;

        debug_assert!(
            new_count <= current_count,
            "Cannot truncate to higher count: current={}, new={}",
            current_count,
            new_count
        );

        // Update in-memory header count
        self.header_mut().count = new_count;

        // Note: We don't commit here - this is an in-memory adjustment only.
        // The next insert() will overwrite ghost nodes and then commit atomically.
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        // Explicitly unlock the file (happens automatically, but being explicit)
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_logical() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp_file.path(), 128).unwrap();

        // Insert 5 vectors
        for i in 0..5 {
            let vec = vec![i as f32; 128];
            storage.insert(&vec).unwrap();
        }

        assert_eq!(storage.count(), 5);

        // Truncate to 3 (simulating ghost node recovery)
        storage.truncate_logical(3);

        assert_eq!(storage.count(), 3);

        // Next insert should reuse ID 3
        let vec = vec![99.0; 128];
        let id = storage.insert(&vec).unwrap();

        assert_eq!(id, 3);
        assert_eq!(storage.count(), 4);
    }

    #[test]
    #[should_panic(expected = "Cannot truncate to higher count")]
    fn test_truncate_logical_invalid() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp_file.path(), 128).unwrap();

        storage.insert(&vec![1.0; 128]).unwrap();

        // Try to truncate to higher count (should panic in debug)
        storage.truncate_logical(5);
    }

    #[test]
    fn test_vector_end_and_graph_offset_helpers() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp_file.path(), 128).unwrap();

        assert_eq!(storage.vector_end().unwrap(), HEADER_SIZE);
        assert_eq!(storage.graph_offset(), None);

        storage.insert(&vec![1.0; 128]).unwrap();
        assert_eq!(storage.vector_end().unwrap(), HEADER_SIZE + 128 * 4);

        storage.set_graph_offset(8192);
        assert_eq!(storage.graph_offset(), Some(8192));
    }

    #[test]
    fn test_move_graph_zone_compacts_file() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut storage = Storage::open(temp_file.path(), 128).unwrap();

        let old_offset = 64 * 1024;
        let new_offset = 8 * 1024;
        let graph_bytes = b"HNSW-test-graph";

        storage.ensure_graph_capacity(old_offset + graph_bytes.len()).unwrap();
        storage.graph_zone_mut(old_offset, graph_bytes.len()).unwrap().copy_from_slice(graph_bytes);

        storage.move_graph_zone(old_offset, new_offset, graph_bytes.len()).unwrap();

        assert_eq!(storage.graph_offset(), Some(new_offset as u64));
        assert_eq!(storage.graph_zone(new_offset, graph_bytes.len()).unwrap(), graph_bytes);
        assert_eq!(
            storage.file.metadata().unwrap().len(),
            Storage::page_align(new_offset + graph_bytes.len()) as u64
        );
    }
}
