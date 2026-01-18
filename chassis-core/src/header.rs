use std::mem;

/// Magic bytes identifying a Chassis index file
pub const MAGIC: &[u8; 8] = b"CHASSIS\0";

/// Current file format version
pub const VERSION: u32 = 1;

/// Maximum supported vector dimensions.
/// This is a sanity check to catch corrupted headers.
/// Typical embeddings are 384-1536 dimensions.
const MAX_DIMENSIONS: u32 = 4096;

/// Header structure for Chassis index files.
/// The header is always 4096 bytes (one page) to ensure proper alignment.
#[repr(C, align(4096))]
pub struct Header {
    /// Magic bytes for file type identification
    pub magic: [u8; 8],
    
    /// File format version
    pub version: u32,
    
    /// Number of dimensions per vector
    pub dimensions: u32,
    
    /// Number of vectors currently stored
    pub count: u64,
    
    /// Reserved space for future use (padding to 4096 bytes)
    pub reserved: [u8; 4072],
}

/// Size of the header in bytes (always one 4KB page)
pub const HEADER_SIZE: usize = mem::size_of::<Header>();

impl Header {
    /// Creates a new header with the specified dimensions
    pub fn new(dimensions: u32) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            dimensions,
            count: 0,
            reserved: [0; 4072],
        }
    }

    /// Validates the header for correctness and compatibility
    pub fn is_valid(&self) -> bool {
        self.magic == *MAGIC
            && self.version > 0
            && self.version <= VERSION
            && self.dimensions > 0
            && self.dimensions <= MAX_DIMENSIONS
    }

    /// Returns the header as a byte slice for writing to disk
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                HEADER_SIZE,
            )
        }
    }

    /// Copies data from another header
    pub fn copy_from(&mut self, other: &Header) {
        self.magic = other.magic;
        self.version = other.version;
        self.dimensions = other.dimensions;
        self.count = other.count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(HEADER_SIZE, 4096, "Header must be exactly one page (4KB)");
    }

    #[test]
    fn test_header_alignment() {
        let header = Header::new(768);
        let addr = &header as *const _ as usize;
        assert_eq!(addr % 4096, 0, "Header must be 4KB aligned");
    }

    #[test]
    fn test_new_header_is_valid() {
        let header = Header::new(768);
        assert!(header.is_valid());
        assert_eq!(header.dimensions, 768);
        assert_eq!(header.count, 0);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut header = Header::new(768);
        header.dimensions = 0;
        assert!(!header.is_valid());

        header.dimensions = MAX_DIMENSIONS + 1;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_invalid_magic() {
        let mut header = Header::new(768);
        header.magic = *b"INVALID\0";
        assert!(!header.is_valid());
    }
}