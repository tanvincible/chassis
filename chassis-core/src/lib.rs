//! Chassis - Embeddable on-disk vector storage engine
//!
//! Chassis is a local-first vector storage engine designed for embedding-based
//! search in edge devices, mobile apps, and local-first software. It's built
//! in Rust and runs anywhere from a Raspberry Pi to a data center.
//!
//! # Features
//!
//! - On-disk storage using memory-mapped I/O
//! - Page-aligned file format (4KB boundaries)
//! - Single-writer, multi-reader concurrency (SWMR)
//! - Explicit durability control via commit()
//! - Zero external dependencies (no daemons or services)
//!
//! # Example
//!
//! ```no_run
//! use chassis_core::Storage;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Open or create an index
//! let mut storage = Storage::open("embeddings.chassis", 768)?;
//!
//! // Insert vectors
//! let embedding = vec![0.1; 768];
//! let id = storage.insert(&embedding)?;
//!
//! // Commit to disk for durability
//! storage.commit()?;
//!
//! // Retrieve a vector
//! let retrieved = storage.get_vector(id)?;
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

mod distance;
mod header;
mod hnsw;
mod storage;

pub use distance::{cosine_distance, euclidean_distance, DistanceMetric};
pub use header::{Header, HEADER_SIZE, MAGIC, VERSION};
pub use hnsw::{HnswBuilder, HnswGraph, HnswParams, SearchResult};
pub use storage::Storage;
