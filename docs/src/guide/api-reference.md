# API Reference

## Storage Layer

### `Storage`

Low-level vector persistence using memory-mapped files.

#### Opening a Storage File

```rust
use chassis_core::Storage;

// Create or open existing index
let mut storage = Storage::open("vectors.chassis", 768)?;

// File is exclusively locked - prevents concurrent writers
// Returns error if another process has it open for writing
```

**Parameters**:
- `path`: File path (created if doesn't exist)
- `dimensions`: Vector dimensionality (immutable once set)

**Errors**:
- File already locked by another process
- Dimension mismatch (file has different dimensions)
- File corruption (invalid magic bytes or header)

#### Inserting Vectors

```rust
let vector: Vec<f32> = vec![0.5; 768];

// Fast insert (memory-speed writes)
let index = storage.insert(&vector)?;
println!("Inserted at index: {}", index);

// Returns sequential index: 0, 1, 2, ...
// Does NOT guarantee durability until commit()
```

**Performance**: ~7.4 µs per vector (no fsync).

#### Batch Insertion with Durability

```rust
// Insert many vectors
for vector in vectors {
    storage.insert(&vector)?;
}

// Guarantee durability (fsync)
storage.commit()?;
```

**Performance**: ~171K vectors/sec with batching + commit.

**Critical**: `commit()` is expensive (1-50 ms). Batch many inserts before calling.

#### Reading Vectors

```rust
// Zero-copy slice (preferred)
let slice: &[f32] = storage.get_vector_slice(42)?;
let sum: f32 = slice.iter().sum();

// Owned copy
let vector: Vec<f32> = storage.get_vector(42)?;
```

**Performance**:
- `get_vector_slice()`: 72 ns (cached), 15 µs (cold)
- `get_vector()`: Adds allocation overhead

**Safety**: Slice lifetime tied to `&self`. Cannot call `insert()` or `commit()` while slice is alive (enforced by borrow checker).

#### Querying Metadata

```rust
let count = storage.count();  // Number of vectors
let dims = storage.dimensions();  // Vector dimensions
```

---

## HNSW Graph

### `HnswGraph`

Navigable small-world graph for approximate nearest neighbor search.

#### Creating a Graph

```rust
use chassis_core::{HnswGraph, HnswParams, Storage};

let storage = Storage::open("index.chassis", 768)?;

let params = HnswParams {
    max_connections: 16,        // M: edges per node (higher = better recall)
    ef_construction: 200,       // Build quality (higher = better graph)
    ef_search: 50,              // Search quality (higher = better recall)
    ml: 1.0 / 16.0_f32.ln(),   // Layer selection (default)
    max_layers: 16,             // Max graph layers
};

let mut graph = HnswGraph::open(storage, params)?;
```

**Recommendations**:
- `max_connections=16`: Good balance for most use cases
- `ef_construction=200`: Build high-quality graph (construction is one-time)
- `ef_search=50`: Runtime adjustable per query
- `max_layers=16`: Sufficient for billions of vectors

#### Building the Index

```rust
use chassis_core::HnswBuilder;

// Automatic construction from existing vectors
let builder = HnswBuilder::new(params);
let graph = builder.build(storage)?;

// Or manual insertion with custom layer selection
for i in 0..storage.count() {
    let layer = select_layer();  // Your heuristic
    graph.insert(i, layer)?;
}

graph.commit()?;  // Persist graph structure
```

**Performance**: ~16 µs per node insertion (including linking).

#### Manual Node Linking

```rust
// Insert node with explicit neighbors
let node_id = 0;
let layer_count = 2;  // Participates in layers 0 and 1

let neighbors_per_layer = vec![
    vec![],           // Layer 0: no neighbors yet (first node)
    vec![],           // Layer 1: no neighbors yet
];

graph.link_node_bidirectional(node_id, layer_count, &neighbors_per_layer)?;

// Subsequent nodes can link to earlier ones
let neighbors_per_layer = vec![
    vec![0, 1, 2],    // Layer 0: links to nodes 0, 1, 2
    vec![0],          // Layer 1: links to node 0
];
graph.link_node_bidirectional(3, 2, &neighbors_per_layer)?;
```

**Critical Invariants**:
1. `node_id` must equal `graph.node_count()` (sequential IDs)
2. All neighbor IDs must be < `node_id` (no forward references)
3. No self-loops or `INVALID_NODE_ID`

Violations return errors or filter out invalid neighbors.

#### Searching

```rust
let query: Vec<f32> = vec![0.5; 768];
let k = 10;      // Number of nearest neighbors
let ef = 50;     // Search quality parameter

let results = graph.search(&query, k, ef)?;

for result in results {
    println!("ID: {}, Distance: {}", result.id, result.distance);
}
```

**Parameters**:
- `k`: Number of results (≤ ef)
- `ef`: Candidate pool size (higher = better recall, slower search)
  - `ef < k` is silently corrected to `ef = k`
  - Typical range: 32-128

**Performance** (768D, 1000 nodes):
- `ef=8`: 2.8 µs, ~90% recall
- `ef=32`: 5.9 µs, ~95% recall
- `ef=128`: 20 µs, ~99% recall

**Returns**: `Vec<SearchResult>` sorted by distance (ascending).

```rust
pub struct SearchResult {
    pub id: NodeId,      // Vector index in storage
    pub distance: f32,   // Euclidean distance to query
}
```

#### Advanced: Zero-Copy Distance Computation

```rust
// Compute distance without allocating vector
let query = vec![0.5; 768];
let node_id = 42;

let distance = graph.compute_distance_zero_copy(&query, node_id)?;
```

**Use case**: Custom search algorithms or filtering.

#### Advanced: Direct Neighbor Iteration

```rust
// Zero-allocation iteration (preferred in hot paths)
for neighbor_id in graph.neighbors_iter_from_mmap(node_id, layer)? {
    println!("Neighbor: {}", neighbor_id);
}

// Alternative: Read full record (slower)
let record = graph.read_node_record(node_id)?;
let neighbors = record.get_neighbors(layer);
```

**Performance**:
- `neighbors_iter_from_mmap()`: 43 ns
- `read_node_record() + get_neighbors()`: 330 ns

---

## Node Records

### `NodeRecord`

Fixed-size on-disk representation of a graph node.

#### Creating a Record

```rust
use chassis_core::hnsw::{NodeRecord, NodeRecordParams};

let params = NodeRecordParams::new(
    16,  // m: max neighbors per layer (layers > 0)
    32,  // m0: max neighbors at layer 0
    16,  // max_layers
);

let mut record = NodeRecord::new(
    42,  // node_id
    3,   // layer_count (participates in layers 0, 1, 2)
    params,
);
```

#### Setting Neighbors

```rust
// Set layer 0 neighbors
record.set_neighbors(0, &[0, 1, 2, 3, 4]);

// Set layer 1 neighbors
record.set_neighbors(1, &[10, 20]);

// Set layer 2 neighbors
record.set_neighbors(2, &[100]);
```

**Constraints**:
- Layer must be < `layer_count`
- Neighbor count must be ≤ `max_neighbors(layer)`

**Panics**: Invalid layer or too many neighbors (debug and release builds).

#### Reading Neighbors

```rust
// Get all valid neighbors (filters INVALID_NODE_ID)
let neighbors = record.get_neighbors(0);

// Count neighbors
let count = record.neighbor_count(0);

// Zero-allocation iteration
for neighbor in record.neighbors_iter(0) {
    println!("{}", neighbor);
}
```

#### Adding Neighbors

```rust
// Add if space available
let success = record.add_neighbor(0, 99);

if !success {
    println!("Layer 0 is full");
}
```

#### Serialization

```rust
// To bytes
let bytes = record.to_bytes();

// From bytes
let restored = NodeRecord::from_bytes(&bytes, params)?;

assert_eq!(restored.get_neighbors(0), record.get_neighbors(0));
```

**Size**: Determined by `params.record_size()`. Example:
- M=16, M0=32, max_layers=8: 1,312 bytes
- M=16, M0=32, max_layers=16: 2,576 bytes

---

## Distance Metrics

### `euclidean_distance()`

SIMD-accelerated Euclidean distance (L2 norm).

```rust
use chassis_core::distance::euclidean_distance;

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let distance = euclidean_distance(&a, &b);
// sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(27) ≈ 5.196
```

**Features**:
- **AVX2 (x86_64)**: 4-way unrolled FMA, processes 32 floats/iter
- **NEON (aarch64)**: 4-way unrolled, processes 16 floats/iter
- **Scalar fallback**: Portable implementation

**Performance** (768D):
- SIMD: 40 ns (19.1 Gelem/s)
- Scalar: 370 ns (2.07 Gelem/s)
- **Speedup**: 9.2x

**Auto-detection**: Uses `is_x86_feature_detected!("avx2")` at runtime.

### `cosine_distance()`

Cosine distance (1 - cosine similarity).

```rust
use chassis_core::distance::cosine_distance;

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let distance = cosine_distance(&a, &b);
// Orthogonal vectors: distance = 1.0
```

**Note**: Currently no SIMD optimization (uses scalar).

---

## Parameters & Tuning

### `HnswParams`

Graph construction parameters.

```rust
pub struct HnswParams {
    pub max_connections: u16,    // M
    pub ef_construction: usize,  // efConstruction
    pub ef_search: usize,        // efSearch (default)
    pub ml: f32,                 // Layer selection multiplier
    pub max_layers: u8,          // Maximum graph depth
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / 16.0_f32.ln(),  // ≈ 0.36
            max_layers: 16,
        }
    }
}
```

**Tuning Guide**:

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `max_connections` | 8-64 | Higher = better recall, larger index | 16 (default) |
| `ef_construction` | 100-400 | Higher = better graph quality | 200 |
| `ef_search` | 10-500 | Higher = better recall, slower search | 50 |
| `ml` | 0.2-0.5 | Layer distribution | 1/ln(M) |
| `max_layers` | 8-32 | Determines record size | 16 |

### `NodeRecordParams`

Determines fixed record size (immutable once set).

```rust
pub struct NodeRecordParams {
    pub m: u16,          // Max neighbors per layer (layers > 0)
    pub m0: u16,         // Max neighbors at layer 0
    pub max_layers: u8,  // Maximum layers
}

impl Default for NodeRecordParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            max_layers: 16,
        }
    }
}
```

**Record Size Calculation**:

```rust
let size = 16                              // Header
         + (m0 * 8)                        // Layer 0
         + ((max_layers - 1) * m * 8);     // Upper layers
let aligned_size = (size + 7) & !7;        // Round to 8 bytes
```

**Example**: M=16, M0=32, max_layers=16
- Header: 16 bytes
- Layer 0: 32 × 8 = 256 bytes
- Layers 1-15: 15 × 16 × 8 = 1,920 bytes
- **Total**: 2,192 bytes (rounded to 2,192)

---

## Error Handling

All fallible operations return `Result<T, anyhow::Error>`.

### Common Errors

```rust
use anyhow::Result;

// Dimension mismatch
let storage = Storage::open("index.chassis", 768)?;
let vector = vec![0.5; 384];  // Wrong size!
let result = storage.insert(&vector);
// Error: "Vector dimension mismatch: expected 768, got 384"

// File locked
let storage1 = Storage::open("index.chassis", 768)?;
let storage2 = Storage::open("index.chassis", 768);
// Error: "Chassis file is already open by another process"

// Index out of bounds
let vector = storage.get_vector(999999)?;
// Error: "Index out of bounds: 999999 (count is 1000)"

// Node ID invariant violation
graph.insert(5, 0)?;  // Expected node 0
// Error: "Node ID invariant violated: expected 0, got 5"
```

### Handling Errors

```rust
match storage.insert(&vector) {
    Ok(index) => println!("Inserted at {}", index),
    Err(e) => eprintln!("Insert failed: {}", e),
}

// Or with ? operator
let index = storage.insert(&vector)?;
```

---

## Complete Example

```rust
use chassis_core::{Storage, HnswGraph, HnswParams, HnswBuilder};
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Create storage and insert vectors
    let mut storage = Storage::open("embeddings.chassis", 768)?;
    
    for i in 0..1000 {
        let vector: Vec<f32> = (0..768)
            .map(|j| ((i + j) as f32).sin())
            .collect();
        storage.insert(&vector)?;
    }
    storage.commit()?;
    
    // 2. Build HNSW graph
    let params = HnswParams::default();
    let builder = HnswBuilder::new(params);
    let mut graph = builder.build(storage)?;
    graph.commit()?;
    
    // 3. Search
    let query = vec![0.5; 768];
    let results = graph.search(&query, 10, 50)?;
    
    for result in results {
        println!("ID: {}, Distance: {:.4}", result.id, result.distance);
    }
    
    Ok(())
}
```

**Output**:
```
ID: 42, Distance: 0.1234
ID: 17, Distance: 0.1456
ID: 99, Distance: 0.1567
...
```
