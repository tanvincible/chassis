# Basic usage

This page shows the three things you do with an index from Rust: open it, add vectors and keep
them, and search them. [Getting started](./getting-started.md) has the setup, and the
[Rust API](./api-reference.md) has every method.

## Example

```rust
use chassisdb::{IndexOptions, VectorIndex};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Stands for your embedding model: here, each word counts in a dimension picked by its hash.
fn embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; 1536];
    for word in text.split_whitespace() {
        let mut hasher = DefaultHasher::new();
        word.to_lowercase().hash(&mut hasher);
        vector[hasher.finish() as usize % 1536] += 1.0;
    }
    vector
}

fn main() -> chassisdb::Result<()> {
    let options = IndexOptions {
        max_connections: Some(16),
        ef_construction: 200,
        ef_search: 50,
        ..IndexOptions::default()
    };
    let mut index = VectorIndex::open("data.chassis", 1536, options)?;

    let id = index.add(&embed("The cat sat on the mat"))?;
    index.add_with_id(1000, &embed("A dog dug in the garden"))?;
    index.flush()?;
    println!("stored under ids {id} and 1000");

    for result in index.search(&embed("the cat on the mat"), 5)? {
        println!("id {}, distance {}", result.id, result.distance);
    }
    Ok(())
}
```

## Open

`open` creates the file if it doesn't exist. The dimensions, metric, precision and
`max_connections` are fixed when the index is created; opening it again takes each from the file
unless you give it, and checks any you do. Dimensions of 0 open an existing index with its own. The
options above are the defaults.

## Add

Every vector has an id, a `u64`. `add` gives it the next one. To keep a vector under an id your
application already has, such as a database row's, use `add_with_id`, and keep the rest of the
row where it already lives.

A search finds a vector as soon as it is added, but the vector is on disk only after `flush()`.
Dropping the index without a flush discards everything added since the last one, as a crash
would.

## Search

`search` returns the `k` nearest vectors, nearest first, each with its id and distance. Smaller is
nearer.

`search` takes `&self`, so many threads can search one `VectorIndex` at once. Wrap it in an
`Arc<RwLock<VectorIndex>>`: searches take the read lock, and `add`, `delete` and `flush`, which
take `&mut self`, take the write lock. To search while another thread or process writes, open an
`IndexReader` on the same file. It sees what the writer's last `flush()` committed.
