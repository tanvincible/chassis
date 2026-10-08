//! Random sequences of every write operation, checked against a model of what the index should
//! hold: adds, batches, deletes, flushes, compactions and reopens without a flush, with a reader
//! following along. `CHASSIS_MODEL_SEEDS=2000 cargo test --release --test model_tests` runs more.
//!
//! A reopen that loses adds must leave the committed graph as the last flush left it (ADR-0012),
//! so every check holds after one too.

use chassis_core::{
    DistanceMetric, IndexOptions, IndexReader, SearchResult, VectorIndex, cosine_distance,
    euclidean_distance,
};
use std::collections::BTreeMap;
use std::path::Path;
use tempfile::tempdir;

const DIMS: usize = 12;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Never the zero vector, which a cosine index rejects.
    fn vector(&mut self) -> Vec<f32> {
        (0..DIMS).map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 + 0.01).collect()
    }
}

/// What the index should hold: live vectors by id, and one past the largest id ever used.
#[derive(Clone, Default)]
struct State {
    live: BTreeMap<u64, Vec<f32>>,
    next_id: u64,
}

impl State {
    fn add(&mut self, id: u64, vector: Vec<f32>) {
        assert!(self.live.insert(id, vector).is_none());
        self.next_id = self.next_id.max(id + 1);
    }
    /// A live id, if there is one.
    fn pick(&self, rng: &mut Rng) -> Option<u64> {
        let nth = rng.below(self.live.len().max(1) as u64) as usize;
        self.live.keys().nth(nth).copied()
    }
}

struct Run {
    seed: u64,
    metric: DistanceMetric,
    step: usize,
    op: &'static str,
}

impl Run {
    fn options(&self) -> IndexOptions {
        IndexOptions { ef_construction: 40, metric: self.metric, ..IndexOptions::default() }
    }

    fn at(&self) -> String {
        format!("seed {} {:?} step {} ({})", self.seed, self.metric, self.step, self.op)
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => cosine_distance(a, b),
            _ => euclidean_distance(a, b),
        }
    }

    /// Every result is a vector `state` holds, at its true distance, nearest first.
    fn check_results(&self, found: &[SearchResult], query: &[f32], state: &State, k: usize) {
        assert!(found.len() <= k, "{}: {} results for k {k}", self.at(), found.len());
        assert!(
            found.windows(2).all(|w| w[0].distance <= w[1].distance),
            "{}: unsorted",
            self.at()
        );
        for (i, r) in found.iter().enumerate() {
            let vector = state
                .live
                .get(&r.id)
                .unwrap_or_else(|| panic!("{}: returned id {}, which isn't live", self.at(), r.id));
            let want = self.distance(query, vector);
            assert!(
                (r.distance - want).abs() < 1e-3,
                "{}: id {} at {} not {want}",
                self.at(),
                r.id,
                r.distance
            );
            assert!(
                found[..i].iter().all(|other| other.id != r.id),
                "{}: id {} twice",
                self.at(),
                r.id
            );
        }
    }

    /// The `k` nearest of the ids `allow` accepts, by brute force.
    fn exact(
        &self,
        query: &[f32],
        state: &State,
        k: usize,
        allow: impl Fn(u64) -> bool,
    ) -> Vec<u64> {
        let mut all: Vec<(f32, u64)> = state
            .live
            .iter()
            .filter(|(id, _)| allow(**id))
            .map(|(&id, v)| (self.distance(query, v), id))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

fn open(path: &Path, run: &Run) -> VectorIndex {
    VectorIndex::open(path, DIMS as u32, run.options()).unwrap()
}

/// One random history. Returns how many self-searches found their vector, of how many.
fn history(seed: u64, metric: DistanceMetric) -> (u64, u64) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.chassis");
    let mut run = Run { seed, metric, step: 0, op: "open" };
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut index = open(&path, &run);
    index.flush().unwrap();
    let mut reader = Some(IndexReader::open(&path, DIMS as u32, run.options()).unwrap());
    // `now` is what the writer holds; `committed` is what the last flush or compaction made
    // durable, which is all a reopen or a reader may show.
    let (mut now, mut committed) = (State::default(), State::default());
    let (mut found_self, mut self_searches) = (0, 0);

    for step in 0..150 {
        run.step = step;
        match rng.below(100) {
            0..=24 => {
                run.op = "add";
                let vector = rng.vector();
                let id = index.add(&vector).unwrap();
                assert_eq!(id, now.next_id, "{}: add reused or skipped an id", run.at());
                now.add(id, vector);
            }
            25..=34 => {
                run.op = "add_batch";
                let vectors: Vec<Vec<f32>> = (0..rng.below(120)).map(|_| rng.vector()).collect();
                let ids = index.add_batch(&vectors.concat()).unwrap();
                let want: Vec<u64> = (now.next_id..now.next_id + vectors.len() as u64).collect();
                assert_eq!(ids, want, "{}", run.at());
                for (id, vector) in ids.into_iter().zip(vectors) {
                    now.add(id, vector);
                }
            }
            35..=42 => {
                run.op = "add_with_id";
                // Often an id that is live or was used before.
                let id = rng.below(now.next_id + 50);
                let vector = rng.vector();
                let added = index.add_with_id(id, &vector);
                assert_eq!(added.is_ok(), !now.live.contains_key(&id), "{}: id {id}", run.at());
                if added.is_ok() {
                    now.add(id, vector);
                }
            }
            43..=47 => {
                run.op = "add_batch_with_ids";
                let count = rng.below(60);
                let first = now.next_id + rng.below(20);
                let mut ids: Vec<u64> = (first..first + count).map(|id| id * 2 - first).collect();
                // One time in three, an id that is live already: the whole batch must be refused.
                let clash = if rng.below(3) == 0 { now.pick(&mut rng) } else { None };
                ids.extend(clash);
                let vectors: Vec<Vec<f32>> = ids.iter().map(|_| rng.vector()).collect();
                let added = index.add_batch_with_ids(&ids, &vectors.concat());
                assert_eq!(added.is_ok(), clash.is_none(), "{}: clash {clash:?}", run.at());
                if added.is_ok() {
                    for (id, vector) in ids.into_iter().zip(vectors) {
                        now.add(id, vector);
                    }
                }
            }
            48..=67 => {
                run.op = "delete";
                // Usually a live id; sometimes one that never was or no longer is.
                let id = match now.pick(&mut rng) {
                    Some(id) if rng.below(5) != 0 => id,
                    _ => rng.below(now.next_id + 5),
                };
                let deleted = index.delete(id).unwrap();
                assert_eq!(deleted, now.live.remove(&id).is_some(), "{}: id {id}", run.at());
            }
            68..=75 => {
                run.op = "flush";
                index.flush().unwrap();
                committed = now.clone();
            }
            76..=79 => {
                run.op = "compact";
                // Windows can't replace a file a reader has open.
                if cfg!(windows) {
                    reader = None;
                }
                index.compact().unwrap();
                committed = now.clone();
            }
            80..=83 => {
                run.op = "reopen";
                // Without a flush: everything since the last commit is gone, as after a crash.
                drop(index);
                index = open(&path, &run);
                now = committed.clone();
            }
            84..=91 => {
                run.op = "search";
                let (query, k) = (rng.vector(), 1 + rng.below(12) as usize);
                run.check_results(&index.search(&query, k).unwrap(), &query, &now, k);
                if let Some(id) = now.pick(&mut rng) {
                    self_searches += 1;
                    let nearest = index.search(&now.live[&id], 1).unwrap();
                    found_self += u64::from(nearest.first().is_some_and(|r| r.id == id));
                }
            }
            92..=95 => {
                run.op = "search_filtered";
                let (query, k) = (rng.vector(), 1 + rng.below(6) as usize);
                // A broad filter, served by the graph or a scan: results must respect it.
                let modulus = 2 + rng.below(3);
                let broad =
                    index.search_filtered(&query, k, |id| id.is_multiple_of(modulus)).unwrap();
                run.check_results(&broad, &query, &now, k);
                assert!(
                    broad.iter().all(|r| r.id.is_multiple_of(modulus)),
                    "{}: filter ignored",
                    run.at()
                );
                // A filter passing at most 3 ids is always scanned, so it must be exact.
                let few: Vec<u64> = (0..3).filter_map(|_| now.pick(&mut rng)).collect();
                let found = index.search_filtered(&query, k, |id| few.contains(&id)).unwrap();
                let ids: Vec<u64> = found.iter().map(|r| r.id).collect();
                let exact = run.exact(&query, &now, k, |id| few.contains(&id));
                assert_eq!(ids, exact, "{}", run.at());
            }
            _ => {
                run.op = "reader";
                let reader = reader.get_or_insert_with(|| {
                    IndexReader::open(&path, DIMS as u32, run.options()).unwrap()
                });
                let (query, k) = (rng.vector(), 1 + rng.below(12) as usize);
                let found = reader.search(&query, k).unwrap();
                run.check_results(&found, &query, &committed, k);
                assert_eq!(reader.len(), committed.live.len() as u64, "{}: reader len", run.at());
            }
        }
        assert_eq!(index.len(), now.live.len() as u64, "{}: len", run.at());
        // A search that returns fewer than it could has lost its way through the graph.
        let reachable = index.search(&rng.vector(), 10).unwrap().len();
        let all = now.live.len().min(10);
        assert_eq!(reachable, all, "{}: a search lost its way", run.at());
    }

    // The end state survives a commit and a reopen, vector for vector, and a compaction after
    // that, which also leaves every vector within reach again.
    index.flush().unwrap();
    drop((index, reader));
    let mut index = open(&path, &run);
    for op in ["final", "final compaction"] {
        run.op = op;
        assert_eq!(index.len(), now.live.len() as u64, "{}", run.at());
        for (&id, vector) in &now.live {
            let exact = index.search_filtered(vector, 1, |other| other == id).unwrap();
            assert_eq!(exact.len(), 1, "{}: id {id} is gone", run.at());
            assert!(exact[0].distance.abs() < 1e-3, "{}: id {id} holds another vector", run.at());
        }
        index.compact().unwrap();
    }
    for vector in now.live.values() {
        let reachable = index.search(vector, 10).unwrap().len();
        assert_eq!(reachable, now.live.len().min(10), "{}: too few reached", run.at());
    }
    (found_self, self_searches)
}

#[test]
fn test_random_histories_match_the_model() {
    let seeds = std::env::var("CHASSIS_MODEL_SEEDS").map_or(12, |s| s.parse().unwrap());
    let (mut found, mut searched) = (0, 0);
    for seed in 1..=seeds {
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
            let (f, s) = history(seed, metric);
            found += f;
            searched += s;
        }
    }
    // HNSW is approximate, so a vector may miss itself now and then, but rarely.
    assert!(found * 100 >= searched * 98, "{found} of {searched} vectors found themselves");
}
