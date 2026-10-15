# Ideas

A dump of ideas for Chassis that have not been validated. This file lives on the `ideas` branch,
which is not merged.

An idea moves to `ROADMAP.md` on `main` once it has been measured and found worth building. One
that is measured and isn't worth it is recorded where it was measured, under "What Was Tried and
Left Out" in an ADR, and removed from here.

Each entry says what the idea is, what is known, and what would validate it. "Unmeasured" means
nobody has run it; an expectation is only that.

## Cold starts and the operating system's pages

### Ask for a node's neighbors' pages at once

A search knows a node's neighbors before it reads their vectors. On a file not in memory each
vector is then its own blocking disk read; asking the system for all of them first lets the disk
serve them together.

* **Known**, from a prototype on `lab/exp-cold` (99,000 vectors of 1,536 dimensions): the first
  result arrives at 31 ms where it takes 110 on an Apple M5, and at 47 to 68 ms where it takes
  204 to 435 on Linux runners. But on Linux the next hundred queries are about twice as slow,
  because the requests replace the kernel's reading of 128 KB around each fault; on a file
  already in memory the first hundred queries take 20 to 50% longer at 1,536 dimensions and up
  to twice as long at 128; and together with a background read it is slower than the background
  read alone.
* **To validate**: a rule for when to do it that keeps the gain and not the costs, such as only
  for a process's first query or only when the file is found out of memory at open. Measure on a
  laptop's NVMe under Linux, and on an index larger than memory, which is the case it is for.

### Lay neighbors out on the same pages when compacting

Compaction rewrites the file anyway. Ordering slots so that nodes linked in the graph sit near
each other would mean fewer page faults on a cold file and fewer TLB misses on a warm one.

* **Known**: nothing measured. It can only help where several vectors fit in a page: at 128
  dimensions eight do in 4 KB, and twice that in half precision; a full-precision vector of 1,536
  dimensions is a page and a half on its own.
* **Known elsewhere**: on HNSW graphs, Gorder and RCM orderings made queries 10 to 40% faster for
  about a tenth of the build time (Coleman et al., NeurIPS 2022, arXiv 2104.03221); Lucene's
  reordering cut latency 20% (apache/lucene#13683).
* **To validate**: reorder an existing index in the experiment branch, then count faults a query
  on a cold file and measure warm search speed, at 128 and 768 dimensions.

### An index larger than memory

Nothing has been measured with one. Reading only what a query touches is the design's claim
there, and huge pages and `warm()` are both wrong for it.

* **To validate**: build an index several times the size of memory (or cap the page cache), and
  measure search speed, what the system reads per query, and what it keeps resident.

## Doing more at once

### `search_batch` across every core

`search_batch` (PR #55) takes many queries in one call, on one thread. Searches are independent and
read-only, and `add_batch` already has the thread pattern.

* **Known**: in one call on one thread, Python reaches Rust's speed: 0.95 to 1.05 times it, where
  one call per query was 0.68 to 0.79 (2026-10-10). Across cores, near-linear scaling is the
  expectation, unmeasured. It helps bulk jobs (deduplication, evaluation, re-ranking many chunks),
  not one interactive query.
* **To validate**: queries a second against cores, in cache and out of it, where memory
  bandwidth may be the limit before the cores are.

### Scan on every core when a filter passes few vectors

When walking the graph would cost more, a filtered search checks every slot, on one thread.

* **Known**: nothing measured.
* **To validate**: time the scan on a large index with a filter passing a thousandth of it, on
  one thread and on all.

### Build the id lookup on every core

With caller-chosen ids, the table from id to slot is built on first use by one thread reading
every slot.

* **Known**: nothing measured, at any size.
* **To validate**: time it at one and ten million ids; only worth doing if it is long enough to
  notice.

### Write a batch's vectors on every core

`add_batch` writes its vectors on one thread before linking on all of them, and in half
precision converts each component there too.

* **Known**: linking is nearly all of a build's time, so the expectation is that this gives
  little.
* **To validate**: the share of a batch build spent before linking starts.

## Smaller vectors

### One bit a component, re-ranked

Keep one bit a component (RaBitQ) to search with and the half-precision vectors to rank the
final candidates: a sixteenth of half precision in what a search reads most.

* **Known** (lab `quant.yml`, 2026-10-11, stage 1, no graph; plan and results in
  `bench/lab/quant-plan.md` on `lab/x86`): 0.998 recall@10 at 1,536 dimensions with 40
  candidates re-ranked, and 0.981 to 0.991 at 384 with 100. The sign alone fails on vectors
  that aren't centered (SIFT: 0.0001); RaBitQ centers and rotates first. Eight bits, centered,
  needed no re-ranking and is on the roadmap.
* **To validate**: a graph searched by these distances, which was not measured, and whether the
  re-ranking reads, one random page each, cost more than the smaller index saves. A second
  distance in Chassis, and the file gets larger, not smaller.

### Change an index's precision when compacting

The precision is fixed at creation; to change it the vectors are added to a new index.
Compaction already copies every live vector into a new file.

* **Known**: full to half is a rounding; half to full cannot bring back what was rounded away.
* **To validate**: little to measure. It is a question of whether anyone needs it.

### A faster half-precision kernel on ARM servers

On Neoverse-N2 the half-precision kernel takes 1.6 times as long as the `f32` one when nothing
waits for memory, and an index of long vectors that fits in cache searches 19% more slowly.

* **Known**: the kernel widens four components at a time with one instruction per four.
* **To validate**: try widening from wider loads, or SVE where a CPU has it, in the kernel
  benchmark.

## Smaller files without smaller vectors

A vector costs its own bytes plus 160: a 24-byte slot header, an 8-byte head word and 32 level-0
neighbors of 4 bytes. That is 38% of a 128-dimension half-precision file and 5% of a
1,536-dimension one. hnswlib keeps about 140 bytes a vector, pgvector about 288.

### A narrower slot header

The header is an id, a metadata reference and a deleted epoch, 8 bytes each. Nothing uses the
metadata reference yet, and an epoch of 4 bytes lasts 4 billion flushes: 12 bytes a vector, 3% of
a 128-dimension half-precision file.

* **Known**: arithmetic only. It is a format change, worth making only with another one.
* **To validate**: nothing to measure but the size; decide it when metadata is designed.

### Neighbor ids in three bytes

Up to 16.7 million vectors, three bytes name a neighbor: 32 bytes a vector at 32 neighbors.

* **Known**: readers in other processes read lists while the writer changes them, as 4-byte
  words that are each written at once. Three bytes are not, so a reader could see half an id.
* **To validate**: a way to read a list that never sees half a write first (a version per list),
  then its cost in search speed.

### Compressed neighbor lists in a read-only copy

Lucene stores lists as differences between sorted ids, in as few bytes as each needs: its graph
of SIFT at M=16 went from 137 to 60 MB with the same latency (apache/lucene#11860). Qdrant 1.13
cut its graph up to 30%; Meta's lossless id coding takes HNSW links from 32 to about 17 bits
(arXiv 2501.10479).

* **Known**: every one of them does it only to files that never change afterwards. Chassis changes
  lists in place while other processes read them, so it fits only a compacted copy that is opened
  read-only from then on, which Chassis doesn't have. With a reordering (above) the differences,
  and so the lists, get smaller.
* **To validate**: whether anyone wants an index that is built once and then only read.

## Fewer neighbors for long vectors

At 1,536 dimensions, 24 level-0 neighbors instead of 32 searched 1.07 to 1.19 times as fast for
0.90 and 0.95 recall, 0.97 to 1.04 for 0.99, and built 9 to 14% faster (lab `m0.yml`,
2026-10-10, half precision, on Neoverse-N2, a Xeon 8573C and an EPYC 9V74). The longer the
vector, the more each distance skipped saves. M=12 did the same.

* **Known**: one dataset of long vectors, in half precision only. At 128 and 384 dimensions the
  same change was up to 16% slower for 0.99 recall (ADR-0018, "What Was Left Out").
* **To validate**: full precision, 768 and 3,072 dimensions and another dataset; if it holds, a
  default M that depends on the dimensions.

## Builds of long vectors on x86

A batch build of 1,536-dimension vectors in full precision takes 1.4 to 1.7 times as long as
hnswlib's on x86 with four cores, where at 128 dimensions it takes half to nine tenths, and in
half precision it is level.

* **Known**, from a profile on 2026-10-10 (20,000 vectors, EPYC 7763 and 9V74): it is not the
  file. Kernel time is 1% of the build, and building on tmpfs takes as long as on disk. 64% of the
  time is the search for a new node's neighbors and 29% the distances computed while choosing
  which neighbors to keep; one thread spends 1.6 ms a vector where hnswlib spends 1.0 to 1.2. So
  it is the same memory-bound distance work as a query, where full precision trails hnswlib on
  these CPUs at the same `ef`. hnswlib builds 15 to 25% slower without its huge pages.
* **Since**: PR #42 computes a search's distances in groups, and builds of 384 dimensions and up
  take 7 to 17% less time.
* **To validate**: what remains of the gap against hnswlib after #42; then count distances per
  insert in both engines, to tell how much is more distances and how much is slower ones.

## Other engines from a cold start

* **Known**, from 2026-10-10 (every engine through Python on one thread, files verified out of
  memory first, `warm()` as in PR #40): from opening to the hundredth query, on 99,000 vectors of
  1,536 dimensions Chassis took 1.3 to 1.5 s in full precision either way, and 0.57 to 0.61 s in
  half precision with `warm` (0.70 to 0.76 s without); hnswlib 1.37 to 1.41 s, FAISS 1.31 to
  1.43 s, usearch 1.35 to 1.44 s, LanceDB 1.13 to 1.19 s. On a million SIFT vectors: Chassis
  1.38 to 1.41 s with `warm` (1.44 to 2.33 s without), 0.74 to 0.77 s in half precision; hnswlib
  2.01 to 2.17 s, FAISS 1.41 to 1.48 s. Chassis's first result came after 0.2 to 0.5 s;
  hnswlib's, FAISS's and usearch's, which load the file first, after 1.3 to 2.2 s; LanceDB's after
  0.9 to 1.4 s; usearch's memory-mapped view's after 0.2 to 0.8 s, with its hundredth after 1.4 to
  2.2 s. One Zen 4 runner was slow for every engine (FAISS 3.3 s), and there
  `warm` made Chassis slower: 2.8 s to 3.2 s in full precision, and its first result 0.5 s to
  1.2 s.
* **To validate**: a rule for when `warm` should hold back on a slow disk, for instance reading in
  only while searches aren't waiting on it.

## What apps built around

Four agents, given only a copy of the repository, each built an app on Chassis (2026-10-13): a
notes search CLI and an LLM cache served by four worker processes in Python, storage for a desktop
app in Rust, and a "more like this" catalog tool in C. What they tripped on in the docs and errors
was fixed in PR #59; looking up a vector or an id, leaving ids out of a search in C and Python, and
the count of deleted vectors were built in PR #61.

Six more (2026-10-14) built apps people build now: a team's coding-agent memory on
`chassisdb-mcp`, an MCP server over Markdown notes, docs retrieval with LangChain, semantic code
search, a shop's search API on four web workers with an ingester, and a Rust desktop photo
backend. All six met every requirement in 19 to 35 minutes; Chassis was never the slow part
(0.05 to 0.23 ms a search). Their bugs and docs gaps were fixed in PRs #64, #65 and #68; what they
asked for is below, most asked first.

### Replace a vector in one call

* **Known**: four apps wrote `if id in index: delete(id)` then `add(vector, id=id)`, the same three
  lines each. Under one flush that is already atomic.
* **To validate**: nothing to measure; whether `add(..., replace=True)` or an `upsert` in Rust, C and
  Python is worth its surface.

### Advice on compacting, from the index

* **Known**: four apps had to decide when to compact; the docs now say compact once
  `deleted_count()` nears `len()`, from ADR-0011's measurements (half deleted: searches at about 70%
  of their speed). One found that compacting doesn't shrink the file until the live vectors fit in
  fewer segments.
* **To validate**: whether a call that reports bytes on disk and what a compaction would free,
  with a `should_compact`, is used once the rule is documented.

### "More like this id" in one call

* **Known**: two apps (and the C tool before them) did `get(id)` and then searched with the id
  excluded: a vector's round trip through the binding.
* **To validate**: how often it's asked for beyond the docs' two-line recipe.

### Everything within a distance

* **Known**: the photo app grouped burst shots by searching each photo with k = 16 and doubling k
  while every hit was within 0.25: 1 s for 20,000 photos, the same 2,760 groups as a brute-force
  pass over 200 million pairs. The memory app wanted recall to stop at a distance, so its harness
  cut-offs survive a change of model.
* **To validate**: trimming top-k at a `max_distance` against a true range search that widens k
  itself, and the widening's cost in dense clusters.

### The same calls in every binding

* **Known**: Python has no `snapshot()` or `refresh()`, which Rust's reader has; the shop's workers
  wanted to report which flush each saw. Rust has no `search_batch`, which the photo app's
  duplicates pass would have used.
* **To validate**: nothing to measure for the Python half; Rust `search_batch` with the
  across-cores idea above.

### A cap on a build's threads

* **Known**: in the photo app, a background `add_batch` on every core while the UI searched gave
  search a 0.3 ms median and a 20 ms p99, on a machine loaded by other jobs; one-at-a-time adds
  brought p99 to 7 ms and imported five times slower.
* **To validate**: UI search latency against build threads on an idle machine, before an option.

### Large allowed sets, cheaper

* **Known**: the shop built its `allowed` ids per request, from SQLite through numpy: about 1.5 ms,
  ten times the filtered search it fed (0.13 to 0.23 ms).
* **Done in part** (ADR-0022, #87): sorted ids are read where they lie, others go into a
  multiply-hash set, and the guide stops passing every id. A reusable set handle and a bitmap over
  slots wait for callers that pass large unsorted filters in a loop.

### A full disk as an error, not a SIGBUS

* **Known** (2026-10-15): a region is added with `set_len`, which leaves the file sparse, so on
  Linux and macOS a full disk shows when a write first touches a page of the new region: SIGBUS
  ends the process instead of an error. ADR-0008 lists "allocating space before writing to it" as
  not built.
* **Options**: reserve each region whole when it is added (`posix_fallocate`, `F_PREALLOCATE`):
  an error instead, but the file then takes its last segment's full size on disk, up to 256 MiB,
  where today it stays sparse. Or reserve in steps a little ahead of the slots written: bounded
  extra space, one call per step. Catching SIGBUS in a library would take over a process-wide
  signal, which is no way to treat the app.
* **To validate**: what a step costs an add and a flush, and how much ahead is enough.

### Smaller ones, asked once each

* **Collections in one file**: the notes server kept sections and whole notes in two files.
* **String keys**: the shop mapped SKUs to integer row ids in SQLite, which the docs advise.
* **A reader threads can share**: Rust's `IndexReader::search` takes `&mut self`, one per thread.
* **A smaller minimum file**: 21 vectors took 2.1 MB, its header and first segment.
* **Error help in each binding's spelling**: "open it with the cosine metric" reads as Rust to a
  Python user, who writes `IndexOptions(metric="cosine")`.
* **The LangChain store**: listing and counting documents (the app used LangChain's record
  manager as a second database), `compact()`, a context manager, and replacing a file's chunks at
  once so readers never see both.
* **The MCP memory**: forgetting by tag or many ids, `update(id, text)`, export and import, a
  `max_distance` on recall, and a hint on remember's `similar` when two memories are one fact.

### Several processes that each write

* **Known**: a second writer fails at once with Locked. The LLM cache's four workers each opened the
  writer for each write, under a lock of their own on a side file: on 100,000 vectors of 384
  dimensions, open took 0.3 ms, an add 10 to 13 and the flush 15 to 17, on an Apple M5. In the
  second round the shop ran one ingester and readers in its four workers, as the docs advise, and
  needed nothing more.
* **To validate**: an open that waits for the lock, with a timeout, against a documented recipe;
  and the per-write cost at a million vectors and on Linux.

### Reopen with the options in the file

* **Known**: a writer reopened without options uses the defaults, so a cosine index fails to open
  until its options are given again. A reader takes the metric and precision from the file but not
  `max_connections`, so in Python a `read_only=True` index with no options fails on an index built
  with 32. Both errors now say what to pass (PR #59). The code-search app asked for it again: every
  writer open restates `metric="cosine"`, and a reader still passes `dimensions`.
* **To validate**: nothing to measure. Taking every option from the file when none are given is an
  API decision; C's `chassis_open_reader` would keep a `max_connections` it no longer needs.

## Where Chassis could plug in

Frameworks and tools whose vector store can be swapped, from a survey (2026-10-14). What was
checked against their code says so; the rest is the survey's.

### A txtai backend

* **Known** (checked): txtai takes a backend by name or by full class path, so a `txtai-chassisdb`
  package needs nothing from upstream; zvec is already a backend. Its ids are positions from 0 that
  continue on append, as Chassis's own are; scores are 1 − distance; txtai keeps the text itself.
  `save(path)` puts the index at a path txtai picks, which wants Python `snapshot()` (above).
* **To validate**: txtai's benchmarks with Chassis against its hnsw and faiss backends, then a PR
  to name `chassis` as a built-in.

### A Rig vector store

* **Known**: Rig is the main Rust LLM framework; its stores are companion crates implementing
  `VectorStoreIndex`, and the embedded ones are LanceDB and SQLite. A search returns each hit's
  document, so like the LangChain store it would keep documents in SQLite (unchecked against the
  current trait). Outside the root workspace, Rig's dependencies and MSRV don't bind the core's.
* **To validate**: the trait as it is now, and whether Rig lists outside crates.

### LlamaIndex

* **Known**: Hnswlib and Faiss are among its local stores. A store with `stores_text = False`
  leaves node text in LlamaIndex's docstore, so Chassis keeps only vectors.
* **To validate**: where LlamaIndex lists outside packages now.

### mem0

* **Known** (checked): its stores are a fixed map from names to classes, so getting in is a PR to
  mem0. Memories are filtered by user and agent: payloads in SQLite and `allowed`, as
  `chassisdb-mcp` does.
* **To validate**: whether mem0 takes new providers.

### A Node binding

* **Known**: the local-first agent tools with the most users are TypeScript and run Python or a
  server for their vectors: claude-mem (SQLite and Chroma, installing uv for it), Roo Code (Qdrant
  only), Kilo Code (LanceDB, which its docs say doesn't work on Intel Macs). zvec has a Node SDK.
  `chassisdb` is free on npm. napi-rs keeps the Rust and npm sides in one folder, with a prebuilt
  binary per platform; it would be a third binding to keep in step with the others.
* **To validate**: ask claude-mem's and Kilo's maintainers whether they'd take it, before building.

### VIBE

* **Known**: ann-benchmarks says it's no longer maintained and points to VIBE.
* **To validate**: run VIBE locally first; submit where Chassis holds up against hnswlib and FAISS.

### One payload store for every integration

* **Known**: the survey suggests a shared "Chassis and SQLite payloads" helper. txtai and
  LlamaIndex keep text themselves; the LangChain store and `chassisdb-mcp` each have their own.
* **To validate**: once mem0 or Rig makes a third copy, whether the copies are the same code, and
  whether a helper fits one file, not a database.

### Where they'd live

* **Known**: `integrations/` already holds `langchain-chassisdb`, and new packages fit there and
  beside `chassis-ffi` without moving anything. Moving `chassis-core`, `chassis-ffi` and
  `chassisdb` into `crates/` and `bindings/` touches CI, releases, wheels, `build.rs`'s path to the
  skill and every doc link, for nothing a user sees. One name everywhere: `txtai-chassisdb`,
  `rig-chassisdb`, `llama-index-vector-stores-chassisdb`, `chassisdb` on npm.
* **To validate**: integrations now release with the core's tag; with several, a tag of their own
  lets each follow its framework's releases.

## Results nobody has explained

* **FAISS's HNSW still searches long vectors faster on x86**: at 1,536 dimensions in full precision,
  every engine given its queries in one call (2026-10-10, `incumbents.yml` RUN 7), FAISS made 1.28
  times Chassis's queries a second for 0.95 recall on Zen 4 and 1.05 on a Xeon 6973P-C; Chassis was
  1.09 times FAISS on Neoverse-N2. In half precision Chassis led on all three, 1.09 to 1.40 times.
  What FAISS still does differently is unknown.
* **Zen 5 with huge pages at 128 dimensions** gained nothing or lost up to 7% (ADR-0016).
* **On a Xeon 6973P-C, hints into L1 beat L2 by 2 to 10%** up to 100,000 vectors (ADR-0014's
  runs): one machine, and nothing at a million.
* **One Zen 3 ran the grouped half-precision kernel at 2.4 times its time** in cache, at 768 and
  1,536 dimensions, against two other jobs on the same CPU (`bytebench.yml`, 2026-10-12, job f).
  The same kernel called twice in a row took its usual time there, and so did the eight-bit
  kernels. On x86 a group of four keeps sixteen sums, all sixteen registers, and spills three to
  the stack inside its loop; whether that is the cause was not looked into.

## Not yet measured at all

* Anything past a million vectors, or a file past about a gigabyte.
* A consumer x86 laptop or desktop. Every x86 figure is from a server CPU.
* Apple silicon out of Low Power Mode, and more than one machine of it.
* Real power loss; it is only simulated.
