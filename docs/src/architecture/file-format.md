# File Format

A Chassis file is a single memory-mapped file with a fixed header, dense vector
data, and a relocatable HNSW graph zone. The vector and graph zones share one
mmap so vector reads and neighbor iteration remain zero-copy.

## Top-Level Layout

| Region | Offset | Size |
|--------|--------|------|
| Header | `0` | `4096` bytes |
| Vector zone | `HEADER_SIZE` | `count * dimensions * 4` bytes |
| Slack / padding | End of vector zone | Variable, page-aligned |
| Graph header | `graph_offset` from the header metadata | `64` bytes |
| Node records | `graph_offset + 64` | `node_count * record_size` bytes |

The graph zone is placed after the vector zone with allocation slack. If vector
growth would overlap the graph zone, Chassis moves the graph zone farther into
the file, updates `graph_offset`, and remaps the file. The graph remains directly
addressable after relocation.

## Header Structure

The header is exactly 4096 bytes and begins with the stable fields below.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | Magic | `CHASSIS\0` identifies the file type |
| 8 | 4 | Version | File format version, currently `1` |
| 12 | 4 | Dimensions | Number of `f32` dimensions per vector |
| 16 | 8 | Count | Number of logical vectors currently stored |
| 24 | 4072 | Reserved | Extended layout metadata and future padding |

The extended layout metadata stored at the start of `reserved` is:

| Reserved offset | Size | Field | Description |
|-----------------|------|-------|-------------|
| 0 | 8 | Layout magic | `CHLAYOUT` |
| 8 | 4 | Layout version | Current extended layout version |
| 16 | 8 | Graph offset | Byte offset of the graph header |

Files without this extended metadata are treated as legacy files. If a legacy
HNSW graph header is found at the old 1 GiB graph offset, Chassis compacts it
into the dynamic layout on open.

## Vector Zone

Vectors are stored sequentially after the header. Each vector is an array of
`f32` values.

For a file with dimension `d`, each vector occupies `d * 4` bytes. The vector at
index `i` is located at:

```text
HEADER_SIZE + (i * dimensions * 4)
```

There is no padding between vectors.

## Graph Zone

The graph zone starts at `graph_offset` and begins with a 64-byte graph header.

| Offset in graph header | Size | Field | Description |
|------------------------|------|-------|-------------|
| 0 | 4 | Magic | `HNSW` |
| 4 | 4 | Version | Graph format version |
| 8 | 8 | Entry point | Node ID of the current entry point, or `u64::MAX` |
| 16 | 8 | Node count | Number of published graph nodes |
| 24 | 4 | Max layer | Highest layer currently present |
| 28 | 2 | M | Max upper-layer connections |
| 30 | 2 | M0 | Max layer-0 connections |
| 32 | 1 | Max layers | Fixed layer capacity for node records |
| 33 | 31 | Reserved | Future padding |

Node records are fixed-width for O(1) addressing:

```text
node_offset = graph_offset + 64 + (node_id * record_size)
```

For the default parameters (`M = 16`, `M0 = 32`, `max_layers = 16`):

```text
record_size = 16 + (32 * 8) + ((16 - 1) * 16 * 8) = 2192 bytes
```

The fixed record reserves neighbor slots for every configured layer. This keeps
neighbor iteration zero-copy and allocation-free, at the cost of unused slots for
nodes that only participate in lower layers.

## Size Example

For 10,000 vectors with 768 dimensions and default HNSW parameters:

| Region | Approximate size |
|--------|------------------|
| Header | 4 KiB |
| Vectors | `10,000 * 768 * 4` = 30.7 MB |
| Graph header | 64 bytes |
| Node records | `10,000 * 2192` = 21.9 MB |
| Slack / page padding | Small allocation slack and page rounding |

The expected logical size is under 100 MiB. Older files could report a logical
size near 1 GiB because the graph zone was hard-coded to start at byte
`1,073,741,824`, leaving a large unused gap between vectors and graph data.

## Alignment

The header is 4096 bytes, so vector data begins on a page boundary. File growth
is page-aligned to 4096-byte boundaries. Graph offsets are also page-aligned.

## Validation

On open, Chassis checks:

- The main magic bytes match `CHASSIS\0`.
- The main version is greater than 0 and less than or equal to the current version.
- The dimensions are greater than 0 and less than or equal to 4096.
- The file size is at least `HEADER_SIZE` bytes.
- If a graph header exists, its magic, version, and record parameters match the requested index options.

If any check fails, the file is considered corrupted and open returns an error.

## Future Changes

The file format is not stable. Breaking changes may occur before version 1.0.
When the format stabilizes, the version fields will be used to detect
incompatible files.
