# File Format

A Chassis file consists of a header followed by raw vector data.

## Header Structure

The header is exactly 4096 bytes and contains the following fields:

| Offset | Size | Field       | Description                          |
|--------|------|-------------|--------------------------------------|
| 0      | 8    | Magic       | `CHASSIS\0` (identifies file type)   |
| 8      | 4    | Version     | File format version (currently 1)    |
| 12     | 4    | Dimensions  | Number of dimensions per vector      |
| 16     | 8    | Count       | Number of vectors currently stored   |
| 24     | 4072 | Reserved    | Padding to reach 4096 bytes          |

All integers are stored in native byte order. The file is not portable across different endianness.

## Vector Data

Vectors are stored sequentially after the header. Each vector is an array of `f32` values in native byte order.

For a file with dimension `d`, each vector occupies `d * 4` bytes (since `f32` is 4 bytes).

The vector at index `i` is located at offset:
```
HEADER_SIZE + (i * dimensions * 4)
```

There is no padding between vectors. They are densely packed.

## Alignment

The header is 4096 bytes, ensuring vector data begins on a page boundary.

The file size is always a multiple of 4096 bytes. When the file grows, it grows by whole pages. This may leave unused space at the end of the file.

## Validation

On open, Chassis checks:

- The magic bytes match `CHASSIS\0`
- The version is greater than 0 and less than or equal to the current version
- The dimensions are greater than 0 and less than or equal to 4096
- The file size is at least `HEADER_SIZE` bytes

If any check fails, the file is considered corrupted and `Storage::open` returns an error.

## Future Changes

The file format is not stable. Breaking changes may occur before version 1.0. When the format stabilizes, the version field will be used to detect incompatible files.
