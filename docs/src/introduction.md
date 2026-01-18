# Introduction

Chassis is an embeddable vector storage engine written in Rust. It stores high-dimensional vectors on disk and provides basic insert and retrieval operations.

## What Chassis Does

Chassis manages a single file that contains vector data. You can open this file, insert vectors, retrieve them by index, and commit changes to ensure they persist across power loss or crashes.

The current implementation provides the storage foundation. Search capabilities are not yet implemented.

## What Chassis Does Not Do

Chassis is not a database server, a cloud service, or a distributed system. It does not provide networking, authentication, or query languages. These are left to the application.

## Current Status

The project is in early development. The file format and API are not stable and will change. This documentation describes the current implementation as it exists today.

## Requirements

- Rust 1.85 or later
- A filesystem that supports memory mapping (Linux, macOS, Windows)

## License

Chassis is dual-licensed under MIT and Apache 2.0. You may use either license.