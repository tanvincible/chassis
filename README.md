# Chassis

Chassis is an embeddable, on disk vector engine written in Rust.

It is designed to be used as a local storage component for vector similarity search. Chassis runs in process, stores data on disk, and does not require a server or external dependencies.

The project is early stage and focused on establishing a stable core.

## Features

- On disk storage for high dimensional vectors
- Approximate nearest neighbor search
- Memory mapped read paths
- SIMD accelerated distance computation where available
- Optional vector compression using product quantization

## Non Goals

Chassis does not aim to be:

- A database server
- A cloud service
- A distributed system
- A query engine

These concerns are intentionally left to the application.

## Status

Chassis is under active development.

The API and file format are not yet stable and may change. Performance characteristics and supported features will evolve as the core design solidifies.

## License

Chassis is dual licensed under:

- Apache License 2.0
- MIT License

You may use either license at your option.

## Contributing

Contributions and design discussion are welcome.  
The project currently prioritizes correctness, simplicity, and clear invariants over feature breadth.
