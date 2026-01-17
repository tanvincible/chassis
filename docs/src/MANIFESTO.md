# Chassis

Chassis is built around a simple observation: most software does not need a service to perform similarity search. It needs reliable local state.

Vector search has largely been introduced through cloud systems and networked databases. This has shaped assumptions about how it must be deployed and operated. Chassis takes a different approach. It treats vector search as a storage and systems problem, not as an infrastructure product.

Chassis is designed to be embedded directly into applications. It runs in process, stores its data on disk, and relies on the operating system for memory management and caching. There is no server to run and no external system to depend on.

The intent is to make vector search a component rather than a destination.

## Approach

Chassis assumes that data should live on disk first and that memory should be treated as a cache rather than a requirement. Reads are performed through memory mapping to avoid unnecessary copying and to allow the operating system to manage working sets naturally.

Search performance is shaped by physical constraints such as disk I O behavior, cache locality, and CPU execution characteristics. Optimizations focus on predictable behavior under constrained memory rather than peak performance on specialized hardware.

Writes are designed to be crash safe and consistent. A write is either fully committed or not visible at all. Partial state is avoided.

## Scope

Chassis assumes a single process owns the data at a time. It is designed for single-writer concurrency, relying on the host application or file-system level locking to coordinate access.

Chassis provides a minimal set of capabilities needed for persistent vector search. It intentionally avoids higher level concerns such as networking, query languages, authentication, or distributed coordination.

These responsibilities are left to the host application, where they can be addressed in a context appropriate way.

## Stability

**Data created by Chassis today must remain accessible by Chassis in the future.**

Chassis is intended to be depended on. This places a higher value on correctness, clarity, and conservative change than on rapid feature expansion.

The storage format and public interfaces are treated as long lived commitments. Evolution is expected, but it is approached cautiously and with attention to existing data.

## Open Source

Chassis is open source and permissively licensed so it can be embedded into a wide range of systems. The goal is to make the engine easy to adopt, inspect, and maintain over time.

Chassis is not designed to be a platform or a service. It is meant to become part of other software and largely disappear from view.
