"""Batch insertion example for Chassis Python bindings.

This script demonstrates:
- Efficient batch insertion of vectors
- Progress tracking
- Memory-efficient processing
- Performance measurement
"""

import time
import numpy as np
from chassis import VectorIndex


def main():
    print("Chassis Python Bindings - Batch Insertion Example")
    print("=" * 60)

    # Configuration
    DIMENSIONS = 768
    NUM_VECTORS = 10_000
    BATCH_SIZE = 1000
    K = 10
    NUM_QUERIES = 100

    print("\nConfiguration:")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Total vectors: {NUM_VECTORS:,}")
    print(f"  Batch size: {BATCH_SIZE:,}")
    print(f"  Search k: {K}")
    print(f"  Search queries: {NUM_QUERIES}")

    print("\nCreating index...")
    with VectorIndex("batch_example.chassis", dimensions=DIMENSIONS) as index:
        print(f"Index created: {index.path}")

        print(f"\nInserting {NUM_VECTORS:,} vectors...")
        start_time = time.time()

        for batch_start in range(0, NUM_VECTORS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_VECTORS)
            batch_size = batch_end - batch_start

            batch = np.random.rand(batch_size, DIMENSIONS).astype(np.float32)

            batch_start_time = time.time()
            for vec in batch:
                index.add(vec)
            batch_time = time.time() - batch_start_time

            progress = (batch_end / NUM_VECTORS) * 100
            vec_per_sec = batch_size / batch_time if batch_time > 0 else 0
            print(
                f"  [{progress:5.1f}%] Inserted {batch_end:,}/{NUM_VECTORS:,} "
                f"({vec_per_sec:,.0f} vec/sec)"
            )

        insert_time = time.time() - start_time

        print("\nInsertion complete:")
        print(f"  Total time: {insert_time:.2f}s")
        print(f"  Average: {NUM_VECTORS / insert_time:,.0f} vectors/sec")
        print(f"  Index size: {len(index):,} vectors")

        print("\nFlushing to disk...")
        flush_start = time.time()
        index.flush()
        flush_time = time.time() - flush_start
        print(f"  Flush time: {flush_time:.2f}s")

        print("\nTesting search performance...")

        # Pre-generate queries to avoid timing RNG
        queries = np.random.rand(NUM_QUERIES, DIMENSIONS).astype(np.float32)

        search_times = []

        for q in queries:
            t0 = time.perf_counter()
            results = index.search(q, k=K)
            t1 = time.perf_counter()

            # Touch results to prevent dead-code elimination
            _ = results[0].id
            search_times.append(t1 - t0)

        avg = sum(search_times) / len(search_times)
        min_t = min(search_times)
        max_t = max(search_times)

        print(f"\nSearch performance ({NUM_QUERIES} queries, k={K}):")
        print(f"  Average: {avg * 1000:.2f} ms")
        print(f"  Min: {min_t * 1000:.2f} ms")
        print(f"  Max: {max_t * 1000:.2f} ms")
        print(f"  Throughput: {1 / avg:,.0f} queries/sec")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
