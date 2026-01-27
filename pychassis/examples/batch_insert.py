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

    print("\nConfiguration:")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Total vectors: {NUM_VECTORS:,}")
    print(f"  Batch size: {BATCH_SIZE:,}")

    # Create index with context manager
    print("\nCreating index...")
    with VectorIndex("batch_example.chassis", dimensions=DIMENSIONS) as index:
        print(f"Index created: {index.path}")

        # Generate and insert vectors in batches
        print(f"\nInserting {NUM_VECTORS:,} vectors...")
        start_time = time.time()

        for batch_start in range(0, NUM_VECTORS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_VECTORS)
            batch_size = batch_end - batch_start

            # Generate batch of random vectors
            batch = np.random.rand(batch_size, DIMENSIONS).astype(np.float32)

            # Insert each vector in the batch
            batch_start_time = time.time()
            for vec in batch:
                index.add(vec)
            batch_time = time.time() - batch_start_time

            # Progress update
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

        # Flush to disk
        print("\nFlushing to disk...")
        flush_start = time.time()
        index.flush()
        flush_time = time.time() - flush_start
        print(f"  Flush time: {flush_time:.2f}s")

        # Test search performance
        print("\nTesting search performance...")
        num_queries = 100
        search_times = []

        for _ in range(num_queries):
            search_start = time.time()
            search_time = time.time() - search_start

            search_times.append(search_time)

        avg_search_time = sum(search_times) / len(search_times)
        min_search_time = min(search_times)
        max_search_time = max(search_times)

        print(f"\nSearch performance ({num_queries} queries):")
        print(f"  Average: {avg_search_time * 1000:.2f}ms")
        print(f"  Min: {min_search_time * 1000:.2f}ms")
        print(f"  Max: {max_search_time * 1000:.2f}ms")
        print(f"  Throughput: {1 / avg_search_time:,.0f} queries/sec")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
