"""Basic usage example for Chassis Python bindings.

This script demonstrates:
- Creating an index
- Adding vectors
- Searching for nearest neighbors
- Flushing to disk
"""

import numpy as np
from chassis import VectorIndex


def main():
    print("Chassis Python Bindings - Basic Usage Example")
    print("=" * 50)

    # Create a 128-dimensional index
    print("\n1. Creating index...")
    index = VectorIndex("basic_example.chassis", dimensions=128)
    print(f"   Created index with {index.dimensions} dimensions")

    # Add some vectors
    print("\n2. Adding vectors...")
    np.random.seed(42)  # For reproducibility

    for i in range(20):
        # Generate a random vector
        vector = np.random.rand(128).astype(np.float32)

        # Add to index
        vector_id = index.add(vector)

        if i % 5 == 0:
            print(f"   Added vector {vector_id}")

    print(f"   Total vectors in index: {len(index)}")

    # Flush to disk
    print("\n3. Flushing to disk...")
    index.flush()
    print("   Flush complete")

    # Search for nearest neighbors
    print("\n4. Searching for nearest neighbors...")
    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, k=5)

    print(f"   Found {len(results)} neighbors:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. ID: {result.id:2d}, Distance: {result.distance:.6f}")

    # Index statistics
    print("\n5. Index statistics:")
    print(f"   Path: {index.path}")
    print(f"   Dimensions: {index.dimensions}")
    print(f"   Total vectors: {len(index)}")
    print(f"   Empty: {index.is_empty()}")

    # Close the index
    index.close()
    print("\n6. Index closed successfully")

    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
