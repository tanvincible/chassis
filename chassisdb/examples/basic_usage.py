"""Create an index, add vectors, flush them and search."""

import numpy as np

from chassisdb import VectorIndex


def main():
    print("Chassis Python Bindings - Basic Usage Example")
    print("=" * 50)

    print("\n1. Creating index...")
    index = VectorIndex("basic_example.chassis", dimensions=128)
    print(f"   Created index with {index.dimensions} dimensions")

    print("\n2. Adding vectors...")
    np.random.seed(42)

    for i in range(20):
        vector = np.random.rand(128).astype(np.float32)

        vector_id = index.add(vector)

        if i % 5 == 0:
            print(f"   Added vector {vector_id}")

    print(f"   Total vectors in index: {len(index)}")

    print("\n3. Flushing to disk...")
    index.flush()
    print("   Flush complete")

    print("\n4. Searching for nearest neighbors...")
    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, k=5)

    print(f"   Found {len(results)} neighbors:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. ID: {result.id:2d}, Distance: {result.distance:.6f}")

    print("\n5. Index statistics:")
    print(f"   Path: {index.path}")
    print(f"   Dimensions: {index.dimensions}")
    print(f"   Total vectors: {len(index)}")
    print(f"   Empty: {index.is_empty()}")

    index.close()
    print("\n6. Index closed successfully")

    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
