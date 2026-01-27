"""Tests for VectorIndex class."""

import numpy as np
import pytest
from pathlib import Path

from chassis import VectorIndex, IndexOptions, SearchResult
from chassis.exceptions import (
    ChassisError,
    DimensionMismatchError,
)


@pytest.fixture
def temp_index_path(tmp_path):
    """Create a temporary index path."""
    return tmp_path / "test.chassis"


@pytest.fixture
def simple_index(temp_index_path):
    """Create a simple 3D index for testing."""
    return VectorIndex(temp_index_path, dimensions=3)


class TestVectorIndexBasics:
    """Test basic VectorIndex functionality."""

    def test_create_index(self, temp_index_path):
        """Test creating a new index."""
        index = VectorIndex(temp_index_path, dimensions=128)
        assert index.dimensions == 128
        assert len(index) == 0
        assert index.is_empty()
        index.close()

    def test_context_manager(self, temp_index_path):
        """Test using index as context manager."""
        with VectorIndex(temp_index_path, dimensions=64) as index:
            assert index.dimensions == 64
        # Index should be closed after context

    def test_add_single_vector(self, simple_index):
        """Test adding a single vector."""
        vec = [0.1, 0.2, 0.3]
        vector_id = simple_index.add(vec)
        assert vector_id == 0
        assert len(simple_index) == 1
        assert not simple_index.is_empty()

    def test_add_multiple_vectors(self, simple_index):
        """Test adding multiple vectors."""
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        ids = []
        for vec in vectors:
            vector_id = simple_index.add(vec)
            ids.append(vector_id)

        assert ids == [0, 1, 2]
        assert len(simple_index) == 3

    def test_add_numpy_array(self, simple_index):
        """Test adding NumPy arrays."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector_id = simple_index.add(vec)
        assert vector_id == 0

    def test_add_numpy_array_wrong_dtype(self, simple_index):
        """Test adding NumPy array with wrong dtype (should auto-convert)."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        vector_id = simple_index.add(vec)
        assert vector_id == 0

    def test_flush(self, simple_index):
        """Test flushing changes to disk."""
        simple_index.add([0.1, 0.2, 0.3])
        simple_index.flush()  # Should not raise


class TestVectorIndexSearch:
    """Test search functionality."""

    def test_search_empty_index(self, simple_index):
        """Test searching an empty index."""
        query = [0.1, 0.2, 0.3]
        results = simple_index.search(query, k=10)
        assert results == []

    def test_search_single_result(self, simple_index):
        """Test searching with one vector in index."""
        vec = [0.1, 0.2, 0.3]
        simple_index.add(vec)

        results = simple_index.search(vec, k=10)
        assert len(results) == 1
        assert results[0].id == 0
        assert results[0].distance < 1e-6  # Should be very close to 0

    def test_search_multiple_results(self, simple_index):
        """Test searching with multiple vectors."""
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        for vec in vectors:
            simple_index.add(vec)

        # Search for vector closest to [1, 0, 0]
        query = [0.9, 0.1, 0.1]
        results = simple_index.search(query, k=3)

        assert len(results) == 3
        assert results[0].id == 0  # [1, 0, 0] should be closest
        assert all(isinstance(r, SearchResult) for r in results)

        # Results should be sorted by distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_search_k_parameter(self, simple_index):
        """Test k parameter limits results."""
        for i in range(10):
            simple_index.add([float(i), 0.0, 0.0])

        results_5 = simple_index.search([5.0, 0.0, 0.0], k=5)
        results_3 = simple_index.search([5.0, 0.0, 0.0], k=3)

        assert len(results_5) == 5
        assert len(results_3) == 3

    def test_search_numpy_query(self, simple_index):
        """Test searching with NumPy query."""
        simple_index.add([1.0, 0.0, 0.0])

        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = simple_index.search(query, k=1)

        assert len(results) == 1
        assert results[0].id == 0


class TestVectorIndexErrors:
    """Test error handling."""

    def test_dimension_mismatch_add(self, simple_index):
        """Test adding vector with wrong dimensions."""
        with pytest.raises(DimensionMismatchError):
            simple_index.add([0.1, 0.2])  # Only 2D, expects 3D

    def test_dimension_mismatch_search(self, simple_index):
        """Test searching with wrong dimensions."""
        simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(DimensionMismatchError):
            simple_index.search([0.1, 0.2], k=1)  # Only 2D

    def test_closed_index_operations(self, simple_index):
        """Test operations on closed index."""
        simple_index.close()

        with pytest.raises(ChassisError, match="closed"):
            simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(ChassisError, match="closed"):
            simple_index.search([0.1, 0.2, 0.3], k=1)

        with pytest.raises(ChassisError, match="closed"):
            len(simple_index)

    def test_invalid_k(self, simple_index):
        """Test search with invalid k."""
        simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(ValueError):
            simple_index.search([0.1, 0.2, 0.3], k=0)

        with pytest.raises(ValueError):
            simple_index.search([0.1, 0.2, 0.3], k=-1)


class TestIndexOptions:
    """Test IndexOptions configuration."""

    def test_default_options(self, temp_index_path):
        """Test creating index with default options."""
        index = VectorIndex(temp_index_path, dimensions=128)
        assert index.options.max_connections == 16
        assert index.options.ef_construction == 200
        assert index.options.ef_search == 50

    def test_custom_options(self, temp_index_path):
        """Test creating index with custom options."""
        options = IndexOptions(
            max_connections=32,
            ef_construction=400,
            ef_search=100,
        )

        index = VectorIndex(temp_index_path, dimensions=128, options=options)
        assert index.options.max_connections == 32
        assert index.options.ef_construction == 400
        assert index.options.ef_search == 100

    def test_invalid_options(self):
        """Test validation of invalid options."""
        # max_connections too large
        options = IndexOptions(max_connections=100000)
        with pytest.raises(ValueError):
            options.validate()

        # ef_construction too small
        options = IndexOptions(ef_construction=0)
        with pytest.raises(ValueError):
            options.validate()

        # ef_search too small
        options = IndexOptions(ef_search=-1)
        with pytest.raises(ValueError):
            options.validate()


class TestIndexPersistence:
    """Test index persistence across sessions."""

    def test_reopen_index(self, temp_index_path):
        """Test reopening an index."""
        # Create and populate index
        with VectorIndex(temp_index_path, dimensions=3) as index:
            index.add([1.0, 0.0, 0.0])
            index.add([0.0, 1.0, 0.0])
            index.flush()

        # Reopen and verify
        with VectorIndex(temp_index_path, dimensions=3) as index:
            assert len(index) == 2
            assert index.dimensions == 3

            results = index.search([0.9, 0.1, 0.0], k=1)
            assert len(results) == 1
            assert results[0].id == 0

    def test_dimension_mismatch_reopen(self, temp_index_path):
        """Test reopening with wrong dimensions."""
        # Create 3D index
        with VectorIndex(temp_index_path, dimensions=3) as index:
            index.add([1.0, 0.0, 0.0])
            index.flush()

        # Try to reopen as 5D (should fail)
        with pytest.raises(DimensionMismatchError):
            VectorIndex(temp_index_path, dimensions=5)


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating SearchResult."""
        result = SearchResult(id=42, distance=1.5)
        assert result.id == 42
        assert result.distance == 1.5

    def test_search_result_repr(self):
        """Test SearchResult string representation."""
        result = SearchResult(id=10, distance=2.345678)
        repr_str = repr(result)
        assert "10" in repr_str
        assert "2.345678" in repr_str


class TestVectorIndexProperties:
    """Test VectorIndex properties and methods."""

    def test_len(self, simple_index):
        """Test __len__ method."""
        assert len(simple_index) == 0

        simple_index.add([0.1, 0.2, 0.3])
        assert len(simple_index) == 1

        simple_index.add([0.4, 0.5, 0.6])
        assert len(simple_index) == 2

    def test_is_empty(self, simple_index):
        """Test is_empty method."""
        assert simple_index.is_empty()

        simple_index.add([0.1, 0.2, 0.3])
        assert not simple_index.is_empty()

    def test_dimensions_property(self, simple_index):
        """Test dimensions property."""
        assert simple_index.dimensions == 3

    def test_path_property(self, simple_index):
        """Test path property."""
        assert isinstance(simple_index.path, Path)
        assert simple_index.path.name == "test.chassis"

    def test_options_property(self, simple_index):
        """Test options property."""
        assert isinstance(simple_index.options, IndexOptions)
        assert simple_index.options.max_connections == 16

    def test_repr(self, simple_index):
        """Test __repr__ method."""
        repr_str = repr(simple_index)
        assert "VectorIndex" in repr_str
        assert "dimensions=3" in repr_str
        assert "test.chassis" in repr_str


class TestBatchOperations:
    """Test batch operations and performance patterns."""

    def test_batch_insert_numpy(self, temp_index_path):
        """Test batch inserting NumPy arrays."""
        index = VectorIndex(temp_index_path, dimensions=128)

        # Generate 100 random vectors
        vectors = np.random.rand(100, 128).astype(np.float32)

        ids = []
        for vec in vectors:
            vector_id = index.add(vec)
            ids.append(vector_id)

        assert len(ids) == 100
        assert ids == list(range(100))
        assert len(index) == 100

        index.flush()

    def test_batch_search(self, temp_index_path):
        """Test batch searching."""
        index = VectorIndex(temp_index_path, dimensions=64)

        # Add vectors
        for i in range(50):
            vec = np.random.rand(64).astype(np.float32)
            index.add(vec)

        # Batch search
        queries = [np.random.rand(64).astype(np.float32) for _ in range(10)]
        all_results = []

        for query in queries:
            results = index.search(query, k=5)
            all_results.append(results)

        assert len(all_results) == 10
        assert all(len(results) <= 5 for results in all_results)
