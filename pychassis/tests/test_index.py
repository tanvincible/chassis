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


class TestIdsAndDelete:
    """Caller-chosen ids and deletes."""

    def test_add_with_id(self, simple_index):
        assert simple_index.add([1.0, 0.0, 0.0], id=500) == 500
        assert simple_index.search([1.0, 0.0, 0.0], k=1)[0].id == 500
        assert simple_index.add([0.0, 1.0, 0.0]) == 501

    def test_duplicate_id_raises(self, simple_index):
        simple_index.add([1.0, 0.0, 0.0], id=7)
        with pytest.raises(ChassisError):
            simple_index.add([0.0, 1.0, 0.0], id=7)

    def test_out_of_range_id_raises(self, simple_index):
        with pytest.raises(ValueError):
            simple_index.add([1.0, 0.0, 0.0], id=-1)
        with pytest.raises(ValueError):
            simple_index.delete(2**64 - 1)

    def test_delete(self, simple_index):
        for i in range(5):
            simple_index.add([float(i), 0.0, 0.0])
        assert simple_index.delete(2) is True
        assert simple_index.delete(2) is False
        assert len(simple_index) == 4
        results = simple_index.search([2.0, 0.0, 0.0], k=5)
        assert all(r.id != 2 for r in results)

    def test_delete_persists_after_flush(self, temp_index_path):
        with VectorIndex(temp_index_path, dimensions=3) as index:
            index.add([1.0, 0.0, 0.0], id=10)
            index.add([0.0, 1.0, 0.0], id=20)
            index.delete(10)
            index.flush()
        with VectorIndex(temp_index_path, dimensions=3) as index:
            assert len(index) == 1
            assert [r.id for r in index.search([1.0, 0.0, 0.0], k=2)] == [20]


class TestReadOnly:
    """An index opened read-only next to a writer."""

    def test_reader_sees_flushed_writes(self, tmp_path):
        path = tmp_path / "shared.chassis"
        writer = VectorIndex(path, dimensions=3)
        writer.add([1.0, 0.0, 0.0], id=10)
        writer.flush()

        reader = VectorIndex(path, dimensions=3, read_only=True)
        assert reader.search([1.0, 0.0, 0.0], k=1)[0].id == 10
        with pytest.raises(ChassisError, match="read-only"):
            reader.add([0.0, 1.0, 0.0])

        writer.add([0.0, 1.0, 0.0], id=11)
        assert len(reader) == 1
        writer.flush()
        assert len(reader) == 2
        assert reader.search([0.0, 1.0, 0.0], k=1)[0].id == 11
        reader.close()
        writer.close()

    def test_reader_needs_an_existing_index(self, tmp_path):
        with pytest.raises(ChassisError):
            VectorIndex(tmp_path / "missing.chassis", dimensions=3, read_only=True)
        assert not (tmp_path / "missing.chassis").exists()


class TestCosine:
    """Cosine distance."""

    def test_cosine_distances_ignore_length(self, tmp_path):
        index = VectorIndex(tmp_path / "cos.chassis", dimensions=2, options=IndexOptions(metric="cosine"))
        index.add([3.0, 0.0], id=1)
        index.add([0.0, 0.5], id=2)
        results = index.search([1.0, 1.0], k=2)
        assert {r.id for r in results} == {1, 2}
        assert all(abs(r.distance - (1 - 0.5**0.5)) < 1e-5 for r in results)
        with pytest.raises(ChassisError):
            index.add([0.0, 0.0])
        index.close()

    def test_huge_pages_on_request_change_no_result(self, tmp_path):
        path = tmp_path / "huge.chassis"
        index = VectorIndex(path, dimensions=4, options=IndexOptions(huge_pages=True))
        for i in range(50):
            index.add([float(i), 1.0, 2.0, 3.0])
        index.flush()
        reader = VectorIndex(path, dimensions=4, options=IndexOptions(huge_pages=True), read_only=True)
        plain = VectorIndex(path, dimensions=4, read_only=True)
        query = [17.2, 1.0, 2.0, 3.0]
        assert [r.id for r in index.search(query, k=3)] == [17, 18, 16]
        assert reader.search(query, k=3) == plain.search(query, k=3) == index.search(query, k=3)
        for handle in (reader, plain, index):
            handle.close()

    def test_unknown_metric_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            VectorIndex(tmp_path / "x.chassis", dimensions=2, options=IndexOptions(metric="dot"))


class TestMetric:
    """Which metric an index uses, and refusing another."""

    def test_metric_is_reported_and_enforced(self, tmp_path):
        path = tmp_path / "m.chassis"
        writer = VectorIndex(path, dimensions=2, options=IndexOptions(metric="cosine"))
        writer.add([1.0, 0.0])
        writer.flush()
        assert writer.metric == "cosine"
        reader = VectorIndex(path, dimensions=2, read_only=True)
        assert reader.metric == "cosine"
        reader.close()
        with pytest.raises(ChassisError):
            VectorIndex(path, dimensions=2, options=IndexOptions(), read_only=True)
        writer.close()


class TestPrecision:
    """Vectors kept as 16-bit floats."""

    def test_half_precision_is_reported_enforced_and_half_the_size(self, tmp_path):
        vectors = np.random.default_rng(7).random((300, 256), dtype=np.float32)
        sizes = {}
        for precision in ("full", "half"):
            path = tmp_path / f"{precision}.chassis"
            index = VectorIndex(path, dimensions=256, options=IndexOptions(precision=precision))
            index.add_batch(vectors)
            index.flush()
            assert index.precision == precision
            nearest = index.search(vectors[17], k=1)[0]
            assert nearest.id == 17 and nearest.distance < 1e-2
            index.close()
            sizes[precision] = path.stat().st_size
        assert 0.5 < sizes["half"] / sizes["full"] < 0.7

        path = tmp_path / "half.chassis"
        reader = VectorIndex(path, dimensions=256, read_only=True)
        assert reader.precision == "half"
        assert reader.search(vectors[17], k=1)[0].id == 17
        reader.close()
        for options in (None, IndexOptions()):
            with pytest.raises(ChassisError, match="precision"):
                VectorIndex(path, dimensions=256, options=options)

    def test_what_half_precision_cannot_hold_is_refused(self, tmp_path):
        options = IndexOptions(precision="half")
        index = VectorIndex(tmp_path / "half.chassis", dimensions=2, options=options)
        index.add([65504.0, -1.0])
        with pytest.raises(ChassisError, match="too large"):
            index.add([1e6, 0.0])
        assert len(index) == 1
        index.close()

    def test_unknown_precision_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            VectorIndex(tmp_path / "x.chassis", dimensions=2, options=IndexOptions(precision="int8"))


class TestFiltered:
    """Search restricted to allowed ids."""

    def test_only_allowed_ids_come_back(self, tmp_path):
        index = VectorIndex(tmp_path / "f.chassis", dimensions=2)
        for i in range(20):
            index.add([float(i), 0.0], id=100 + i)
        results = index.search([0.0, 0.0], k=4, allowed=[119, 105, 110, 7])
        assert [r.id for r in results] == [105, 110, 119]
        as_array = index.search([0.0, 0.0], k=4, allowed=np.array([110, 105], dtype=np.uint64))
        assert [r.id for r in as_array] == [105, 110]
        assert index.search([0.0, 0.0], k=4, allowed=[]) == []
        assert [r.id for r in index.search([0.0, 0.0], k=4, allowed={110, 105})] == [105, 110]
        for bad in (np.array([-6]), [105.9], ["105"], np.array([[105, 110]]), np.array([True])):
            with pytest.raises((TypeError, ValueError)):
                index.search([0.0, 0.0], k=4, allowed=bad)
        index.close()


class TestAddBatch:
    """Adding many vectors at once."""

    def test_batch_ids_and_search(self, tmp_path):
        index = VectorIndex(tmp_path / "b.chassis", dimensions=4)
        rng = np.random.default_rng(7)
        vectors = rng.random((500, 4), dtype=np.float32)
        ids = index.add_batch(vectors)
        assert list(ids) == list(range(500))
        assert index.search(vectors[123], k=1)[0].id == 123
        assert list(index.add_batch(vectors[:2] + 5)) == [500, 501]
        assert len(index) == 502
        index.close()

    def test_batch_with_ids_is_all_or_nothing(self, tmp_path):
        index = VectorIndex(tmp_path / "b.chassis", dimensions=2)
        index.add_batch([[0.0, 0.0], [1.0, 1.0]], ids=[10, 20])
        assert index.search([1.0, 1.0], k=1)[0].id == 20
        with pytest.raises(ChassisError):
            index.add_batch([[2.0, 2.0], [3.0, 3.0]], ids=[30, 20])
        with pytest.raises(ValueError):
            index.add_batch([[2.0, 2.0]], ids=[-1])
        with pytest.raises(DimensionMismatchError):
            index.add_batch([[2.0, 2.0, 2.0]])
        assert len(index) == 2
        index.close()


class TestCompact:
    """Reclaiming deleted vectors' space."""

    def test_compact_keeps_live_ids_and_shrinks_the_file(self, tmp_path):
        path = tmp_path / "c.chassis"
        index = VectorIndex(path, dimensions=8)
        rng = np.random.default_rng(3)
        vectors = rng.random((3000, 8), dtype=np.float32)
        index.add_batch(vectors)
        # Keep one in three, few enough to need one segment less.
        for i in range(3000):
            if i % 3:
                index.delete(i)
        index.flush()
        before = path.stat().st_size
        index.compact()
        assert len(index) == 1000
        assert path.stat().st_size < before
        assert index.search(vectors[0], k=1)[0].id == 0
        assert index.search(vectors[1], k=1)[0].id != 1
        assert index.add(vectors[1]) == 3000
        index.close()
