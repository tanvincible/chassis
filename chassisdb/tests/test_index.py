from pathlib import Path

import numpy as np
import pytest

from chassisdb import Check, IndexOptions, SearchResult, VectorIndex
from chassisdb.exceptions import (
    ChassisError,
    DimensionMismatchError,
    IndexLockedError,
    InvalidArgumentError,
    OptionsMismatchError,
    ReadOnlyError,
)


@pytest.fixture
def temp_index_path(tmp_path):
    return tmp_path / "test.chassis"


@pytest.fixture
def simple_index(temp_index_path):
    return VectorIndex(temp_index_path, dimensions=3)


class TestVectorIndexBasics:

    def test_create_index(self, temp_index_path):
        index = VectorIndex(temp_index_path, dimensions=128)
        assert index.dimensions == 128
        assert len(index) == 0
        assert index.is_empty()
        index.close()

    def test_context_manager(self, temp_index_path):
        with VectorIndex(temp_index_path, dimensions=64) as index:
            assert index.dimensions == 64

    def test_context_manager_flushes_unless_raising(self, temp_index_path):
        with VectorIndex(temp_index_path, dimensions=2) as index:
            index.add([1, 0])
        with pytest.raises(KeyError):
            with VectorIndex(temp_index_path, dimensions=2) as index:
                index.add([0, 1])
                raise KeyError
        index = VectorIndex(temp_index_path, dimensions=2)
        index.add([0, 1])
        index.close()
        with VectorIndex(temp_index_path, dimensions=2, read_only=True) as reader:
            assert len(reader) == 1

    def test_add_single_vector(self, simple_index):
        vec = [0.1, 0.2, 0.3]
        vector_id = simple_index.add(vec)
        assert vector_id == 0
        assert len(simple_index) == 1
        assert not simple_index.is_empty()

    def test_add_multiple_vectors(self, simple_index):
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
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector_id = simple_index.add(vec)
        assert vector_id == 0

    def test_add_numpy_array_wrong_dtype(self, simple_index):
        """A float64 array is converted, not refused."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        vector_id = simple_index.add(vec)
        assert vector_id == 0

    def test_flush(self, simple_index):
        simple_index.add([0.1, 0.2, 0.3])
        simple_index.flush()


class TestVectorIndexSearch:

    def test_search_empty_index(self, simple_index):
        query = [0.1, 0.2, 0.3]
        results = simple_index.search(query, k=10)
        assert results == []

    def test_search_single_result(self, simple_index):
        vec = [0.1, 0.2, 0.3]
        simple_index.add(vec)

        results = simple_index.search(vec, k=10)
        assert len(results) == 1
        assert results[0].id == 0
        assert results[0].distance < 1e-6

    def test_search_multiple_results(self, simple_index):
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        for vec in vectors:
            simple_index.add(vec)

        query = [0.9, 0.1, 0.1]
        results = simple_index.search(query, k=3)

        assert len(results) == 3
        assert results[0].id == 0  # [1, 0, 0] should be closest
        assert all(isinstance(r, SearchResult) for r in results)

        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_search_k_parameter(self, simple_index):
        for i in range(10):
            simple_index.add([float(i), 0.0, 0.0])

        results_5 = simple_index.search([5.0, 0.0, 0.0], k=5)
        results_3 = simple_index.search([5.0, 0.0, 0.0], k=3)

        assert len(results_5) == 5
        assert len(results_3) == 3

    def test_search_numpy_query(self, simple_index):
        simple_index.add([1.0, 0.0, 0.0])

        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = simple_index.search(query, k=1)

        assert len(results) == 1
        assert results[0].id == 0


class TestVectorIndexErrors:

    def test_dimension_mismatch_add(self, simple_index):
        with pytest.raises(DimensionMismatchError):
            simple_index.add([0.1, 0.2])  # Only 2D, expects 3D

    def test_dimension_mismatch_search(self, simple_index):
        simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(DimensionMismatchError):
            simple_index.search([0.1, 0.2], k=1)  # Only 2D

    def test_closed_index_operations(self, simple_index):
        simple_index.close()

        with pytest.raises(ChassisError, match="closed"):
            simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(ChassisError, match="closed"):
            simple_index.search([0.1, 0.2, 0.3], k=1)

        with pytest.raises(ChassisError, match="closed"):
            len(simple_index)

    def test_invalid_k(self, simple_index):
        simple_index.add([0.1, 0.2, 0.3])

        with pytest.raises(ValueError):
            simple_index.search([0.1, 0.2, 0.3], k=0)

        with pytest.raises(ValueError):
            simple_index.search([0.1, 0.2, 0.3], k=-1)


class TestIndexOptions:

    def test_default_options(self, temp_index_path):
        index = VectorIndex(temp_index_path, dimensions=128)
        assert index.options.max_connections == 16
        assert index.options.ef_construction == 200
        assert index.options.ef_search == 50

    def test_custom_options(self, temp_index_path):
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

    def test_reopen_index(self, temp_index_path):
        with VectorIndex(temp_index_path, dimensions=3) as index:
            index.add([1.0, 0.0, 0.0])
            index.add([0.0, 1.0, 0.0])
            index.flush()

        with VectorIndex(temp_index_path, dimensions=3) as index:
            assert len(index) == 2
            assert index.dimensions == 3

            results = index.search([0.9, 0.1, 0.0], k=1)
            assert len(results) == 1
            assert results[0].id == 0

    def test_dimension_mismatch_reopen(self, temp_index_path):
        with VectorIndex(temp_index_path, dimensions=3) as index:
            index.add([1.0, 0.0, 0.0])
            index.flush()

        with pytest.raises(DimensionMismatchError):
            VectorIndex(temp_index_path, dimensions=5)


class TestSearchResult:

    def test_search_result_creation(self):
        result = SearchResult(id=42, distance=1.5)
        assert result.id == 42
        assert result.distance == 1.5

    def test_search_result_repr(self):
        result = SearchResult(id=10, distance=2.345678)
        repr_str = repr(result)
        assert "10" in repr_str
        assert "2.345678" in repr_str


class TestVectorIndexProperties:

    def test_len(self, simple_index):
        assert len(simple_index) == 0

        simple_index.add([0.1, 0.2, 0.3])
        assert len(simple_index) == 1

        simple_index.add([0.4, 0.5, 0.6])
        assert len(simple_index) == 2

    def test_is_empty(self, simple_index):
        assert simple_index.is_empty()

        simple_index.add([0.1, 0.2, 0.3])
        assert not simple_index.is_empty()

    def test_dimensions_property(self, simple_index):
        assert simple_index.dimensions == 3

    def test_path_property(self, simple_index):
        assert isinstance(simple_index.path, Path)
        assert simple_index.path.name == "test.chassis"

    def test_options_property(self, simple_index):
        assert isinstance(simple_index.options, IndexOptions)
        assert simple_index.options.max_connections == 16

    def test_repr(self, simple_index):
        repr_str = repr(simple_index)
        assert "VectorIndex" in repr_str
        assert "dimensions=3" in repr_str
        assert "test.chassis" in repr_str


class TestBatchOperations:

    def test_batch_insert_numpy(self, temp_index_path):
        index = VectorIndex(temp_index_path, dimensions=128)

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
        index = VectorIndex(temp_index_path, dimensions=64)

        for _ in range(50):
            vec = np.random.rand(64).astype(np.float32)
            index.add(vec)

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

    def test_lookups(self, simple_index):
        for i in range(5):
            simple_index.add([float(i), 0.0, 0.0], id=10 + i)
        simple_index.delete(12)
        assert simple_index.get(11).tolist() == [1.0, 0.0, 0.0]
        assert simple_index.get(12) is None
        assert (11 in simple_index, 12 in simple_index, -1 in simple_index) == (True, False, False)
        assert simple_index.ids().tolist() == [10, 11, 13, 14]
        assert simple_index.deleted_count() == 1
        with pytest.raises(TypeError):
            assert "11" in simple_index
        with pytest.raises(ValueError):
            simple_index.get(-1)


class TestReadOnly:
    """An index opened read-only next to a writer."""

    def test_reader_sees_flushed_writes(self, tmp_path):
        path = tmp_path / "shared.chassis"
        writer = VectorIndex(path, dimensions=3)
        writer.add([1.0, 0.0, 0.0], id=10)
        writer.flush()

        reader = VectorIndex(path, dimensions=3, read_only=True)
        assert reader.search([1.0, 0.0, 0.0], k=1)[0].id == 10
        with pytest.raises(ReadOnlyError, match="opened with read_only=True"):
            reader.add([0.0, 1.0, 0.0])

        writer.add([0.0, 1.0, 0.0], id=11)
        assert len(reader) == 1
        writer.flush()
        assert len(reader) == 2
        assert reader.search([0.0, 1.0, 0.0], k=1)[0].id == 11

        writer.delete(10)
        assert (10 in reader, reader.ids().tolist(), reader.deleted_count()) == (True, [10, 11], 0)
        writer.flush()
        assert (10 in reader, reader.ids().tolist(), reader.deleted_count()) == (False, [11], 1)
        assert reader.get(11).tolist() == [0.0, 1.0, 0.0]
        reader.close()
        writer.close()

    def test_reader_reports_the_files_options(self, tmp_path):
        path = tmp_path / "cos.chassis"
        cosine = IndexOptions(metric="cosine", precision="half")
        VectorIndex(path, dimensions=3, options=cosine).close()
        with VectorIndex(path, dimensions=3, read_only=True) as reader:
            assert (reader.options.metric, reader.options.precision) == ("cosine", "half")
        with pytest.raises(OptionsMismatchError, match="help: open it with precision='half'"):
            VectorIndex(path, dimensions=3, options=IndexOptions(metric="cosine"), read_only=True)

    def test_reader_needs_an_existing_index(self, tmp_path):
        with pytest.raises(ChassisError):
            VectorIndex(tmp_path / "missing.chassis", dimensions=3, read_only=True)
        assert not (tmp_path / "missing.chassis").exists()


class TestSearchBatch:
    """Many queries in one call: each row is what search returns for its query."""

    def test_rows_are_searches(self, tmp_path):
        rng = np.random.default_rng(0)
        index = VectorIndex(tmp_path / "batch.chassis", dimensions=8)
        index.add_batch(rng.standard_normal((50, 8), dtype=np.float32))
        queries = rng.standard_normal((6, 8), dtype=np.float32)
        ids, distances = index.search_batch(queries, k=5)
        assert ids.shape == distances.shape == (6, 5)
        assert ids.dtype == np.uint64 and distances.dtype == np.float32
        for row, query in enumerate(queries):
            results = index.search(query, k=5)
            assert list(ids[row]) == [r.id for r in results]
            assert list(distances[row]) == [r.distance for r in results]
        index.close()

    def test_short_rows_are_filled_out(self, simple_index):
        simple_index.add([1.0, 0.0, 0.0])
        simple_index.add([0.0, 1.0, 0.0])
        ids, distances = simple_index.search_batch([[1.0, 0.0, 0.0]], k=4)
        assert list(ids[0][:2]) == [0, 1]
        assert list(ids[0][2:]) == [2**64 - 1] * 2
        assert np.isinf(distances[0][2:]).all()

    def test_empty_batch(self, simple_index):
        ids, distances = simple_index.search_batch(np.empty((0, 3), dtype=np.float32), k=2)
        assert ids.shape == distances.shape == (0, 2)

    def test_reader(self, tmp_path):
        path = tmp_path / "shared.chassis"
        writer = VectorIndex(path, dimensions=3)
        writer.add_batch(np.eye(3, dtype=np.float32))
        writer.flush()
        reader = VectorIndex(path, dimensions=3, read_only=True)
        queries = np.eye(3, dtype=np.float32)
        got_and_want = zip(
            reader.search_batch(queries, k=2), writer.search_batch(queries, k=2), strict=True
        )
        for got, want in got_and_want:
            assert (got == want).all()
        reader.close()
        writer.close()


class TestCosine:
    """Cosine distance."""

    def test_cosine_distances_ignore_length(self, tmp_path):
        index = VectorIndex(
            tmp_path / "cos.chassis", dimensions=2, options=IndexOptions(metric="cosine")
        )
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
        reader = VectorIndex(
            path, dimensions=4, options=IndexOptions(huge_pages=True), read_only=True
        )
        plain = VectorIndex(
            path, dimensions=4, options=IndexOptions(huge_pages=False), read_only=True
        )
        query = [17.2, 1.0, 2.0, 3.0]
        assert [r.id for r in index.search(query, k=3)] == [17, 18, 16]
        assert reader.search(query, k=3) == plain.search(query, k=3) == index.search(query, k=3)
        for handle in (reader, plain, index):
            handle.close()

    def test_huge_pages_false_is_said_before_warming(self, tmp_path, monkeypatch):
        from chassisdb import _ffi

        calls = []
        for name in ("chassis_use_huge_pages", "chassis_keep_small_pages", "chassis_warm"):
            real = getattr(_ffi._lib, name)
            monkeypatch.setattr(
                _ffi._lib, name, lambda ptr, real=real, name=name: calls.append(name) or real(ptr)
            )
        for huge_pages in (False, None, True):
            options = IndexOptions(huge_pages=huge_pages, warm=True)
            VectorIndex(tmp_path / f"{huge_pages}.chassis", dimensions=4, options=options).close()
        assert calls == [
            "chassis_keep_small_pages",
            "chassis_warm",
            "chassis_warm",
            "chassis_use_huge_pages",
            "chassis_warm",
        ]

    def test_warming_changes_no_result(self, tmp_path):
        path = tmp_path / "warm.chassis"
        index = VectorIndex(path, dimensions=4, options=IndexOptions(warm=True))
        for i in range(50):
            index.add([float(i), 1.0, 2.0, 3.0])
        index.flush()
        index.warm()
        reader = VectorIndex(path, dimensions=4, options=IndexOptions(warm=True), read_only=True)
        plain = VectorIndex(
            path, dimensions=4, options=IndexOptions(huge_pages=False), read_only=True
        )
        plain.warm()
        query = [17.2, 1.0, 2.0, 3.0]
        assert [r.id for r in index.search(query, k=3)] == [17, 18, 16]
        assert reader.search(query, k=3) == plain.search(query, k=3) == index.search(query, k=3)
        for handle in (reader, plain, index):
            handle.close()
        with pytest.raises(ChassisError):
            index.warm()

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
            VectorIndex(
                tmp_path / "x.chassis", dimensions=2, options=IndexOptions(precision="int8")
            )


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

    def test_excluded_ids_never_come_back(self, tmp_path):
        index = VectorIndex(tmp_path / "f.chassis", dimensions=2)
        for i in range(20):
            index.add([float(i), 0.0], id=100 + i)
        results = index.search([0.0, 0.0], k=3, excluded=[100, 102, 7])
        assert [r.id for r in results] == [101, 103, 104]
        assert [r.id for r in index.search([0.0, 0.0], k=2, excluded=[])] == [100, 101]
        both = index.search([0.0, 0.0], k=4, allowed=[119, 105, 110], excluded={105})
        assert [r.id for r in both] == [110, 119]
        with pytest.raises(TypeError, match="excluded is a function"):
            index.search([0.0, 0.0], k=4, excluded=lambda i: i == 100)
        with pytest.raises(TypeError, match=r"help: pass excluded=\[100\]"):
            index.search([0.0, 0.0], k=4, excluded=100)
        # 64-bit ids, as from a hash: numpy alone would read this list as floats.
        index.add([0.5, 0.0], id=2**63 + 7)
        near = index.search([0.5, 0.0], k=2, excluded=[100, 2**63 + 7])
        assert 2**63 + 7 not in [r.id for r in near]
        with pytest.raises(TypeError, match="get\\(id\\) gives None"):
            index.search(index.get(12345), k=1)
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


class TestBackup:
    """A copy of the index, from its writer or from a reader beside it (ADR-0024)."""

    def test_backup_from_the_writer_and_from_a_reader(self, tmp_path):
        path = tmp_path / "i.chassis"
        index = VectorIndex(path, dimensions=8)
        vectors = np.random.default_rng(5).random((200, 8), dtype=np.float32)
        index.add_batch(vectors[:150])
        index.flush()
        index.add_batch(vectors[150:])
        reader = VectorIndex(path, dimensions=8, read_only=True)
        reader.backup(tmp_path / "reader.chassis")
        index.backup(str(tmp_path / "writer.chassis"))
        # The reader's holds the last flush; the writer flushes first.
        for name, count in [("reader.chassis", 150), ("writer.chassis", 200)]:
            with VectorIndex(tmp_path / name, dimensions=8) as copy:
                assert len(copy) == count
                assert copy.search(vectors[count - 1], k=1)[0].id == count - 1
        reader.close()
        index.close()

    def test_a_backup_onto_the_index_itself_is_refused(self, tmp_path):
        path = tmp_path / "i.chassis"
        with VectorIndex(path, dimensions=8) as index:
            with pytest.raises(InvalidArgumentError, match="the index's own file"):
                index.backup(path)


class TestUserVersion:
    """The application's number, committed with the data (ADR-0023)."""

    def test_a_flush_commits_it_and_closing_without_one_rolls_it_back(self, tmp_path):
        path = tmp_path / "versioned.chassis"
        with VectorIndex(path, dimensions=3) as index:
            assert index.user_version == 0
            index.add([1.0, 0.0, 0.0])
            index.user_version = 5
        index = VectorIndex(path, dimensions=3)
        assert index.user_version == 5
        index.add([0.0, 1.0, 0.0])
        index.user_version = 6
        index.close()
        with VectorIndex(path, dimensions=3) as index:
            assert (index.user_version, len(index)) == (5, 1)

    def test_rollback_discards_what_a_close_would_and_keeps_the_lock(self, tmp_path):
        path = tmp_path / "rolled.chassis"
        index = VectorIndex(path, dimensions=3)
        index.add([1.0, 0.0, 0.0], id=7)
        index.user_version = 1
        index.flush()
        index.add([0.0, 1.0, 0.0], id=8)
        index.delete(7)
        index.user_version = 2
        index.rollback()
        assert (index.user_version, list(index.ids())) == (1, [7])
        with pytest.raises(IndexLockedError):
            VectorIndex(path, dimensions=3)
        index.close()
        reader = VectorIndex(path, dimensions=3, read_only=True)
        with pytest.raises(ReadOnlyError, match="roll back"):
            reader.rollback()
        reader.close()

    def test_a_read_only_index_sees_each_flush_and_cannot_set_it(self, tmp_path):
        path = tmp_path / "versioned.chassis"
        with VectorIndex(path, dimensions=3) as writer:
            writer.user_version = 1
            writer.flush()
            reader = VectorIndex(path, dimensions=3, read_only=True)
            assert reader.user_version == 1
            writer.user_version = 2
            writer.flush()
            assert reader.user_version == 2
            with pytest.raises(ReadOnlyError, match="user_version"):
                reader.user_version = 3
            reader.close()

    @pytest.mark.parametrize("bad", [-1, 2**64])
    def test_out_of_range_is_refused(self, tmp_path, bad):
        with VectorIndex(tmp_path / "v.chassis", dimensions=3) as index:
            with pytest.raises(InvalidArgumentError, match="help: "):
                index.user_version = bad

    @pytest.mark.parametrize("bad", [1.5, True, "1"])
    def test_not_an_int_is_refused(self, tmp_path, bad):
        with VectorIndex(tmp_path / "v.chassis", dimensions=3) as index:
            with pytest.raises(TypeError, match="help: "):
                index.user_version = bad


class TestCheck:
    """What opening can't find: vectors no search reaches, and vectors that aren't finite."""

    def test_an_index_in_good_order_checks_clean(self, tmp_path):
        path = tmp_path / "checked.chassis"
        rng = np.random.default_rng(0)
        with VectorIndex(path, dimensions=8) as index:
            assert index.check() == Check(unreachable=[], not_finite=[])
            index.add_batch(rng.random((500, 8), dtype=np.float32))
            index.delete(7)
            assert index.check().ok
        with VectorIndex(path, dimensions=8, read_only=True) as reader:
            assert reader.check().ok

    def test_a_vector_damaged_into_nan_is_reported(self, tmp_path):
        path = tmp_path / "damaged.chassis"
        marked = np.full(8, 1234.5678, dtype=np.float32)
        with VectorIndex(path, dimensions=8) as index:
            index.add_batch(np.random.default_rng(0).random((50, 8), dtype=np.float32))
            assert index.add(marked) == 50
        # Damage the file where it keeps the marked vector, as a bad disk could.
        data = bytearray(path.read_bytes())
        at = data.find(marked.tobytes())
        data[at : at + 4] = np.float32(np.nan).tobytes()
        path.write_bytes(bytes(data))
        with VectorIndex(path, dimensions=8) as index:
            check = index.check()
        assert (check.unreachable, check.not_finite, check.ok) == ([], [50], False)

    def test_a_closed_index_says_so(self, simple_index):
        simple_index.close()
        with pytest.raises(ChassisError, match="closed"):
            simple_index.check()
