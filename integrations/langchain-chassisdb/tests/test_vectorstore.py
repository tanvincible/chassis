from collections.abc import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_chassisdb import ChassisVectorStore


class TestChassisStandard(VectorStoreIntegrationTests):
    """LangChain's own tests for every vector store, sync and async."""

    @pytest.fixture()
    def vectorstore(self, tmp_path) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        store = ChassisVectorStore(tmp_path / "test.chassis", self.get_embeddings())
        yield store
        store.close()


@pytest.fixture()
def embeddings():
    return TestChassisStandard.get_embeddings()


def test_reopens_with_what_was_stored(tmp_path, embeddings):
    store = ChassisVectorStore(tmp_path / "docs.chassis", embeddings)
    store.add_texts(["refunds take five days"], metadatas=[{"topic": "billing"}], ids=["a"])
    store.close()
    again = ChassisVectorStore(tmp_path / "docs.chassis", embeddings)
    [doc] = again.similarity_search("refunds take five days", k=1)
    assert (doc.id, doc.page_content, doc.metadata) == (
        "a",
        "refunds take five days",
        {"topic": "billing"},
    )


def test_filters_on_metadata(tmp_path, embeddings):
    store = ChassisVectorStore(tmp_path / "docs.chassis", embeddings)
    store.add_texts(
        ["one", "two", "three"],
        metadatas=[
            {"topic": "billing", "n": 1},
            {"topic": "shipping", "n": 2},
            {"topic": "billing", "n": 3},
        ],
        ids=["1", "2", "3"],
    )
    found = store.similarity_search("two", k=3, filter={"topic": "billing"})
    assert sorted(doc.id for doc in found) == ["1", "3"]
    assert [
        doc.id for doc in store.similarity_search("x", k=3, filter={"topic": "billing", "n": 3})
    ] == ["3"]
    assert store.similarity_search("x", k=3, filter={"topic": "nothing"}) == []


def test_mmr_and_scores(tmp_path, embeddings):
    store = ChassisVectorStore.from_texts(
        ["apple pie", "apple pie", "car engine"],
        embeddings,
        ids=["a", "b", "c"],
        path=tmp_path / "s.chassis",
    )
    picked = store.max_marginal_relevance_search("apple pie", k=2, fetch_k=3, lambda_mult=0.1)
    assert {doc.id for doc in picked} == {"a", "c"} or {doc.id for doc in picked} == {"b", "c"}
    [(doc, score)] = store.similarity_search_with_relevance_scores("apple pie", k=1)
    assert doc.page_content == "apple pie" and score == pytest.approx(1.0, abs=1e-5)


def test_readers_and_their_errors(tmp_path, embeddings):
    writer = ChassisVectorStore(tmp_path / "docs.chassis", embeddings)
    writer.add_texts(["hello world"], ids=["h"])
    reader = ChassisVectorStore(tmp_path / "docs.chassis", embeddings, read_only=True)
    assert reader.similarity_search("hello world", k=1)[0].id == "h"
    with pytest.raises(PermissionError, match="help: open it without read_only"):
        reader.add_texts(["no"])
    with pytest.raises(ValueError, match="help: pass path="):
        ChassisVectorStore.from_texts(["x"], embeddings)
