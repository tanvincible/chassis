"""A LangChain vector store on Chassis: the vectors in a Chassis index file, the documents and their
metadata in SQLite beside it, in your own process with no server."""

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from chassisdb import IndexNotFoundError, IndexOptions, VectorIndex
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance


class ChassisVectorStore(VectorStore):
    """Documents searched by meaning, kept in two files: `path` (say `docs.chassis`) holds the
    vectors, and the same name ending in `.db` holds the documents and their metadata.

    Example:
        ```python
        from langchain_chassisdb import ChassisVectorStore

        store = ChassisVectorStore("docs.chassis", embedding=OpenAIEmbeddings())
        store.add_texts(["Refunds take 5 days"], metadatas=[{"topic": "billing"}])
        store.similarity_search("how long do refunds take", k=4, filter={"topic": "billing"})
        ```

    Args:
        path: The Chassis index file; it is created if it doesn't exist.
        embedding: The model that embeds documents and queries. Reopen a store with the same one.
        metric: "cosine" (the default; what most text embedding models want) or "euclidean". An
            existing store keeps the metric it was created with.
        precision: "full" or "half", which keeps vectors in half the space at the same recall.
        read_only: Open to search only, while another process writes the store. A reader takes
            no lock and sees what the writer last saved.
    """

    def __init__(
        self,
        path: str | Path,
        embedding: Embeddings,
        *,
        metric: str = "cosine",
        precision: str = "full",
        read_only: bool = False,
    ):
        self.path = Path(path)
        self._embedding = embedding
        self._options = IndexOptions(metric=metric, precision=precision)
        self._read_only = read_only
        # LangChain runs the async methods on worker threads: one call at a time, through `_lock`.
        self._lock = threading.RLock()
        documents = self.path.with_suffix(".db")
        if read_only and not documents.exists():
            raise IndexNotFoundError(
                f"No store at {self.path}: {documents} doesn't exist\nhelp: check the path, or "
                "create the store first by opening it without read_only where it is written"
            )
        self._db = sqlite3.connect(
            documents, isolation_level=None, check_same_thread=False, timeout=30
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                key INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self._index: VectorIndex | None = None
        dimensions = self._dimensions()
        if dimensions is not None:
            self._open(dimensions)

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _dimensions(self) -> int | None:
        row = self._db.execute("SELECT value FROM meta WHERE name = 'dimensions'").fetchone()
        return int(row[0]) if row else None

    def _current(self) -> VectorIndex | None:
        """The index, opened once the store holds a vector: a reader may open the store before
        its writer has added anything."""
        if self._index is None:
            dimensions = self._dimensions()
            if dimensions is not None:
                self._open(dimensions)
        return self._index

    def _open(self, dimensions: int) -> VectorIndex:
        if self._index is None:
            if self._read_only:
                self._index = VectorIndex(self.path, dimensions=dimensions, read_only=True)
            else:
                self._index = VectorIndex(self.path, dimensions=dimensions, options=self._options)
                self._mend()
        return self._index

    def _version(self) -> int | None:
        """The number of the last committed write; None in a store written before it was kept."""
        row = self._db.execute("SELECT value FROM meta WHERE name = 'version'").fetchone()
        return int(row[0]) if row else None

    def _mend(self) -> None:
        """Brings the index level with the documents after a write that flushed the index but
        never committed, by a crash or a failed commit: its vectors without a document go, and
        documents without a vector are embedded again. Each text has its own key, so comparing
        keys finds every difference."""
        index, version = self._index, self._version()
        assert index is not None
        if index.user_version == version:
            return
        texts = dict(self._db.execute("SELECT key, text FROM documents"))
        held = set(index.ids())
        for key in held - texts.keys():
            index.delete(key)
        missing = sorted(texts.keys() - held)
        if missing:
            vectors = self._embedding.embed_documents([texts[key] for key in missing])
            index.add_batch(np.asarray(vectors, dtype=np.float32), ids=missing)
        if version is None:
            version = index.user_version
            self._db.execute("INSERT INTO meta VALUES ('version', ?)", (str(version),))
        index.user_version = version
        index.flush()

    @contextmanager
    def _writing(self, index: VectorIndex) -> Iterator[None]:
        """A write: the documents and the index change together, or neither does. The index is
        flushed before the commit, with the write's number, so a write that never commits shows
        as an index one ahead, which the next write or the next open mends."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._mend()
            yield
            version = (self._version() or 0) + 1
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(version),))
            index.user_version = version
            index.flush()
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def close(self) -> None:
        """Close the files. Everything added or deleted was saved already."""
        with self._lock:
            if self._index is not None:
                self._index.close()
                self._index = None
            self._db.close()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and store `texts`. A text with the id of a stored document replaces it."""
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        ids = [
            str(id) if id is not None else uuid.uuid4().hex for id in (ids or [None] * len(texts))
        ]
        if not len(texts) == len(metadatas) == len(ids):
            raise ValueError(
                f"{len(texts)} texts were given with {len(metadatas)} metadatas and "
                f"{len(ids)} ids\nhelp: give one of each per text, in the same order"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("An id is given twice\nhelp: give each text its own id")
        vectors = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        with self._lock:
            self._writable("add")
            self._db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('dimensions', ?)", (str(vectors.shape[1]),)
            )
            index = self._open(vectors.shape[1])
            with self._writing(index):
                replaced = self._keys(ids)
                self._db.executemany("DELETE FROM documents WHERE id = ?", [(id,) for id in ids])
                keys = [
                    self._db.execute(
                        "INSERT INTO documents (id, text, metadata) VALUES (?, ?, ?)",
                        (id, text, json.dumps(metadata)),
                    ).lastrowid
                    for id, text, metadata in zip(ids, texts, metadatas, strict=True)
                ]
                index.add_batch(vectors, ids=keys)
                for key in replaced:
                    index.delete(key)
        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete the documents with these ids; ids that aren't stored are skipped. With no ids,
        delete every document."""
        with self._lock:
            self._writable("delete")
            index = self._current()
            if index is None:  # Nothing was ever added
                return True
            with self._writing(index):
                if ids is None:
                    ids = [row[0] for row in self._db.execute("SELECT id FROM documents")]
                keys = self._keys(ids)
                self._db.executemany("DELETE FROM documents WHERE id = ?", [(id,) for id in ids])
                for key in keys:
                    index.delete(key)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        with self._lock:
            found = {
                row[0]: row
                for row in self._select(
                    "SELECT id, text, metadata, key FROM documents WHERE id IN", ids
                )
            }
        return [_document(found[id]) for id in ids if id in found]

    def similarity_search(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """The `k` documents closest to `query`. `filter` keeps only documents whose metadata has
        every key with the value given, such as `{"topic": "billing"}`."""
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter=filter)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Like `similarity_search`, with each document's distance: smaller is closer."""
        vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(vector, k, filter=filter)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(embedding, k, filter=filter)
        ]

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        with self._lock:
            if self._current() is None:
                return []
            allowed = self._matching(filter)
            if allowed == []:
                return []
            hits = self._index.search(embedding, k=k, allowed=allowed)
            rows = self._rows([hit.id for hit in hits])
        return [(_document(rows[hit.id]), hit.distance) for hit in hits if hit.id in rows]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            vector, k, fetch_k, lambda_mult, filter=filter
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """`k` documents close to `embedding` and different from each other, chosen among the
        `fetch_k` closest. `lambda_mult` 1 is closest only, 0 most different."""
        with self._lock:
            if self._current() is None:
                return []
            allowed = self._matching(filter)
            if allowed == []:
                return []
            hits = self._index.search(embedding, k=fetch_k, allowed=allowed)
            vectors = [self._index.get(hit.id) for hit in hits]
            rows = self._rows([hit.id for hit in hits])
        chosen = maximal_marginal_relevance(
            np.asarray(embedding, dtype=np.float32), vectors, lambda_mult=lambda_mult, k=k
        )
        return [_document(rows[hits[i].id]) for i in chosen if hits[i].id in rows]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # The file's metric: a read_only store takes it from there, whatever was asked for.
        index = self._current()
        if (index.metric if index is not None else self._options.metric) == "cosine":
            return self._cosine_relevance_score_fn
        return self._euclidean_relevance_score_fn

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> "ChassisVectorStore":
        """A store at `path` holding `texts`. `path` is required: `path="docs.chassis"`."""
        if path is None:
            raise ValueError(
                "from_texts needs the file to keep the store in\nhelp: pass path=, such as "
                'path="docs.chassis"'
            )
        store = cls(path, embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def _writable(self, action: str) -> None:
        if self._read_only:
            raise PermissionError(
                f"This store was opened with read_only=True, so it can't {action}\nhelp: open it "
                "without read_only in the one process that writes it"
            )

    def _select(self, query: str, values: Sequence) -> list[tuple]:
        """`query` ending in `IN`, for every value, in batches under SQLite's variable limit."""
        rows: list[tuple] = []
        values = list(values)
        for start in range(0, len(values), 500):
            batch = values[start : start + 500]
            rows += self._db.execute(f"{query} ({','.join('?' * len(batch))})", batch).fetchall()
        return rows

    def _keys(self, ids: Sequence[str]) -> list[int]:
        return [row[0] for row in self._select("SELECT key FROM documents WHERE id IN", ids)]

    def _rows(self, keys: list[int]) -> dict[int, tuple]:
        query = "SELECT id, text, metadata, key FROM documents WHERE key IN"
        return {row[3]: row for row in self._select(query, keys)}

    def _matching(self, filter: dict | None) -> list[int] | None:
        """The keys of the documents `filter` keeps, or None for every document."""
        if not filter:
            return None
        where = " AND ".join("json_extract(metadata, ?) = json_extract(?, '$')" for _ in filter)
        args = [x for name, value in filter.items() for x in (f'$."{name}"', json.dumps(value))]
        return [
            row[0] for row in self._db.execute(f"SELECT key FROM documents WHERE {where}", args)
        ]


def _document(row: tuple) -> Document:
    id, text, metadata, _ = row
    return Document(id=id, page_content=text, metadata=json.loads(metadata))
