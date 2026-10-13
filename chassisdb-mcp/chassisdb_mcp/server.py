"""An MCP server that gives agents a memory on this machine: the text of each memory in SQLite, its
embedding in a Chassis index, under the same id. Any number of servers, one per agent session, can
share one memory folder: each write opens the index for as long as it takes, and searches read the
last flush without a lock."""

import argparse
import json
import os
import sqlite3
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from chassisdb import IndexLockedError, IndexOptions, VectorIndex
from mcp.server.mcpserver import MCPServer

Embed = Callable[[list[str]], np.ndarray]

EMBEDDERS = """\
  fastembed:<model>              local, no PyTorch: pip install 'chassisdb-mcp[fastembed]'
                                 e.g. fastembed:BAAI/bge-small-en-v1.5
  sentence-transformers:<model>  local, with PyTorch: pip install 'chassisdb-mcp[sentence-transformers]'
                                 e.g. sentence-transformers:all-MiniLM-L6-v2
  openai:<model>                 any OpenAI-compatible embeddings API: OpenAI, Ollama, LM Studio, vLLM.
                                 OPENAI_BASE_URL picks the server (default https://api.openai.com/v1;
                                 Ollama: http://localhost:11434/v1), OPENAI_API_KEY the key if it needs one
                                 e.g. openai:text-embedding-3-small, openai:nomic-embed-text"""

# Text embedding models compare by cosine; it also suits models whose vectors aren't unit length.
OPTIONS = IndexOptions(metric="cosine")
# How long a write waits for another session's write to finish.
LOCK_WAIT_SECONDS = 10.0


def load_embedder(spec: str) -> Embed:
    """The embedding function `spec` names. The model loads on first use, so the server starts at
    once even when the model has to be downloaded first."""
    kind, _, model = spec.partition(":")
    if kind not in ("fastembed", "sentence-transformers", "openai") or not model:
        raise ValueError(
            f"The embedder is {spec!r}, but has to be <kind>:<model>, one of\n{EMBEDDERS}"
        )
    loaded: list[Embed] = []

    def embed(texts: list[str]) -> np.ndarray:
        if not loaded:
            loaded.append(_load(kind, model))
        return np.asarray(loaded[0](texts), dtype=np.float32)

    return embed


def _load(kind: str, model: str) -> Embed:
    if kind == "openai":
        return _openai(model)
    try:
        if kind == "fastembed":
            from fastembed import TextEmbedding

            fast = TextEmbedding(model)
            return lambda texts: np.array(list(fast.embed(texts)))
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model)
        return lambda texts: st.encode(texts, convert_to_numpy=True)
    except ImportError as e:
        raise RuntimeError(
            f"{kind} isn't installed\nhelp: pip install 'chassisdb-mcp[{kind}]', or run the server "
            f"with uvx --from 'chassisdb-mcp[{kind}]' chassisdb-mcp"
        ) from e


def _openai(model: str) -> Embed:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")

    def embed(texts: list[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = json.dumps({"model": model, "input": texts}).encode()
        request = urllib.request.Request(f"{base}/embeddings", body, headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)["data"]
        return np.array([row["embedding"] for row in sorted(data, key=lambda row: row["index"])])

    return embed


class Memory:
    """Memories in `folder`: `memory.db` (SQLite) holds their text and tags, `memory.chassis` their
    embeddings, and the embedder they were made with, which every later run has to use too."""

    def __init__(self, folder: Path, embedder: str, embed: Embed):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "memory.chassis"
        self.embed = embed
        # The MCP server runs tools on worker threads: one at a time, through `lock`.
        self.lock = threading.Lock()
        self.db = sqlite3.connect(
            folder / "memory.db", isolation_level=None, timeout=30, check_same_thread=False
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('embedder', ?)", (embedder,))
        held = self._meta("embedder")
        if held != embedder:
            raise ValueError(
                f"The memory in {folder} was made with the embedder {held}, so it has to be read "
                f"with {held} too, not {embedder}: their vectors can't be compared\nhelp: run with "
                f"--embedder {held}, or give {embedder} a folder of its own with --dir"
            )
        self._reader: VectorIndex | None = None

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _writer(self, dimensions: int) -> VectorIndex:
        """The index, opened to write. Another session writing waits its turn: writes take
        milliseconds."""
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                return VectorIndex(self.path, dimensions=dimensions, options=OPTIONS)
            except IndexLockedError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def remember(self, text: str, tags: list[str]) -> int:
        with self.lock:
            return self._remember(text, tags)

    def recall(self, query: str, k: int, tag: str | None) -> list[dict]:
        with self.lock:
            return self._recall(query, k, tag)

    def forget(self, id: int) -> bool:
        with self.lock:
            return self._forget(id)

    def _remember(self, text: str, tags: list[str]) -> int:
        vector = self.embed([text])[0]
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('dimensions', ?)", (len(vector),))
        id = self.db.execute(
            "INSERT INTO memories (text, tags) VALUES (?, ?)", (text, json.dumps(tags))
        ).lastrowid
        try:
            index = self._writer(len(vector))
            try:
                index.add(vector, id=id)
                index.flush()
            finally:
                index.close()
        except BaseException:
            self.db.execute("DELETE FROM memories WHERE id = ?", (id,))
            raise
        return id

    def _recall(self, query: str, k: int, tag: str | None) -> list[dict]:
        if not self.path.exists():
            return []
        allowed = None
        if tag is not None:
            allowed = [
                row[0]
                for row in self.db.execute(
                    "SELECT memories.id FROM memories, json_each(memories.tags) WHERE value = ?",
                    (tag,),
                )
            ]
            if not allowed:
                return []
        vector = self.embed([query])[0]
        if self._reader is None:
            self._reader = VectorIndex(self.path, dimensions=len(vector), read_only=True)
        hits = self._reader.search(vector, k=k, allowed=allowed)
        if not hits:
            return []
        rows = {
            row[0]: row
            for row in self.db.execute(
                f"SELECT id, text, tags, created FROM memories WHERE id IN "
                f"({','.join('?' * len(hits))})",
                [hit.id for hit in hits],
            )
        }
        # A memory whose row is gone was forgotten, or never finished being remembered.
        return [
            {
                "id": hit.id,
                "text": rows[hit.id][1],
                "tags": json.loads(rows[hit.id][2]),
                "created": rows[hit.id][3],
                "distance": round(hit.distance, 4),
            }
            for hit in hits
            if hit.id in rows
        ]

    def _forget(self, id: int) -> bool:
        dimensions = self._meta("dimensions")
        if dimensions is None or not self.path.exists():
            return False
        index = self._writer(int(dimensions))
        try:
            deleted = index.delete(id)
            index.flush()
        finally:
            index.close()
        self.db.execute("DELETE FROM memories WHERE id = ?", (id,))
        return deleted


def build_server(memory: Memory) -> MCPServer:
    server = MCPServer(
        "chassis-memory",
        instructions=(
            "Long-term memory that lasts across sessions, on this machine. Recall before answering "
            "anything earlier sessions may have settled; remember decisions, preferences and facts "
            "about the user or project worth knowing next time."
        ),
    )

    @server.tool()
    def remember(text: str, tags: list[str] | None = None) -> int:
        """Store a memory: a fact, decision, preference or note worth recalling in later sessions.
        Write it to stand on its own. Tags, such as a project's name, let recall look among them
        only. Returns the memory's id."""
        return memory.remember(text, tags or [])

    @server.tool()
    def recall(query: str, k: int = 5, tag: str | None = None) -> list[dict]:
        """Find the k memories closest in meaning to the query, nearest first, with their ids, text,
        tags, when they were made and their distance (0 is the same meaning, 2 the opposite). With
        a tag, only memories that have it."""
        return memory.recall(query, k, tag)

    @server.tool()
    def forget(id: int) -> bool:
        """Delete the memory with this id, as recall returned it. Returns whether there was one."""
        return memory.forget(id)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chassisdb-mcp",
        description="Long-term memory for agents on this machine, as an MCP server over stdio.",
        epilog=f"embedders:\n{EMBEDDERS}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embedder",
        default=os.environ.get("CHASSISDB_MCP_EMBEDDER"),
        help="the embedding model, as <kind>:<model> (or CHASSISDB_MCP_EMBEDDER); see below",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(os.environ.get("CHASSISDB_MCP_DIR", Path.home() / ".chassisdb-mcp")),
        help="the folder memories are kept in (or CHASSISDB_MCP_DIR; default ~/.chassisdb-mcp)",
    )
    args = parser.parse_args(argv)
    if not args.embedder:
        parser.error(f"pick an embedding model with --embedder <kind>:<model>, one of\n{EMBEDDERS}")
    try:
        memory = Memory(args.dir, args.embedder, load_embedder(args.embedder))
    except ValueError as e:
        parser.error(str(e))
    build_server(memory).run()


if __name__ == "__main__":
    main()
