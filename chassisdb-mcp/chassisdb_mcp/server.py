"""chassisdb-mcp: long-term memory for agents on this machine, as an MCP server and a command line.

Each memory's text and tags are in SQLite (`memory.db`) and its embedding in a Chassis index
(`memory.chassis`), under the same id, in one folder. Any number of sessions can share the folder:
every write is one transaction on `memory.db`, so sessions take turns writing and a memory's row
and vector are saved together, while searches read the index without a lock.
"""

import argparse
import difflib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NotRequired

import numpy as np
from chassisdb import (
    ChassisError,
    DimensionMismatchError,
    IndexLockedError,
    IndexOptions,
    VectorIndex,
)
from typing_extensions import TypedDict  # pydantic needs this one before Python 3.12

try:
    VERSION = version("chassisdb-mcp")
except PackageNotFoundError:  # Run from a checkout without installing
    VERSION = "0+checkout"

DEFAULT_DIR = Path.home() / ".chassisdb-mcp"
SUGGESTED = "fastembed:BAAI/bge-small-en-v1.5"
EMBEDDERS = """\
  fastembed:<model>              local, no PyTorch; needs chassisdb-mcp[fastembed]
                                 e.g. fastembed:BAAI/bge-small-en-v1.5 (about 70 MB)
  sentence-transformers:<model>  local, with PyTorch; needs chassisdb-mcp[sentence-transformers]
                                 e.g. sentence-transformers:all-MiniLM-L6-v2
  openai:<model>                 an OpenAI-compatible API: OpenAI, Ollama, LM Studio, vLLM.
                                 OPENAI_BASE_URL says where (OpenAI's by default; Ollama's is
                                 http://localhost:11434/v1), OPENAI_API_KEY gives a key
                                 e.g. openai:text-embedding-3-small, openai:nomic-embed-text"""
KINDS = ("fastembed", "sentence-transformers", "openai")

# Text embedding models compare by cosine; it also suits models whose vectors aren't unit length.
OPTIONS = IndexOptions(metric="cosine")
# How long a write waits for other sessions' writes.
WAIT_SECONDS = 30.0
# Most embedding models read the first few hundred words and drop the rest.
LONG_TEXT = 2000


class Remembered(TypedDict):
    id: int
    text: str
    tags: list[str]
    created: str


class Found(Remembered):
    distance: float


class Stored(TypedDict):
    id: int
    similar: list[Found]
    warning: NotRequired[str]


class Forgotten(TypedDict):
    forgotten: Remembered


class Overview(TypedDict):
    folder: str
    embedder: str | None
    total: int
    tags: dict[str, int]
    memories: list[Remembered]
    more: bool
    unindexed: int
    stray: int


class Problem(Exception):
    """Something the user can fix: what was wrong, then what to do on a `help:` line."""


VERBOSE = True


def log(message: str) -> None:
    """MCP clients keep a server's stderr in their logs; stdout carries the protocol."""
    if VERBOSE:
        print(f"chassisdb-mcp: {message}", file=sys.stderr, flush=True)


def check_spec(spec: str) -> str:
    """`spec` if it reads as <kind>:<model>, or a Problem that says what was meant."""
    kind, _, model = spec.partition(":")
    if kind not in KINDS or not model:
        guess = difflib.get_close_matches(kind, KINDS, n=1, cutoff=0.5)
        did_you_mean = (
            f"did you mean {guess[0]}:{model or '<model>'}? " if guess else ""
        )
        raise Problem(
            f"The embedder is {spec!r}, but has to be <kind>:<model>\nhelp: {did_you_mean}"
            f"pick one of\n{EMBEDDERS}"
        )
    return spec


class Embedder:
    """The embedding model `spec` names. It loads on first use, or on a thread from `start()`, so
    the server answers at once even while a model downloads."""

    def __init__(self, spec: str):
        self.spec = check_spec(spec)
        self.kind, _, self.model = spec.partition(":")
        self._embed: Callable[[list[str]], object] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        def load() -> None:
            try:
                self._load()
            except (
                Exception
            ) as e:  # The first call that needs the model reports it again.
                log(str(e).replace("\n", " "))

        threading.Thread(target=load, daemon=True).start()

    def _load(self) -> Callable[[list[str]], object]:
        with self._lock:
            if self._embed is None:
                log(f"loading {self.spec}")
                start = time.monotonic()
                self._embed = LOADERS[self.kind](self.model)
                log(f"{self.spec} ready in {time.monotonic() - start:.1f} s")
            return self._embed

    def __call__(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._load()(texts), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise Problem(
                f"{self.spec} gave {vectors.shape[0] if vectors.ndim else 0} vectors for "
                f"{len(texts)} texts\nhelp: check that the model makes text embeddings"
            )
        return vectors


def _missing(kind: str) -> Problem:
    return Problem(
        f"{kind} isn't installed, and the embedder {kind}:<model> needs it\nhelp: run the server "
        f"with uvx --from 'chassisdb-mcp[{kind}]' chassisdb-mcp, or pip install "
        f"'chassisdb-mcp[{kind}]'"
    )


def _fastembed(model: str) -> Callable[[list[str]], object]:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise _missing("fastembed") from None
    names = [m["model"] for m in TextEmbedding.list_supported_models()]
    if model.lower() not in {name.lower() for name in names}:
        close = difflib.get_close_matches(model, names, n=3, cutoff=0.5)
        did_you_mean = f"did you mean {' or '.join(close)}? " if close else ""
        raise Problem(
            f"fastembed has no model {model!r}\nhelp: {did_you_mean}fastembed's models are "
            f"{', '.join(names)}"
        )
    try:
        fast = TextEmbedding(model)
    except Exception as e:
        raise Problem(
            f"fastembed couldn't load {model}: {e}\nhelp: the first time, the model downloads from "
            "Hugging Face: check the connection, or set FASTEMBED_CACHE_PATH to a folder with it"
        ) from e
    return lambda texts: list(fast.embed(texts))


def _sentence_transformers(model: str) -> Callable[[list[str]], object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise _missing("sentence-transformers") from None
    try:
        st = SentenceTransformer(model)
    except Exception as e:
        raise Problem(
            f"sentence-transformers couldn't load {model}: {e}\nhelp: name a model from "
            "https://huggingface.co/models?library=sentence-transformers, such as "
            "all-MiniLM-L6-v2; the first time, it downloads, so check the connection"
        ) from e
    return lambda texts: st.encode(texts)


def _openai(model: str) -> Callable[[list[str]], object]:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not key and urllib.parse.urlsplit(base).hostname == "api.openai.com":
        raise Problem(
            "OPENAI_API_KEY isn't set, and OpenAI's API needs it\nhelp: give the server the key "
            "(claude mcp add ... -e OPENAI_API_KEY=...), or point OPENAI_BASE_URL at a local "
            "server such as Ollama: http://localhost:11434/v1"
        )

    def embed(texts: list[str]) -> object:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = json.dumps({"model": model, "input": texts}).encode()
        request = urllib.request.Request(f"{base}/embeddings", body, headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                said = response.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            said = e.read().decode(errors="replace").strip()[:300]
            fix = {
                401: "check OPENAI_API_KEY in the server's environment",
                403: "check OPENAI_API_KEY in the server's environment",
                404: f"check that the server has {model} (Ollama: ollama pull {model}) and that "
                "OPENAI_BASE_URL ends in /v1",
                429: "the API is limiting requests or is out of credit: wait, or check the account",
            }.get(e.code, "the server's own logs say more")
            raise Problem(
                f"The embeddings API at {base} answered {e.code} {e.reason}: {said}\nhelp: {fix}"
            ) from None
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise Problem(
                f"Nothing answered at {base} ({reason})\nhelp: start the server (Ollama: ollama "
                "serve), or set OPENAI_BASE_URL to where it listens"
            ) from None
        try:
            data = json.loads(said)["data"]
            return [
                row["embedding"] for row in sorted(data, key=lambda row: row["index"])
            ]
        except (ValueError, KeyError, TypeError):
            raise Problem(
                f"The embeddings API at {base} answered with no embeddings: {said.strip()[:300]}\n"
                "help: check that OPENAI_BASE_URL is an OpenAI-compatible API, ending in /v1"
            ) from None

    return embed


LOADERS = {
    "fastembed": _fastembed,
    "sentence-transformers": _sentence_transformers,
    "openai": _openai,
}


def _tags(tags: list[str] | str | None) -> list[str]:
    if isinstance(tags, str):
        tags = [tags]
    return sorted({tag.strip() for tag in tags or [] if tag.strip()})


class Memory:
    """The memories in `folder`. The folder keeps the model its memories were made with, and every
    session uses that one: `embedder` is for a folder with none yet, and the model `reembed` moves
    the memories to."""

    def __init__(self, folder: Path, embedder: str | None = None):
        self.folder = folder.expanduser()
        self.path = self.folder / "memory.chassis"
        self.default = check_spec(embedder) if embedder else None
        self._embedders: dict[str, Embedder] = {}
        self._warned = False
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise Problem(
                f"The memory folder {self.folder} can't be made: {e.strerror}\nhelp: pass another "
                "with --dir, or set CHASSISDB_MCP_DIR"
            ) from e
        # MCP runs tools on worker threads: one at a time on this connection, through `lock`.
        self.lock = threading.Lock()
        db = self.folder / "memory.db"
        try:
            self.db = sqlite3.connect(
                db, isolation_level=None, timeout=WAIT_SECONDS, check_same_thread=False
            )
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """)
        except sqlite3.Error as e:
            raise Problem(
                f"{db} can't be used as a memory: {e}\nhelp: check that the folder can be "
                "written, or give the memory another folder with --dir"
            ) from e

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: object) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    def embedder(self) -> Embedder:
        """The folder's model, or for a folder with none yet, the one this was opened with."""
        with self.lock:
            held = self._meta("embedder")
            if held and self.default and held != self.default and not self._warned:
                log(
                    f"the memories in {self.folder} were made with {held}, so that model is used, "
                    f"not {self.default}; `chassisdb-mcp --dir {self.folder} --embedder "
                    f"{self.default} reembed` moves them"
                )
                self._warned = True
            spec = held or self.default
            if spec is None:
                raise Problem(
                    f"No embedding model is set for the memory in {self.folder}\nhelp: start the "
                    f"server with --embedder <kind>:<model>, such as --embedder {SUGGESTED}; "
                    "chassisdb-mcp --help lists the kinds"
                )
            if spec not in self._embedders:
                self._embedders[spec] = Embedder(spec)
            return self._embedders[spec]

    @contextmanager
    def _writing(self) -> Iterator[None]:
        """A write transaction on memory.db, which every write takes: sessions take turns, and a
        memory's row and vector are saved, or not, together."""
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as e:
            raise Problem(
                f"Another session has been writing the memory in {self.folder} for "
                f"{WAIT_SECONDS:.0f} s\nhelp: a reembed may be running there; try again when it is "
                "done"
            ) from e
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _open(self, dimensions: int, read_only: bool = False) -> VectorIndex:
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            try:
                if read_only:
                    return VectorIndex(self.path, dimensions=dimensions, read_only=True)
                return VectorIndex(self.path, dimensions=dimensions, options=OPTIONS)
            except DimensionMismatchError as e:
                raise self._mismatch(e) from e
            except IndexLockedError as e:
                if time.monotonic() > deadline:
                    raise Problem(
                        f"Another program has {self.path} open to write\nhelp: chassisdb-mcp "
                        "sessions take turns by themselves; close the other program"
                    ) from e
                time.sleep(0.02)

    def _mismatch(self, e: Exception) -> Problem:
        first = str(e).split("\n")[0]
        return Problem(
            f"The memory's index in {self.folder} doesn't fit its model: {first}\nhelp: rebuild "
            f"the index from the memories' text: chassisdb-mcp --dir {self.folder} reembed"
        )

    def _check_model(self, embedder: Embedder) -> None:
        """The folder still has `embedder`'s model: a reembed elsewhere may have moved it."""
        held = self._meta("embedder")
        if held is not None and held != embedder.spec:
            raise Problem(
                f"The memory in {self.folder} moved to {held} while this was embedded with "
                f"{embedder.spec}\nhelp: try again"
            )

    def _ids(self, tag: str | None) -> list[int]:
        if tag is None:
            return [row[0] for row in self.db.execute("SELECT id FROM memories")]
        # In id order, which Chassis searches without copying (ADR-0022).
        ids = [
            row[0] for row in self.db.execute(_TAGGED + " ORDER BY memories.id", (tag,))
        ]
        if not ids and self.db.execute("SELECT 1 FROM memories LIMIT 1").fetchone():
            counts = self._tag_counts()
            close = difflib.get_close_matches(
                tag.lower(), [t.lower() for t in counts], n=1
            )
            meant = next((t for t in counts if close and t.lower() == close[0]), None)
            tags = ", ".join(f"{t} ({n})" for t, n in counts.items())
            raise Problem(
                f"No memory has the tag {tag!r}\nhelp: "
                + (f"did you mean {meant!r}? " if meant else "")
                + (f"the tags are {tags}" if tags else "no memory has a tag yet")
            )
        return ids

    def _tag_counts(self) -> dict[str, int]:
        return dict(
            self.db.execute(
                "SELECT json_each.value, count(*) FROM memories, json_each(memories.tags) "
                "GROUP BY json_each.value ORDER BY count(*) DESC, json_each.value"
            ).fetchall()
        )

    def _count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM memories").fetchone()[0])

    def _no_index(self, count: int) -> Problem:
        return Problem(
            f"The {count} memories in {self.folder} have no index ({self.path.name} is missing)\n"
            f"help: rebuild it from their text: chassisdb-mcp --dir {self.folder} reembed"
        )

    def _rows(self, hits: list) -> list[Found]:
        """The memories `hits` name, nearest first."""
        if not hits:
            return []
        ids = [hit.id for hit in hits]
        rows = {
            row[0]: row
            for row in self.db.execute(
                f"SELECT id, text, tags, created FROM memories WHERE id IN "
                f"({','.join('?' * len(ids))})",
                ids,
            )
        }
        return [
            {**_memory(rows[hit.id]), "distance": round(float(hit.distance), 4)}
            for hit in hits
            if hit.id in rows
        ]

    def remember(self, text: str, tags: list[str] | str | None = None) -> Stored:
        text, tags = text.strip(), _tags(tags)
        if not text:
            raise Problem("The memory is empty\nhelp: pass the text to remember")
        embedder = self.embedder()
        vector = embedder([text])[0]  # Outside the lock: a model can take seconds
        with self.lock, self._writing():
            self._check_model(embedder)
            others = self._count()
            if self._meta("embedder") is None:
                self._set_meta("embedder", embedder.spec)
                self._set_meta("dimensions", len(vector))
            elif others and not self.path.exists():
                raise self._no_index(others)
            id = self.db.execute(
                "INSERT INTO memories (text, tags) VALUES (?, ?)",
                (text, json.dumps(tags)),
            ).lastrowid
            with self._open(len(vector)) as index:
                # A vector a crash left under this id has no row yet; _rows drops any other.
                nearest = index.search(vector, k=3, excluded=[id]) if others else []
                if id in index:  # A vector a crash left before its row was saved
                    index.delete(id)
                index.add(vector, id=id)
                index.flush()
            stored: Stored = {"id": id, "similar": self._rows(nearest)}
        if len(text) > LONG_TEXT:
            stored["warning"] = (
                f"This memory is {len(text)} characters long; most embedding models read only the "
                "first few hundred words, so the rest can't be found. Store one fact per memory."
            )
        return stored

    def recall(self, query: str, k: int = 5, tag: str | None = None) -> list[Found]:
        if not query.strip():
            raise Problem("The query is empty\nhelp: pass what to look for")
        embedder = self.embedder()
        vector = embedder([query])[0]
        with self.lock:
            self._check_model(embedder)
            # Every memory needs no filter: _rows drops a vector a crash left without its row.
            ids = self._ids(tag) if tag is not None else None
            count = len(ids) if ids is not None else self._count()
            if not count:
                return []
            if not self.path.exists():
                raise self._no_index(count)
            try:
                with self._open(
                    int(self._meta("dimensions") or 0), read_only=True
                ) as reader:
                    hits = reader.search(vector, k=k, allowed=ids)
            except DimensionMismatchError as e:
                raise self._mismatch(e) from e
            return self._rows(hits)

    def forget(self, id: int | str) -> Forgotten:
        if isinstance(id, str) and id.lstrip("#").isdigit():
            id = int(id.lstrip("#"))  # As ids are shown: "#23"
        with self.lock, self._writing():
            row = None
            if isinstance(id, int) and 0 <= id < 2**63:
                row = self.db.execute(
                    "SELECT id, text, tags, created FROM memories WHERE id = ?", (id,)
                ).fetchone()
            if row is None:
                raise Problem(
                    f"No memory has id {id}\nhelp: recall, or list_memories (chassisdb-mcp list), "
                    "shows the ids there are"
                )
            dimensions = self._meta("dimensions")
            if dimensions is not None and self.path.exists():
                with self._open(int(dimensions)) as index:
                    index.delete(id)
                    index.flush()
            self.db.execute("DELETE FROM memories WHERE id = ?", (id,))
            return {"forgotten": _memory(row)}

    def overview(
        self, tag: str | None = None, limit: int = 20, offset: int = 0
    ) -> Overview:
        with self.lock:
            where, args = (
                (f"WHERE id IN ({_TAGGED})", (tag,)) if tag is not None else ("", ())
            )
            total = self.db.execute(
                f"SELECT count(*) FROM memories {where}", args
            ).fetchone()[0]
            rows = self.db.execute(
                f"SELECT id, text, tags, created FROM memories {where} ORDER BY id DESC "
                "LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            ids = set(self._ids(None))
            indexed: set[int] = set()
            dimensions = self._meta("dimensions")
            if dimensions is not None and self.path.exists():
                with self._open(int(dimensions), read_only=True) as reader:
                    indexed = {int(i) for i in reader.ids()}
            return {
                "folder": str(self.folder),
                "embedder": self._meta("embedder") or self.default,
                "total": total,
                "tags": self._tag_counts(),
                "memories": [_memory(row) for row in rows],
                "more": total > offset + len(rows),
                "unindexed": len(ids - indexed),
                "stray": len(indexed - ids),
            }

    def reembed(self) -> int:
        """Moves every memory to the model this was opened with, from their text, and returns how
        many. Sessions keep writing while it embeds; what they wrote meanwhile is embedded while
        the new index replaces the old one, and they wait for that."""
        if self.default is None:
            raise Problem(
                "reembed needs the model to move the memories to\nhelp: name it, as in "
                f"chassisdb-mcp --embedder {SUGGESTED} reembed"
            )
        target = Embedder(self.default)
        with self.lock:
            first = self.db.execute(
                "SELECT id, text FROM memories ORDER BY id"
            ).fetchall()
        vectors: dict[int, np.ndarray] = {}
        for start in range(0, len(first), 64):
            batch = first[start : start + 64]
            vectors.update(
                zip([id for id, _ in batch], target([t for _, t in batch]), strict=True)
            )
            print(
                f"embedded {len(vectors)} of {len(first)}", file=sys.stderr, flush=True
            )
        fresh = self.folder / "memory-reembed.chassis"
        try:
            with self.lock, self._writing():
                rows = self.db.execute(
                    "SELECT id, text FROM memories ORDER BY id"
                ).fetchall()
                added = [(id, text) for id, text in rows if id not in vectors]
                if added:
                    vectors.update(
                        zip(
                            [id for id, _ in added],
                            target([t for _, t in added]),
                            strict=True,
                        )
                    )
                Path(f"{self.path}.undo").unlink(
                    missing_ok=True
                )  # The old index's, if any
                if rows:
                    ids = [id for id, _ in rows]
                    matrix = np.stack([vectors[id] for id in ids])
                    _remove_index(fresh)
                    with VectorIndex(
                        fresh, dimensions=matrix.shape[1], options=OPTIONS
                    ) as index:
                        index.add_batch(matrix, ids=ids)
                        index.flush()
                    _replace(fresh, self.path)
                    self._set_meta("dimensions", matrix.shape[1])
                else:
                    _remove_index(self.path)
                    self.db.execute("DELETE FROM meta WHERE key = 'dimensions'")
                self._set_meta("embedder", target.spec)
            return len(rows)
        finally:
            _remove_index(fresh)


def _remove_index(path: Path) -> None:
    for name in (path, Path(f"{path}.undo")):
        name.unlink(missing_ok=True)


def _replace(source: Path, target: Path) -> None:
    """Windows can't replace a file another session is reading this moment: try for 5 s."""
    for _ in range(50):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            time.sleep(0.1)
    raise Problem(
        f"{target} stayed open in another program for 5 s, so it couldn't be replaced\nhelp: close "
        "the agent sessions that use this memory, then run reembed again"
    )


_TAGGED = (
    "SELECT memories.id FROM memories, json_each(memories.tags) "
    "WHERE lower(json_each.value) = lower(?)"
)


def _memory(row: tuple) -> Remembered:
    id, text, tags, created = row
    return {"id": id, "text": text, "tags": json.loads(tags), "created": created}


def build_server(memory: "Memory | Problem"):  # -> MCPServer
    """The MCP server with its tools on `memory` (`_serve`). The MCP SDK takes half a second to
    import, so only serving imports it: the other commands start without it."""
    from chassisdb_mcp._serve import build_server as build

    return build(memory)


def _plural(n: int, noun: str) -> str:
    if n == 1:
        return f"1 {noun}"
    return f"{n} {noun[:-1]}ies" if noun.endswith("y") else f"{n} {noun}s"


def _show(value: object) -> None:
    """A command's result, for a person."""
    memories = value if isinstance(value, list) else value.get("memories", [])
    if isinstance(value, dict) and "total" in value:
        tags = ", ".join(f"{tag} ({n})" for tag, n in value["tags"].items()) or "none"
        print(f"{_plural(value['total'], 'memory')} in {value['folder']}")
        print(f"embedder: {value['embedder'] or 'not set yet'}")
        print(f"tags: {tags}")
        if value["unindexed"] or value["stray"]:
            print(
                f"\n{_plural(value['unindexed'], 'memory')} without a vector and "
                f"{_plural(value['stray'], 'vector')} without a memory, as a crash can leave\n"
                f"help: rebuild the index from the memories' text: chassisdb-mcp --dir "
                f"{value['folder']} reembed"
            )
        if memories:
            print()
    elif not memories:
        print("No memory matches.")
    for m in memories:
        distance = f"{m['distance']:.3f}  " if "distance" in m else ""
        tags = f"  [{', '.join(m['tags'])}]" if m["tags"] else ""
        print(f"{distance}#{m['id']}  {m['text']}{tags}  ({m['created']})")
    if memories and isinstance(value, dict) and value.get("more"):
        print("... more with --offset")


def main(argv: list[str] | None = None) -> None:
    # The same options before the command or after it.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument(
        "--embedder",
        metavar="KIND:MODEL",
        help="the model for a folder with none yet, and the one reembed moves to "
        "(or CHASSISDB_MCP_EMBEDDER)",
    )
    common.add_argument(
        "--dir",
        type=Path,
        help="the folder memories are kept in (or CHASSISDB_MCP_DIR; default ~/.chassisdb-mcp)",
    )
    common.add_argument("--json", action="store_true", help="print results as JSON")
    parser = argparse.ArgumentParser(
        prog="chassisdb-mcp",
        parents=[common],
        description=(
            "Long-term memory for agents on this machine. With no command, it runs as an MCP "
            "server over stdio; the commands show and change what it remembers."
        ),
        epilog=(
            f"embedders (--embedder <kind>:<model>):\n{EMBEDDERS}\n\nA folder keeps the model "
            "its memories were made with; reembed moves them to another.\n\nexample:\n  claude "
            "mcp add --scope user chassis-memory -- uvx --from 'chassisdb-mcp[fastembed]' "
            f"chassisdb-mcp --embedder {SUGGESTED}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", metavar="command")

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help, parents=[common])

    command("serve", "run the MCP server over stdio, as no command does")
    command("status", "the folder, its model, how many memories and tags")
    listing = command("list", "the memories, newest first")
    listing.add_argument("--tag")
    listing.add_argument("-n", "--limit", type=int, default=20)
    listing.add_argument("--offset", type=int, default=0)
    recall = command("recall", "the memories closest in meaning to a query")
    recall.add_argument("query")
    recall.add_argument("-k", type=int, default=5)
    recall.add_argument("--tag")
    remember = command("remember", "store a memory")
    remember.add_argument("text")
    remember.add_argument("--tag", action="append", default=[], dest="tags")
    forget = command("forget", "delete a memory by id")
    forget.add_argument("id", help="as shown: 23 or #23")
    command("reembed", "move every memory to --embedder's model")
    args = parser.parse_args(argv)
    # Not set_defaults: the subcommands share these options, defaults and all.
    args.embedder = getattr(args, "embedder", os.environ.get("CHASSISDB_MCP_EMBEDDER"))
    args.dir = getattr(
        args, "dir", Path(os.environ.get("CHASSISDB_MCP_DIR", DEFAULT_DIR))
    )
    args.json = getattr(args, "json", False)
    serving = args.command in (None, "serve")

    global VERBOSE
    VERBOSE = serving or args.command == "reembed"
    try:
        memory: Memory | Problem = Memory(args.dir, args.embedder)
    except Problem as e:
        if not serving:
            print(f"chassisdb-mcp: error: {e}", file=sys.stderr)
            sys.exit(2)
        log(str(e).replace("\n", " "))
        memory = e
    if serving:
        # The SDK logs at INFO; a model's download would log every HTTP request.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        if isinstance(memory, Memory):
            overview = memory.overview(limit=0)
            log(
                f"{_plural(overview['total'], 'memory')} in {memory.folder}; embedder "
                f"{overview['embedder'] or 'not set: the tools say how to set one'}"
            )
            try:
                memory.embedder().start()
            except Problem:
                pass  # The tools say what's missing
        build_server(memory).run()
        return
    run = {
        "status": lambda: memory.overview(limit=0),
        "list": lambda: memory.overview(args.tag, args.limit, args.offset),
        "recall": lambda: memory.recall(args.query, args.k, args.tag),
        "remember": lambda: memory.remember(args.text, args.tags),
        "forget": lambda: memory.forget(args.id),
        "reembed": lambda: {"moved": memory.reembed(), "embedder": args.embedder},
    }[args.command]
    try:
        result = run()
    except (Problem, ChassisError, sqlite3.Error, OSError) as e:
        print(f"chassisdb-mcp: error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "remember":
        print(f"remembered #{result['id']}")
        if result.get("warning"):
            print(f"warning: {result['warning']}")
        if result["similar"]:
            print("\nsimilar memories already kept:")
            _show(result["similar"])
    elif args.command == "forget":
        print(f"forgot #{result['forgotten']['id']}: {result['forgotten']['text']}")
    elif args.command == "reembed":
        print(f"moved {_plural(result['moved'], 'memory')} to {result['embedder']}")
    else:
        _show(result)


if __name__ == "__main__":
    main()
