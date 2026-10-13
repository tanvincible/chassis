"""chassisdb-mcp: long-term memory for agents on this machine, as an MCP server and a command line.

Each memory's text and tags are in SQLite (`memory.db`) and its embedding in a Chassis index
(`memory.chassis`), under the same id, in one folder. Any number of sessions can share the folder:
a write holds the index for milliseconds, and searches read the last write without a lock.
"""

import argparse
import difflib
import functools
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing_extensions import TypedDict  # pydantic needs this one before Python 3.12

import numpy as np
from chassisdb import ChassisError, IndexLockedError, IndexOptions, VectorIndex
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

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
# How long a write waits for another session's write to finish.
LOCK_WAIT_SECONDS = 10.0


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


class Forgotten(TypedDict):
    forgotten: Remembered


class Overview(TypedDict):
    folder: str
    embedder: str | None
    total: int
    tags: dict[str, int]
    memories: list[Remembered]
    more: bool


class Problem(Exception):
    """Something the user can fix: what was wrong, then what to do on a `help:` line."""


def log(message: str) -> None:
    """MCP clients keep a server's stderr in their logs; stdout carries the protocol."""
    print(f"chassisdb-mcp: {message}", file=sys.stderr, flush=True)


class Embedder:
    """The embedding model `spec` names, as <kind>:<model>. It loads on first use, or on a thread
    from `start()`, so the server answers at once even while a model downloads."""

    def __init__(self, spec: str):
        kind, _, model = spec.partition(":")
        if kind not in KINDS or not model:
            guess = difflib.get_close_matches(kind, KINDS, n=1, cutoff=0.5)
            did_you_mean = f"did you mean {guess[0]}:{model or '<model>'}? " if guess else ""
            raise Problem(
                f"The embedder is {spec!r}, but has to be <kind>:<model>\nhelp: {did_you_mean}"
                f"pick one of\n{EMBEDDERS}"
            )
        self.spec, self.kind, self.model = spec, kind, model
        self._embed: Callable[[list[str]], object] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        def load() -> None:
            try:
                self._load()
            except Exception as e:  # The first call that needs the model reports it again.
                log(str(e).replace("\n", " "))

        threading.Thread(target=load, daemon=True).start()

    def _load(self) -> Callable[[list[str]], object]:
        with self._lock:
            if self._embed is None:
                self._embed = LOADERS[self.kind](self.model)
            return self._embed

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load()(texts), dtype=np.float32)


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
    log(f"loading {model} (the first time, it downloads from Hugging Face)")
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
    log(f"loading {model} (the first time, it downloads from Hugging Face)")
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
    if not key and "api.openai.com" in base:
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
                data = json.load(response)["data"]
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
        except urllib.error.URLError as e:
            raise Problem(
                f"Nothing answered at {base} ({e.reason})\nhelp: start the server (Ollama: ollama "
                "serve), or set OPENAI_BASE_URL to where it listens"
            ) from None
        return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]

    return embed


LOADERS = {
    "fastembed": _fastembed,
    "sentence-transformers": _sentence_transformers,
    "openai": _openai,
}


class Memory:
    """The memories in `folder`. The folder keeps the embedder its memories were made with: with
    none given, that one is used, and another is refused until `reembed` moves them to it."""

    def __init__(self, folder: Path, embedder: Embedder | None = None):
        self.folder = folder.expanduser()
        self.path = self.folder / "memory.chassis"
        self.embedder = embedder
        self.explicit = embedder is not None
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise Problem(
                f"The memory folder {self.folder} can't be made: {e.strerror}\nhelp: pass another "
                "with --dir, or set CHASSISDB_MCP_DIR"
            ) from e
        # MCP runs tools on worker threads: one at a time, through `lock`.
        self.lock = threading.Lock()
        self.db = sqlite3.connect(
            self.folder / "memory.db", isolation_level=None, timeout=30, check_same_thread=False
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

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: object) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    def _embedder(self) -> Embedder:
        held = self._meta("embedder")
        if not self.explicit and held and (self.embedder is None or self.embedder.spec != held):
            self.embedder = Embedder(held)  # The folder's, which a reembed may have changed
        if self.embedder is None:
            raise Problem(
                f"No embedding model is set for the memory in {self.folder}\nhelp: start the "
                f"server with --embedder <kind>:<model>, such as --embedder {SUGGESTED}; "
                "chassisdb-mcp --help lists the kinds"
            )
        if held is None:
            self.db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('embedder', ?)", (self.embedder.spec,)
            )
            held = self._meta("embedder")
        if held != self.embedder.spec:
            raise Problem(
                f"The memory in {self.folder} was made with {held}, and {self.embedder.spec}'s "
                f"vectors can't be compared with its\nhelp: use --embedder {held}, or leave "
                f"--embedder out to use it; to move every memory to {self.embedder.spec}, run "
                f"chassisdb-mcp --dir {self.folder} --embedder {self.embedder.spec} reembed"
            )
        return self.embedder

    @contextmanager
    def _writing(self, dimensions: int) -> Iterator[VectorIndex]:
        """The index, opened to write. Another session's write is waited for: writes take
        milliseconds."""
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                index = VectorIndex(self.path, dimensions=dimensions, options=OPTIONS)
                break
            except IndexLockedError as e:
                if time.monotonic() > deadline:
                    raise Problem(
                        f"Another process has held the memory in {self.folder} for "
                        f"{LOCK_WAIT_SECONDS:.0f} s, longer than any write takes\nhelp: a reembed "
                        "may be running there; try again when it is done"
                    ) from e
                time.sleep(0.02)
        try:
            yield index
        finally:
            index.close()

    def _rows(self, hits: list) -> list[Found]:
        """The memories `hits` name, nearest first. One whose row is gone was forgotten, or never
        finished being remembered."""
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

    def remember(self, text: str, tags: list[str]) -> Stored:
        with self.lock:
            text, tags = text.strip(), sorted({tag.strip() for tag in tags if tag.strip()})
            if not text:
                raise Problem("The memory is empty\nhelp: pass the text to remember")
            vector = self._embedder()([text])[0]
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('dimensions', ?)", (len(vector),))
            id = self.db.execute(
                "INSERT INTO memories (text, tags) VALUES (?, ?)", (text, json.dumps(tags))
            ).lastrowid
            try:
                with self._writing(len(vector)) as index:
                    index.add(vector, id=id)
                    index.flush()
                    nearest = index.search(vector, k=4, excluded=[id]) if len(index) > 1 else []
            except BaseException:
                self.db.execute("DELETE FROM memories WHERE id = ?", (id,))
                raise
            return {"id": id, "similar": self._rows(nearest[:3])}

    def recall(self, query: str, k: int = 5, tag: str | None = None) -> list[Found]:
        with self.lock:
            if not query.strip():
                raise Problem("The query is empty\nhelp: pass what to look for")
            embed = self._embedder()
            dimensions = self._meta("dimensions")
            if dimensions is None or not self.path.exists():
                return []
            allowed = None
            if tag is not None:
                allowed = [row[0] for row in self.db.execute(_TAGGED, (tag,))]
                if not allowed:
                    return []
            vector = embed([query])[0]
            with VectorIndex(self.path, dimensions=int(dimensions), read_only=True) as reader:
                hits = reader.search(vector, k=k, allowed=allowed)
            return self._rows(hits)

    def forget(self, id: int) -> Forgotten:
        with self.lock:
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
                with self._writing(int(dimensions)) as index:
                    index.delete(id)
                    index.flush()
            self.db.execute("DELETE FROM memories WHERE id = ?", (id,))
            return {"forgotten": _memory(row)}

    def overview(self, tag: str | None = None, limit: int = 20, offset: int = 0) -> Overview:
        with self.lock:
            where, args = (f"WHERE id IN ({_TAGGED})", (tag,)) if tag is not None else ("", ())
            total = self.db.execute(f"SELECT count(*) FROM memories {where}", args).fetchone()[0]
            rows = self.db.execute(
                f"SELECT id, text, tags, created FROM memories {where} ORDER BY id DESC "
                "LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            tags = dict(
                self.db.execute(
                    "SELECT json_each.value, count(*) FROM memories, json_each(memories.tags) "
                    "GROUP BY json_each.value ORDER BY count(*) DESC, json_each.value"
                ).fetchall()
            )
            held = self._meta("embedder") or (self.embedder.spec if self.embedder else None)
            return {
                "folder": str(self.folder),
                "embedder": held,
                "total": total,
                "tags": tags,
                "memories": [_memory(row) for row in rows],
                "more": total > offset + len(rows),
            }

    def reembed(self) -> int:
        """Moves every memory to the embedder this was opened with, and returns how many."""
        with self.lock:
            if self.embedder is None:
                raise Problem(
                    "reembed needs the model to move the memories to\nhelp: name it, as in "
                    f"chassisdb-mcp --embedder {SUGGESTED} reembed"
                )
            rows = self.db.execute("SELECT id, text FROM memories ORDER BY id").fetchall()
            held, dimensions = self._meta("embedder"), self._meta("dimensions")
            fresh = self.folder / "memory-reembed.chassis"
            vectors = np.zeros((0, 0), dtype=np.float32)
            if rows:
                batches = []
                for start in range(0, len(rows), 64):
                    batches.append(self.embedder([text for _, text in rows[start : start + 64]]))
                    log(f"embedded {min(start + 64, len(rows))} of {len(rows)}")
                vectors = np.concatenate(batches)
                fresh.unlink(missing_ok=True)
                with VectorIndex(fresh, dimensions=vectors.shape[1], options=OPTIONS) as index:
                    index.add_batch(vectors, ids=[id for id, _ in rows])
                    index.flush()
            # Waits for a write under way elsewhere; a later one waits on the new index.
            if dimensions is not None and self.path.exists():
                with self._writing(int(dimensions)):
                    pass
            if rows:
                os.replace(fresh, self.path)
                self._set_meta("dimensions", vectors.shape[1])
            self._set_meta("embedder", self.embedder.spec)
            log(f"moved {len(rows)} memories from {held or 'no embedder'} to {self.embedder.spec}")
            return len(rows)


_TAGGED = "SELECT memories.id FROM memories, json_each(memories.tags) WHERE json_each.value = ?"


def _memory(row: tuple) -> Remembered:
    id, text, tags, created = row
    return {"id": id, "text": text, "tags": json.loads(tags), "created": created}


def _tool(fn: Callable) -> Callable:
    """MCP shows the model only "Error executing tool" for an exception other than a ToolError:
    this hands it the message, help line and all."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (Problem, ChassisError) as e:
            raise ToolError(str(e)) from e
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return run


def build_server(memory: Memory) -> MCPServer:
    server = MCPServer(
        "chassis-memory",
        instructions=(
            "Long-term memory that lasts across sessions, on this machine: use it for anything to "
            "keep for later sessions. Recall before answering "
            "anything earlier sessions may have settled, such as the user's preferences or a "
            "project's decisions. Remember what is worth knowing next time, one fact per memory, "
            "written to stand on its own. When remember returns a similar memory that the new one "
            "repeats or makes wrong, forget it."
        ),
    )

    @server.tool()
    @_tool
    def remember(text: str, tags: list[str] | None = None) -> Stored:
        """Store a memory: one fact, decision, preference or note worth knowing in later sessions,
        written to stand on its own. Tags, such as a project's name, let recall look among them
        only. Returns its id and the most similar memories already kept, with their distance
        (0 is the same meaning): forget one that the new memory repeats or makes wrong."""
        return memory.remember(text, tags or [])

    @server.tool()
    @_tool
    def recall(query: str, k: int = 5, tag: str | None = None) -> list[Found]:
        """Find the k memories closest in meaning to the query, nearest first: their ids, text,
        tags, when each was made, and distance (smaller is closer; 0 is the same meaning). With a
        tag, only memories that have it."""
        return memory.recall(query, k, tag)

    @server.tool()
    @_tool
    def forget(id: int) -> Forgotten:
        """Delete the memory with this id, as recall or list_memories showed it. Returns what was
        forgotten."""
        return memory.forget(id)

    @server.tool()
    @_tool
    def list_memories(tag: str | None = None, limit: int = 20, offset: int = 0) -> Overview:
        """The memories, newest first, with how many there are, every tag and its count, the
        folder they are kept in and the embedding model: to show the user what is remembered."""
        return memory.overview(tag, limit, offset)

    return server


def _show(value: object) -> None:
    """A command's result, for a person."""
    memories = value if isinstance(value, list) else value.get("memories", [])
    if isinstance(value, dict) and "total" in value:
        tags = ", ".join(f"{tag} ({n})" for tag, n in value["tags"].items()) or "none"
        print(f"{value['total']} memories in {value['folder']}")
        print(f"embedder: {value['embedder'] or 'not set yet'}")
        print(f"tags: {tags}")
        if memories:
            print()
    for m in memories:
        distance = f"{m['distance']:.3f}  " if "distance" in m else ""
        tags = f"  [{', '.join(m['tags'])}]" if m["tags"] else ""
        print(f"{distance}#{m['id']}  {m['text']}{tags}  ({m['created']})")
    if memories and isinstance(value, dict) and value.get("more"):
        print("... more with --offset")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chassisdb-mcp",
        description=(
            "Long-term memory for agents on this machine. With no command, it runs as an MCP "
            "server over stdio; the commands show and change what it remembers."
        ),
        epilog=(
            f"embedders (--embedder <kind>:<model>):\n{EMBEDDERS}\n\nThe first run in a folder "
            "needs --embedder; later runs use the folder's.\n\nexample:\n  claude mcp add --scope "
            "user chassis-memory -- uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp "
            f"--embedder {SUGGESTED}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=version("chassisdb-mcp"))
    parser.add_argument(
        "--embedder",
        default=os.environ.get("CHASSISDB_MCP_EMBEDDER"),
        metavar="KIND:MODEL",
        help="the embedding model (or CHASSISDB_MCP_EMBEDDER); by default, the folder's",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(os.environ.get("CHASSISDB_MCP_DIR", DEFAULT_DIR)),
        help="the folder memories are kept in (or CHASSISDB_MCP_DIR; default ~/.chassisdb-mcp)",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser("serve", help="run the MCP server over stdio, as no command does")
    commands.add_parser("status", help="the folder, its embedder, how many memories and tags")
    listing = commands.add_parser("list", help="the memories, newest first")
    listing.add_argument("--tag")
    listing.add_argument("-n", "--limit", type=int, default=20)
    listing.add_argument("--offset", type=int, default=0)
    recall = commands.add_parser("recall", help="the memories closest in meaning to a query")
    recall.add_argument("query")
    recall.add_argument("-k", type=int, default=5)
    recall.add_argument("--tag")
    remember = commands.add_parser("remember", help="store a memory")
    remember.add_argument("text")
    remember.add_argument("--tag", action="append", default=[], dest="tags")
    forget = commands.add_parser("forget", help="delete a memory by id")
    forget.add_argument("id", type=int)
    commands.add_parser("reembed", help="move every memory to --embedder's model")
    args = parser.parse_args(argv)

    try:
        memory = Memory(args.dir, Embedder(args.embedder) if args.embedder else None)
    except Problem as e:
        print(f"chassisdb-mcp: error: {e}", file=sys.stderr)
        sys.exit(2)
    if args.command in (None, "serve"):
        # The SDK logs at INFO; a model's download would log every HTTP request.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        overview = memory.overview(limit=0)
        held = overview["embedder"]
        log(
            f"{overview['total']} memories in {memory.folder}; embedder "
            f"{held or 'not set: the tools say how to set one'}"
        )
        if memory.embedder is None and held:
            memory.embedder = Embedder(held)
        if memory.embedder is not None:
            memory.embedder.start()
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
    except (Problem, ChassisError) as e:
        print(f"chassisdb-mcp: error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "remember":
        print(f"remembered #{result['id']}")
        if result["similar"]:
            print("\nsimilar memories already kept:")
            _show(result["similar"])
    elif args.command == "forget":
        print(f"forgot #{result['forgotten']['id']}: {result['forgotten']['text']}")
    elif args.command == "reembed":
        print(f"moved {result['moved']} memories to {result['embedder']}")
    else:
        _show(result)


if __name__ == "__main__":
    main()
