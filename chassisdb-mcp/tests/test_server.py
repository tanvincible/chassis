import hashlib
import json
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anyio
import numpy as np
import pytest
from chassisdb import IndexOptions, VectorIndex
from mcp import Client

from chassisdb_mcp import server
from chassisdb_mcp.server import Embedder, Memory, Problem, build_server


def hashed(dimensions):
    """Stands in for a model: words of 4 letters or more, hashed into buckets, so texts that share
    words are near."""

    def embed(texts):
        vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]{4,}", text.lower()):
                bucket = hashlib.blake2b(word.encode(), digest_size=2).digest()
                vectors[row, int.from_bytes(bucket, "little") % dimensions] += 1
        return vectors

    return embed


@pytest.fixture(autouse=True)
def models(monkeypatch):
    """`openai:<n>` embeds into n dimensions, with no API behind it."""
    monkeypatch.setitem(server.LOADERS, "openai", lambda model: hashed(int(model)))
    monkeypatch.delenv("CHASSISDB_MCP_EMBEDDER", raising=False)
    monkeypatch.delenv("CHASSISDB_MCP_DIR", raising=False)


def index_ids(folder, dimensions):
    with VectorIndex(folder / "memory.chassis", dimensions=dimensions, read_only=True) as index:
        return sorted(int(i) for i in index.ids())


def test_remember_recall_forget(tmp_path):
    memory = Memory(tmp_path, "openai:64")
    assert memory.recall("anything") == []
    assert memory.overview()["embedder"] == "openai:64"
    tea = memory.remember("The user drinks green tea, never coffee", ["prefs", " prefs "])
    assert tea["similar"] == []
    tabs = memory.remember("This project indents with tabs", "chassis")
    hits = memory.recall("what does the user drink, coffee or tea", 1)
    assert [(h["id"], h["text"], h["tags"]) for h in hits] == [
        (tea["id"], "The user drinks green tea, never coffee", ["prefs"])
    ]
    assert [h["id"] for h in memory.recall("coffee", 5, "CHASSIS")] == [tabs["id"]]
    with pytest.raises(Problem, match=r"No memory has the tag 'nope'\nhelp: the tags are chassis"):
        memory.recall("coffee", 5, "nope")
    with pytest.raises(Problem, match=r"help: did you mean 'chassis'\? the tags are"):
        memory.recall("coffee", 5, "chasis")

    # A new memory shows the old one it contradicts, to forget.
    black = memory.remember("The user drinks black coffee now, never green tea", ["prefs"])
    assert black["similar"][0]["id"] == tea["id"]
    assert memory.forget(tea["id"])["forgotten"]["text"].startswith("The user drinks green tea")
    for missing in (tea["id"], 10**21, -1):
        with pytest.raises(Problem, match=f"No memory has id {missing}\nhelp: recall, or"):
            memory.forget(missing)
    assert memory.overview()["total"] == 2
    with pytest.raises(Problem, match="The memory is empty"):
        memory.remember("   ", [])
    assert "warning" in memory.remember("word " * 600, [])


def test_a_new_folder_takes_no_model_until_its_first_memory(tmp_path):
    with pytest.raises(Problem, match="help: start the server with --embedder <kind>:<model>"):
        Memory(tmp_path).remember("anything")
    Memory(tmp_path, "openai:64").recall("nothing here yet")
    assert Memory(tmp_path).overview()["embedder"] is None
    for spec, hint in [("fastembd:x", "did you mean fastembed:x?"), ("openai", "did you mean")]:
        with pytest.raises(Problem, match=hint):
            Embedder(spec)


def test_sessions_share_a_folder(tmp_path):
    first, second = Memory(tmp_path, "openai:64"), Memory(tmp_path)
    a = first.remember("Deploys go out on Thursdays")["id"]
    # The second session named no model: it uses the folder's.
    assert [h["id"] for h in second.recall("when do deploys go out", 1)] == [a]
    b = second.remember("Releases need a changelog entry")["id"]
    assert {h["id"] for h in first.recall("deploys releases")} == {a, b}

    # A write waits while another session's write holds memory.db.
    other = sqlite3.connect(tmp_path / "memory.db", isolation_level=None, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    threading.Timer(0.3, other.execute, ("COMMIT",)).start()
    c = second.remember("Waited for the other session")["id"]
    assert c in {h["id"] for h in first.recall("waited other session")}


def test_the_folder_decides_the_model_and_reembed_moves_it(tmp_path):
    old = Memory(tmp_path, "openai:64")
    ids = [old.remember(f"note about {topic}")["id"] for topic in ("cats", "dogs", "tea")]
    # Another model named for this folder is set aside for the folder's.
    assert Memory(tmp_path, "openai:32").recall("cats", 1)[0]["id"] == ids[0]

    assert Memory(tmp_path, "openai:32").reembed() == 3
    assert index_ids(tmp_path, 32) == ids
    # Every session follows the folder, whatever it was started with.
    assert old.recall("cats", 1)[0]["id"] == ids[0]
    assert old.overview()["embedder"] == "openai:32"


def test_writes_during_a_reembed_are_kept(tmp_path, monkeypatch):
    memory = Memory(tmp_path, "openai:64")
    keep = memory.remember("bananas are yellow")["id"]
    drop = memory.remember("plums are purple")["id"]
    other = Memory(tmp_path)

    def slow(model):
        embed = hashed(int(model))

        def during(texts):
            if texts == ["bananas are yellow", "plums are purple"]:  # The first pass
                other.remember("cherries are red")
                other.forget(drop)
            return embed(texts)

        return during

    monkeypatch.setitem(server.LOADERS, "openai", slow)
    assert Memory(tmp_path, "openai:32").reembed() == 2
    added = [h["id"] for h in memory.recall("cherries red", 1)]
    assert index_ids(tmp_path, 32) == sorted([keep, *added])
    assert memory.overview()["unindexed"] == memory.overview()["stray"] == 0


def test_a_reembed_waits_for_a_reader_to_let_go(tmp_path):
    # Windows can't replace a file that is open; another session's search holds it briefly.
    memory = Memory(tmp_path, "openai:64")
    memory.remember("held open")
    reader = VectorIndex(tmp_path / "memory.chassis", dimensions=64, read_only=True)
    threading.Timer(0.5, reader.close).start()
    assert Memory(tmp_path, "openai:32").reembed() == 1
    assert memory.recall("held open", 1)[0]["text"] == "held open"


def test_a_reembed_of_nothing_frees_the_folder(tmp_path):
    memory = Memory(tmp_path, "openai:64")
    memory.forget(memory.remember("soon gone")["id"])
    assert Memory(tmp_path, "openai:32").reembed() == 0
    assert Memory(tmp_path).remember("now in 32 dimensions")["similar"] == []
    assert index_ids(tmp_path, 32) == [2]


def test_what_a_crash_leaves_is_found_and_mended(tmp_path):
    memory = Memory(tmp_path, "openai:64")
    a = memory.remember("apples are red")["id"]
    # A vector whose row was never saved, under the id the next memory will get, and another.
    with VectorIndex(tmp_path / "memory.chassis", 64, IndexOptions(metric="cosine")) as index:
        index.add(hashed(64)(["apples are red"])[0], id=a + 1)
        index.add(hashed(64)(["apples are red"])[0], id=99)
    assert [h["id"] for h in memory.recall("apples red")] == [a]  # Strays take no place
    assert memory.overview()["stray"] == 2
    b = memory.remember("apples are green")["id"]
    assert b == a + 1 and memory.overview()["stray"] == 1

    (tmp_path / "memory.chassis").unlink()
    with pytest.raises(
        Problem, match=r"have no index \(memory\.chassis is missing\)\nhelp: .*reembed"
    ):
        memory.recall("apples")
    assert memory.overview()["unindexed"] == 2
    Memory(tmp_path, "openai:64").reembed()
    assert {h["id"] for h in memory.recall("apples")} == {a, b}


def test_a_model_that_changed_dimensions_says_to_reembed(tmp_path, monkeypatch):
    Memory(tmp_path, "openai:64").remember("first")
    monkeypatch.setitem(server.LOADERS, "openai", lambda model: hashed(32))
    for act in (lambda m: m.remember("second"), lambda m: m.recall("first")):
        with pytest.raises(Problem, match="doesn't fit its model.*\nhelp: .*reembed"):
            act(Memory(tmp_path))


def call(memory_server, tool, arguments):
    async def session():
        async with Client(memory_server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool(tool, arguments)
            return names, result

    return anyio.run(session)


def test_the_tools_and_their_errors(tmp_path):
    mcp = build_server(Memory(tmp_path, "openai:64"))
    names, result = call(mcp, "remember", {"text": "The user's name is Sam", "tags": ["user"]})
    assert names == {"remember", "recall", "forget", "list_memories"}
    assert not result.is_error
    _, found = call(mcp, "recall", {"query": "name of the user"})
    assert "Sam" in json.dumps(found.structured_content)
    _, listed = call(mcp, "list_memories", {})
    assert listed.structured_content["tags"] == {"user": 1}

    # The model reads the error and what to do, not just "Error executing tool".
    _, failed = call(build_server(Memory(tmp_path / "new")), "remember", {"text": "x"})
    assert failed.is_error
    assert "help: start the server with --embedder" in failed.content[0].text
    _, failed = call(mcp, "recall", {"query": "x", "k": 0})
    assert "k is 0, but has to be at least 1" in failed.content[0].text
    _, failed = call(mcp, "recall", {"k": "five"})
    assert failed.content[0].text == (
        "The arguments don't fit recall: query: Field required; k: "
        "Input should be a valid integer, unable to parse string as an integer (given 'five')\n"
        "help: pass them as the tool's input schema says"
    )
    # A server set up wrong still starts, and its tools say what to fix.
    broken = build_server(Problem("The embedder is 'x'\nhelp: pick one"))
    _, failed = call(broken, "recall", {"query": "x"})
    assert failed.content[0].text.endswith("The embedder is 'x'\nhelp: pick one")


def test_the_command_line(tmp_path, capsys):
    def run(*argv):
        server.main(list(argv))
        return capsys.readouterr().out

    d = ("--dir", str(tmp_path))
    assert "remembered #1" in run(
        "remember", "Tabs, not spaces", "--tag", "style", *d, "--embedder", "openai:64"
    )
    similar = run(*d, "remember", "Spaces, not tabs, since October")
    assert re.search(
        r"similar memories already kept:\n0\.\d+  #1  Tabs, not spaces  \[style\]", similar
    )
    assert run("status", *d).startswith(
        f"2 memories in {tmp_path}\nembedder: openai:64\ntags: style (1)"
    )
    assert "#1  Tabs, not spaces  [style]" in run("list", "--tag", "style", *d)
    assert json.loads(run("recall", "tabs or spaces", "-k", "1", "--json", *d))[0]["id"] in (1, 2)
    empty = ("--dir", str(tmp_path / "empty"), "--embedder", "openai:64")
    assert run("recall", "anything", *empty) == "No memory matches.\n"
    assert run("forget", "#1", *d).startswith("forgot #1: Tabs, not spaces")
    assert run("status", *d).startswith("1 memory in")
    with pytest.raises(SystemExit):
        run("forget", "1", *d)
    assert "No memory has id 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run("status", "--embedder", "fasttext:model", *d)
    assert "did you mean fastembed:model?" in capsys.readouterr().err
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "memory.db").write_text("not a database")
    with pytest.raises(SystemExit):
        run("status", "--dir", str(tmp_path / "broken"))
    assert "can't be used as a memory" in capsys.readouterr().err


class FakeAPI(BaseHTTPRequestHandler):
    status, reply, asked = 200, None, []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.asked.append((self.path, self.headers.get("Authorization"), body))
        self.send_response(FakeAPI.status)
        self.end_headers()
        if FakeAPI.reply is not None:
            self.wfile.write(FakeAPI.reply)
        else:
            data = [{"index": i, "embedding": [1.0, float(i)]} for i in range(len(body["input"]))]
            self.wfile.write(json.dumps({"data": data[::-1]}).encode())

    def log_message(self, *args):
        pass


def test_openai_compatible_api(monkeypatch):
    monkeypatch.setitem(server.LOADERS, "openai", server._openai)
    http = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{http.server_port}/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", base + "/")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    assert Embedder("openai:some-model")(["a", "b"]).tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert FakeAPI.asked == [
        ("/v1/embeddings", "Bearer key", {"model": "some-model", "input": ["a", "b"]})
    ]

    FakeAPI.status, FakeAPI.reply = 404, b'{"error":"model \\"nomic\\" not found, try pulling it"}'
    with pytest.raises(Problem, match=r"answered 404 .*not found.*\nhelp: .*ollama pull nomic"):
        Embedder("openai:nomic")(["a"])
    FakeAPI.status = 401
    with pytest.raises(Problem, match="help: check OPENAI_API_KEY"):
        Embedder("openai:m")(["a"])
    FakeAPI.status, FakeAPI.reply = 200, b"<html>a login page</html>"
    with pytest.raises(Problem, match=r"no embeddings: <html>.*\nhelp: check that OPENAI_BASE_URL"):
        Embedder("openai:m")(["a"])
    FakeAPI.reply = b'{"data": [{"index": 0, "embedding": [1.0, 2.0]}]}'
    with pytest.raises(Problem, match="gave 1 vectors for 2 texts"):
        Embedder("openai:m")(["a", "b"])
    http.shutdown()
    http.server_close()
    with pytest.raises(Problem, match=f"Nothing answered at {base} .*\nhelp: start the server"):
        Embedder("openai:m")(["a"])

    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(Problem, match="OPENAI_API_KEY isn't set"):
        Embedder("openai:text-embedding-3-small")(["a"])
    # Only OpenAI's own host needs a key, wherever else its name appears.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/api.openai.com")
    with pytest.raises(Problem, match="Nothing answered"):
        Embedder("openai:m")(["a"])


def test_fastembed_suggests_the_model_meant():
    pytest.importorskip("fastembed")
    with pytest.raises(Problem, match="did you mean BAAI/bge-small-en-v1.5"):
        Embedder("fastembed:BAAI/bge-smol-en-v1.5")(["a"])
