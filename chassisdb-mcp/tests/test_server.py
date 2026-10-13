import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anyio
import numpy as np
import pytest
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


def test_remember_recall_forget(tmp_path):
    memory = Memory(tmp_path, Embedder("openai:64"))
    assert memory.recall("anything") == []
    tea = memory.remember("The user drinks green tea, never coffee", ["prefs", " prefs "])
    assert tea["similar"] == []
    tabs = memory.remember("This project indents with tabs", ["chassis"])
    hits = memory.recall("what does the user drink, coffee or tea", 1)
    assert [(h["id"], h["text"], h["tags"]) for h in hits] == [
        (tea["id"], "The user drinks green tea, never coffee", ["prefs"])
    ]
    assert [h["id"] for h in memory.recall("coffee", 5, "chassis")] == [tabs["id"]]
    assert memory.recall("coffee", 5, "no such tag") == []

    # A new memory shows the old one it contradicts, to forget.
    green = memory.remember("The user drinks black coffee now, never green tea", ["prefs"])
    assert green["similar"][0]["id"] == tea["id"]
    assert memory.forget(tea["id"])["forgotten"]["text"].startswith("The user drinks green tea")
    with pytest.raises(
        Problem, match=f"No memory has id {tea['id']}\nhelp: recall, or list_memories"
    ):
        memory.forget(tea["id"])
    assert memory.overview()["total"] == 2
    with pytest.raises(Problem, match="The memory is empty"):
        memory.remember("   ", [])


def test_sessions_share_a_folder(tmp_path):
    first, second = Memory(tmp_path, Embedder("openai:64")), Memory(tmp_path)
    a = first.remember("Deploys go out on Thursdays", [])["id"]
    # The second session named no embedder: it uses the folder's.
    assert [h["id"] for h in second.recall("when do deploys go out", 1)] == [a]
    b = second.remember("Releases need a changelog entry", [])["id"]
    assert {h["id"] for h in first.recall("deploys releases")} == {a, b}

    # A write waits while another session holds the index.
    held = first._writing(64)
    held.__enter__()
    threading.Timer(0.2, held.__exit__, (None, None, None)).start()
    c = second.remember("Waited for the lock", [])["id"]
    assert c in {h["id"] for h in first.recall("waited lock")}


def test_a_folder_keeps_its_embedder_until_reembed(tmp_path):
    old = Memory(tmp_path, Embedder("openai:64"))
    ids = [old.remember(f"note about {topic}", [])["id"] for topic in ("cats", "dogs", "tea")]
    with pytest.raises(Problem) as refused:
        Memory(tmp_path, Embedder("openai:32")).recall("cats")
    assert str(refused.value).endswith(
        f"to move every memory to openai:32, run chassisdb-mcp --dir {tmp_path} --embedder "
        "openai:32 reembed"
    )

    assert Memory(tmp_path, Embedder("openai:32")).reembed() == 3
    # The session that named the old model is refused now; one that named none follows along.
    assert "help: use --embedder openai:32, or leave --embedder out" in str(
        pytest.raises(Problem, old.recall, "cats").value
    )
    follower = Memory(tmp_path)
    assert follower.recall("cats", 1)[0]["id"] == ids[0]
    assert follower.overview()["embedder"] == "openai:32"


def test_no_embedder_says_how_to_set_one(tmp_path):
    with pytest.raises(Problem, match="help: start the server with --embedder <kind>:<model>"):
        Memory(tmp_path).remember("anything", [])
    for spec, hint in [("fastembd:x", "did you mean fastembed:x?"), ("openai", "did you mean")]:
        with pytest.raises(Problem, match=hint):
            Embedder(spec)


def call(memory_server, tool, arguments):
    async def session():
        async with Client(memory_server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool(tool, arguments)
            return names, result

    return anyio.run(session)


def test_the_tools_and_their_errors(tmp_path):
    mcp = build_server(Memory(tmp_path, Embedder("openai:64")))
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


def test_the_command_line(tmp_path, capsys):
    def run(*argv):
        server.main(["--dir", str(tmp_path), *argv])
        return capsys.readouterr().out

    assert "remembered #1" in run(
        "--embedder", "openai:64", "remember", "Tabs, not spaces", "--tag", "style"
    )
    similar = run("remember", "Spaces, not tabs, since October")
    assert re.search(
        r"similar memories already kept:\n0\.\d+  #1  Tabs, not spaces  \[style\]", similar
    )
    assert run("status").startswith(
        f"2 memories in {tmp_path}\nembedder: openai:64\ntags: style (1)"
    )
    assert "#1  Tabs, not spaces  [style]" in run("list", "--tag", "style")
    assert json.loads(run("--json", "recall", "tabs or spaces", "-k", "1"))[0]["id"] in (1, 2)
    assert run("forget", "1").startswith("forgot #1: Tabs, not spaces")
    with pytest.raises(SystemExit):
        run("forget", "1")
    assert "No memory has id 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run("--embedder", "fasttext:model", "status")
    assert "did you mean fastembed:model?" in capsys.readouterr().err


class FakeAPI(BaseHTTPRequestHandler):
    status, reply, asked = 200, b"", []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.asked.append((self.path, self.headers.get("Authorization"), body))
        self.send_response(FakeAPI.status)
        self.end_headers()
        if FakeAPI.status == 200:
            data = [{"index": i, "embedding": [1.0, float(i)]} for i in range(len(body["input"]))]
            self.wfile.write(json.dumps({"data": data[::-1]}).encode())
        else:
            self.wfile.write(FakeAPI.reply)

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
    http.shutdown()
    http.server_close()
    with pytest.raises(Problem, match=f"Nothing answered at {base} .*\nhelp: start the server"):
        Embedder("openai:m")(["a"])

    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(Problem, match="OPENAI_API_KEY isn't set"):
        Embedder("openai:text-embedding-3-small")(["a"])


def test_fastembed_suggests_the_model_meant():
    pytest.importorskip("fastembed")
    with pytest.raises(Problem, match="did you mean BAAI/bge-small-en-v1.5"):
        Embedder("fastembed:BAAI/bge-smol-en-v1.5")(["a"])
