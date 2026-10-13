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
from chassisdb_mcp.server import Memory, build_server, load_embedder


def embed(texts):
    """Words of 4 letters or more, hashed into 64 buckets: texts sharing words are near."""
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            vectors[row, hashlib.blake2b(word.encode(), digest_size=1).digest()[0] % 64] += 1
    return vectors


def test_remember_recall_forget(tmp_path):
    memory = Memory(tmp_path, "test:hash", embed)
    assert memory.recall("anything", 5, None) == []
    tea = memory.remember("The user drinks green tea, never coffee", ["prefs"])
    tabs = memory.remember("This project indents with tabs", ["chassis"])
    hits = memory.recall("what does the user drink, coffee or tea", 1, None)
    assert [(h["id"], h["text"], h["tags"]) for h in hits] == [
        (tea, "The user drinks green tea, never coffee", ["prefs"])
    ]
    assert [h["id"] for h in memory.recall("coffee", 5, "chassis")] == [tabs]
    assert memory.recall("coffee", 5, "no such tag") == []

    assert memory.forget(tea) is True
    assert memory.forget(tea) is False
    assert [h["id"] for h in memory.recall("tea coffee", 5, None)] == [tabs]


def test_sessions_share_a_folder(tmp_path):
    first, second = Memory(tmp_path, "test:hash", embed), Memory(tmp_path, "test:hash", embed)
    a = first.remember("Deploys go out on Thursdays", [])
    assert [h["id"] for h in second.recall("when do deploys go out", 1, None)] == [a]
    b = second.remember("Releases need a changelog entry", [])
    assert {h["id"] for h in first.recall("deploys releases", 5, None)} == {a, b}

    # A write waits while another session holds the index.
    holder = first._writer(64)
    threading.Timer(0.2, holder.close).start()
    c = second.remember("Waited for the lock", [])
    assert c in {h["id"] for h in first.recall("waited lock", 5, None)}


def test_another_embedder_is_refused(tmp_path):
    Memory(tmp_path, "test:hash", embed)
    with pytest.raises(ValueError, match="--embedder test:hash"):
        Memory(tmp_path, "openai:other", embed)


def test_the_server_has_three_tools(tmp_path):
    mcp = build_server(Memory(tmp_path, "test:hash", embed))

    async def session():
        async with Client(mcp) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            await client.call_tool("remember", {"text": "The user's name is Sam", "tags": []})
            found = await client.call_tool("recall", {"query": "name of the user"})
            return tools, json.dumps(found.structured_content or [c.text for c in found.content])

    tools, found = anyio.run(session)
    assert tools == {"remember", "recall", "forget"}
    assert "Sam" in found


def test_openai_compatible_api(tmp_path, monkeypatch):
    asked = []

    class Embeddings(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            asked.append((self.path, self.headers.get("Authorization"), body))
            data = [{"index": i, "embedding": [1.0, float(i)]} for i in range(len(body["input"]))]
            reply = json.dumps({"data": data[::-1]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    http = HTTPServer(("127.0.0.1", 0), Embeddings)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{http.server_port}/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    vectors = load_embedder("openai:some-model")(["a", "b"])
    http.shutdown()
    assert vectors.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert asked == [("/v1/embeddings", "Bearer key", {"model": "some-model", "input": ["a", "b"]})]


def test_the_embedder_has_to_be_named(capsys):
    for argv in ([], ["--embedder", "fasttext:model"], ["--embedder", "fastembed"]):
        with pytest.raises(SystemExit):
            server.main(argv)
        assert "sentence-transformers:<model>" in capsys.readouterr().err
