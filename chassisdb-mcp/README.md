# chassisdb-mcp

Long-term memory for coding agents, on your machine, as an MCP server. An agent remembers decisions,
preferences and facts in one session and recalls them by meaning in the next. Each memory's text is
kept in SQLite and its embedding in a [Chassis](https://github.com/tanvincible/chassis) index, in
one folder. Nothing leaves the machine unless the embedding model you pick is a remote API.

## Set it up

You pick the embedding model. Locally, with no PyTorch (about 70 MB, downloaded on first use):

```sh
claude mcp add --scope user chassis-memory -- \
  uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp --embedder fastembed:BAAI/bge-small-en-v1.5
```

With [Ollama](https://ollama.com) (`ollama pull nomic-embed-text` first):

```sh
claude mcp add --scope user chassis-memory -e OPENAI_BASE_URL=http://localhost:11434/v1 -- \
  uvx chassisdb-mcp --embedder openai:nomic-embed-text
```

Cursor, Codex, Claude Desktop and other clients take the same command in their MCP settings:

```json
{
  "mcpServers": {
    "chassis-memory": {
      "command": "uvx",
      "args": ["--from", "chassisdb-mcp[fastembed]", "chassisdb-mcp",
               "--embedder", "fastembed:BAAI/bge-small-en-v1.5"]
    }
  }
}
```

## Embedders

| `--embedder` | Runs | Install |
| --- | --- | --- |
| `fastembed:<model>`, e.g. `fastembed:BAAI/bge-small-en-v1.5` | locally, ONNX | `chassisdb-mcp[fastembed]` |
| `sentence-transformers:<model>`, e.g. `sentence-transformers:all-MiniLM-L6-v2` | locally, PyTorch | `chassisdb-mcp[sentence-transformers]` |
| `openai:<model>`, e.g. `openai:text-embedding-3-small` | any OpenAI-compatible API: OpenAI, Ollama, LM Studio, vLLM | nothing more; set `OPENAI_BASE_URL`, and `OPENAI_API_KEY` if it needs one |

A memory folder keeps the embedder it was made with: vectors from two models can't be compared, so
the server refuses another one there and says which to use.

## Tools

- `remember(text, tags=None)`: store a memory; returns its id. Tags, such as a project's name, let
  `recall` look among them only.
- `recall(query, k=5, tag=None)`: the `k` memories closest in meaning, nearest first, with id, text,
  tags, when each was made, and its distance (0 is the same meaning).
- `forget(id)`: delete a memory.

## Where memories live

In `~/.chassisdb-mcp`, or the folder `--dir` (or `CHASSISDB_MCP_DIR`) names: `memory.db` and
`memory.chassis`. Give a project a memory of its own with `--dir`. Several agent sessions can use
one folder at once: a write holds the index for milliseconds, and searches don't wait for it.
