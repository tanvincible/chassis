# chassisdb-mcp

Long-term memory for your coding agents, on your machine. Tell an agent something once, and the
next session, or another agent, finds it by meaning:

```text
session 1   you:    Keep in mind: I format Python with ruff, never black.
            agent:  remember("User preference: always format Python code with ruff (ruff format),
                             never black.", tags=["preferences", "python"])

session 2   you:    Which formatter should I set up in this new project?
            agent:  recall("Python formatter preference")  →  #1, at distance 0.229
                    Ruff, as you prefer: uv add --dev ruff, then ruff format .
```

An MCP server built on [Chassis](https://github.com/tanvincible/chassis): each memory's text is
kept in SQLite and its embedding in a Chassis index, in one folder you can read. Nothing leaves
your machine unless the embedding model you pick is a remote API.

## Set it up

One command, with a local model (about 70 MB, downloaded the first time; needs
[uv](https://docs.astral.sh/uv/)):

```sh
claude mcp add --scope user chassis-memory -- \
  uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp --embedder fastembed:BAAI/bge-small-en-v1.5
```

Claude Code keeps a memory of its own, and an agent uses that one when both are there. To have it
use this one, turn Claude Code's off in `~/.claude/settings.json`:

```json
{ "autoMemoryEnabled": false }
```

Other clients take the same command. Cursor (`~/.cursor/mcp.json`), Claude Desktop and most
others:

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

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.chassis-memory]
command = "uvx"
args = ["--from", "chassisdb-mcp[fastembed]", "chassisdb-mcp", "--embedder", "fastembed:BAAI/bge-small-en-v1.5"]
```

## Pick the embedding model

| `--embedder` | Runs | Needs |
| --- | --- | --- |
| `fastembed:<model>`, such as `fastembed:BAAI/bge-small-en-v1.5` | on your machine, ONNX | `chassisdb-mcp[fastembed]` |
| `sentence-transformers:<model>`, such as `sentence-transformers:all-MiniLM-L6-v2` | on your machine, PyTorch | `chassisdb-mcp[sentence-transformers]` |
| `openai:<model>`, such as `openai:text-embedding-3-small` | any OpenAI-compatible API: OpenAI, Ollama, LM Studio, vLLM | nothing more |

For an API, `OPENAI_BASE_URL` says where it is (OpenAI's by default) and `OPENAI_API_KEY` gives
the key. With [Ollama](https://ollama.com), after `ollama pull nomic-embed-text`:

```sh
claude mcp add --scope user chassis-memory -e OPENAI_BASE_URL=http://localhost:11434/v1 -- \
  uvx chassisdb-mcp --embedder openai:nomic-embed-text
```

Only the first run in a folder needs `--embedder`: the folder keeps its model, and later runs use
it. Vectors from two models can't be compared, so the folder refuses another model until you move
its memories, as below.

## What agents get

- `remember(text, tags)` stores a memory and returns its id with the most similar memories already
  kept, so the agent can forget one the new memory repeats or makes wrong.
- `recall(query, k=5, tag=None)` returns the closest memories by meaning, nearest first.
- `forget(id)` deletes one and returns what it was.
- `list_memories(tag=None, limit=20)` shows what is remembered: the memories, newest first, every
  tag and its count, the folder and the model.

## See and change what it remembers

The same command, with a command after it, works on the memory folder directly:

```console
$ uvx chassisdb-mcp list
2 memories in /Users/you/.chassisdb-mcp
embedder: fastembed:BAAI/bge-small-en-v1.5
tags: infra (1), preferences (1), python (1), staging (1)

#2  The user's staging server is staging.example.internal.  [infra, staging]  (2026-10-13 17:42:28)
#1  User preference: always format Python code with ruff (ruff format), never black.  [preferences, python]  (2026-10-13 17:42:26)

$ uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp recall "where do I deploy" -k 1
0.350  #2  The user's staging server is staging.example.internal.  [infra, staging]  (2026-10-13 17:42:28)

$ uvx chassisdb-mcp forget 2
forgot #2: The user's staging server is staging.example.internal.
```

`status`, `list`, `recall`, `remember`, `forget` and `reembed`; `--json` prints JSON.

## Where memories live

In `~/.chassisdb-mcp`: `memory.db` holds the text, `memory.chassis` the vectors. Give a project a
memory of its own with `--dir` (or `CHASSISDB_MCP_DIR`). Any number of agent sessions can share a
folder: a write holds it for milliseconds, and searches don't wait for writes.

To move to another model, every memory is embedded again from its text:

```sh
uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp --embedder fastembed:BAAI/bge-base-en-v1.5 reembed
```

Sessions that named no `--embedder` follow the folder to the new model.

## When something is wrong

The server starts even when it is set up wrong, and the agent's first call says what was wrong and
what to do, as every error here does:

```text
The memory in /Users/you/.chassisdb-mcp was made with fastembed:BAAI/bge-small-en-v1.5, and
fastembed:BAAI/bge-base-en-v1.5's vectors can't be compared with its
help: use --embedder fastembed:BAAI/bge-small-en-v1.5, or leave --embedder out to use it; to move
every memory to fastembed:BAAI/bge-base-en-v1.5, run chassisdb-mcp --dir /Users/you/.chassisdb-mcp
--embedder fastembed:BAAI/bge-base-en-v1.5 reembed
```

The server also logs to its client's MCP log what it opened and which model it loads.
