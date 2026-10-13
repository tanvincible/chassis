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

The first launch installs the package and the model's runtime, which can take a minute, longer
than a client waits for a server to connect: run the command once in a terminal first (`uvx
--from 'chassisdb-mcp[fastembed]' chassisdb-mcp --version`). For a memory the whole team's
sessions in one project use, `--scope project` writes the setup to `.mcp.json` in the repository
instead, and `--dir` names the folder.

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

A folder keeps the model its first memory was made with, and every session uses that one, since
vectors from two models can't be compared: `--embedder` only picks the model for a new folder.
Name another for a folder that has one, and the server says so in its log and goes on with the
folder's, until you move the memories, as below.

## What agents get

- `remember(text, tags)` stores a memory and returns its id with the most similar memories already
  kept, so the agent can forget one the new memory repeats or makes wrong.
- `recall(query, k=5, tag=None)` returns the closest memories by meaning, nearest first. Tags match
  whatever their case; one no memory has is an error that lists the tags there are.
- `forget(id)` deletes one and returns what it was.
- `list_memories(tag=None, limit=20)` shows what is remembered: the memories, newest first, every
  tag and its count, the folder and the model.

A distance is 0 for the same meaning and grows as meanings part, but how far apart is "related"
depends on the model, and changes with a `reembed`: rank by it rather than cut at a fixed number
where you can. As a rough guide, on one project's memories, bge-small-en-v1.5 put the right
answer within about 0.30 of a question and two wordings of one fact within about 0.15;
bge-base-en-v1.5 within 0.40 and 0.19.

From your own code, the MCP Python SDK (2.x) talks to the server like this:

```python
import anyio
from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command="uvx",
    args=["--from", "chassisdb-mcp[fastembed]", "chassisdb-mcp",
          "--embedder", "fastembed:BAAI/bge-small-en-v1.5"],
)

async def main():
    async with Client(server) as client:
        await client.call_tool("remember", {"text": "Releases go out on Thursdays", "tags": ["ops"]})
        found = await client.call_tool("recall", {"query": "when do we release", "k": 3})
        if found.is_error:
            print(found.content[0].text)  # What was wrong, then what to do
        else:
            print(found.structured_content["result"])  # A list comes wrapped in "result"

anyio.run(main)
```

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

`status`, `list`, `recall`, `remember`, `forget` and `reembed`. `--dir`, `--embedder` and `--json`
(print JSON) go before or after the command. `list` and `forget` need no model, so plain
`uvx chassisdb-mcp` does; `recall`, `remember` and `reembed` need the model's extra, as above.

## Where memories live

In `~/.chassisdb-mcp`: `memory.db` holds the text, `memory.chassis` the vectors. Give a project a
memory of its own with `--dir` (or `CHASSISDB_MCP_DIR`). Any number of agent sessions can share a
folder: each write is one transaction on `memory.db`, so sessions take turns writing for the
milliseconds a write takes, and a memory's text and vector are saved together or not at all.
Searches don't wait for writes.

To move to another model, every memory is embedded again from its text:

```sh
uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp --embedder fastembed:BAAI/bge-base-en-v1.5 reembed
```

It embeds while sessions go on using the memory, then picks up what they wrote meanwhile as it
swaps the new index in. Every session follows the folder to the new model; nothing in their setup
needs changing. `status` also says when a crash left a memory without its vector, and `reembed`
mends that too.

## When something is wrong

The server starts however it is set up: a misspelled `--embedder`, a folder it can't write, a
model that can't load. The agent's first call then says what was wrong and what to do, as every
error here does:

```text
The embedder is 'fasttext:BAAI/bge-small-en-v1.5', but has to be <kind>:<model>
help: did you mean fastembed:BAAI/bge-small-en-v1.5? pick one of ...
```

```text
The memory's index in /Users/you/.chassisdb-mcp doesn't fit its model: The query has 768
components, but this index holds vectors of 384
help: rebuild the index from the memories' text: chassisdb-mcp --dir /Users/you/.chassisdb-mcp reembed
```

The server also logs to its client's MCP log what it opened and which model it loads.
