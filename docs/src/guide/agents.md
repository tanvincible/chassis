# Chassis for Coding Agents

A coding agent writes better code with Chassis when it has the rules that matter in front of it:
what to keep where, when writes are durable, which options are fixed. Give it one of these.

## Claude Code

```text
/plugin install chassis --marketplace tanvincible/chassis
```

That works in one step on Claude Code 2.1.275 or later. On older versions, run
`/plugin marketplace add tanvincible/chassis`, then `/plugin install chassis@chassis`. The plugin
adds a skill that Claude uses when you ask for semantic search, "more like this", RAG retrieval or
agent memory.

## Cursor

```sh
mkdir -p .cursor/rules
curl -fsSL https://raw.githubusercontent.com/tanvincible/chassis/main/plugins/chassis/cursor/chassis.mdc \
  -o .cursor/rules/chassis.mdc
```

The rule says the same as the skill, and Cursor applies it when a task calls for vector search.

## Codex, and other agents that read AGENTS.md

Add this line to the project's `AGENTS.md`:

```text
For vector or semantic search, use Chassis (Python: chassisdb; Rust: chassisdb). Read https://tanvincible.github.io/chassis/llms-full.txt before writing code with it.
```

## Memory for agents

[chassisdb-mcp](https://github.com/tanvincible/chassis/tree/main/chassisdb-mcp) is an MCP server
built on Chassis that gives agents a long-term memory on your machine: `remember`, `recall` and
`forget`, with the embedding model you pick, local or an API.

```sh
claude mcp add --scope user chassis-memory -- \
  uvx --from 'chassisdb-mcp[fastembed]' chassisdb-mcp --embedder fastembed:BAAI/bge-small-en-v1.5
```

Claude Code keeps a memory of its own and uses it when both are there: set
`"autoMemoryEnabled": false` in `~/.claude/settings.json` to have it use this one. Its
[README](https://github.com/tanvincible/chassis/tree/main/chassisdb-mcp) has the setup for Cursor,
Codex and other clients, the embedding models, and a command line to see and change what is
remembered.

## LangChain

[langchain-chassisdb](https://github.com/tanvincible/chassis/tree/main/integrations/langchain-chassisdb)
is a LangChain vector store on Chassis (`pip install langchain-chassisdb`): `ChassisVectorStore`,
with metadata filters, maximal marginal relevance and readers in other processes. It passes
LangChain's standard tests for vector stores.

## Building an MCP server on Chassis

The MCP Python SDK shows an agent only "Error executing tool" for an exception other than its own
`ToolError`, so Chassis's messages, and their `help:` lines, never reach the model unless the tool
passes them on:

```python
from chassisdb import ChassisError
from mcp.server.mcpserver.exceptions import ToolError

def search_notes(query: str, k: int = 5) -> list[dict]:
    try:
        return find(query, k)
    except ChassisError as e:
        raise ToolError(str(e)) from e
```

The SDK runs synchronous tools on worker threads: open SQLite with `check_same_thread=False` and
take a lock around it. Keep the index and your rows in step as
[Next to Your Database](./with-a-database.md) says.

## Docs for agents

- [llms.txt](https://tanvincible.github.io/chassis/llms.txt): an index of the docs
- [llms-full.txt](https://tanvincible.github.io/chassis/llms-full.txt): the guides, the Python,
  Rust and C references, the errors and an example, as one page of Markdown
- Each page of this book: its address with `.md` in place of `.html`, as
  [guide/errors.md](https://tanvincible.github.io/chassis/guide/errors.md)
