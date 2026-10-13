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

## Docs for agents

- [llms.txt](https://tanvincible.github.io/chassis/llms.txt): an index of the docs
- [llms-full.txt](https://tanvincible.github.io/chassis/llms-full.txt): the guides, the Python,
  Rust and C references, the errors and an example, as one page of Markdown
