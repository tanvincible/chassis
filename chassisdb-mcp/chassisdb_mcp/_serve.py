"""The MCP server: chassisdb-mcp's tools on a `Memory`, over the MCP SDK."""

from collections.abc import Callable

from chassisdb import ChassisError
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from chassisdb_mcp.server import (
    VERSION,
    Forgotten,
    Found,
    Memory,
    Overview,
    Problem,
    Stored,
)


class _Server(MCPServer):
    """Arguments that don't fit a tool's schema are said plainly, with what to do, and without
    the link pydantic adds to each."""

    async def call_tool(self, name, arguments, context=None):  # type: ignore[no-untyped-def]
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as e:
            if not isinstance(e.__cause__, ValidationError):
                raise
            wrong = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                + ("" if error["type"] == "missing" else f" (given {error['input']!r:.40})")
                for error in e.__cause__.errors()
            )
            raise ToolError(
                f"The arguments don't fit {name}: {wrong}\nhelp: pass them as the tool's input "
                "schema says"
            ) from e.__cause__


def build_server(memory: "Memory | Problem") -> MCPServer:
    """The server, with its tools on `memory`; or, when the memory couldn't be opened, tools that
    say why, so an agent learns what to fix instead of seeing a server that failed to start."""
    server = _Server(
        "chassis-memory",
        version=VERSION,
        instructions=(
            "Long-term memory that lasts across sessions, on this machine: use it for anything to "
            "keep for later sessions. Recall before answering anything earlier sessions may have "
            "settled, such as the user's preferences or a project's decisions. Remember what is "
            "worth knowing next time, one fact per memory, written to stand on its own. When "
            "remember returns a similar memory that the new one repeats or makes wrong, forget it. "
            "Tags match whatever their case; list_memories shows them."
        ),
    )

    def run(call: Callable[[Memory], object]) -> object:
        try:
            if isinstance(memory, Problem):
                raise memory
            return call(memory)
        except (Problem, ChassisError) as e:
            raise ToolError(str(e)) from e
        except Exception as e:  # MCP would show only "Error executing tool"
            raise ToolError(f"{type(e).__name__}: {e}") from e

    @server.tool()
    def remember(text: str, tags: list[str] | str | None = None) -> Stored:
        """Store a memory: one fact, decision, preference or note worth knowing in later sessions,
        written to stand on its own. Tags, such as a project's name, let recall look among them
        only. Returns its id and the most similar memories already kept, with their distance
        (0 is the same meaning): forget one that the new memory repeats or makes wrong."""
        return run(lambda m: m.remember(text, tags))  # type: ignore[return-value]

    @server.tool()
    def recall(query: str, k: int = 5, tag: str | None = None) -> list[Found]:
        """Find the k memories closest in meaning to the query, nearest first: their ids, text,
        tags, when each was made, and distance (smaller is closer; 0 is the same meaning). With a
        tag, only memories that have it."""
        return run(lambda m: m.recall(query, k, tag))  # type: ignore[return-value]

    @server.tool()
    def forget(id: int | str) -> Forgotten:
        """Delete the memory with this id, as recall or list_memories showed it. Returns what was
        forgotten."""
        return run(lambda m: m.forget(id))  # type: ignore[return-value]

    @server.tool()
    def list_memories(tag: str | None = None, limit: int = 20, offset: int = 0) -> Overview:
        """The memories, newest first, with how many there are, every tag and its count, the
        folder they are kept in and the embedding model: to show the user what is remembered."""
        return run(lambda m: m.overview(tag, limit, offset))  # type: ignore[return-value]

    return server
