"""``reward_lens.operate`` — the operator surface.

The CLI and the MCP server are how a human operator and an agent operator drive the kernel. Both are
torch-free to import: the CLI's model-touching commands import their kernel dependencies lazily and
are GPU-gated, and the MCP server dispatches in-process over the evidence store. Importing this
package pulls nothing heavier than typer and the torch-free artifacts and studies layers, so an
agent can introspect the operator surface without loading a model.
"""

from __future__ import annotations

from reward_lens.operate.cli import app, main
from reward_lens.operate.mcp import MCPServer, Tool, build_server

__all__ = ["app", "main", "MCPServer", "Tool", "build_server"]
