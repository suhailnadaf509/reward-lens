"""A minimal MCP server exposing observables, cards, and the auditing game.

This is the surface through which an external agent operator consumes the library and through which
the blind auditing game runs with an agent on the blue side. It is deliberately minimal and says so:
the tool protocol shape is real and the pure tools are fully wired over the evidence store, but the
transport is not. A production server adds a JSON-RPC 2.0 loop over stdio or a socket (the ``mcp``
Python SDK, or a hand-rolled loop), the ``initialize`` capability handshake, resource endpoints so a
card is addressable as an MCP resource, oracle-provenance stamping on any judge-backed tool, and the
live wiring of the auditing game to a loaded signal and an organism with an answer key. Those
are the notes at the bottom of this module; what is here is enough to exercise the tool contract and
to show exactly where the model-touching tool is gated.

The dispatch is in-process and torch-free: ``list_tools`` returns the tool schemas an MCP client
reads, and ``call_tool`` runs a tool by name and returns MCP-shaped content. The two read-only tools
(``list_observables``, ``get_card``) are views over the store; the ``auditing_game`` tool is
GPU-gated and returns a clearly marked notice rather than fabricating a game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from reward_lens.artifacts.card import build_card
from reward_lens.core.store import EvidenceStore, default_store

# What a full server must add on top of this in-process dispatch, stated once so it is not mistaken
# for complete. Kept as data so an operator command can print it.
PRODUCTION_NOTES = (
    "A production MCP server adds: (1) a JSON-RPC 2.0 transport over stdio or a socket, via the "
    "`mcp` Python SDK or an equivalent loop; (2) the `initialize` handshake with capability "
    "negotiation; (3) MCP resources so each RM card is addressable by URI, not only via a tool; "
    "(4) oracle-provenance stamping (model id, prompt hash, date) on any judge-backed tool; "
    "(5) live wiring of the auditing game to a loaded signal and an organism answer key, which is "
    "the GPU-gated part; and (6) authentication and rate limiting for untrusted agent operators."
)


@dataclass(frozen=True)
class Tool:
    """One MCP tool: its name, description, JSON-Schema for arguments, and its handler.

    ``input_schema`` is the JSON Schema an MCP client reads to call the tool. ``handler`` takes the
    parsed arguments and returns a string; ``call_tool`` wraps that in MCP content and marks whether
    it errored, so a handler that raises becomes an ``isError`` result rather than crashing the loop.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]
    gpu_gated: bool = False


class GpuGatedTool(RuntimeError):
    """Raised by a tool handler that needs a real model, so ``call_tool`` marks it an error result."""


class MCPServer:
    """A minimal, in-process MCP tool server over the evidence store.

    Holds the store and a tool registry. ``list_tools`` returns the schema list an MCP client reads;
    ``call_tool`` dispatches by name and returns MCP-shaped content. This class carries no transport;
    wiring it to stdio JSON-RPC is what ``PRODUCTION_NOTES`` describes and what a full server adds.
    """

    def __init__(self, store: EvidenceStore | None = None) -> None:
        self.store = store if store is not None else default_store()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        """The ``tools/list`` response: name, description, and input schema for each tool."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "gpuGated": t.gpu_gated,
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """The ``tools/call`` response: run a tool and return MCP content with an error flag.

        An unknown tool, or a handler that raises, returns ``isError = True`` with the message as
        text content, which is the MCP convention for a failed call. A GPU-gated tool raises
        ``GpuGatedTool`` and so returns an error result naming the model work it would do.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"unknown tool '{name}'")
        try:
            text = tool.handler(arguments or {})
        except GpuGatedTool as exc:
            return _error(f"GPU-gated: {exc}")
        except Exception as exc:  # a handler fault is an error result, not a server crash
            return _error(f"{type(exc).__name__}: {exc}")
        return {"content": [{"type": "text", "text": text}], "isError": False}


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _tool_list_observables(server: MCPServer) -> Tool:
    def handler(_: dict[str, Any]) -> str:
        import json

        names = sorted({ev.observable for ev in server.store.find()})
        return json.dumps({"observables": names}, indent=2)

    return Tool(
        name="list_observables",
        description="List the observable names the evidence store holds measurements for.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
    )


def _tool_get_card(server: MCPServer) -> Tool:
    def handler(args: dict[str, Any]) -> str:
        signal = args.get("signal")
        if not signal:
            raise ValueError("get_card requires a 'signal' fingerprint argument")
        return build_card(str(signal), server.store).to_json()

    return Tool(
        name="get_card",
        description="Assemble the RM card for a signal fingerprint (a view over the store).",
        input_schema={
            "type": "object",
            "properties": {
                "signal": {"type": "string", "description": "Model fingerprint, e.g. mfp:..."}
            },
            "required": ["signal"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _tool_auditing_game(server: MCPServer) -> Tool:
    def handler(args: dict[str, Any]) -> str:
        signal = args.get("signal", "<signal>")
        organism = args.get("organism", "<organism>")
        raise GpuGatedTool(
            f"the auditing game needs a loaded signal ({signal}) and an organism with an answer key "
            f"({organism}). It dispatches to reward_lens.organisms.game.AuditingGame(signal, "
            "organism).run(); run it on hardware. This tool marks the boundary, it does not fake a "
            "round."
        )

    return Tool(
        name="auditing_game",
        description="Start a blind auditing-game round against an organism (GPU-gated).",
        input_schema={
            "type": "object",
            "properties": {
                "signal": {"type": "string"},
                "organism": {"type": "string"},
            },
            "required": ["signal", "organism"],
            "additionalProperties": False,
        },
        handler=handler,
        gpu_gated=True,
    )


def build_server(store: EvidenceStore | None = None) -> MCPServer:
    """Build the minimal MCP server with the observables, card, and auditing-game tools.

    The first two tools are read-only views over the store and run here; the auditing-game tool is
    GPU-gated and returns a marked error naming its dispatch. See ``PRODUCTION_NOTES`` for what a
    transport-complete server adds.
    """
    server = MCPServer(store=store)
    server.register(_tool_list_observables(server))
    server.register(_tool_get_card(server))
    server.register(_tool_auditing_game(server))
    return server


__all__ = ["Tool", "MCPServer", "GpuGatedTool", "build_server", "PRODUCTION_NOTES"]
