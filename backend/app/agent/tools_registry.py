"""Unified tool registry: local Python tools + tools discovered from the MCP server."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..config import get_settings
from ..tools.mcp_client import call_mcp_tool, list_mcp_tools
from ..tools.url_reader import read_url
from ..tools.web_search import web_search


@dataclass
class Tool:
    name: str
    description: str
    args_hint: str
    write: bool
    run: Callable[[dict], Awaitable[str]]
    schema: dict | None = None


async def _run_local(fn, args: dict, timeout: int) -> str:
    try:
        return await asyncio.wait_for(asyncio.to_thread(lambda: fn(**args)), timeout)
    except asyncio.TimeoutError:
        return f"ERROR: tool timed out after {timeout}s."
    except TypeError as e:
        return f"ERROR: bad arguments for tool: {e}"
    except Exception as e:
        return f"ERROR: tool failed: {e}"


async def build_registry() -> dict[str, Tool]:
    s = get_settings()
    timeout = s.tool_timeout_seconds
    registry: dict[str, Tool] = {}

    registry["web_search"] = Tool(
        name="web_search",
        description="Search the web for up-to-date information. Returns titles, URLs, snippets.",
        args_hint='{"query": "search terms"}',
        write=False,
        run=lambda args: _run_local(web_search, args, timeout),
    )
    registry["read_url"] = Tool(
        name="read_url",
        description="Fetch a web page and return its cleaned text. Use after web_search to read a result.",
        args_hint='{"url": "https://..."}',
        write=False,
        run=lambda args: _run_local(read_url, args, timeout),
    )

    try:
        for t in await list_mcp_tools():
            name = t["name"]
            registry[name] = Tool(
                name=name,
                description=t["description"],
                args_hint=_schema_hint(t.get("schema")),
                write=t["write"],
                run=_make_mcp_runner(name, timeout),
                schema=t.get("schema"),
            )
    except Exception as e:
        logging.warning("MCP tools unavailable, continuing with web tools only: %s", e)

    return registry


def _make_mcp_runner(name: str, timeout: int):
    async def run(args: dict) -> str:
        try:
            return await asyncio.wait_for(call_mcp_tool(name, args), timeout)
        except asyncio.TimeoutError:
            return f"ERROR: MCP tool '{name}' timed out."
        except Exception as e:
            return f"ERROR: MCP tool '{name}' failed: {e}"
    return run


def _schema_hint(schema: dict | None) -> str:
    if not schema or "properties" not in schema:
        return "{}"
    props = schema["properties"]
    required = set(schema.get("required", []))
    parts = []
    for k, v in props.items():
        typ = v.get("type", "any")
        star = "*" if k in required else ""
        parts.append(f'"{k}{star}": <{typ}>')
    return "{" + ", ".join(parts) + "}"
