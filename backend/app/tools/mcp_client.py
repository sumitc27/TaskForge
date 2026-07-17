"""MCP client — a persistent stdio session to our custom MCP server."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import get_settings

logger = logging.getLogger("taskforge.mcp")


def _server_params() -> StdioServerParameters:
    s = get_settings()
    env = {**os.environ, "TASKFORGE_DB": str(s.db_path), "PYTHONIOENCODING": "utf-8"}
    return StdioServerParameters(
        command=sys.executable,
        args=[str(s.mcp_server_script)],
        env=env,
        cwd=str(s.backend_root),
    )


def _content_to_text(result) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured)
    return "(tool returned no content)"


def _is_write_tool(tool) -> bool:
    ann = getattr(tool, "annotations", None)
    return not (ann and getattr(ann, "readOnlyHint", False))


class _MCPSession:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._ready: asyncio.Event | None = None
        self._shutdown: asyncio.Event | None = None
        self._supervisor: asyncio.Task | None = None
        self._start_lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _rebind_if_new_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        self._loop = loop
        self._session = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self._supervisor = None

    async def _run(self) -> None:
        try:
            async with stdio_client(_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception:
            logger.exception("[mcp] session supervisor crashed")
        finally:
            self._session = None
            self._ready.set()

    async def ensure_started(self) -> None:
        self._rebind_if_new_loop()
        async with self._start_lock:
            if self._supervisor is None or self._supervisor.done():
                self._ready.clear()
                self._shutdown.clear()
                self._supervisor = asyncio.create_task(self._run())
        await self._ready.wait()

    async def restart(self) -> None:
        self._rebind_if_new_loop()
        async with self._start_lock:
            supervisor = self._supervisor
        if supervisor and not supervisor.done():
            self._shutdown.set()
            try:
                await asyncio.wait_for(supervisor, timeout=5)
            except Exception:
                supervisor.cancel()
        async with self._start_lock:
            self._session = None
            self._supervisor = None
        await self.ensure_started()

    async def shutdown(self) -> None:
        self._rebind_if_new_loop()
        async with self._start_lock:
            supervisor = self._supervisor
        if supervisor and not supervisor.done():
            self._shutdown.set()
            try:
                await asyncio.wait_for(supervisor, timeout=5)
            except Exception:
                supervisor.cancel()

    async def get(self) -> ClientSession:
        await self.ensure_started()
        if self._session is None:
            raise RuntimeError("MCP session failed to start")
        return self._session


_session = _MCPSession()


async def _with_session(fn):
    try:
        return await fn(await _session.get())
    except Exception as e:
        logger.warning("[mcp] call failed (%s) — restarting session and retrying once", e)
        await _session.restart()
        return await fn(await _session.get())


async def list_mcp_tools() -> list[dict]:
    async def go(session: ClientSession):
        resp = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                "schema": t.inputSchema,
                "write": _is_write_tool(t),
            }
            for t in resp.tools
        ]
    return await _with_session(go)


async def call_mcp_tool(name: str, args: dict) -> str:
    async def go(session: ClientSession):
        result = await session.call_tool(name, args or {})
        if getattr(result, "isError", False):
            return f"ERROR from tool {name}: {_content_to_text(result)}"
        return _content_to_text(result)
    return await _with_session(go)


async def shutdown_mcp_session() -> None:
    await _session.shutdown()
