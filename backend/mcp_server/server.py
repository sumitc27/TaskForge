from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from db import add_task as _add_task
from db import complete_task as _complete_task
from db import init_db
from db import list_tasks as _list_tasks

mcp = FastMCP("taskforge-tools")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def add_task(title: str, notes: str = "", due: str = "") -> dict:
    """Create a new task in the tracker."""
    return _add_task(title, notes, due)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_tasks(status: str = "") -> list[dict]:
    """List tasks, optionally filtered by status ('open' or 'done')."""
    return _list_tasks(status)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def complete_task(task_id: int) -> dict:
    """Mark the task with the given id as done."""
    return _complete_task(task_id)


if __name__ == "__main__":
    init_db()
    mcp.run()
