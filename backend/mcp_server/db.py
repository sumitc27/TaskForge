from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def db_path() -> Path:
    raw = os.getenv("TASKFORGE_DB", "tasks.db")
    p = Path(raw)
    return p if p.is_absolute() else _BACKEND_ROOT / p


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                notes TEXT DEFAULT '',
                due TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            )
            """
        )


def add_task(title: str, notes: str = "", due: str = "") -> dict:
    init_db()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (title, notes, due, status, created_at) "
            "VALUES (?, ?, ?, 'open', ?)",
            (title.strip(), notes.strip(), due.strip(), now),
        )
        task_id = cur.lastrowid
    return {"id": task_id, "title": title.strip(), "notes": notes.strip(),
            "due": due.strip(), "status": "open", "created_at": now}


def list_tasks(status: str = "") -> list[dict]:
    init_db()
    with _conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def complete_task(task_id: int) -> dict:
    init_db()
    with _conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,)
        )
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise ValueError(f"No task with id {task_id}")
    return dict(row)
