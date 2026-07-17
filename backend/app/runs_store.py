"""Run history: persists every streamed SSE frame."""
from __future__ import annotations

import functools
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_settings().agent_db_path, timeout=2)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _retry_on_lock(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        delay = 0.1
        for attempt in range(4):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or attempt == 3:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 1.0)

    return wrapper


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                created_at TEXT NOT NULL,
                elapsed_s REAL,
                final_preview TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event TEXT NOT NULL,
                data TEXT NOT NULL,
                ts TEXT NOT NULL,
                PRIMARY KEY (run_id, seq)
            )
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@_retry_on_lock
def create_run(run_id: str, task: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (id, task, status, created_at) VALUES (?, ?, 'running', ?)",
            (run_id, task, _now()),
        )


@_retry_on_lock
def set_status(run_id: str, status: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


@_retry_on_lock
def finish_run(run_id: str, status: str, elapsed_s: float, final_preview: str = "") -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, elapsed_s = ?, final_preview = ? WHERE id = ?",
            (status, elapsed_s, final_preview, run_id),
        )


@_retry_on_lock
def append_event(run_id: str, event: str, data: dict) -> None:
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM run_events WHERE run_id = ?", (run_id,)
        ).fetchone()
        seq = row[0]
        conn.execute(
            "INSERT INTO run_events (run_id, seq, event, data, ts) VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, event, json.dumps(data, ensure_ascii=False), _now()),
        )


def list_runs(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    init_db()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_run_events(run_id: str) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT event, data, ts FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
    return [{"event": r["event"], "data": json.loads(r["data"]), "ts": r["ts"]} for r in rows]


def delete_run(run_id: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
