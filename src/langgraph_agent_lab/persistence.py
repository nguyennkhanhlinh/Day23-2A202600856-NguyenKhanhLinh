"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _sqlite_path(database_url: str | None) -> str:
    """Normalize a sqlite database URL/path into a sqlite3 connection target."""
    if not database_url:
        return "outputs/checkpoints.sqlite"
    if database_url == ":memory:":
        return database_url
    if database_url.startswith("sqlite:///"):
        return database_url.removeprefix("sqlite:///")
    if database_url.startswith("sqlite://"):
        return database_url.removeprefix("sqlite://")
    return database_url


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supported values:
    - ``none``: no checkpointing
    - ``memory``: in-process MemorySaver for tests and local runs
    - ``sqlite``: durable SQLite checkpointing via langgraph-checkpoint-sqlite
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointing requires the optional dependency: "
                "pip install -e .[sqlite]"
            ) from exc

        db_path = _sqlite_path(database_url)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(connection)
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointing is an optional extension. Install "
            "langgraph-checkpoint-postgres and add a PostgresSaver adapter."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
