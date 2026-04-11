"""SQLite storage backend — persistent, zero external dependencies."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from eachmind.serialization import JSONSerializer, Serializer


@dataclass
class SQLiteBackend:
    """Stores data in a SQLite database. Persistent across restarts.
    Args:
        db_path: Path to SQLite file, or ":memory:" for in-memory.
    """
    db_path: str = "eachmind.db"
    serializer: Serializer = field(default_factory=JSONSerializer)
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS store ("
            "  collection TEXT NOT NULL,"
            "  key TEXT NOT NULL,"
            "  value TEXT NOT NULL,"
            "  PRIMARY KEY (collection, key)"
            ")"
        )
        self._conn.commit()

    def save(self, collection: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO store (collection, key, value) VALUES (?, ?, ?)",
            (collection, key, self.serializer.serialize(value)),
        )
        self._conn.commit()

    def get(self, collection: str, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value FROM store WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        return self.serializer.deserialize(row[0]) if row else None

    def list(self, collection: str) -> list[Any]:
        rows = self._conn.execute(
            "SELECT value FROM store WHERE collection = ?", (collection,),
        ).fetchall()
        return [self.serializer.deserialize(row[0]) for row in rows]

    def delete(self, collection: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM store WHERE collection = ? AND key = ?", (collection, key),
        )
        self._conn.commit()

    def clear(self, collection: str) -> None:
        self._conn.execute("DELETE FROM store WHERE collection = ?", (collection,))
        self._conn.commit()
