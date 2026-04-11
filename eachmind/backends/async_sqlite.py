"""Async SQLite storage backend.

Install: pip install eachmind[async]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import aiosqlite
except ImportError as e:
    raise ImportError(
        "Async SQLite backend requires 'aiosqlite'. "
        "Install with: pip install eachmind[async]"
    ) from e

from eachmind.serialization import JSONSerializer, Serializer


@dataclass
class AsyncSQLiteBackend:
    """Async SQLite storage. Requires aiosqlite."""

    db_path: str = "eachmind.db"
    serializer: Serializer = field(default_factory=JSONSerializer)
    _conn: Any = field(init=False, repr=False, default=None)

    async def initialize(self) -> None:
        """Must be called before use (can't do async in __post_init__)."""
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS store ("
            "  collection TEXT NOT NULL,"
            "  key TEXT NOT NULL,"
            "  value TEXT NOT NULL,"
            "  PRIMARY KEY (collection, key)"
            ")"
        )
        await self._conn.commit()

    async def save(self, collection: str, key: str, value: Any) -> None:
        await self._conn.execute(
            "INSERT OR REPLACE INTO store (collection, key, value) "
            "VALUES (?, ?, ?)",
            (collection, key, self.serializer.serialize(value)),
        )
        await self._conn.commit()

    async def get(self, collection: str, key: str) -> Any | None:
        cursor = await self._conn.execute(
            "SELECT value FROM store WHERE collection = ? AND key = ?",
            (collection, key),
        )
        row = await cursor.fetchone()
        return self.serializer.deserialize(row[0]) if row else None

    async def list(self, collection: str) -> list[Any]:
        cursor = await self._conn.execute(
            "SELECT value FROM store WHERE collection = ?",
            (collection,),
        )
        rows = await cursor.fetchall()
        return [self.serializer.deserialize(row[0]) for row in rows]

    async def delete(self, collection: str, key: str) -> None:
        await self._conn.execute(
            "DELETE FROM store WHERE collection = ? AND key = ?",
            (collection, key),
        )
        await self._conn.commit()

    async def clear(self, collection: str) -> None:
        await self._conn.execute(
            "DELETE FROM store WHERE collection = ?",
            (collection,),
        )
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
