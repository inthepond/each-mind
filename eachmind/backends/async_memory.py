"""Async in-memory storage backend."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AsyncInMemoryBackend:
    """Async in-memory storage. Concurrency-safe via asyncio.Lock."""

    _store: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def save(self, collection: str, key: str, value: Any) -> None:
        async with self._lock:
            self._store.setdefault(collection, {})[key] = value

    async def get(self, collection: str, key: str) -> Any | None:
        async with self._lock:
            return self._store.get(collection, {}).get(key)

    async def list(self, collection: str) -> list[Any]:
        async with self._lock:
            return list(self._store.get(collection, {}).values())

    async def delete(self, collection: str, key: str) -> None:
        async with self._lock:
            coll = self._store.get(collection, {})
            coll.pop(key, None)

    async def clear(self, collection: str) -> None:
        async with self._lock:
            self._store.pop(collection, None)
