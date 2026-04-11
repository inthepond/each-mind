"""In-memory storage backend — the default."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryBackend:
    """Stores data in plain Python dicts. Fast, ephemeral, zero dependencies."""
    _store: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, collection: str, key: str, value: Any) -> None:
        with self._lock:
            self._store.setdefault(collection, {})[key] = value

    def get(self, collection: str, key: str) -> Any | None:
        with self._lock:
            return self._store.get(collection, {}).get(key)

    def list(self, collection: str) -> list[Any]:
        with self._lock:
            return list(self._store.get(collection, {}).values())

    def delete(self, collection: str, key: str) -> None:
        with self._lock:
            coll = self._store.get(collection, {})
            coll.pop(key, None)

    def clear(self, collection: str) -> None:
        with self._lock:
            self._store.pop(collection, None)
