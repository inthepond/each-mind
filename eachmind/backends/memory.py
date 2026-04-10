"""In-memory storage backend — the default."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryBackend:
    """Stores data in plain Python dicts. Fast, ephemeral, zero dependencies."""
    _store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, collection: str, key: str, value: Any) -> None:
        self._store.setdefault(collection, {})[key] = value

    def get(self, collection: str, key: str) -> Any | None:
        return self._store.get(collection, {}).get(key)

    def list(self, collection: str) -> list[Any]:
        return list(self._store.get(collection, {}).values())

    def delete(self, collection: str, key: str) -> None:
        coll = self._store.get(collection, {})
        coll.pop(key, None)

    def clear(self, collection: str) -> None:
        self._store.pop(collection, None)
