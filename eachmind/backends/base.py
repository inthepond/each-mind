"""Base protocol for storage backends."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends.
    Manages collections of key-value pairs. Collections isolate different
    data types (e.g., "private_memories", "shared_entries").
    """
    def save(self, collection: str, key: str, value: Any) -> None: ...
    def get(self, collection: str, key: str) -> Any | None: ...
    def list(self, collection: str) -> list[Any]: ...
    def delete(self, collection: str, key: str) -> None: ...
    def clear(self, collection: str) -> None: ...
