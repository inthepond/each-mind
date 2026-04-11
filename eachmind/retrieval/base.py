"""Base protocol for memory retrieval."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from eachmind.primitives.perspective import EncodedMemory


@runtime_checkable
class Retriever(Protocol):
    """Protocol for searching/ranking encoded memories by relevance."""

    def search(
        self,
        query: str,
        memories: list[EncodedMemory],
        limit: int = 10,
    ) -> list[EncodedMemory]:
        """Return memories ranked by relevance to query."""
        ...
