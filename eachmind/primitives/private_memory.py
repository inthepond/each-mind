"""
PrivateMemory — Each agent's own memory store.

Private memory is encoded from the agent's own Perspective and is never
automatically shared. This is the foundation of eachmind's "private by
default" principle. Agents must explicitly choose to share via SharedMemory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eachmind.primitives.perspective import EncodedMemory

if TYPE_CHECKING:
    from eachmind.retrieval.base import Retriever


@dataclass
class PrivateMemory:
    """An agent's personal, private memory store.

    PrivateMemory holds EncodedMemory objects — events that have been
    filtered through the agent's Perspective. Nothing in PrivateMemory
    is visible to other agents unless explicitly shared.

    Attributes:
        agent_id: The owning agent's identifier.
        memories: Ordered list of encoded memories.
        capacity: Maximum number of memories to retain (0 = unlimited).
        backend: Optional storage backend for persistence.
    """

    agent_id: str
    memories: list[EncodedMemory] = field(default_factory=list)
    capacity: int = 0  # 0 = unlimited
    backend: Any = None
    retriever: Retriever | None = None

    def __post_init__(self) -> None:
        """Hydrate memories from backend if one is provided."""
        if self.backend is not None and not self.memories:
            for data in self.backend.list("private_memories"):
                self.memories.append(EncodedMemory(**data))
        # Enforce capacity in case the backend held more than capacity allows.
        while self.capacity > 0 and len(self.memories) > self.capacity:
            self._evict()

    def store(self, encoded: EncodedMemory) -> None:
        """Store an encoded memory.

        If capacity is set and exceeded, the least salient memory
        is evicted to make room.

        Args:
            encoded: An EncodedMemory to store.
        """
        self.memories.append(encoded)

        if self.backend is not None:
            self.backend.save(
                "private_memories",
                encoded.event_id,
                dataclasses.asdict(encoded),
            )

        if self.capacity > 0 and len(self.memories) > self.capacity:
            self._evict()

    def recall(
        self,
        *,
        event_id: str | None = None,
        min_salience: float = 0.0,
        limit: int = 10,
    ) -> list[EncodedMemory]:
        """Retrieve memories matching the given criteria.

        Args:
            event_id: Filter by specific event ID.
            min_salience: Minimum salience threshold.
            limit: Maximum number of memories to return.

        Returns:
            List of matching EncodedMemory objects, sorted by salience (desc).
        """
        results = self.memories

        if event_id is not None:
            results = [m for m in results if m.event_id == event_id]

        if min_salience > 0.0:
            results = [m for m in results if m.salience >= min_salience]

        results = sorted(results, key=lambda m: m.salience, reverse=True)
        return results[:limit]

    def search(self, query: str) -> list[EncodedMemory]:
        """Search memories by content similarity.

        If a retriever is configured, delegates to it for ranked search.
        Otherwise falls back to simple substring matching.

        Args:
            query: Search query string.

        Returns:
            List of matching memories.
        """
        if self.retriever is not None:
            return self.retriever.search(query, self.memories)

        # Fallback: existing substring search
        query_lower = query.lower()
        results = []

        for memory in self.memories:
            content_str = str(memory.encoded_content).lower()
            if query_lower in content_str:
                results.append(memory)

        return sorted(results, key=lambda m: m.salience, reverse=True)

    @property
    def size(self) -> int:
        """Number of memories currently stored."""
        return len(self.memories)

    def _evict(self) -> None:
        """Remove the least salient memory when capacity is exceeded."""
        if not self.memories:
            return
        least_salient = min(self.memories, key=lambda m: m.salience)
        self.memories.remove(least_salient)
        if self.backend is not None:
            self.backend.delete("private_memories", least_salient.event_id)

    def __repr__(self) -> str:
        return f"PrivateMemory(agent={self.agent_id!r}, size={self.size})"
