"""
PrivateMemory — Each agent's own memory store.

Private memory is encoded from the agent's own Perspective and is never
automatically shared. This is the foundation of eachmind's "private by
default" principle. Agents must explicitly choose to share via SharedMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import EncodedMemory, Perspective


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
    """

    agent_id: str
    memories: list[EncodedMemory] = field(default_factory=list)
    capacity: int = 0  # 0 = unlimited

    def store(self, encoded: EncodedMemory) -> None:
        """Store an encoded memory.

        If capacity is set and exceeded, the least salient memory
        is evicted to make room.

        Args:
            encoded: An EncodedMemory to store.
        """
        self.memories.append(encoded)

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

        A simple text-based search over encoded content. Production
        implementations should override this with vector similarity
        or more sophisticated retrieval.

        Args:
            query: Search query string.

        Returns:
            List of matching memories.
        """
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

    def __repr__(self) -> str:
        return f"PrivateMemory(agent={self.agent_id!r}, size={self.size})"
