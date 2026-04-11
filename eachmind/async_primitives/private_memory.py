"""Async private memory store."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from eachmind.primitives.perspective import EncodedMemory


@dataclass
class AsyncPrivateMemory:
    """Async version of PrivateMemory. Requires an AsyncStorageBackend.

    All data is persisted via the backend. No in-memory cache.

    Attributes:
        agent_id: The owning agent's identifier.
        backend: An AsyncStorageBackend instance (required).
        capacity: Maximum memories to retain (0 = unlimited).
    """

    agent_id: str
    backend: Any  # AsyncStorageBackend
    capacity: int = 0

    async def store(self, encoded: EncodedMemory) -> None:
        """Store an encoded memory."""
        await self.backend.save(
            "private_memories",
            encoded.event_id,
            dataclasses.asdict(encoded),
        )
        if self.capacity > 0:
            await self._evict_if_needed()

    async def recall(
        self,
        *,
        event_id: str | None = None,
        min_salience: float = 0.0,
        limit: int = 10,
    ) -> list[EncodedMemory]:
        """Retrieve memories matching criteria."""
        all_data = await self.backend.list("private_memories")
        memories = [EncodedMemory(**d) for d in all_data]

        if event_id is not None:
            memories = [m for m in memories if m.event_id == event_id]
        if min_salience > 0.0:
            memories = [m for m in memories if m.salience >= min_salience]

        memories.sort(key=lambda m: m.salience, reverse=True)
        return memories[:limit]

    async def search(self, query: str) -> list[EncodedMemory]:
        """Search memories by content substring."""
        all_data = await self.backend.list("private_memories")
        memories = [EncodedMemory(**d) for d in all_data]
        query_lower = query.lower()
        results = [
            m for m in memories
            if query_lower in str(m.encoded_content).lower()
        ]
        results.sort(key=lambda m: m.salience, reverse=True)
        return results

    async def size(self) -> int:
        """Number of memories stored."""
        items = await self.backend.list("private_memories")
        return len(items)

    async def _evict_if_needed(self) -> None:
        """Remove least salient memory if over capacity."""
        all_data = await self.backend.list("private_memories")
        if len(all_data) <= self.capacity:
            return
        memories = [EncodedMemory(**d) for d in all_data]
        least = min(memories, key=lambda m: m.salience)
        await self.backend.delete("private_memories", least.event_id)

    def __repr__(self) -> str:
        return f"AsyncPrivateMemory(agent={self.agent_id!r})"
