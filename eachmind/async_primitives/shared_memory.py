"""Async shared memory store."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from eachmind.primitives.shared_memory import SharedEntry, ShareScope


@dataclass
class AsyncSharedMemory:
    """Async version of SharedMemory. Requires an AsyncStorageBackend.

    Attributes:
        team_id: Team identifier.
        backend: An AsyncStorageBackend instance (required).
        default_ttl: Default TTL in seconds for new entries (None = no expiry).
    """

    team_id: str = "default"
    backend: Any = None  # AsyncStorageBackend
    default_ttl: int | None = None

    async def publish(
        self,
        content: Any,
        shared_by: str,
        scope: ShareScope = ShareScope.TEAM,
        targets: list[str] | None = None,
        reason: str = "",
        ttl: int | None = None,
        **metadata: Any,
    ) -> SharedEntry:
        """Publish knowledge to shared memory."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        entry = SharedEntry(
            content=content,
            shared_by=shared_by,
            scope=scope,
            targets=targets or [],
            reason=reason,
            ttl=effective_ttl,
            metadata=metadata,
        )
        entry_data = dataclasses.asdict(entry)
        entry_data["scope"] = entry.scope.value
        await self.backend.save("shared_entries", entry.entry_id, entry_data)
        return entry

    async def recall(
        self,
        *,
        agent_id: str | None = None,
        source: ShareScope | None = None,
        shared_by: str | None = None,
        limit: int = 20,
    ) -> list[SharedEntry]:
        """Retrieve accessible shared entries."""
        all_data = await self.backend.list("shared_entries")
        entries = []
        for d in all_data:
            d = dict(d)
            d["scope"] = ShareScope(d["scope"])
            entries.append(SharedEntry(**d))

        # Filter expired
        entries = [e for e in entries if not e.is_expired()]

        if agent_id is not None:
            entries = [
                e for e in entries
                if e.scope != ShareScope.TARGETED or agent_id in e.targets
            ]
        if source is not None:
            entries = [e for e in entries if e.scope == source]
        if shared_by is not None:
            entries = [e for e in entries if e.shared_by == shared_by]

        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    async def size(self) -> int:
        """Number of entries in shared memory."""
        items = await self.backend.list("shared_entries")
        return len(items)

    async def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        all_data = await self.backend.list("shared_entries")
        entries = []
        for d in all_data:
            d = dict(d)
            d["scope"] = ShareScope(d["scope"])
            entries.append(SharedEntry(**d))

        expired = [e for e in entries if e.is_expired()]
        for e in expired:
            await self.backend.delete("shared_entries", e.entry_id)
        return len(expired)

    def __repr__(self) -> str:
        return f"AsyncSharedMemory(team={self.team_id!r})"
