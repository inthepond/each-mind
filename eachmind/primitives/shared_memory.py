"""
SharedMemory — What agents explicitly publish to the collective.

Shared memory is opt-in, not default. Agents must deliberately choose
to share a piece of knowledge. This preserves the "private by default"
principle while enabling genuine collaboration when agents decide it's
valuable for the team.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ShareScope(Enum):
    """Defines who can access a shared memory entry."""

    TEAM = "team"           # All agents in the team
    TARGETED = "targeted"   # Specific named agents only
    BROADCAST = "broadcast" # All agents across all teams


@dataclass
class SharedEntry:
    """A single entry in shared memory.

    Represents something an agent chose to share with others.
    Unlike PrivateMemory, shared entries include attribution so
    recipients know who shared what and why.
    """

    content: Any
    shared_by: str  # agent_id of the sharer
    scope: ShareScope = ShareScope.TEAM
    targets: list[str] = field(default_factory=list)  # for TARGETED scope
    reason: str = ""  # why the agent chose to share this
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl: int | None = None  # time to live in seconds; None = never expires

    def is_expired(self) -> bool:
        """Check if this entry has expired based on its TTL."""
        if self.ttl is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
        return elapsed >= self.ttl


@dataclass
class SharedMemory:
    """The collective memory space where agents publish knowledge.

    SharedMemory is a team-level store that agents can write to (share)
    and read from (recall). It implements the opt-in sharing protocol:
    nothing enters SharedMemory without an agent explicitly publishing it.

    Attributes:
        team_id: Identifier for the team this shared memory belongs to.
        entries: All shared memory entries.
    """

    TEAM = ShareScope.TEAM
    TARGETED = ShareScope.TARGETED
    BROADCAST = ShareScope.BROADCAST

    team_id: str = "default"
    entries: list[SharedEntry] = field(default_factory=list)
    backend: Any = None
    default_ttl: int | None = None  # default TTL for new entries

    def __post_init__(self) -> None:
        """Hydrate entries from backend if one is provided."""
        if self.backend is not None and not self.entries:
            for data in self.backend.list("shared_entries"):
                data = dict(data)  # shallow copy to avoid mutating backend data
                data["scope"] = ShareScope(data["scope"])
                self.entries.append(SharedEntry(**data))

    def publish(
        self,
        content: Any,
        shared_by: str,
        scope: ShareScope = ShareScope.TEAM,
        targets: list[str] | None = None,
        reason: str = "",
        ttl: int | None = None,
        **metadata: Any,
    ) -> SharedEntry:
        """Publish a piece of knowledge to shared memory.

        Args:
            content: The knowledge to share.
            shared_by: Agent ID of the sharing agent.
            scope: Who should be able to see this.
            targets: Specific agent IDs (for TARGETED scope).
            reason: Why the agent is sharing this.
            ttl: Time to live in seconds. Falls back to default_ttl if not set.
            **metadata: Additional context.

        Returns:
            The created SharedEntry.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        entry = SharedEntry(
            content=content,
            shared_by=shared_by,
            scope=scope,
            targets=targets or [],
            reason=reason,
            metadata=metadata,
            ttl=effective_ttl,
        )
        self.entries.append(entry)

        if self.backend is not None:
            entry_data = dataclasses.asdict(entry)
            entry_data["scope"] = entry.scope.value
            self.backend.save("shared_entries", entry.entry_id, entry_data)

        return entry

    def recall(
        self,
        *,
        agent_id: str | None = None,
        source: ShareScope | None = None,
        shared_by: str | None = None,
        limit: int = 20,
    ) -> list[SharedEntry]:
        """Retrieve shared knowledge accessible to the requesting agent.

        Args:
            agent_id: The agent requesting recall (for access filtering).
            source: Filter by share scope.
            shared_by: Filter by who shared the entry.
            limit: Maximum entries to return.

        Returns:
            List of accessible SharedEntry objects, most recent first.
        """
        results = [e for e in self.entries if not e.is_expired()]

        if agent_id is not None:
            results = [
                e for e in results
                if e.scope != ShareScope.TARGETED or agent_id in e.targets
            ]

        if source is not None:
            results = [e for e in results if e.scope == source]

        if shared_by is not None:
            results = [e for e in results if e.shared_by == shared_by]

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def cleanup(self) -> int:
        """Remove expired entries. Returns count of entries removed."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        # Also clean from backend if present
        if self.backend is not None:
            # Re-persist the cleaned list
            self.backend.clear("shared_entries")
            for entry in self.entries:
                entry_data = dataclasses.asdict(entry)
                entry_data["scope"] = entry.scope.value
                self.backend.save("shared_entries", entry.entry_id, entry_data)
        return before - len(self.entries)

    @property
    def size(self) -> int:
        """Number of entries in shared memory."""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"SharedMemory(team={self.team_id!r}, entries={self.size})"
