"""
Agent — The primary interface for using eachmind.

An Agent wraps the core primitives into a coherent interface. Each Agent
has its own PrivateMemory, Perspective, and Consolidation engine. Agents
interact with SharedMemory to selectively publish knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import EncodedMemory, Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import SharedEntry, SharedMemory, ShareScope


@dataclass
class Agent:
    """An agent with its own private memory and perspective.

    The Agent class is the primary user-facing interface for eachmind.
    It combines a Perspective (how the agent sees the world), PrivateMemory
    (what the agent remembers), and Consolidation (how memories become beliefs).

    Agents observe events, build private understanding, and selectively
    share knowledge with others through SharedMemory.

    Attributes:
        name: Human-readable agent name.
        role: The agent's designated role.
        perspective: The agent's encoding lens.
        private_memory: The agent's personal memory store.
        consolidation: The agent's belief formation engine.
        shared_memory: Reference to the team's shared memory (optional).
    """

    name: str
    role: str = "general"
    perspective: Perspective = field(default=None)
    private_memory: PrivateMemory = field(default=None)
    consolidation: Consolidation = field(default=None)
    shared_memory: SharedMemory | None = None

    def __post_init__(self) -> None:
        if self.perspective is None:
            self.perspective = Perspective(role=self.role)
        if self.private_memory is None:
            self.private_memory = PrivateMemory(agent_id=self.name)
        if self.consolidation is None:
            self.consolidation = Consolidation()

    def observe(self, event: MemoryEvent) -> EncodedMemory:
        """Observe an event and encode it through this agent's Perspective.

        The event is encoded privately — no other agent sees this encoding.
        The encoding reflects this agent's role, history, and accumulated beliefs.

        Args:
            event: The event to observe.

        Returns:
            The EncodedMemory stored in private memory.
        """
        encoded = self.perspective.encode(event)
        self.private_memory.store(encoded)
        return encoded

    def recall(
        self,
        *,
        source: ShareScope | type[SharedMemory] | None = None,
        min_salience: float = 0.0,
        limit: int = 10,
    ) -> list[EncodedMemory] | list[SharedEntry]:
        """Recall memories from private or shared memory.

        Args:
            source: If SharedMemory.TEAM or similar, recall from shared memory.
                    If None, recall from private memory.
            min_salience: Minimum salience for private memory recall.
            limit: Maximum results.

        Returns:
            List of memories or shared entries.
        """
        if isinstance(source, ShareScope) and self.shared_memory is not None:
            return self.shared_memory.recall(agent_id=self.name, source=source, limit=limit)

        if source is SharedMemory and self.shared_memory is not None:
            return self.shared_memory.recall(agent_id=self.name, limit=limit)

        return self.private_memory.recall(min_salience=min_salience, limit=limit)

    def share(
        self,
        content: Any,
        to: ShareScope = ShareScope.TEAM,
        targets: list[str] | None = None,
        reason: str = "",
    ) -> SharedEntry | None:
        """Share a piece of knowledge with other agents.

        This is a deliberate act — the agent actively decides to publish
        something to shared memory. Nothing is shared automatically.

        Args:
            content: The knowledge to share.
            to: The sharing scope.
            targets: Specific agents (for TARGETED scope).
            reason: Why the agent is sharing this.

        Returns:
            The SharedEntry if shared memory is connected, else None.
        """
        if self.shared_memory is None:
            return None

        return self.shared_memory.publish(
            content=content,
            shared_by=self.name,
            scope=to,
            targets=targets,
            reason=reason,
        )

    def consolidate(self) -> None:
        """Consolidate private memories into durable beliefs.

        Triggers the consolidation process, which scans private memory
        for patterns and forms (or reinforces) beliefs that update
        this agent's Perspective over time.
        """
        beliefs = self.consolidation.process(self.private_memory.memories)

        # Feed beliefs back into perspective as priors
        for belief in beliefs:
            for tag in belief.tags:
                self.perspective.priors[tag] = belief.confidence

    def connect(self, shared_memory: SharedMemory) -> None:
        """Connect this agent to a shared memory space.

        Args:
            shared_memory: The team's shared memory to connect to.
        """
        self.shared_memory = shared_memory

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, role={self.role!r}, "
            f"memories={self.private_memory.size})"
        )
