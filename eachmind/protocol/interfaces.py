"""
Protocol interfaces for eachmind.

Defines structural contracts (PEP 544 Protocols) that the core primitives
satisfy. These enable structural subtyping — any class that implements the
required methods is considered compliant, even without explicit inheritance.

All protocols are decorated with @runtime_checkable so isinstance() checks
work at runtime, not just at static analysis time.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import EncodedMemory, Perspective
from eachmind.primitives.shared_memory import ShareScope, SharedEntry
from eachmind.primitives.consolidation import Belief
from eachmind.primitives.drift import DriftMeasurement


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Contract for a private, per-agent memory store.

    Any class implementing store, recall, search, and size satisfies
    this protocol. The canonical implementation is PrivateMemory.
    """

    def store(self, encoded: EncodedMemory) -> None:
        """Store an encoded memory."""
        ...

    def recall(
        self,
        *,
        event_id: str | None = None,
        min_salience: float = 0.0,
        limit: int = 10,
    ) -> list[EncodedMemory]:
        """Retrieve memories matching the given criteria."""
        ...

    def search(self, query: str) -> list[EncodedMemory]:
        """Search memories by content similarity."""
        ...

    @property
    def size(self) -> int:
        """Number of memories currently stored."""
        ...


@runtime_checkable
class SharedStoreProtocol(Protocol):
    """Contract for a collective, opt-in shared memory store.

    Any class implementing publish, recall, and size satisfies this
    protocol. The canonical implementation is SharedMemory.
    """

    def publish(
        self,
        content: Any,
        shared_by: str,
        scope: ShareScope = ShareScope.TEAM,
        targets: list[str] | None = None,
        reason: str = "",
        **metadata: Any,
    ) -> SharedEntry:
        """Publish a piece of knowledge to shared memory."""
        ...

    def recall(
        self,
        *,
        agent_id: str | None = None,
        source: ShareScope | None = None,
        shared_by: str | None = None,
        limit: int = 20,
    ) -> list[SharedEntry]:
        """Retrieve shared knowledge accessible to the requesting agent."""
        ...

    @property
    def size(self) -> int:
        """Number of entries in shared memory."""
        ...


@runtime_checkable
class MemoryEncoderProtocol(Protocol):
    """Contract for the encoding lens of an individual agent.

    Any class implementing role, encode, and drift_from satisfies this
    protocol. The canonical implementation is Perspective.
    """

    @property
    def role(self) -> str:
        """The agent's designated role."""
        ...

    def encode(self, event: MemoryEvent) -> EncodedMemory:
        """Encode a MemoryEvent through this perspective."""
        ...

    def drift_from(self, other: Perspective) -> float:
        """Measure how much this encoder has diverged from another."""
        ...


@runtime_checkable
class ConsolidationEngineProtocol(Protocol):
    """Contract for consolidating private memories into durable beliefs.

    Any class implementing process and get_beliefs satisfies this
    protocol. The canonical implementation is Consolidation.
    """

    def process(self, memories: list[EncodedMemory]) -> list[Belief]:
        """Analyze memories and consolidate patterns into beliefs."""
        ...

    def get_beliefs(
        self,
        *,
        min_confidence: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[Belief]:
        """Retrieve beliefs matching criteria."""
        ...


@runtime_checkable
class DriftTrackerProtocol(Protocol):
    """Contract for tracking and analyzing perspective divergence.

    Any class implementing measure and team_diversity satisfies this
    protocol. The canonical implementation is Drift.
    """

    def measure(
        self,
        agent_a_id: str,
        perspective_a: Perspective,
        agent_b_id: str,
        perspective_b: Perspective,
    ) -> DriftMeasurement:
        """Measure current drift between two agents."""
        ...

    def team_diversity(self, perspectives: dict[str, Perspective]) -> float:
        """Compute overall team cognitive diversity."""
        ...
