"""
Consolidation — How repeated private experiences abstract into durable beliefs.

Over time, patterns in an agent's private memory consolidate into stable
beliefs and priors. This mirrors how human memory works: individual episodes
fade, but the patterns they form persist as intuition and expertise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from eachmind.primitives.perspective import EncodedMemory


@dataclass
class Belief:
    """A consolidated belief derived from repeated experiences.

    Beliefs are durable abstractions that emerge when an agent observes
    consistent patterns across multiple memory events. They update the
    agent's Perspective over time, shaping how future events are encoded.

    Attributes:
        content: The belief statement or pattern.
        confidence: How strongly held (0.0 to 1.0), based on evidence count.
        evidence_count: How many memories support this belief.
        source_events: Event IDs that contributed to this belief.
        tags: Categorization labels.
    """

    content: str
    confidence: float = 0.5
    evidence_count: int = 1
    source_events: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def reinforce(self, event_id: str) -> None:
        """Strengthen this belief with additional evidence."""
        self.evidence_count += 1
        self.source_events.append(event_id)
        # Confidence grows logarithmically — diminishing returns
        self.confidence = min(1.0, 0.5 + 0.15 * (self.evidence_count ** 0.5))

    def decay(self, factor: float = 0.05) -> None:
        """Slightly weaken this belief over time without reinforcement."""
        self.confidence = max(0.1, self.confidence - factor)


@dataclass
class Consolidation:
    """Manages the consolidation of private memories into durable beliefs.

    Consolidation is the process by which repeated patterns in an agent's
    PrivateMemory abstract into stable beliefs. These beliefs then feed
    back into the agent's Perspective, influencing how future events are
    encoded — creating a self-reinforcing cycle of learning.

    Attributes:
        beliefs: Currently held beliefs.
        consolidation_threshold: Minimum pattern frequency to form a belief.
    """

    beliefs: list[Belief] = field(default_factory=list)
    consolidation_threshold: int = 3

    def process(self, memories: list[EncodedMemory]) -> list[Belief]:
        """Analyze memories and consolidate patterns into beliefs.

        Scans through a set of encoded memories, identifies recurring
        patterns (sources, associations, high-salience themes), and
        either creates new beliefs or reinforces existing ones.

        Args:
            memories: The encoded memories to analyze for patterns.

        Returns:
            List of newly created or reinforced beliefs.
        """
        updated: list[Belief] = []

        # Consolidate by recurring associations
        association_counts: Counter[str] = Counter()
        association_events: dict[str, list[str]] = {}

        for memory in memories:
            for assoc in memory.associations:
                association_counts[assoc] += 1
                association_events.setdefault(assoc, []).append(memory.event_id)

        for pattern, count in association_counts.items():
            if count >= self.consolidation_threshold:
                belief = self._find_or_create_belief(
                    content=f"Pattern: {pattern} is recurrently relevant",
                    tags=[pattern],
                )
                for eid in association_events[pattern]:
                    belief.reinforce(eid)
                updated.append(belief)

        # Consolidate by high-salience source patterns
        source_counts: Counter[str] = Counter()
        for memory in memories:
            role = memory.metadata.get("role", "")
            if memory.salience >= 0.7:
                source_counts[role] += 1

        for role_pattern, count in source_counts.items():
            if count >= self.consolidation_threshold and role_pattern:
                belief = self._find_or_create_belief(
                    content=f"High-salience events frequently come from role context: {role_pattern}",
                    tags=["salience", role_pattern],
                )
                updated.append(belief)

        return updated

    def get_beliefs(
        self,
        *,
        min_confidence: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[Belief]:
        """Retrieve beliefs matching criteria.

        Args:
            min_confidence: Minimum confidence threshold.
            tags: Filter by tags (any match).

        Returns:
            Matching beliefs sorted by confidence (desc).
        """
        results = self.beliefs

        if min_confidence > 0.0:
            results = [b for b in results if b.confidence >= min_confidence]

        if tags:
            tag_set = set(tags)
            results = [b for b in results if tag_set & set(b.tags)]

        return sorted(results, key=lambda b: b.confidence, reverse=True)

    def decay_all(self, factor: float = 0.05) -> None:
        """Apply time-based decay to all beliefs."""
        for belief in self.beliefs:
            belief.decay(factor)

    def _find_or_create_belief(self, content: str, tags: list[str]) -> Belief:
        """Find an existing belief or create a new one."""
        for belief in self.beliefs:
            if set(belief.tags) & set(tags):
                return belief

        belief = Belief(content=content, tags=tags)
        self.beliefs.append(belief)
        return belief

    def __repr__(self) -> str:
        return f"Consolidation(beliefs={len(self.beliefs)})"
