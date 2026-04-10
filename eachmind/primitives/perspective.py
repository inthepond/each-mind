"""
Perspective — The lens through which an agent encodes events.

Each agent's Perspective is shaped by its role, history, and accumulated
experience. Two agents observing the same event will produce different
encodings because their Perspectives differ. Over time, Perspectives
naturally diverge (Drift) even among agents on the same team.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from eachmind.primitives.memory_event import MemoryEvent


@dataclass
class EncodedMemory:
    """The result of encoding a MemoryEvent through a Perspective.

    This is what actually gets stored in an agent's PrivateMemory —
    not the raw event, but the agent's interpretation of it.
    """

    event_id: str
    encoded_content: Any
    perspective_hash: str
    salience: float  # 0.0 to 1.0 — how important this felt to the agent
    associations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Perspective:
    """The encoding lens of an individual agent.

    A Perspective defines how an agent interprets and stores experiences.
    It is shaped by the agent's role, its accumulated beliefs, and the
    patterns it has seen before. Perspectives are never shared directly —
    they are inferred from an agent's behavior and outputs.

    Attributes:
        role: The agent's designated role (e.g., "analyst", "writer").
        priors: Accumulated beliefs and biases from past experience.
        encoding_weights: How much emphasis the agent places on different
            aspects of an event (e.g., {"numerical_data": 0.9, "sentiment": 0.3}).
        history_hash: A rolling hash of encoded memories, representing the
            cumulative effect of experience on this Perspective.
    """

    role: str
    priors: dict[str, Any] = field(default_factory=dict)
    encoding_weights: dict[str, float] = field(default_factory=dict)
    history_hash: str = field(default_factory=lambda: hashlib.sha256(b"genesis").hexdigest()[:16])
    _encoding_count: int = field(default=0, repr=False)

    def encode(self, event: MemoryEvent) -> EncodedMemory:
        """Encode a MemoryEvent through this Perspective.

        The encoding process applies role-specific interpretation,
        weights different aspects of the event, and incorporates
        accumulated priors. The result is a subjective encoding
        unique to this agent's viewpoint.

        Args:
            event: The raw event to encode.

        Returns:
            An EncodedMemory representing this agent's interpretation.
        """
        salience = self._compute_salience(event)
        associations = self._find_associations(event)
        encoded_content = self._apply_encoding(event)

        self._update_history(event)
        self._encoding_count += 1

        return EncodedMemory(
            event_id=event.event_id,
            encoded_content=encoded_content,
            perspective_hash=self.history_hash,
            salience=salience,
            associations=associations,
            metadata={
                "role": self.role,
                "encoding_number": self._encoding_count,
            },
        )

    def drift_from(self, other: Perspective) -> float:
        """Measure how much this Perspective has diverged from another.

        Returns a value between 0.0 (identical) and 1.0 (maximally diverged).
        Drift is computed based on differences in encoding weights, priors,
        and accumulated history.

        Args:
            other: Another Perspective to compare against.

        Returns:
            A float representing the degree of divergence.
        """
        if self.history_hash == other.history_hash:
            return 0.0

        # Weight divergence
        all_keys = set(self.encoding_weights) | set(other.encoding_weights)
        if not all_keys:
            weight_drift = 0.0
        else:
            weight_diffs = []
            for key in all_keys:
                w1 = self.encoding_weights.get(key, 0.0)
                w2 = other.encoding_weights.get(key, 0.0)
                weight_diffs.append(abs(w1 - w2))
            weight_drift = sum(weight_diffs) / len(weight_diffs)

        # Prior divergence
        all_priors = set(self.priors) | set(other.priors)
        if not all_priors:
            prior_drift = 0.0
        else:
            shared = set(self.priors) & set(other.priors)
            prior_drift = 1.0 - (len(shared) / len(all_priors))

        # History divergence (binary — either same path or not)
        history_drift = 0.0 if self.history_hash == other.history_hash else 0.5

        return min(1.0, (weight_drift + prior_drift + history_drift) / 3.0)

    def _compute_salience(self, event: MemoryEvent) -> float:
        """Determine how important this event is to this agent's role."""
        base_salience = 0.5

        # Role-relevant sources boost salience
        role_sources = self.priors.get("relevant_sources", [])
        if event.source in role_sources:
            base_salience += 0.3

        return min(1.0, base_salience)

    def _find_associations(self, event: MemoryEvent) -> list[str]:
        """Find connections between this event and existing knowledge."""
        associations = []
        content_str = str(event.content).lower()

        for prior_key, _prior_value in self.priors.items():
            if prior_key.lower() in content_str:
                associations.append(prior_key)

        return associations

    def _apply_encoding(self, event: MemoryEvent) -> dict[str, Any]:
        """Transform the raw event content through this Perspective's lens."""
        return {
            "raw": event.content,
            "role_context": self.role,
            "weighted_aspects": {
                k: v for k, v in self.encoding_weights.items()
            },
            "source_trust": self.priors.get("source_trust", {}).get(event.source, 0.5),
        }

    def _update_history(self, event: MemoryEvent) -> None:
        """Update the rolling history hash after encoding an event."""
        combined = f"{self.history_hash}:{event.event_id}"
        self.history_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
