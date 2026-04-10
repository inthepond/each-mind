"""
Drift — Measuring how agents naturally diverge in perspective over time.

Even agents on the same team, observing the same events, will develop
different perspectives. This divergence — Drift — is a feature, not a bug.
It means the team develops genuine cognitive diversity. Drift is measurable,
trackable, and can be used to understand team dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from eachmind.primitives.perspective import Perspective


@dataclass
class DriftMeasurement:
    """A single measurement of drift between two agents at a point in time.

    Attributes:
        agent_a: First agent's identifier.
        agent_b: Second agent's identifier.
        drift_value: The measured divergence (0.0 = identical, 1.0 = max).
        timestamp: When the measurement was taken.
        components: Breakdown of what's contributing to drift.
    """

    agent_a: str
    agent_b: str
    drift_value: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Drift:
    """Tracks and analyzes perspective divergence across agents over time.

    Drift provides tools for measuring, recording, and analyzing how
    agents' perspectives diverge. This is useful for understanding team
    cognitive diversity, detecting groupthink (too little drift), or
    detecting fragmentation (too much drift).

    Attributes:
        measurements: History of drift measurements.
    """

    measurements: list[DriftMeasurement] = field(default_factory=list)

    def measure(
        self,
        agent_a_id: str,
        perspective_a: Perspective,
        agent_b_id: str,
        perspective_b: Perspective,
    ) -> DriftMeasurement:
        """Measure current drift between two agents.

        Args:
            agent_a_id: First agent's identifier.
            perspective_a: First agent's current Perspective.
            agent_b_id: Second agent's identifier.
            perspective_b: Second agent's current Perspective.

        Returns:
            A DriftMeasurement capturing the current divergence.
        """
        drift_value = perspective_a.drift_from(perspective_b)

        measurement = DriftMeasurement(
            agent_a=agent_a_id,
            agent_b=agent_b_id,
            drift_value=drift_value,
            components={
                "roles_differ": 0.0 if perspective_a.role == perspective_b.role else 1.0,
                "history_divergence": (
                    0.0
                    if perspective_a.history_hash == perspective_b.history_hash
                    else 1.0
                ),
            },
        )

        self.measurements.append(measurement)
        return measurement

    def history(
        self,
        agent_a: str | None = None,
        agent_b: str | None = None,
        limit: int = 50,
    ) -> list[DriftMeasurement]:
        """Retrieve drift measurement history.

        Args:
            agent_a: Filter by first agent.
            agent_b: Filter by second agent.
            limit: Maximum measurements to return.

        Returns:
            Matching measurements, most recent first.
        """
        results = self.measurements

        if agent_a is not None:
            results = [m for m in results if m.agent_a == agent_a or m.agent_b == agent_a]

        if agent_b is not None:
            results = [m for m in results if m.agent_a == agent_b or m.agent_b == agent_b]

        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results[:limit]

    def team_diversity(self, perspectives: dict[str, Perspective]) -> float:
        """Compute overall team cognitive diversity.

        Measures the average pairwise drift across all agents. Higher
        values mean more diverse thinking; lower values suggest groupthink.

        Args:
            perspectives: Mapping of agent_id to their current Perspective.

        Returns:
            Average drift value (0.0 to 1.0).
        """
        agents = list(perspectives.keys())
        if len(agents) < 2:
            return 0.0

        total_drift = 0.0
        pair_count = 0

        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                measurement = self.measure(
                    agent_a, perspectives[agent_a],
                    agent_b, perspectives[agent_b],
                )
                total_drift += measurement.drift_value
                pair_count += 1

        return total_drift / pair_count if pair_count > 0 else 0.0

    def __repr__(self) -> str:
        return f"Drift(measurements={len(self.measurements)})"
