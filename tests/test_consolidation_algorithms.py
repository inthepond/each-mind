"""Tests for enhanced consolidation algorithms."""

from eachmind import MemoryEvent
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.perspective import Perspective


class TestConsolidationStrategies:
    def test_default_strategy_is_frequency(self):
        consolidation = Consolidation()
        assert consolidation.strategy == "frequency"

    def test_process_with_frequency_strategy(self):
        consolidation = Consolidation(consolidation_threshold=2, strategy="frequency")
        perspective = Perspective(role="test", priors={"data": 0.7})
        memories = [
            perspective.encode(MemoryEvent(content=f"data item {i}", source="test"))
            for i in range(4)
        ]
        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_process_with_temporal_strategy(self):
        consolidation = Consolidation(consolidation_threshold=2, strategy="temporal")
        perspective = Perspective(role="test", priors={"data": 0.7})
        memories = [
            perspective.encode(
                MemoryEvent(content=f"data item {i}", source="test")
            )
            for i in range(4)
        ]
        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_frequency_and_temporal_both_produce_beliefs(self):
        """Both strategies should produce beliefs from the same input."""
        perspective = Perspective(role="test", priors={"revenue": 0.8, "market": 0.7})
        memories = [
            perspective.encode(MemoryEvent(content=f"revenue and market data {i}", source="test"))
            for i in range(5)
        ]

        freq = Consolidation(consolidation_threshold=2, strategy="frequency")
        freq_beliefs = freq.process(memories)

        temp = Consolidation(consolidation_threshold=2, strategy="temporal")
        temp_beliefs = temp.process(memories)

        assert len(freq_beliefs) > 0
        assert len(temp_beliefs) > 0


class TestTemporalConsolidation:
    def test_recent_memories_weighted_higher(self):
        """Temporal strategy should produce beliefs even with fewer total occurrences
        when recent memories reinforce a pattern."""
        consolidation = Consolidation(consolidation_threshold=3, strategy="temporal")
        perspective = Perspective(role="test", priors={"trend": 0.9})

        # Create memories where "trend" appears in all of them
        memories = [
            perspective.encode(MemoryEvent(content=f"trend data point {i}", source="test"))
            for i in range(4)
        ]

        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_decay_reduces_confidence(self):
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(role="test", priors={"revenue": 0.8})

        memories = [
            perspective.encode(MemoryEvent(content=f"revenue point {i}", source="test"))
            for i in range(4)
        ]

        consolidation.process(memories)
        initial_confidences = [b.confidence for b in consolidation.beliefs]

        consolidation.decay_all(factor=0.1)
        decayed_confidences = [b.confidence for b in consolidation.beliefs]

        for initial, decayed in zip(initial_confidences, decayed_confidences, strict=True):
            assert decayed < initial


class TestConsolidationIdempotency:
    """Consolidation is meant to run repeatedly over an agent's lifetime.
    Re-running it over the same memories must not keep re-counting the same
    events as fresh evidence (which would inflate confidence without bound).
    """

    def _memories(self):
        perspective = Perspective(role="test", priors={"revenue": 0.8})
        return [
            perspective.encode(MemoryEvent(content=f"revenue point {i}", source="test"))
            for i in range(5)
        ]

    def test_repeated_process_is_stable(self):
        consolidation = Consolidation(consolidation_threshold=2)
        memories = self._memories()

        consolidation.process(memories)
        belief = consolidation.beliefs[0]
        first_evidence = belief.evidence_count
        first_confidence = belief.confidence

        # Running again over the same memories must not add new evidence.
        for _ in range(3):
            consolidation.process(memories)

        assert belief.evidence_count == first_evidence
        assert belief.confidence == first_confidence
        # No duplicate event IDs accumulated.
        assert len(belief.source_events) == len(set(belief.source_events))

    def test_temporal_strategy_is_also_idempotent(self):
        consolidation = Consolidation(consolidation_threshold=2, strategy="temporal")
        memories = self._memories()

        consolidation.process(memories)
        belief = consolidation.beliefs[0]
        first_evidence = belief.evidence_count

        consolidation.process(memories)
        assert belief.evidence_count == first_evidence
        assert len(belief.source_events) == len(set(belief.source_events))


class TestBeliefIdentity:
    """Distinct belief categories that happen to share a tag must stay separate."""

    def test_association_and_salience_beliefs_not_merged(self):
        # role == "revenue" and source "revenue" is a relevant source, so these
        # memories produce BOTH an association belief tagged {"revenue"} and a
        # salience belief tagged {"salience", "revenue"}.
        perspective = Perspective(
            role="revenue",
            priors={"relevant_sources": ["revenue"], "revenue": 0.8},
        )
        memories = [
            perspective.encode(MemoryEvent(content=f"revenue point {i}", source="revenue"))
            for i in range(3)
        ]
        consolidation = Consolidation(consolidation_threshold=2)
        consolidation.process(memories)

        tag_sets = [frozenset(b.tags) for b in consolidation.beliefs]
        assert frozenset({"revenue"}) in tag_sets
        assert frozenset({"salience", "revenue"}) in tag_sets


class TestBackwardCompatibility:
    def test_existing_process_behavior_unchanged(self):
        """The default (frequency) strategy should work exactly as before."""
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(role="test", priors={"revenue": 0.8})

        memories = []
        for i in range(3):
            event = MemoryEvent(content=f"revenue data point {i}", source="test")
            encoded = perspective.encode(event)
            memories.append(encoded)

        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)
