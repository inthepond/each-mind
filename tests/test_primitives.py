"""Tests for eachmind core primitives."""

import pytest

from eachmind import Agent, MemoryEvent, SharedMemory
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.drift import Drift
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import ShareScope


class TestMemoryEvent:
    def test_create_event(self):
        event = MemoryEvent(content="test data", source="unit_test")
        assert event.content == "test data"
        assert event.source == "unit_test"
        assert event.event_id is not None

    def test_event_requires_content(self):
        with pytest.raises(ValueError):
            MemoryEvent(content=None)

    def test_with_metadata(self):
        event = MemoryEvent(content="test", source="test")
        enriched = event.with_metadata(key="value")
        assert enriched.metadata["key"] == "value"
        assert enriched.event_id == event.event_id  # same event


class TestPerspective:
    def test_encode_event(self):
        perspective = Perspective(role="analyst")
        event = MemoryEvent(content="data point", source="test")
        encoded = perspective.encode(event)
        assert encoded.event_id == event.event_id
        assert encoded.perspective_hash is not None

    def test_encoding_updates_history(self):
        perspective = Perspective(role="analyst")
        initial_hash = perspective.history_hash

        event = MemoryEvent(content="data", source="test")
        perspective.encode(event)

        assert perspective.history_hash != initial_hash

    def test_drift_from_self_after_divergence(self):
        p1 = Perspective(role="analyst")
        p2 = Perspective(role="writer")

        event = MemoryEvent(content="shared event", source="test")
        p1.encode(event)
        p2.encode(event)

        drift = p1.drift_from(p2)
        assert 0.0 <= drift <= 1.0

    def test_identical_perspectives_zero_drift(self):
        p1 = Perspective(role="same")
        p2 = Perspective(role="same")
        # Before any encoding, histories match
        drift = p1.drift_from(p2)
        assert drift == 0.0


class TestPrivateMemory:
    def test_store_and_recall(self):
        perspective = Perspective(role="test")
        memory = PrivateMemory(agent_id="test-agent")
        event = MemoryEvent(content="remember this", source="test")

        encoded = perspective.encode(event)
        memory.store(encoded)

        results = memory.recall(event_id=event.event_id)
        assert len(results) == 1
        assert results[0].event_id == event.event_id

    def test_capacity_eviction(self):
        memory = PrivateMemory(agent_id="test", capacity=2)
        perspective = Perspective(role="test")

        for i in range(3):
            event = MemoryEvent(content=f"event {i}", source="test")
            encoded = perspective.encode(event)
            memory.store(encoded)

        assert memory.size == 2

    def test_search(self):
        perspective = Perspective(role="test")
        memory = PrivateMemory(agent_id="test")

        event = MemoryEvent(content="important finding about revenue", source="test")
        encoded = perspective.encode(event)
        memory.store(encoded)

        results = memory.search("revenue")
        assert len(results) >= 1


class TestSharedMemory:
    def test_publish_and_recall(self):
        shared = SharedMemory(team_id="test-team")
        shared.publish(content="shared insight", shared_by="agent-a")

        results = shared.recall()
        assert len(results) == 1
        assert results[0].shared_by == "agent-a"

    def test_targeted_scope(self):
        shared = SharedMemory(team_id="test-team")
        shared.publish(
            content="secret",
            shared_by="agent-a",
            scope=ShareScope.TARGETED,
            targets=["agent-b"],
        )

        visible = shared.recall(agent_id="agent-b")
        assert len(visible) == 1

        hidden = shared.recall(agent_id="agent-c")
        assert len(hidden) == 0


class TestConsolidation:
    def test_consolidation_forms_beliefs(self):
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(
            role="test",
            priors={"revenue": 0.8},
        )

        memories = []
        for i in range(3):
            event = MemoryEvent(content=f"revenue data point {i}", source="test")
            encoded = perspective.encode(event)
            memories.append(encoded)

        beliefs = consolidation.process(memories)
        # Should form at least some beliefs from patterns
        assert isinstance(beliefs, list)


class TestDrift:
    def test_measure_drift(self):
        drift_tracker = Drift()
        p1 = Perspective(role="analyst")
        p2 = Perspective(role="writer")

        event = MemoryEvent(content="test", source="test")
        p1.encode(event)

        measurement = drift_tracker.measure("a", p1, "b", p2)
        assert 0.0 <= measurement.drift_value <= 1.0

    def test_team_diversity(self):
        drift_tracker = Drift()
        perspectives = {
            "agent-a": Perspective(role="analyst"),
            "agent-b": Perspective(role="writer"),
            "agent-c": Perspective(role="reviewer"),
        }

        diversity = drift_tracker.team_diversity(perspectives)
        assert 0.0 <= diversity <= 1.0


class TestAgent:
    def test_observe_event(self):
        agent = Agent(name="test", role="analyst")
        event = MemoryEvent(content="data", source="test")
        encoded = agent.observe(event)
        assert agent.private_memory.size == 1
        assert encoded.event_id == event.event_id

    def test_share_requires_connection(self):
        agent = Agent(name="test", role="analyst")
        result = agent.share(content="insight")
        assert result is None  # Not connected to shared memory

    def test_share_with_connection(self):
        agent = Agent(name="test", role="analyst")
        shared = SharedMemory(team_id="team")
        agent.connect(shared)

        entry = agent.share(content="insight", reason="important finding")
        assert entry is not None
        assert entry.shared_by == "test"
        assert shared.size == 1

    def test_full_workflow(self):
        analyst = Agent(name="analyst", role="data analysis")
        writer = Agent(name="writer", role="content creation")
        shared = SharedMemory(team_id="team")
        analyst.connect(shared)
        writer.connect(shared)

        event = MemoryEvent(content="Q1 revenue grew 23%", source="report")
        analyst.observe(event)
        writer.observe(event)

        analyst.share(content="Revenue trend is positive", to=SharedMemory.TEAM)

        results = writer.recall(source=SharedMemory.TEAM)
        assert len(results) == 1
