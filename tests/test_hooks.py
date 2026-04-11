"""Tests for observability hooks."""
from __future__ import annotations

from eachmind import Agent, MemoryEvent, SharedMemory
from eachmind.hooks import LoggingHook, MemoryHook


class RecordingHook:
    """Test hook that records all calls."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def on_observe(
        self, agent_name: str, event_id: str, salience: float
    ) -> None:
        self.events.append(("observe", {
            "agent": agent_name, "event_id": event_id,
            "salience": salience,
        }))

    def on_share(
        self, agent_name: str, content: str, scope: str
    ) -> None:
        self.events.append(("share", {
            "agent": agent_name, "content": content,
            "scope": scope,
        }))

    def on_recall(
        self, agent_name: str, count: int, source: str
    ) -> None:
        self.events.append(("recall", {
            "agent": agent_name, "count": count,
            "source": source,
        }))

    def on_consolidate(
        self, agent_name: str, belief_count: int
    ) -> None:
        self.events.append(("consolidate", {
            "agent": agent_name, "belief_count": belief_count,
        }))


class TestMemoryHookProtocol:
    def test_recording_hook_satisfies_protocol(self):
        assert isinstance(RecordingHook(), MemoryHook)

    def test_logging_hook_satisfies_protocol(self):
        assert isinstance(LoggingHook(), MemoryHook)


class TestAgentWithHooks:
    def test_observe_triggers_hook(self):
        hook = RecordingHook()
        agent = Agent(name="test", role="analyst", hooks=[hook])
        event = MemoryEvent(content="data", source="test")
        agent.observe(event)
        assert len(hook.events) == 1
        assert hook.events[0][0] == "observe"
        assert hook.events[0][1]["agent"] == "test"

    def test_share_triggers_hook(self):
        hook = RecordingHook()
        shared = SharedMemory(team_id="team")
        agent = Agent(
            name="test", role="analyst",
            shared_memory=shared, hooks=[hook],
        )
        agent.share(content="insight", to=SharedMemory.TEAM)
        share_events = [e for e in hook.events if e[0] == "share"]
        assert len(share_events) == 1

    def test_consolidate_triggers_hook(self):
        hook = RecordingHook()
        agent = Agent(name="test", role="analyst", hooks=[hook])
        for i in range(5):
            agent.observe(
                MemoryEvent(content=f"data {i}", source="test")
            )
        agent.consolidate()
        consolidate_events = [
            e for e in hook.events if e[0] == "consolidate"
        ]
        assert len(consolidate_events) == 1

    def test_multiple_hooks(self):
        hook1 = RecordingHook()
        hook2 = RecordingHook()
        agent = Agent(
            name="test", role="analyst", hooks=[hook1, hook2]
        )
        agent.observe(MemoryEvent(content="data", source="test"))
        assert len(hook1.events) == 1
        assert len(hook2.events) == 1

    def test_no_hooks_by_default(self):
        agent = Agent(name="test", role="analyst")
        event = MemoryEvent(content="data", source="test")
        encoded = agent.observe(event)
        assert encoded is not None

    def test_broken_hook_doesnt_crash(self):
        class BrokenHook:
            def on_observe(self, *args, **kwargs):
                raise RuntimeError("broken!")
            def on_share(self, *args, **kwargs):
                raise RuntimeError("broken!")
            def on_recall(self, *args, **kwargs):
                raise RuntimeError("broken!")
            def on_consolidate(self, *args, **kwargs):
                raise RuntimeError("broken!")

        agent = Agent(name="test", role="analyst", hooks=[BrokenHook()])
        encoded = agent.observe(MemoryEvent(content="data", source="test"))
        assert encoded is not None  # didn't crash


class TestLoggingHook:
    def test_does_not_raise(self):
        hook = LoggingHook()
        hook.on_observe("test", "event-123", 0.8)
        hook.on_share("test", "insight", "team")
        hook.on_recall("test", 5, "private")
        hook.on_consolidate("test", 3)
