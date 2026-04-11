"""Tests for async memory primitives."""
from __future__ import annotations

from eachmind.async_primitives import AsyncPrivateMemory, AsyncSharedMemory
from eachmind.backends.async_memory import AsyncInMemoryBackend
from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.shared_memory import ShareScope


class TestAsyncPrivateMemory:
    async def test_store_and_recall(self):
        backend = AsyncInMemoryBackend()
        memory = AsyncPrivateMemory(agent_id="test", backend=backend)
        perspective = Perspective(role="test")
        event = MemoryEvent(content="remember this", source="test")
        encoded = perspective.encode(event)
        await memory.store(encoded)
        results = await memory.recall(event_id=event.event_id)
        assert len(results) == 1
        assert results[0].event_id == event.event_id

    async def test_size(self):
        backend = AsyncInMemoryBackend()
        memory = AsyncPrivateMemory(agent_id="test", backend=backend)
        perspective = Perspective(role="test")
        await memory.store(perspective.encode(
            MemoryEvent(content="a", source="test")
        ))
        await memory.store(perspective.encode(
            MemoryEvent(content="b", source="test")
        ))
        size = await memory.size()
        assert size == 2

    async def test_search(self):
        backend = AsyncInMemoryBackend()
        memory = AsyncPrivateMemory(agent_id="test", backend=backend)
        perspective = Perspective(role="test")
        await memory.store(perspective.encode(
            MemoryEvent(content="revenue grew 23%", source="test")
        ))
        results = await memory.search("revenue")
        assert len(results) >= 1

    async def test_capacity_eviction(self):
        backend = AsyncInMemoryBackend()
        memory = AsyncPrivateMemory(
            agent_id="test", backend=backend, capacity=2
        )
        perspective = Perspective(role="test")
        for i in range(3):
            await memory.store(perspective.encode(
                MemoryEvent(content=f"event {i}", source="test")
            ))
        size = await memory.size()
        assert size == 2


class TestAsyncSharedMemory:
    async def test_publish_and_recall(self):
        backend = AsyncInMemoryBackend()
        shared = AsyncSharedMemory(team_id="test", backend=backend)
        await shared.publish(content="insight", shared_by="agent-a")
        results = await shared.recall()
        assert len(results) == 1
        assert results[0].shared_by == "agent-a"

    async def test_targeted_scope(self):
        backend = AsyncInMemoryBackend()
        shared = AsyncSharedMemory(team_id="test", backend=backend)
        await shared.publish(
            content="secret",
            shared_by="agent-a",
            scope=ShareScope.TARGETED,
            targets=["agent-b"],
        )
        visible = await shared.recall(agent_id="agent-b")
        assert len(visible) == 1
        hidden = await shared.recall(agent_id="agent-c")
        assert len(hidden) == 0

    async def test_size(self):
        backend = AsyncInMemoryBackend()
        shared = AsyncSharedMemory(team_id="test", backend=backend)
        await shared.publish(content="a", shared_by="x")
        await shared.publish(content="b", shared_by="y")
        size = await shared.size()
        assert size == 2
