"""Tests for storage backend adapters."""
import pytest
from eachmind.backends import InMemoryBackend
from eachmind.backends.base import StorageBackend

class TestStorageBackendProtocol:
    def test_in_memory_satisfies_protocol(self):
        assert isinstance(InMemoryBackend(), StorageBackend)

class TestInMemoryBackend:
    def test_save_and_get(self):
        backend = InMemoryBackend()
        backend.save("memories", "key1", {"content": "hello"})
        result = backend.get("memories", "key1")
        assert result == {"content": "hello"}

    def test_get_missing_key(self):
        backend = InMemoryBackend()
        assert backend.get("memories", "missing") is None

    def test_list_collection(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        items = backend.list("memories")
        assert len(items) == 2

    def test_delete(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.delete("memories", "k1")
        assert backend.get("memories", "k1") is None

    def test_clear_collection(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        backend.clear("memories")
        assert backend.list("memories") == []

    def test_separate_collections(self):
        backend = InMemoryBackend()
        backend.save("private", "k1", {"a": 1})
        backend.save("shared", "k1", {"b": 2})
        assert backend.get("private", "k1") == {"a": 1}
        assert backend.get("shared", "k1") == {"b": 2}


import os
from eachmind.backends import SQLiteBackend
from eachmind.backends.base import StorageBackend

class TestSQLiteBackend:
    def _make_backend(self, tmp_path):
        db_path = os.path.join(tmp_path, "test.db")
        return SQLiteBackend(db_path)

    def test_satisfies_protocol(self, tmp_path):
        assert isinstance(self._make_backend(tmp_path), StorageBackend)

    def test_save_and_get(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"content": "hello"})
        result = backend.get("memories", "k1")
        assert result == {"content": "hello"}

    def test_get_missing_key(self, tmp_path):
        backend = self._make_backend(tmp_path)
        assert backend.get("memories", "missing") is None

    def test_list_collection(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        items = backend.list("memories")
        assert len(items) == 2

    def test_delete(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.delete("memories", "k1")
        assert backend.get("memories", "k1") is None

    def test_clear_collection(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        backend.clear("memories")
        assert backend.list("memories") == []

    def test_persistence_across_instances(self, tmp_path):
        db_path = os.path.join(tmp_path, "persist.db")
        backend1 = SQLiteBackend(db_path)
        backend1.save("memories", "k1", {"persisted": True})
        backend2 = SQLiteBackend(db_path)
        result = backend2.get("memories", "k1")
        assert result == {"persisted": True}

    def test_in_memory_mode(self):
        backend = SQLiteBackend(":memory:")
        backend.save("test", "k1", {"a": 1})
        assert backend.get("test", "k1") == {"a": 1}


# --- Integration tests: backends wired into primitives ---

from eachmind import Agent, MemoryEvent, SharedMemory
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory


class TestPrivateMemoryWithBackend:
    def test_sqlite_backend_stores_and_recalls(self, tmp_path):
        backend = SQLiteBackend(os.path.join(tmp_path, "pm.db"))
        memory = PrivateMemory(agent_id="test", backend=backend)
        perspective = Perspective(role="test")
        event = MemoryEvent(content="remember this", source="test")
        encoded = perspective.encode(event)
        memory.store(encoded)
        results = memory.recall(event_id=event.event_id)
        assert len(results) == 1

    def test_default_backend_still_works(self):
        """Existing code without backend param must keep working."""
        memory = PrivateMemory(agent_id="test")
        perspective = Perspective(role="test")
        event = MemoryEvent(content="data", source="test")
        encoded = perspective.encode(event)
        memory.store(encoded)
        assert memory.size == 1

    def test_persistence_across_instances(self, tmp_path):
        """Data stored via backend survives new PrivateMemory instances."""
        db_path = os.path.join(tmp_path, "pm_persist.db")
        perspective = Perspective(role="test")
        event = MemoryEvent(content="persist me", source="test")

        # Store in first instance
        backend1 = SQLiteBackend(db_path)
        mem1 = PrivateMemory(agent_id="test", backend=backend1)
        encoded = perspective.encode(event)
        mem1.store(encoded)

        # Hydrate in second instance
        backend2 = SQLiteBackend(db_path)
        mem2 = PrivateMemory(agent_id="test", backend=backend2)
        assert mem2.size == 1
        results = mem2.recall(event_id=event.event_id)
        assert len(results) == 1


class TestSharedMemoryWithBackend:
    def test_sqlite_backend_publishes_and_recalls(self, tmp_path):
        backend = SQLiteBackend(os.path.join(tmp_path, "sm.db"))
        shared = SharedMemory(team_id="test", backend=backend)
        shared.publish(content="insight", shared_by="agent-a")
        results = shared.recall()
        assert len(results) == 1
        assert results[0].shared_by == "agent-a"

    def test_default_backend_still_works(self):
        """Existing code without backend param must keep working."""
        shared = SharedMemory(team_id="test")
        shared.publish(content="insight", shared_by="agent-a")
        assert shared.size == 1

    def test_persistence_across_instances(self, tmp_path):
        db_path = os.path.join(tmp_path, "sm_persist.db")

        backend1 = SQLiteBackend(db_path)
        shared1 = SharedMemory(team_id="test", backend=backend1)
        shared1.publish(content="persisted insight", shared_by="agent-a", reason="test")

        backend2 = SQLiteBackend(db_path)
        shared2 = SharedMemory(team_id="test", backend=backend2)
        assert shared2.size == 1
        results = shared2.recall()
        assert results[0].content == "persisted insight"
        assert results[0].shared_by == "agent-a"
