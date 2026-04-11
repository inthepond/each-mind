"""Tests for TTL/expiration on shared memory entries."""
from __future__ import annotations

import time

from eachmind.primitives.shared_memory import SharedEntry, SharedMemory


class TestSharedEntryTTL:
    def test_entry_without_ttl_never_expires(self):
        entry = SharedEntry(content="test", shared_by="agent-a")
        assert not entry.is_expired()

    def test_entry_with_future_ttl_not_expired(self):
        entry = SharedEntry(content="test", shared_by="agent-a", ttl=3600)
        assert not entry.is_expired()

    def test_entry_with_zero_ttl_expires_immediately(self):
        entry = SharedEntry(content="test", shared_by="agent-a", ttl=0)
        time.sleep(0.01)
        assert entry.is_expired()


class TestSharedMemoryTTL:
    def test_recall_filters_expired_entries(self):
        shared = SharedMemory(team_id="test")
        shared.publish(content="ephemeral", shared_by="a", ttl=0)
        shared.publish(content="permanent", shared_by="b")
        time.sleep(0.01)
        results = shared.recall()
        assert len(results) == 1
        assert results[0].content == "permanent"

    def test_cleanup_removes_expired(self):
        shared = SharedMemory(team_id="test")
        shared.publish(content="ephemeral", shared_by="a", ttl=0)
        shared.publish(content="permanent", shared_by="b")
        time.sleep(0.01)
        removed = shared.cleanup()
        assert removed == 1
        assert shared.size == 1

    def test_publish_with_default_ttl(self):
        shared = SharedMemory(team_id="test", default_ttl=3600)
        entry = shared.publish(content="test", shared_by="a")
        assert entry.ttl == 3600

    def test_publish_ttl_overrides_default(self):
        shared = SharedMemory(team_id="test", default_ttl=3600)
        entry = shared.publish(content="test", shared_by="a", ttl=60)
        assert entry.ttl == 60

    def test_publish_no_ttl_no_default(self):
        shared = SharedMemory(team_id="test")
        entry = shared.publish(content="test", shared_by="a")
        assert entry.ttl is None
        assert not entry.is_expired()
