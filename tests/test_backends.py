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
