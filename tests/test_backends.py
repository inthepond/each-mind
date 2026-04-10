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
