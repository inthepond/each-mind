"""Tests for thread-safe backend operations."""
from __future__ import annotations

import os
import threading

from eachmind.backends import InMemoryBackend, SQLiteBackend


class TestInMemoryBackendThreadSafety:
    def test_concurrent_writes(self):
        backend = InMemoryBackend()
        errors: list[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(100):
                    backend.save("test", f"t{thread_id}_k{i}", {"v": i})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        items = backend.list("test")
        assert len(items) == 1000

    def test_concurrent_read_write(self):
        backend = InMemoryBackend()
        backend.save("test", "k1", {"v": 1})
        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(100):
                    backend.get("test", "k1")
                    backend.list("test")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    backend.save("test", f"w_{i}", {"v": i})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestSQLiteBackendThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        db_path = os.path.join(tmp_path, "concurrent.db")
        backend = SQLiteBackend(db_path)
        errors: list[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    backend.save(
                        "test", f"t{thread_id}_k{i}", {"v": i}
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        items = backend.list("test")
        assert len(items) == 250
