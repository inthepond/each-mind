"""Tests for async storage backends."""
from __future__ import annotations

import pytest

from eachmind.backends.async_memory import AsyncInMemoryBackend
from eachmind.backends.base import AsyncStorageBackend


@pytest.mark.asyncio
class TestAsyncStorageBackendProtocol:
    async def test_async_in_memory_satisfies_protocol(self):
        assert isinstance(AsyncInMemoryBackend(), AsyncStorageBackend)


@pytest.mark.asyncio
class TestAsyncInMemoryBackend:
    async def test_save_and_get(self):
        backend = AsyncInMemoryBackend()
        await backend.save("memories", "k1", {"content": "hello"})
        result = await backend.get("memories", "k1")
        assert result == {"content": "hello"}

    async def test_get_missing_key(self):
        backend = AsyncInMemoryBackend()
        result = await backend.get("memories", "missing")
        assert result is None

    async def test_list_collection(self):
        backend = AsyncInMemoryBackend()
        await backend.save("memories", "k1", {"a": 1})
        await backend.save("memories", "k2", {"b": 2})
        items = await backend.list("memories")
        assert len(items) == 2

    async def test_delete(self):
        backend = AsyncInMemoryBackend()
        await backend.save("memories", "k1", {"a": 1})
        await backend.delete("memories", "k1")
        result = await backend.get("memories", "k1")
        assert result is None

    async def test_clear(self):
        backend = AsyncInMemoryBackend()
        await backend.save("memories", "k1", {"a": 1})
        await backend.clear("memories")
        items = await backend.list("memories")
        assert items == []

    async def test_separate_collections(self):
        backend = AsyncInMemoryBackend()
        await backend.save("private", "k1", {"a": 1})
        await backend.save("shared", "k1", {"b": 2})
        assert await backend.get("private", "k1") == {"a": 1}
        assert await backend.get("shared", "k1") == {"b": 2}
