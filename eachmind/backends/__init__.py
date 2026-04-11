"""Storage backend adapters for eachmind."""
from eachmind.backends.async_memory import AsyncInMemoryBackend
from eachmind.backends.memory import InMemoryBackend
from eachmind.backends.sqlite import SQLiteBackend

__all__ = ["AsyncInMemoryBackend", "InMemoryBackend", "SQLiteBackend"]

# RedisBackend requires optional dependency — import explicitly:
# from eachmind.backends.redis import RedisBackend

# AsyncSQLiteBackend requires optional dependency — import explicitly:
# from eachmind.backends.async_sqlite import AsyncSQLiteBackend
