"""Redis storage backend — optional dependency.

Install: pip install eachmind[redis]
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

try:
    import redis as redis_lib
except ImportError as e:
    raise ImportError(
        "Redis backend requires the 'redis' package. "
        "Install it with: pip install eachmind[redis]"
    ) from e

@dataclass
class RedisBackend:
    """Stores data in Redis. Suitable for distributed multi-agent systems.
    Args:
        url: Redis connection URL (default: redis://localhost:6379).
        prefix: Key prefix to namespace eachmind data.
    """
    url: str = "redis://localhost:6379"
    prefix: str = "eachmind"
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = redis_lib.from_url(self.url, decode_responses=True)

    def _key(self, collection: str, key: str) -> str:
        return f"{self.prefix}:{collection}:{key}"

    def _collection_pattern(self, collection: str) -> str:
        return f"{self.prefix}:{collection}:*"

    def save(self, collection: str, key: str, value: Any) -> None:
        self._client.set(self._key(collection, key), json.dumps(value))

    def get(self, collection: str, key: str) -> Any | None:
        raw = self._client.get(self._key(collection, key))
        return json.loads(raw) if raw else None

    def list(self, collection: str) -> list[Any]:
        keys = self._client.keys(self._collection_pattern(collection))
        if not keys:
            return []
        values = self._client.mget(keys)
        return [json.loads(v) for v in values if v is not None]

    def delete(self, collection: str, key: str) -> None:
        self._client.delete(self._key(collection, key))

    def clear(self, collection: str) -> None:
        keys = self._client.keys(self._collection_pattern(collection))
        if keys:
            self._client.delete(*keys)
