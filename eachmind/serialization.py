"""
Serialization — Robust serialization for storage backends.

Handles dataclasses, enums, datetimes, and other non-JSON-native types
that frequently appear in eachmind's data structures.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Serializer(Protocol):
    """Protocol for serializing values to/from strings."""
    def serialize(self, value: Any) -> str: ...
    def deserialize(self, raw: str) -> Any: ...


class _EachmindEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses, enums, and datetimes."""
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        return super().default(o)


class JSONSerializer:
    """Default serializer using JSON with extended type support."""
    def serialize(self, value: Any) -> str:
        return json.dumps(value, cls=_EachmindEncoder)

    def deserialize(self, raw: str) -> Any:
        return json.loads(raw)
