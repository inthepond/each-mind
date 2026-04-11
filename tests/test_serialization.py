"""Tests for serialization module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from eachmind.serialization import JSONSerializer, Serializer


class Color(Enum):
    RED = "red"
    BLUE = "blue"

@dataclass
class Point:
    x: int
    y: int


class TestSerializerProtocol:
    def test_json_serializer_satisfies_protocol(self):
        assert isinstance(JSONSerializer(), Serializer)


class TestJSONSerializer:
    def setup_method(self):
        self.s = JSONSerializer()

    def test_primitives(self):
        for val in [42, 3.14, "hello", True, None]:
            assert self.s.deserialize(self.s.serialize(val)) == val

    def test_dict(self):
        data = {"key": "value", "nested": {"a": 1}}
        assert self.s.deserialize(self.s.serialize(data)) == data

    def test_list(self):
        data = [1, "two", {"three": 3}]
        assert self.s.deserialize(self.s.serialize(data)) == data

    def test_dataclass(self):
        p = Point(x=1, y=2)
        result = self.s.deserialize(self.s.serialize(p))
        assert result == {"x": 1, "y": 2}

    def test_enum(self):
        result = self.s.deserialize(self.s.serialize(Color.RED))
        assert result == "red"

    def test_datetime(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = self.s.deserialize(self.s.serialize(dt))
        assert result == "2026-01-01T00:00:00+00:00"

    def test_nested_non_json_types(self):
        data = {
            "point": Point(x=1, y=2),
            "color": Color.BLUE,
            "items": [1, Color.RED],
        }
        result = self.s.deserialize(self.s.serialize(data))
        assert result["point"] == {"x": 1, "y": 2}
        assert result["color"] == "blue"
        assert result["items"] == [1, "red"]

    def test_bytes_roundtrip(self):
        raw = self.s.serialize({"a": 1})
        assert isinstance(raw, str)
        assert self.s.deserialize(raw) == {"a": 1}
