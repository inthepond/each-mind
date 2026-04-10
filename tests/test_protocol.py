"""Tests for eachmind protocol interfaces.

These tests verify:
1. The Protocol classes exist and are importable from eachmind.protocol
2. The existing primitives satisfy each Protocol (isinstance checks pass)
3. Protocol contracts match existing implementations exactly
"""

from __future__ import annotations

import pytest

from eachmind.protocol import (
    ConsolidationEngineProtocol,
    DriftTrackerProtocol,
    MemoryEncoderProtocol,
    MemoryStoreProtocol,
    SharedStoreProtocol,
)
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.drift import Drift
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import SharedMemory


class TestProtocolImports:
    """All five protocol classes are importable from eachmind.protocol."""

    def test_memory_store_protocol_importable(self):
        assert MemoryStoreProtocol is not None

    def test_shared_store_protocol_importable(self):
        assert SharedStoreProtocol is not None

    def test_memory_encoder_protocol_importable(self):
        assert MemoryEncoderProtocol is not None

    def test_consolidation_engine_protocol_importable(self):
        assert ConsolidationEngineProtocol is not None

    def test_drift_tracker_protocol_importable(self):
        assert DriftTrackerProtocol is not None


class TestMemoryStoreProtocolCompliance:
    """PrivateMemory satisfies MemoryStoreProtocol."""

    def test_private_memory_is_instance(self):
        mem = PrivateMemory(agent_id="test")
        assert isinstance(mem, MemoryStoreProtocol)

    def test_memory_store_protocol_is_runtime_checkable(self):
        # Should not raise TypeError — Protocol must be @runtime_checkable
        mem = PrivateMemory(agent_id="test")
        result = isinstance(mem, MemoryStoreProtocol)
        assert isinstance(result, bool)

    def test_non_compliant_object_fails(self):
        class NotAStore:
            pass

        assert not isinstance(NotAStore(), MemoryStoreProtocol)


class TestSharedStoreProtocolCompliance:
    """SharedMemory satisfies SharedStoreProtocol."""

    def test_shared_memory_is_instance(self):
        shared = SharedMemory(team_id="team")
        assert isinstance(shared, SharedStoreProtocol)

    def test_shared_store_protocol_is_runtime_checkable(self):
        shared = SharedMemory(team_id="team")
        result = isinstance(shared, SharedStoreProtocol)
        assert isinstance(result, bool)

    def test_non_compliant_object_fails(self):
        class NotAStore:
            pass

        assert not isinstance(NotAStore(), SharedStoreProtocol)


class TestMemoryEncoderProtocolCompliance:
    """Perspective satisfies MemoryEncoderProtocol."""

    def test_perspective_is_instance(self):
        p = Perspective(role="analyst")
        assert isinstance(p, MemoryEncoderProtocol)

    def test_memory_encoder_protocol_is_runtime_checkable(self):
        p = Perspective(role="analyst")
        result = isinstance(p, MemoryEncoderProtocol)
        assert isinstance(result, bool)

    def test_non_compliant_object_fails(self):
        class NotAnEncoder:
            pass

        assert not isinstance(NotAnEncoder(), MemoryEncoderProtocol)


class TestConsolidationEngineProtocolCompliance:
    """Consolidation satisfies ConsolidationEngineProtocol."""

    def test_consolidation_is_instance(self):
        c = Consolidation()
        assert isinstance(c, ConsolidationEngineProtocol)

    def test_consolidation_engine_protocol_is_runtime_checkable(self):
        c = Consolidation()
        result = isinstance(c, ConsolidationEngineProtocol)
        assert isinstance(result, bool)

    def test_non_compliant_object_fails(self):
        class NotAnEngine:
            pass

        assert not isinstance(NotAnEngine(), ConsolidationEngineProtocol)


class TestDriftTrackerProtocolCompliance:
    """Drift satisfies DriftTrackerProtocol."""

    def test_drift_is_instance(self):
        d = Drift()
        assert isinstance(d, DriftTrackerProtocol)

    def test_drift_tracker_protocol_is_runtime_checkable(self):
        d = Drift()
        result = isinstance(d, DriftTrackerProtocol)
        assert isinstance(result, bool)

    def test_non_compliant_object_fails(self):
        class NotATracker:
            pass

        assert not isinstance(NotATracker(), DriftTrackerProtocol)


class TestCustomImplementationsCompliance:
    """Custom classes that implement the protocol contracts also pass isinstance."""

    def test_custom_memory_store_passes(self):
        from eachmind.primitives.perspective import EncodedMemory

        class CustomStore:
            def store(self, encoded: EncodedMemory) -> None:
                pass

            def recall(
                self,
                *,
                event_id: str | None = None,
                min_salience: float = 0.0,
                limit: int = 10,
            ) -> list:
                return []

            def search(self, query: str) -> list:
                return []

            @property
            def size(self) -> int:
                return 0

        assert isinstance(CustomStore(), MemoryStoreProtocol)

    def test_custom_shared_store_passes(self):
        class CustomSharedStore:
            def publish(
                self,
                content,
                shared_by: str,
                scope=None,
                targets=None,
                reason: str = "",
                **metadata,
            ):
                pass

            def recall(
                self,
                *,
                agent_id: str | None = None,
                source=None,
                shared_by: str | None = None,
                limit: int = 20,
            ) -> list:
                return []

            @property
            def size(self) -> int:
                return 0

        assert isinstance(CustomSharedStore(), SharedStoreProtocol)

    def test_custom_encoder_passes(self):
        from eachmind.primitives.memory_event import MemoryEvent

        class CustomEncoder:
            @property
            def role(self) -> str:
                return "custom"

            def encode(self, event: MemoryEvent):
                pass

            def drift_from(self, other) -> float:
                return 0.0

        assert isinstance(CustomEncoder(), MemoryEncoderProtocol)

    def test_custom_consolidation_engine_passes(self):
        class CustomEngine:
            def process(self, memories: list) -> list:
                return []

            def get_beliefs(
                self,
                *,
                min_confidence: float = 0.0,
                tags: list | None = None,
            ) -> list:
                return []

        assert isinstance(CustomEngine(), ConsolidationEngineProtocol)

    def test_custom_drift_tracker_passes(self):
        from eachmind.primitives.perspective import Perspective

        class CustomTracker:
            def measure(
                self,
                agent_a_id: str,
                perspective_a: Perspective,
                agent_b_id: str,
                perspective_b: Perspective,
            ):
                pass

            def team_diversity(self, perspectives: dict) -> float:
                return 0.0

        assert isinstance(CustomTracker(), DriftTrackerProtocol)
