"""
Protocol definitions for eachmind.

The protocol module defines the rules and interfaces for how memory
operations work across agents — encoding, sharing, consolidation,
and drift measurement contracts.
"""

from __future__ import annotations

from eachmind.protocol.interfaces import (
    ConsolidationEngineProtocol,
    DriftTrackerProtocol,
    MemoryEncoderProtocol,
    MemoryStoreProtocol,
    SharedStoreProtocol,
)

__all__ = [
    "ConsolidationEngineProtocol",
    "DriftTrackerProtocol",
    "MemoryEncoderProtocol",
    "MemoryStoreProtocol",
    "SharedStoreProtocol",
]
