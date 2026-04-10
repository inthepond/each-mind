"""
eachmind — Per-agent memory protocol for multi-agent systems.

A standalone, framework-agnostic Python library that defines how memory is
stored, differentiated, and selectively shared across agents in a multi-agent
system. It is a memory protocol — not an agent framework, not a task runner.
Any agent system can adopt it.
"""

__version__ = "0.1.0"

from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import SharedMemory
from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.drift import Drift
from eachmind.agent import Agent
from eachmind.backends import InMemoryBackend, SQLiteBackend

__all__ = [
    "PrivateMemory",
    "SharedMemory",
    "MemoryEvent",
    "Perspective",
    "Consolidation",
    "Drift",
    "Agent",
    "InMemoryBackend",
    "SQLiteBackend",
]
