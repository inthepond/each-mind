"""
eachmind — Per-agent memory protocol for multi-agent systems.

A standalone, framework-agnostic Python library that defines how memory is
stored, differentiated, and selectively shared across agents in a multi-agent
system. It is a memory protocol — not an agent framework, not a task runner.
Any agent system can adopt it.
"""

__version__ = "0.1.0"

from eachmind.agent import Agent
from eachmind.async_primitives import AsyncPrivateMemory, AsyncSharedMemory
from eachmind.backends import AsyncInMemoryBackend, InMemoryBackend, SQLiteBackend
from eachmind.hooks import LoggingHook, MemoryHook
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.drift import Drift
from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import SharedMemory
from eachmind.retrieval import TFIDFRetriever
from eachmind.serialization import JSONSerializer

__all__ = [
    "Agent",
    "AsyncInMemoryBackend",
    "AsyncPrivateMemory",
    "AsyncSharedMemory",
    "Consolidation",
    "Drift",
    "InMemoryBackend",
    "JSONSerializer",
    "LoggingHook",
    "MemoryEvent",
    "MemoryHook",
    "Perspective",
    "PrivateMemory",
    "SharedMemory",
    "SQLiteBackend",
    "TFIDFRetriever",
]
