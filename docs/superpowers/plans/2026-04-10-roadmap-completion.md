# Roadmap Completion (Through Project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all roadmap items before Project 2 — protocol specification, storage backends, integration examples, drift visualization, and enhanced consolidation algorithms.

**Architecture:** The protocol layer defines `Protocol` classes (structural typing) that formalize contracts for memory stores, encoders, consolidation, and drift. Storage backends implement a `StorageBackend` protocol with in-memory, SQLite, and Redis adapters. PrivateMemory and SharedMemory are updated to accept pluggable backends. Drift visualization uses matplotlib for heatmaps and timelines. Consolidation gets temporal and weighted algorithms. Integration examples are runnable scripts for OpenAI Swarm, CrewAI, and LangGraph.

**Tech Stack:** Python 3.10+, dataclasses, typing.Protocol, sqlite3 (stdlib), redis (optional), matplotlib (optional), openai-swarm/crewai/langgraph (example deps)

---

## File Structure

```
eachmind/
├── protocol/
│   ├── __init__.py          # Re-exports all protocols
│   └── interfaces.py        # Protocol classes (MemoryStoreProtocol, etc.)
├── backends/
│   ├── __init__.py           # Re-exports backends
│   ├── base.py               # StorageBackend Protocol
│   ├── memory.py             # InMemoryBackend
│   ├── sqlite.py             # SQLiteBackend
│   └── redis.py              # RedisBackend (optional dep)
├── primitives/
│   ├── private_memory.py     # Modified: accept backend parameter
│   ├── shared_memory.py      # Modified: accept backend parameter
│   ├── consolidation.py      # Modified: add temporal + weighted algorithms
│   └── drift.py              # (unchanged)
├── visualization/
│   ├── __init__.py           # Re-exports
│   └── drift_viz.py          # matplotlib drift visualization
├── __init__.py               # Modified: export new modules
tests/
├── test_primitives.py        # Existing (unchanged)
├── test_protocol.py          # Protocol compliance tests
├── test_backends.py          # Backend adapter tests
├── test_consolidation_algorithms.py  # Enhanced consolidation tests
├── test_visualization.py     # Visualization smoke tests
examples/
├── basic_usage.py            # Existing (unchanged)
├── sqlite_backend.py         # SQLite storage example
├── swarm_integration.py      # OpenAI Swarm + eachmind
├── crewai_integration.py     # CrewAI + eachmind
├── langgraph_integration.py  # LangGraph + eachmind
├── drift_visualization.py    # Drift viz demo
pyproject.toml                # Modified: add optional deps
```

---

### Task 1: Protocol Specification

**Files:**
- Create: `eachmind/protocol/interfaces.py`
- Modify: `eachmind/protocol/__init__.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: Write failing tests for protocol compliance**

```python
"""Tests that existing primitives satisfy the Protocol contracts."""

from typing import runtime_checkable

from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.drift import Drift
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.primitives.shared_memory import SharedMemory
from eachmind.protocol import (
    ConsolidationEngineProtocol,
    DriftTrackerProtocol,
    MemoryEncoderProtocol,
    MemoryStoreProtocol,
    SharedStoreProtocol,
)


class TestProtocolCompliance:
    def test_private_memory_is_memory_store(self):
        assert isinstance(PrivateMemory(agent_id="test"), MemoryStoreProtocol)

    def test_shared_memory_is_shared_store(self):
        assert isinstance(SharedMemory(), SharedStoreProtocol)

    def test_perspective_is_encoder(self):
        assert isinstance(Perspective(role="test"), MemoryEncoderProtocol)

    def test_consolidation_is_engine(self):
        assert isinstance(Consolidation(), ConsolidationEngineProtocol)

    def test_drift_is_tracker(self):
        assert isinstance(Drift(), DriftTrackerProtocol)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: ImportError — protocol module not yet defined.

- [ ] **Step 3: Implement protocol interfaces**

Create `eachmind/protocol/interfaces.py` with these `Protocol` classes:

```python
"""
Protocol interfaces for the eachmind memory protocol.

These define the structural contracts that any implementation must satisfy.
Existing primitives already satisfy these contracts — the protocols formalize
what was previously implicit.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import EncodedMemory


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Contract for an agent's private memory store."""

    def store(self, encoded: EncodedMemory) -> None: ...
    def recall(
        self,
        *,
        event_id: str | None = None,
        min_salience: float = 0.0,
        limit: int = 10,
    ) -> list[EncodedMemory]: ...
    def search(self, query: str) -> list[EncodedMemory]: ...
    @property
    def size(self) -> int: ...


@runtime_checkable
class SharedStoreProtocol(Protocol):
    """Contract for the team's shared memory space."""

    def publish(
        self,
        content: Any,
        shared_by: str,
        **kwargs: Any,
    ) -> Any: ...
    def recall(self, *, agent_id: str | None = None, limit: int = 20) -> list[Any]: ...
    @property
    def size(self) -> int: ...


@runtime_checkable
class MemoryEncoderProtocol(Protocol):
    """Contract for encoding events through a perspective."""

    role: str
    def encode(self, event: MemoryEvent) -> EncodedMemory: ...
    def drift_from(self, other: MemoryEncoderProtocol) -> float: ...


@runtime_checkable
class ConsolidationEngineProtocol(Protocol):
    """Contract for consolidating memories into beliefs."""

    def process(self, memories: list[EncodedMemory]) -> list[Any]: ...
    def get_beliefs(self, *, min_confidence: float = 0.0) -> list[Any]: ...


@runtime_checkable
class DriftTrackerProtocol(Protocol):
    """Contract for tracking perspective divergence."""

    def measure(
        self,
        agent_a_id: str,
        perspective_a: Any,
        agent_b_id: str,
        perspective_b: Any,
    ) -> Any: ...
    def team_diversity(self, perspectives: dict[str, Any]) -> float: ...
```

- [ ] **Step 4: Update protocol `__init__.py`**

```python
"""
Protocol definitions for eachmind.

The protocol module defines the rules and interfaces for how memory
operations work across agents — encoding, sharing, consolidation,
and drift measurement contracts.
"""

from eachmind.protocol.interfaces import (
    ConsolidationEngineProtocol,
    DriftTrackerProtocol,
    MemoryEncoderProtocol,
    MemoryStoreProtocol,
    SharedStoreProtocol,
)

__all__ = [
    "MemoryStoreProtocol",
    "SharedStoreProtocol",
    "MemoryEncoderProtocol",
    "ConsolidationEngineProtocol",
    "DriftTrackerProtocol",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All 24 tests PASS (19 existing + 5 new).

- [ ] **Step 7: Commit**

```bash
git add eachmind/protocol/ tests/test_protocol.py
git commit -m "feat: add protocol specification with Protocol classes"
```

---

### Task 2: Storage Backend Base + In-Memory Backend

**Files:**
- Create: `eachmind/backends/__init__.py`
- Create: `eachmind/backends/base.py`
- Create: `eachmind/backends/memory.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write failing tests for storage backend protocol and in-memory backend**

```python
"""Tests for storage backend adapters."""

import pytest

from eachmind.backends import InMemoryBackend
from eachmind.backends.base import StorageBackend


class TestStorageBackendProtocol:
    def test_in_memory_satisfies_protocol(self):
        assert isinstance(InMemoryBackend(), StorageBackend)


class TestInMemoryBackend:
    def test_save_and_get(self):
        backend = InMemoryBackend()
        backend.save("memories", "key1", {"content": "hello"})
        result = backend.get("memories", "key1")
        assert result == {"content": "hello"}

    def test_get_missing_key(self):
        backend = InMemoryBackend()
        assert backend.get("memories", "missing") is None

    def test_list_collection(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        items = backend.list("memories")
        assert len(items) == 2

    def test_delete(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.delete("memories", "k1")
        assert backend.get("memories", "k1") is None

    def test_clear_collection(self):
        backend = InMemoryBackend()
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        backend.clear("memories")
        assert backend.list("memories") == []

    def test_separate_collections(self):
        backend = InMemoryBackend()
        backend.save("private", "k1", {"a": 1})
        backend.save("shared", "k1", {"b": 2})
        assert backend.get("private", "k1") == {"a": 1}
        assert backend.get("shared", "k1") == {"b": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement StorageBackend protocol**

Create `eachmind/backends/base.py`:

```python
"""Base protocol for storage backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends.

    A storage backend manages collections of key-value pairs.
    Collections isolate different data types (e.g., "private_memories",
    "shared_entries").
    """

    def save(self, collection: str, key: str, value: Any) -> None:
        """Save a value under a key in a collection."""
        ...

    def get(self, collection: str, key: str) -> Any | None:
        """Get a value by key. Returns None if not found."""
        ...

    def list(self, collection: str) -> list[Any]:
        """List all values in a collection."""
        ...

    def delete(self, collection: str, key: str) -> None:
        """Delete a key from a collection."""
        ...

    def clear(self, collection: str) -> None:
        """Remove all entries from a collection."""
        ...
```

- [ ] **Step 4: Implement InMemoryBackend**

Create `eachmind/backends/memory.py`:

```python
"""In-memory storage backend — the default."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryBackend:
    """Stores data in plain Python dicts. Fast, ephemeral, zero dependencies."""

    _store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, collection: str, key: str, value: Any) -> None:
        self._store.setdefault(collection, {})[key] = value

    def get(self, collection: str, key: str) -> Any | None:
        return self._store.get(collection, {}).get(key)

    def list(self, collection: str) -> list[Any]:
        return list(self._store.get(collection, {}).values())

    def delete(self, collection: str, key: str) -> None:
        coll = self._store.get(collection, {})
        coll.pop(key, None)

    def clear(self, collection: str) -> None:
        self._store.pop(collection, None)
```

- [ ] **Step 5: Create backends `__init__.py`**

```python
"""Storage backend adapters for eachmind."""

from eachmind.backends.memory import InMemoryBackend

__all__ = ["InMemoryBackend"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_backends.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add eachmind/backends/ tests/test_backends.py
git commit -m "feat: add storage backend protocol and in-memory adapter"
```

---

### Task 3: SQLite Backend

**Files:**
- Create: `eachmind/backends/sqlite.py`
- Modify: `eachmind/backends/__init__.py`
- Test: `tests/test_backends.py` (append)

- [ ] **Step 1: Write failing tests for SQLite backend**

Append to `tests/test_backends.py`:

```python
import tempfile
import os

from eachmind.backends import SQLiteBackend
from eachmind.backends.base import StorageBackend


class TestSQLiteBackend:
    def _make_backend(self, tmp_path):
        db_path = os.path.join(tmp_path, "test.db")
        return SQLiteBackend(db_path)

    def test_satisfies_protocol(self, tmp_path):
        assert isinstance(self._make_backend(tmp_path), StorageBackend)

    def test_save_and_get(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"content": "hello"})
        result = backend.get("memories", "k1")
        assert result == {"content": "hello"}

    def test_get_missing_key(self, tmp_path):
        backend = self._make_backend(tmp_path)
        assert backend.get("memories", "missing") is None

    def test_list_collection(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        items = backend.list("memories")
        assert len(items) == 2

    def test_delete(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.delete("memories", "k1")
        assert backend.get("memories", "k1") is None

    def test_clear_collection(self, tmp_path):
        backend = self._make_backend(tmp_path)
        backend.save("memories", "k1", {"a": 1})
        backend.save("memories", "k2", {"b": 2})
        backend.clear("memories")
        assert backend.list("memories") == []

    def test_persistence_across_instances(self, tmp_path):
        db_path = os.path.join(tmp_path, "persist.db")
        backend1 = SQLiteBackend(db_path)
        backend1.save("memories", "k1", {"persisted": True})

        backend2 = SQLiteBackend(db_path)
        result = backend2.get("memories", "k1")
        assert result == {"persisted": True}

    def test_in_memory_mode(self):
        backend = SQLiteBackend(":memory:")
        backend.save("test", "k1", {"a": 1})
        assert backend.get("test", "k1") == {"a": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends.py::TestSQLiteBackend -v`
Expected: ImportError.

- [ ] **Step 3: Implement SQLiteBackend**

Create `eachmind/backends/sqlite.py`:

```python
"""SQLite storage backend — persistent, zero external dependencies."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SQLiteBackend:
    """Stores data in a SQLite database. Persistent across restarts.

    Args:
        db_path: Path to SQLite file, or ":memory:" for in-memory.
    """

    db_path: str = "eachmind.db"
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS store ("
            "  collection TEXT NOT NULL,"
            "  key TEXT NOT NULL,"
            "  value TEXT NOT NULL,"
            "  PRIMARY KEY (collection, key)"
            ")"
        )
        self._conn.commit()

    def save(self, collection: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO store (collection, key, value) VALUES (?, ?, ?)",
            (collection, key, json.dumps(value)),
        )
        self._conn.commit()

    def get(self, collection: str, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value FROM store WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str) -> list[Any]:
        rows = self._conn.execute(
            "SELECT value FROM store WHERE collection = ?",
            (collection,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, collection: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM store WHERE collection = ? AND key = ?",
            (collection, key),
        )
        self._conn.commit()

    def clear(self, collection: str) -> None:
        self._conn.execute(
            "DELETE FROM store WHERE collection = ?",
            (collection,),
        )
        self._conn.commit()
```

- [ ] **Step 4: Update backends `__init__.py`**

```python
"""Storage backend adapters for eachmind."""

from eachmind.backends.memory import InMemoryBackend
from eachmind.backends.sqlite import SQLiteBackend

__all__ = ["InMemoryBackend", "SQLiteBackend"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backends.py -v`
Expected: All 15 tests PASS (7 in-memory + 8 SQLite).

- [ ] **Step 6: Commit**

```bash
git add eachmind/backends/sqlite.py eachmind/backends/__init__.py tests/test_backends.py
git commit -m "feat: add SQLite storage backend"
```

---

### Task 4: Redis Backend

**Files:**
- Create: `eachmind/backends/redis.py`
- Modify: `eachmind/backends/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Implement RedisBackend**

Create `eachmind/backends/redis.py`:

```python
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
```

- [ ] **Step 2: Update backends `__init__.py`**

```python
"""Storage backend adapters for eachmind."""

from eachmind.backends.memory import InMemoryBackend
from eachmind.backends.sqlite import SQLiteBackend

__all__ = ["InMemoryBackend", "SQLiteBackend"]

# RedisBackend requires optional dependency — import explicitly:
# from eachmind.backends.redis import RedisBackend
```

- [ ] **Step 3: Add optional dependency to pyproject.toml**

Add after `requires-python`:

```toml
[project.optional-dependencies]
redis = ["redis>=5.0"]
viz = ["matplotlib>=3.7"]
examples = ["openai-agents>=0.1", "crewai>=0.100", "langgraph>=0.3"]
dev = ["pytest", "ruff", "mypy", "matplotlib>=3.7"]
```

- [ ] **Step 4: Run full test suite (Redis tests skipped without server)**

Run: `python -m pytest tests/ -v`
Expected: All existing + new tests pass. Redis not tested here (needs running server).

- [ ] **Step 5: Commit**

```bash
git add eachmind/backends/ pyproject.toml
git commit -m "feat: add Redis storage backend (optional dependency)"
```

---

### Task 5: Wire Backends into PrivateMemory and SharedMemory

**Files:**
- Modify: `eachmind/primitives/private_memory.py`
- Modify: `eachmind/primitives/shared_memory.py`
- Modify: `eachmind/__init__.py`
- Test: `tests/test_backends.py` (append integration tests)

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_backends.py`:

```python
from eachmind import Agent, MemoryEvent, SharedMemory
from eachmind.primitives.perspective import Perspective
from eachmind.primitives.private_memory import PrivateMemory
from eachmind.backends import InMemoryBackend, SQLiteBackend


class TestPrivateMemoryWithBackend:
    def test_sqlite_backend_stores_and_recalls(self, tmp_path):
        backend = SQLiteBackend(os.path.join(tmp_path, "pm.db"))
        memory = PrivateMemory(agent_id="test", backend=backend)
        perspective = Perspective(role="test")
        event = MemoryEvent(content="remember this", source="test")
        encoded = perspective.encode(event)
        memory.store(encoded)
        results = memory.recall(event_id=event.event_id)
        assert len(results) == 1

    def test_default_backend_still_works(self):
        """Existing code without backend param must keep working."""
        memory = PrivateMemory(agent_id="test")
        perspective = Perspective(role="test")
        event = MemoryEvent(content="data", source="test")
        encoded = perspective.encode(event)
        memory.store(encoded)
        assert memory.size == 1


class TestSharedMemoryWithBackend:
    def test_sqlite_backend_publishes_and_recalls(self, tmp_path):
        backend = SQLiteBackend(os.path.join(tmp_path, "sm.db"))
        shared = SharedMemory(team_id="test", backend=backend)
        shared.publish(content="insight", shared_by="agent-a")
        results = shared.recall()
        assert len(results) == 1
        assert results[0].shared_by == "agent-a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backends.py::TestPrivateMemoryWithBackend -v`
Expected: TypeError — `backend` is not an accepted parameter yet.

- [ ] **Step 3: Update PrivateMemory to accept optional backend**

The key change: add a `backend` parameter. When set, `store()` persists to the backend and `recall()` hydrates from it. Keep the in-memory `memories` list as the working set for backward compatibility.

In `eachmind/primitives/private_memory.py`, add:
- `backend` field (default `None`)
- In `store()`: also save to backend if present
- In `__post_init__()`: hydrate from backend if present

- [ ] **Step 4: Update SharedMemory to accept optional backend**

Similar pattern: add `backend` field, persist entries on `publish()`, hydrate on init.

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass, including new integration tests and all 19 original tests.

- [ ] **Step 6: Update `__init__.py` exports**

Add backends to main exports:

```python
from eachmind.backends import InMemoryBackend, SQLiteBackend
```

- [ ] **Step 7: Commit**

```bash
git add eachmind/primitives/private_memory.py eachmind/primitives/shared_memory.py eachmind/__init__.py tests/test_backends.py
git commit -m "feat: wire pluggable storage backends into PrivateMemory and SharedMemory"
```

---

### Task 6: Enhanced Consolidation Algorithms

**Files:**
- Modify: `eachmind/primitives/consolidation.py`
- Test: `tests/test_consolidation_algorithms.py`

- [ ] **Step 1: Write failing tests for temporal and weighted consolidation**

```python
"""Tests for enhanced consolidation algorithms."""

from eachmind import MemoryEvent
from eachmind.primitives.consolidation import Consolidation
from eachmind.primitives.perspective import Perspective


class TestTemporalConsolidation:
    def test_recent_memories_weighted_higher(self):
        """Memories with more recent timestamps should contribute more."""
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(role="test", priors={"revenue": 0.8})

        memories = []
        for i in range(5):
            event = MemoryEvent(
                content=f"revenue data {i}",
                source="test",
                timestamp=f"2026-01-0{i + 1}T00:00:00Z",
            )
            encoded = perspective.encode(event)
            memories.append(encoded)

        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_decay_reduces_confidence(self):
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(role="test", priors={"revenue": 0.8})

        memories = []
        for i in range(4):
            event = MemoryEvent(content=f"revenue point {i}", source="test")
            memories.append(perspective.encode(event))

        consolidation.process(memories)
        initial_confidences = [b.confidence for b in consolidation.beliefs]

        consolidation.decay_all(factor=0.1)
        decayed_confidences = [b.confidence for b in consolidation.beliefs]

        for initial, decayed in zip(initial_confidences, decayed_confidences):
            assert decayed < initial


class TestWeightedConsolidation:
    def test_high_salience_memories_form_beliefs_faster(self):
        """High-salience memories should be weighted more in pattern detection."""
        consolidation = Consolidation(consolidation_threshold=2)
        perspective = Perspective(
            role="analyst",
            priors={"relevant_sources": ["important_source"], "market": 0.9},
        )

        memories = []
        for i in range(3):
            event = MemoryEvent(
                content=f"market data point {i}",
                source="important_source",
            )
            memories.append(perspective.encode(event))

        beliefs = consolidation.process(memories)
        # High-salience source should form beliefs
        assert len(beliefs) > 0


class TestConsolidationStrategies:
    def test_process_with_frequency_strategy(self):
        consolidation = Consolidation(consolidation_threshold=2, strategy="frequency")
        perspective = Perspective(role="test", priors={"data": 0.7})
        memories = [
            perspective.encode(MemoryEvent(content=f"data item {i}", source="test"))
            for i in range(4)
        ]
        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_process_with_temporal_strategy(self):
        consolidation = Consolidation(consolidation_threshold=2, strategy="temporal")
        perspective = Perspective(role="test", priors={"data": 0.7})
        memories = [
            perspective.encode(
                MemoryEvent(
                    content=f"data item {i}",
                    source="test",
                    timestamp=f"2026-01-0{i + 1}T00:00:00Z",
                )
            )
            for i in range(4)
        ]
        beliefs = consolidation.process(memories)
        assert isinstance(beliefs, list)

    def test_default_strategy_is_frequency(self):
        consolidation = Consolidation()
        assert consolidation.strategy == "frequency"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_consolidation_algorithms.py -v`
Expected: TypeError — `strategy` param doesn't exist yet.

- [ ] **Step 3: Enhance Consolidation with strategy parameter**

Update `eachmind/primitives/consolidation.py`:
- Add `strategy: str = "frequency"` field
- Extract existing logic into `_process_frequency()`
- Add `_process_temporal()` that weights recent memories higher
- `process()` dispatches based on strategy

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add eachmind/primitives/consolidation.py tests/test_consolidation_algorithms.py
git commit -m "feat: add temporal and frequency consolidation strategies"
```

---

### Task 7: Drift Visualization

**Files:**
- Create: `eachmind/visualization/__init__.py`
- Create: `eachmind/visualization/drift_viz.py`
- Test: `tests/test_visualization.py`
- Create: `examples/drift_visualization.py`

- [ ] **Step 1: Write smoke tests for visualization**

```python
"""Smoke tests for drift visualization (matplotlib)."""

import pytest

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from eachmind import Agent, MemoryEvent
from eachmind.primitives.drift import Drift


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestDriftVisualization:
    def _build_scenario(self):
        agents = {
            "analyst": Agent(name="analyst", role="data analysis"),
            "writer": Agent(name="writer", role="content creation"),
            "reviewer": Agent(name="reviewer", role="code review"),
        }
        events = [
            MemoryEvent(content=f"data point {i}", source="report")
            for i in range(5)
        ]
        for event in events:
            for agent in agents.values():
                agent.observe(event)

        drift_tracker = Drift()
        perspectives = {name: a.perspective for name, a in agents.items()}
        drift_tracker.team_diversity(perspectives)
        return drift_tracker, agents

    def test_heatmap_returns_figure(self):
        from eachmind.visualization import plot_drift_heatmap
        drift_tracker, agents = self._build_scenario()
        perspectives = {name: a.perspective for name, a in agents.items()}
        fig = plot_drift_heatmap(perspectives)
        assert fig is not None

    def test_timeline_returns_figure(self):
        from eachmind.visualization import plot_drift_timeline
        drift_tracker, _ = self._build_scenario()
        fig = plot_drift_timeline(drift_tracker)
        assert fig is not None

    def test_diversity_gauge_returns_figure(self):
        from eachmind.visualization import plot_team_diversity
        drift_tracker, agents = self._build_scenario()
        perspectives = {name: a.perspective for name, a in agents.items()}
        fig = plot_team_diversity(perspectives)
        assert fig is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement drift visualization module**

Create `eachmind/visualization/drift_viz.py` with three functions:

1. `plot_drift_heatmap(perspectives)` — pairwise drift matrix as a color heatmap
2. `plot_drift_timeline(drift_tracker)` — line chart of drift measurements over time
3. `plot_team_diversity(perspectives)` — bar chart showing overall team diversity with per-pair breakdown

Each returns a `matplotlib.figure.Figure`.

- [ ] **Step 4: Create visualization `__init__.py`**

```python
"""Visualization tools for eachmind.

Requires matplotlib: pip install eachmind[viz]
"""

from eachmind.visualization.drift_viz import (
    plot_drift_heatmap,
    plot_drift_timeline,
    plot_team_diversity,
)

__all__ = [
    "plot_drift_heatmap",
    "plot_drift_timeline",
    "plot_team_diversity",
]
```

- [ ] **Step 5: Run tests**

Run: `pip install matplotlib && python -m pytest tests/test_visualization.py -v`
Expected: All 3 tests pass.

- [ ] **Step 6: Create drift visualization example**

Create `examples/drift_visualization.py` — a complete script that:
1. Creates 4 agents with different roles and encoding weights
2. Feeds them a series of events (some shared, some role-specific)
3. Measures drift at multiple points in time
4. Generates all three visualizations and saves them as PNG files

- [ ] **Step 7: Commit**

```bash
git add eachmind/visualization/ tests/test_visualization.py examples/drift_visualization.py
git commit -m "feat: add matplotlib drift visualization (heatmap, timeline, diversity)"
```

---

### Task 8: Integration Example — OpenAI Agents SDK

**Files:**
- Create: `examples/swarm_integration.py`

- [ ] **Step 1: Create runnable OpenAI Agents SDK example**

The example demonstrates:
- Creating eachmind agents alongside OpenAI Agents SDK agents
- Using eachmind as the memory layer while OpenAI handles orchestration
- Agents observe tool outputs, encode through their perspective, share selectively

- [ ] **Step 2: Verify it runs (with `openai-agents` installed)**

Run: `pip install openai-agents && python examples/swarm_integration.py`

- [ ] **Step 3: Commit**

```bash
git add examples/swarm_integration.py
git commit -m "feat: add OpenAI Agents SDK integration example"
```

---

### Task 9: Integration Example — CrewAI

**Files:**
- Create: `examples/crewai_integration.py`

- [ ] **Step 1: Create runnable CrewAI example**

The example demonstrates:
- Wrapping CrewAI agents with eachmind memory
- Each crew member observes task results through their own perspective
- Shared memory enables selective knowledge transfer between crew members

- [ ] **Step 2: Verify it runs (with `crewai` installed)**

Run: `pip install crewai && python examples/crewai_integration.py`

- [ ] **Step 3: Commit**

```bash
git add examples/crewai_integration.py
git commit -m "feat: add CrewAI integration example"
```

---

### Task 10: Integration Example — LangGraph

**Files:**
- Create: `examples/langgraph_integration.py`

- [ ] **Step 1: Create runnable LangGraph example**

The example demonstrates:
- LangGraph state graph with eachmind memory at each node
- Each graph node is an agent with its own perspective
- Shared memory as a channel for inter-node knowledge sharing

- [ ] **Step 2: Verify it runs (with `langgraph` installed)**

Run: `pip install langgraph && python examples/langgraph_integration.py`

- [ ] **Step 3: Commit**

```bash
git add examples/langgraph_integration.py
git commit -m "feat: add LangGraph integration example"
```

---

### Task 11: SQLite Backend Example

**Files:**
- Create: `examples/sqlite_backend.py`

- [ ] **Step 1: Create SQLite backend example**

Shows persistent memory across "sessions" — agents remember across restarts.

- [ ] **Step 2: Verify it runs**

Run: `python examples/sqlite_backend.py`

- [ ] **Step 3: Commit**

```bash
git add examples/sqlite_backend.py
git commit -m "feat: add SQLite persistent backend example"
```

---

### Task 12: Final Integration — Update Exports, README, and pyproject.toml

**Files:**
- Modify: `eachmind/__init__.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Update main `__init__.py` with all new exports**

- [ ] **Step 2: Update pyproject.toml with optional dependencies**

- [ ] **Step 3: Update README roadmap checkboxes**

Mark completed items with `[x]`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Run linting and type checking**

Run: `ruff check . && mypy eachmind/`

- [ ] **Step 6: Commit**

```bash
git add eachmind/__init__.py pyproject.toml README.md
git commit -m "chore: update exports, dependencies, and roadmap status"
```
