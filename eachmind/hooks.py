"""
Observability hooks for eachmind.

Hooks allow monitoring agent memory operations without modifying
the core logic. Attach hooks to Agent instances to receive
callbacks on observe, share, recall, and consolidate events.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger("eachmind")


@runtime_checkable
class MemoryHook(Protocol):
    """Protocol for memory operation observers."""

    def on_observe(
        self, agent_name: str, event_id: str, salience: float
    ) -> None: ...

    def on_share(
        self, agent_name: str, content: str, scope: str
    ) -> None: ...

    def on_recall(
        self, agent_name: str, count: int, source: str
    ) -> None: ...

    def on_consolidate(
        self, agent_name: str, belief_count: int
    ) -> None: ...


class LoggingHook:
    """Default hook that logs memory operations via stdlib logging."""

    def __init__(self, level: int = logging.DEBUG) -> None:
        self.level = level

    def on_observe(
        self, agent_name: str, event_id: str, salience: float
    ) -> None:
        logger.log(
            self.level,
            "[%s] observed event %s (salience=%.2f)",
            agent_name,
            event_id,
            salience,
        )

    def on_share(
        self, agent_name: str, content: str, scope: str
    ) -> None:
        logger.log(
            self.level,
            "[%s] shared to %s: %s",
            agent_name,
            scope,
            content[:100],
        )

    def on_recall(
        self, agent_name: str, count: int, source: str
    ) -> None:
        logger.log(
            self.level,
            "[%s] recalled %d items from %s",
            agent_name,
            count,
            source,
        )

    def on_consolidate(
        self, agent_name: str, belief_count: int
    ) -> None:
        logger.log(
            self.level,
            "[%s] consolidated into %d beliefs",
            agent_name,
            belief_count,
        )
