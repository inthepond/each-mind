"""
MemoryEvent — A discrete experience in the eachmind protocol.

The same event can be observed by multiple agents, but each encodes it
differently based on their own context, role, and history. This is
fundamental to how eachmind creates genuine perspective divergence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryEvent:
    """A discrete experience that agents can observe and encode.

    A MemoryEvent represents something that happened — a piece of data,
    an observation, a result, an interaction. When multiple agents observe
    the same event, each produces its own private encoding shaped by their
    Perspective. The raw event itself is neutral; meaning comes from encoding.

    Attributes:
        content: The raw content of the event (text, data, or structured object).
        source: Where the event originated (e.g., "user_input", "api_response").
        timestamp: When the event occurred (defaults to now).
        event_id: Unique identifier for this event instance.
        metadata: Optional additional context about the event.
    """

    content: Any
    source: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.content is None:
            raise ValueError("MemoryEvent content cannot be None")

    def with_metadata(self, **kwargs: Any) -> MemoryEvent:
        """Return a new event with additional metadata merged in."""
        merged = {**self.metadata, **kwargs}
        return MemoryEvent(
            content=self.content,
            source=self.source,
            timestamp=self.timestamp,
            event_id=self.event_id,
            metadata=merged,
        )

    def __repr__(self) -> str:
        content_preview = str(self.content)[:60]
        return f"MemoryEvent(source={self.source!r}, content={content_preview!r})"
