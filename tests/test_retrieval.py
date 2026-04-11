"""Tests for retrieval module."""
from __future__ import annotations

from eachmind.primitives.memory_event import MemoryEvent
from eachmind.primitives.perspective import EncodedMemory, Perspective
from eachmind.retrieval import TFIDFRetriever
from eachmind.retrieval.base import Retriever


class TestRetrieverProtocol:
    def test_tfidf_satisfies_protocol(self):
        assert isinstance(TFIDFRetriever(), Retriever)


class TestTFIDFRetriever:
    def _make_memories(self) -> list[EncodedMemory]:
        perspective = Perspective(role="test")
        events = [
            MemoryEvent(
                content="quarterly revenue grew 23% year over year",
                source="finance",
            ),
            MemoryEvent(
                content="customer churn decreased to 3.2% this quarter",
                source="analytics",
            ),
            MemoryEvent(
                content="new product launch scheduled for Q3",
                source="product",
            ),
            MemoryEvent(
                content="revenue forecast indicates continued growth",
                source="finance",
            ),
            MemoryEvent(
                content="engineering team expanded by 15 engineers",
                source="hr",
            ),
        ]
        return [perspective.encode(e) for e in events]

    def test_search_returns_relevant_results(self):
        retriever = TFIDFRetriever()
        memories = self._make_memories()
        results = retriever.search("revenue growth", memories, limit=2)
        assert len(results) <= 2
        contents = [str(r.encoded_content) for r in results]
        assert any("revenue" in c.lower() for c in contents)

    def test_search_empty_query(self):
        retriever = TFIDFRetriever()
        memories = self._make_memories()
        results = retriever.search("", memories, limit=5)
        assert results == []

    def test_search_no_matches(self):
        retriever = TFIDFRetriever()
        memories = self._make_memories()
        results = retriever.search(
            "xyzzy quantum entanglement", memories, limit=5
        )
        assert isinstance(results, list)

    def test_search_respects_limit(self):
        retriever = TFIDFRetriever()
        memories = self._make_memories()
        results = retriever.search("revenue", memories, limit=1)
        assert len(results) <= 1

    def test_search_empty_memories(self):
        retriever = TFIDFRetriever()
        results = retriever.search("test", [], limit=5)
        assert results == []


class TestPrivateMemoryWithRetriever:
    def test_search_uses_retriever(self):
        from eachmind.primitives.private_memory import PrivateMemory

        retriever = TFIDFRetriever()
        memory = PrivateMemory(agent_id="test", retriever=retriever)
        perspective = Perspective(role="test")

        events = [
            MemoryEvent(content="revenue grew 23%", source="finance"),
            MemoryEvent(content="new hire onboarding", source="hr"),
            MemoryEvent(
                content="revenue forecast positive", source="finance"
            ),
        ]
        for e in events:
            memory.store(perspective.encode(e))

        results = memory.search("revenue")
        assert len(results) >= 1
        contents = [str(r.encoded_content) for r in results]
        assert any("revenue" in c.lower() for c in contents)

    def test_search_without_retriever_uses_substring(self):
        from eachmind.primitives.private_memory import PrivateMemory

        memory = PrivateMemory(agent_id="test")
        perspective = Perspective(role="test")
        memory.store(perspective.encode(
            MemoryEvent(content="revenue data", source="test")
        ))
        results = memory.search("revenue")
        assert len(results) >= 1
