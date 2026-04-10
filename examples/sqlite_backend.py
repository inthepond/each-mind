"""
eachmind SQLite Backend Example — Persistent Agent Memory

Demonstrates how agents can persist memories across sessions using
the SQLite storage backend. Without persistence, agents start fresh
every run. With SQLite, they accumulate knowledge over time.

Usage: python examples/sqlite_backend.py
"""

import os
import tempfile

from eachmind import Agent, MemoryEvent, PrivateMemory, SharedMemory, SQLiteBackend


def run_session_1(db_path: str) -> None:
    """First session: agents observe events and share insights."""
    print("--- Session 1: First run ---\n")

    backend = SQLiteBackend(db_path)

    analyst = Agent(
        name="analyst",
        role="data analysis",
        private_memory=PrivateMemory(agent_id="analyst", backend=backend),
    )
    analyst.perspective.encoding_weights = {
        "numerical_data": 0.9,
        "trends": 0.8,
        "sentiment": 0.2,
    }

    researcher = Agent(
        name="researcher",
        role="market research",
        private_memory=PrivateMemory(agent_id="researcher", backend=backend),
    )
    researcher.perspective.encoding_weights = {
        "sentiment": 0.85,
        "trends": 0.75,
        "numerical_data": 0.4,
    }

    shared = SharedMemory(team_id="insight-team", backend=backend)
    analyst.connect(shared)
    researcher.connect(shared)

    events = [
        MemoryEvent(
            content="Q1 revenue grew 23% YoY, driven by enterprise expansion",
            source="quarterly_report",
        ),
        MemoryEvent(
            content="Enterprise ARR crossed $50M milestone",
            source="quarterly_report",
        ),
        MemoryEvent(
            content="Customer churn decreased to 3.2%, lowest in company history",
            source="quarterly_report",
        ),
        MemoryEvent(
            content="Market sentiment toward SaaS remains cautiously optimistic",
            source="news",
        ),
    ]

    print("Observing events...")
    for event in events:
        analyst.observe(event)
        researcher.observe(event)

    # Analyst shares a key finding to shared memory
    analyst.share(
        content="Revenue growth + declining churn signals strong product-market fit",
        to=SharedMemory.TEAM,
        reason="Critical strategic insight for next planning cycle",
    )

    # Researcher adds a complementary view
    researcher.share(
        content="Market sentiment is positive — good timing for expansion push",
        to=SharedMemory.TEAM,
        reason="External conditions support growth strategy",
    )

    analyst.consolidate()

    print(f"Analyst private memories:    {analyst.private_memory.size}")
    print(f"Researcher private memories: {researcher.private_memory.size}")
    print(f"Shared memory entries:       {shared.size}")
    print(f"Analyst beliefs formed:      {len(analyst.consolidation.beliefs)}")
    print()
    print("Session 1 complete. Memories persisted to SQLite.")
    print(f"Database: {db_path}")


def run_session_2(db_path: str) -> None:
    """Second session: new agent instances hydrate from the same SQLite database."""
    print("--- Session 2: After 'restart' ---\n")

    # New backend connection to the SAME database file
    backend = SQLiteBackend(db_path)

    # Completely fresh Agent instances — but backed by the same SQLite file
    analyst = Agent(
        name="analyst",
        role="data analysis",
        private_memory=PrivateMemory(agent_id="analyst", backend=backend),
    )

    researcher = Agent(
        name="researcher",
        role="market research",
        private_memory=PrivateMemory(agent_id="researcher", backend=backend),
    )

    shared = SharedMemory(team_id="insight-team", backend=backend)
    analyst.connect(shared)
    researcher.connect(shared)

    print("Memories hydrated from SQLite (no events observed yet in this session):")
    print(f"Analyst private memories:    {analyst.private_memory.size}")
    print(f"Researcher private memories: {researcher.private_memory.size}")
    print(f"Shared memory entries:       {shared.size}")
    print()

    # Recall previously shared knowledge — it persisted across the restart
    shared_knowledge = analyst.recall(source=SharedMemory.TEAM)
    print("Shared knowledge retrieved from SQLite:")
    for entry in shared_knowledge:
        print(f"  [{entry.shared_by}] {entry.content}")
    print()

    # Observe new events in this session — memories accumulate
    new_events = [
        MemoryEvent(
            content="Q2 pipeline looks strong; enterprise deals up 40%",
            source="sales_report",
        ),
        MemoryEvent(
            content="Competitor pricing pressure increasing in mid-market segment",
            source="news",
        ),
    ]

    print("Observing new events in Session 2...")
    for event in new_events:
        analyst.observe(event)
        researcher.observe(event)

    analyst.share(
        content="Session 2 update: pipeline strength offsets competitive pressure",
        to=SharedMemory.TEAM,
        reason="Updating team with latest data",
    )

    print()
    print("Memories after Session 2 observations (cumulative across both sessions):")
    print(f"Analyst private memories:    {analyst.private_memory.size}")
    print(f"Researcher private memories: {researcher.private_memory.size}")
    print(f"Shared memory entries:       {shared.size}")


def main() -> None:
    db_path = os.path.join(tempfile.gettempdir(), "eachmind_demo.db")

    # Clean up any previous run so the demo starts fresh
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=" * 60)
    print("  eachmind — SQLite Persistent Memory Demo")
    print("=" * 60)
    print()

    run_session_1(db_path)

    print()
    print("=" * 60)
    print()

    run_session_2(db_path)

    print()
    print("=" * 60)
    print("Demo complete. Key takeaway:")
    print("  Without a backend, agents start fresh every run.")
    print("  With SQLiteBackend, memories survive across restarts.")
    print("=" * 60)

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main()
