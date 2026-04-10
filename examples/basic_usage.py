"""
Basic eachmind usage — two agents observe the same event, develop
different perspectives, and selectively share knowledge.
"""

from eachmind import Agent, MemoryEvent, SharedMemory
from eachmind.primitives.drift import Drift


def main():
    # --- Setup: create agents with different roles ---
    analyst = Agent(name="analyst", role="data analysis")
    analyst.perspective.encoding_weights = {
        "numerical_data": 0.9,
        "trends": 0.8,
        "sentiment": 0.2,
    }
    analyst.perspective.priors = {
        "relevant_sources": ["quarterly_report", "market_data"],
        "source_trust": {"quarterly_report": 0.95, "news": 0.4},
    }

    writer = Agent(name="writer", role="content creation")
    writer.perspective.encoding_weights = {
        "narrative": 0.9,
        "sentiment": 0.8,
        "numerical_data": 0.3,
    }
    writer.perspective.priors = {
        "relevant_sources": ["interview", "news"],
        "source_trust": {"interview": 0.9, "quarterly_report": 0.6},
    }

    # --- Connect both to shared memory ---
    shared = SharedMemory(team_id="content-team")
    analyst.connect(shared)
    writer.connect(shared)

    # --- Both observe the same event ---
    event = MemoryEvent(
        content="Q1 revenue grew 23% YoY, driven by enterprise expansion",
        source="quarterly_report",
    )

    encoding_a = analyst.observe(event)
    encoding_w = writer.observe(event)

    print("=== Same event, different encodings ===")
    print(f"Analyst salience: {encoding_a.salience:.2f}")
    print(f"Writer salience:  {encoding_w.salience:.2f}")
    print()

    # --- Feed more events to build divergence ---
    events = [
        MemoryEvent(content="Enterprise ARR crossed $50M milestone", source="quarterly_report"),
        MemoryEvent(content="Customer churn decreased to 3.2%", source="quarterly_report"),
        MemoryEvent(content="CEO interview: 'We're just getting started'", source="interview"),
        MemoryEvent(content="Competitor launched similar product at lower price", source="news"),
    ]

    for e in events:
        analyst.observe(e)
        writer.observe(e)

    # --- Analyst shares a finding ---
    analyst.share(
        content="Revenue growth acceleration + declining churn = strong unit economics",
        to=SharedMemory.TEAM,
        reason="Key insight for team strategy discussion",
    )

    # --- Writer retrieves shared knowledge ---
    shared_knowledge = writer.recall(source=SharedMemory.TEAM)
    print("=== Shared knowledge accessible to writer ===")
    for entry in shared_knowledge:
        print(f"  From {entry.shared_by}: {entry.content}")
    print()

    # --- Measure drift between perspectives ---
    drift_tracker = Drift()
    measurement = drift_tracker.measure(
        "analyst", analyst.perspective,
        "writer", writer.perspective,
    )
    print("=== Perspective drift ===")
    print(f"Drift between analyst and writer: {measurement.drift_value:.3f}")
    print()

    # --- Consolidate memories into beliefs ---
    analyst.consolidate()
    print("=== Analyst beliefs after consolidation ===")
    for belief in analyst.consolidation.beliefs:
        print(f"  [{belief.confidence:.2f}] {belief.content}")
    print()

    # --- Private memory stats ---
    print("=== Memory stats ===")
    print(f"Analyst private memories: {analyst.private_memory.size}")
    print(f"Writer private memories:  {writer.private_memory.size}")
    print(f"Shared memory entries:    {shared.size}")


if __name__ == "__main__":
    main()
