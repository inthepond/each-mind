#!/usr/bin/env python3
"""Drift Visualization Demo — eachmind.

Demonstrates the core insight of eachmind: agents observing the same events
develop genuinely different perspectives over time. This script creates four
agents with distinct roles, feeds them a stream of events, and generates
rich visualizations showing how their perspectives diverge.

Output: three PNG files in the current directory.
  - drift_heatmap.png
  - drift_timeline.png
  - team_diversity.png

Usage:
    python examples/drift_visualization.py
"""
from __future__ import annotations

from eachmind import Agent, MemoryEvent
from eachmind.primitives.drift import Drift
from eachmind.visualization import (
    plot_drift_heatmap,
    plot_drift_timeline,
    plot_team_diversity,
)


def main() -> None:
    print("=" * 64)
    print("  eachmind — Drift Visualization Demo")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Stage 1: Create agents with distinct cognitive profiles
    # ------------------------------------------------------------------
    print("[1/5] Creating agents with distinct roles and encoding weights...")

    analyst = Agent(name="analyst", role="data analysis")
    analyst.perspective.encoding_weights = {
        "numerical_data": 0.95,
        "trends": 0.85,
        "anomalies": 0.80,
        "sentiment": 0.20,
    }
    analyst.perspective.priors = {
        "relevant_sources": ["metrics", "report"],
        "source_trust": {"metrics": 0.95, "social": 0.3},
        "focus": "quantitative patterns",
    }

    writer = Agent(name="writer", role="content creation")
    writer.perspective.encoding_weights = {
        "narrative": 0.95,
        "sentiment": 0.90,
        "tone": 0.80,
        "numerical_data": 0.25,
    }
    writer.perspective.priors = {
        "relevant_sources": ["social", "interview"],
        "source_trust": {"social": 0.85, "metrics": 0.4},
        "focus": "human stories",
    }

    reviewer = Agent(name="reviewer", role="code review")
    reviewer.perspective.encoding_weights = {
        "correctness": 0.95,
        "clarity": 0.80,
        "edge_cases": 0.85,
        "sentiment": 0.10,
    }
    reviewer.perspective.priors = {
        "relevant_sources": ["code", "report"],
        "source_trust": {"code": 0.95, "social": 0.2},
        "focus": "logical rigour",
    }

    strategist = Agent(name="strategist", role="strategic planning")
    strategist.perspective.encoding_weights = {
        "trends": 0.90,
        "competitive_signal": 0.85,
        "narrative": 0.60,
        "numerical_data": 0.55,
    }
    strategist.perspective.priors = {
        "relevant_sources": ["report", "social"],
        "source_trust": {"report": 0.8, "metrics": 0.7, "social": 0.6},
        "focus": "big-picture opportunity",
    }

    agents = {
        "analyst": analyst,
        "writer": writer,
        "reviewer": reviewer,
        "strategist": strategist,
    }

    print(f"  Created {len(agents)} agents: {', '.join(agents.keys())}")
    print()

    # ------------------------------------------------------------------
    # Stage 2: Feed events & take drift measurements over time
    # ------------------------------------------------------------------
    print("[2/5] Feeding events and measuring drift at each time step...")

    drift_tracker = Drift()

    # Events are tagged by source so we can route some selectively.
    # Shared events go to everyone; domain-specific events go to a subset.
    # This creates genuine history divergence (different history hashes)
    # which is what Perspective.drift_from uses to detect drift.

    shared_events = [
        MemoryEvent(content="Q3 all-hands: company pivoting to enterprise", source="report"),
        MemoryEvent(content="Team retro highlights communication gaps", source="report"),
    ]

    role_specific_batches: list[dict[str, list[MemoryEvent]]] = [
        # Batch 1 — each agent gets shared + role-specific events
        {
            "analyst": [
                MemoryEvent(content="Revenue increased 12% quarter-over-quarter", source="metrics"),
                MemoryEvent(content="Customer churn dropped to 2.1%", source="metrics"),
            ],
            "writer": [
                MemoryEvent(
                    content="Users praise the new onboarding flow on social media",
                    source="social",
                ),
                MemoryEvent(content="Draft blog post on Q3 results", source="social"),
            ],
            "reviewer": [
                MemoryEvent(content="Code review found edge case in billing module", source="code"),
                MemoryEvent(content="PR #421 has failing type checks", source="code"),
            ],
            "strategist": [
                MemoryEvent(content="New competitor entered the market last week", source="report"),
                MemoryEvent(content="Strategic partnership discussion ongoing", source="report"),
            ],
        },
        # Batch 2 — more specialisation
        {
            "analyst": [
                MemoryEvent(content="A/B test shows 8% lift with new copy", source="metrics"),
            ],
            "writer": [
                MemoryEvent(
                    content="Interview with power user reveals unexpected workflow",
                    source="interview",
                ),
            ],
            "reviewer": [
                MemoryEvent(content="Refactored auth module passes all tests", source="code"),
            ],
            "strategist": [
                MemoryEvent(content="Board meeting summary: focus on retention", source="report"),
            ],
        },
        # Batch 3 — cross-pollination (some overlap)
        {
            "analyst": [
                MemoryEvent(content="Monthly active users hit 50k milestone", source="metrics"),
                MemoryEvent(
                    content="Support tickets mention confusing pricing page",
                    source="social",
                ),
            ],
            "writer": [
                MemoryEvent(content="Blog post on product vision goes viral", source="social"),
                MemoryEvent(content="Monthly active users hit 50k milestone", source="metrics"),
            ],
            "reviewer": [
                MemoryEvent(content="Dependency audit completed, 3 upgrades needed", source="code"),
                MemoryEvent(
                    content="Load testing reveals 99.5% uptime over 30 days",
                    source="metrics",
                ),
            ],
            "strategist": [
                MemoryEvent(
                    content="User interviews suggest need for better search",
                    source="interview",
                ),
                MemoryEvent(
                    content="Support tickets mention confusing pricing page",
                    source="social",
                ),
            ],
        },
        # Batch 4 — deep specialisation
        {
            "analyst": [
                MemoryEvent(
                    content="Funnel conversion rate up 3pp after redesign",
                    source="metrics",
                ),
                MemoryEvent(content="NPS score dipped from 62 to 58", source="metrics"),
            ],
            "writer": [
                MemoryEvent(
                    content="Customer success story: Acme Corp saves 40%",
                    source="interview",
                ),
                MemoryEvent(content="Newsletter open rate trending down", source="social"),
            ],
            "reviewer": [
                MemoryEvent(content="Security audit: no critical issues found", source="code"),
                MemoryEvent(content="Codebase complexity score improved 12%", source="code"),
            ],
            "strategist": [
                MemoryEvent(
                    content="Competitor acquired a key integration partner",
                    source="report",
                ),
                MemoryEvent(content="Enterprise pilot with BigCorp progressing", source="report"),
            ],
        },
        # Batch 5 — final round
        {
            "analyst": [
                MemoryEvent(
                    content="Cohort analysis: Jan users retain 15% better",
                    source="metrics",
                ),
            ],
            "writer": [
                MemoryEvent(content="Product launch announcement drafted", source="social"),
            ],
            "reviewer": [
                MemoryEvent(content="CI pipeline now 40% faster after caching", source="code"),
            ],
            "strategist": [
                MemoryEvent(content="Investor update: Series B timeline set", source="report"),
            ],
        },
    ]

    for batch_idx, role_batch in enumerate(role_specific_batches):
        # Shared events on first batch only
        if batch_idx == 0:
            for event in shared_events:
                for agent in agents.values():
                    agent.observe(event)

        # Role-specific events
        for agent_name, events in role_batch.items():
            for event in events:
                agents[agent_name].observe(event)

        # Consolidate beliefs periodically
        if batch_idx % 2 == 1:
            for agent in agents.values():
                agent.consolidate()

        # Record a pairwise drift snapshot (for the timeline) and report diversity
        perspectives = {name: a.perspective for name, a in agents.items()}
        drift_tracker.snapshot(perspectives)
        diversity = drift_tracker.team_diversity(perspectives)

        total_events = sum(len(evts) for evts in role_batch.values())
        print(f"  Batch {batch_idx + 1}: {total_events} role-specific events  |  "
              f"Team diversity = {diversity:.4f}")

    total_measurements = len(drift_tracker.measurements)
    print(f"\n  Total drift measurements recorded: {total_measurements}")
    print()

    # ------------------------------------------------------------------
    # Stage 3: Generate the heatmap
    # ------------------------------------------------------------------
    print("[3/5] Generating drift heatmap...")

    perspectives = {name: a.perspective for name, a in agents.items()}
    fig_heatmap = plot_drift_heatmap(perspectives)
    fig_heatmap.savefig("drift_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: drift_heatmap.png")
    print("  -> Shows pairwise drift between all agents as a colour matrix.")
    print("     Green cells = similar perspectives, red cells = divergent.")
    import matplotlib.pyplot as plt
    plt.close(fig_heatmap)
    print()

    # ------------------------------------------------------------------
    # Stage 4: Generate the timeline
    # ------------------------------------------------------------------
    print("[4/5] Generating drift timeline...")

    fig_timeline = plot_drift_timeline(drift_tracker)
    fig_timeline.savefig("drift_timeline.png", dpi=150, bbox_inches="tight")
    print("  Saved: drift_timeline.png")
    print("  -> Tracks how each agent pair's drift evolves over successive")
    print("     measurement rounds. Rising lines indicate increasing divergence.")
    plt.close(fig_timeline)
    print()

    # ------------------------------------------------------------------
    # Stage 5: Generate the team diversity overview
    # ------------------------------------------------------------------
    print("[5/5] Generating team diversity overview...")

    fig_diversity = plot_team_diversity(perspectives)
    fig_diversity.savefig("team_diversity.png", dpi=150, bbox_inches="tight")
    print("  Saved: team_diversity.png")
    print("  -> Comprehensive view: bar chart of all pairwise drift values")
    print("     plus a gauge showing overall cognitive diversity score.")
    plt.close(fig_diversity)
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Visualization complete!")
    print()
    print("  Key insight: agents observing overlapping but role-specific")
    print("  events develop genuinely divergent perspectives. Each agent")
    print("  encodes through its own lens — shaped by role, priors, history.")
    print()
    print("  This is what eachmind measures: genuine cognitive diversity")
    print("  that emerges naturally from differentiated memory encoding.")
    print("=" * 64)


if __name__ == "__main__":
    main()
