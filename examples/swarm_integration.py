"""
eachmind + OpenAI Agents SDK Integration Example

Demonstrates how eachmind provides per-agent memory alongside
the OpenAI Agents SDK. Each SDK agent gets its own memory,
perspective, and belief system through eachmind.

The OpenAI Agents SDK handles orchestration and tool definitions,
while eachmind handles per-agent memory, perspective encoding,
and knowledge sharing between agents.

Usage: python examples/swarm_integration.py
Requires: pip install openai-agents
"""

from agents import Agent as SDKAgent
from eachmind import Agent as MindAgent, MemoryEvent, SharedMemory
from eachmind.primitives.drift import Drift


# ---------------------------------------------------------------------------
# Simulated agent "thinking" — no API keys needed
# ---------------------------------------------------------------------------

RESEARCHER_FINDINGS = [
    MemoryEvent(
        content="Primary research indicates 68% of enterprise users cite integration complexity as top barrier",
        source="user_interviews",
    ),
    MemoryEvent(
        content="Market size for AI orchestration tooling estimated at $4.2B by 2027, CAGR 31%",
        source="market_report",
    ),
    MemoryEvent(
        content="Three direct competitors lack per-agent memory isolation; shared context causes role confusion",
        source="competitive_analysis",
    ),
    MemoryEvent(
        content="Early adopters report 40% reduction in agent hallucinations when using perspective-aware memory",
        source="user_interviews",
    ),
]

ANALYST_INSIGHT = (
    "Integration complexity + competitor gap + measurable hallucination reduction = "
    "strong product-market fit signal for per-agent memory tooling"
)

WRITER_DRAFT_PROMPT = MemoryEvent(
    content="Draft audience: technical decision-makers evaluating multi-agent frameworks",
    source="editorial_brief",
)


def main():
    print("=" * 60)
    print("eachmind + OpenAI Agents SDK Integration Demo")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # 1. Define OpenAI Agents SDK agents (orchestration layer)
    #    These carry instructions and tool definitions for real runs.
    #    In production you would call Runner.run(sdk_researcher, prompt).
    # -----------------------------------------------------------------------
    sdk_researcher = SDKAgent(
        name="researcher",
        instructions=(
            "You are a market researcher. Gather data, run web searches, "
            "and surface key findings about the target market."
        ),
    )

    sdk_analyst = SDKAgent(
        name="analyst",
        instructions=(
            "You are a data analyst. Interpret research findings, identify "
            "patterns, and produce strategic insights."
        ),
    )

    sdk_writer = SDKAgent(
        name="writer",
        instructions=(
            "You are a technical writer. Transform insights into clear, "
            "compelling narratives for a developer audience."
        ),
    )

    print("--- OpenAI Agents SDK agents defined ---")
    for sdk_agent in (sdk_researcher, sdk_analyst, sdk_writer):
        print(f"  SDK agent: {sdk_agent.name!r}")
    print()

    # -----------------------------------------------------------------------
    # 2. Create paired eachmind agents — one per SDK agent
    #    Each carries its own perspective and private memory store.
    # -----------------------------------------------------------------------
    mind_researcher = MindAgent(name="researcher", role="market research")
    mind_researcher.perspective.encoding_weights = {
        "user_interviews": 0.95,
        "market_report": 0.85,
        "competitive_analysis": 0.80,
        "editorial_brief": 0.2,
    }
    mind_researcher.perspective.priors = {
        "relevant_sources": ["user_interviews", "market_report", "competitive_analysis"],
        "source_trust": {"user_interviews": 0.9, "market_report": 0.8},
    }

    mind_analyst = MindAgent(name="analyst", role="data analysis")
    mind_analyst.perspective.encoding_weights = {
        "market_report": 0.95,
        "competitive_analysis": 0.90,
        "user_interviews": 0.70,
        "editorial_brief": 0.1,
    }
    mind_analyst.perspective.priors = {
        "relevant_sources": ["market_report", "competitive_analysis"],
        "source_trust": {"market_report": 0.95, "competitive_analysis": 0.85},
    }

    mind_writer = MindAgent(name="writer", role="content creation")
    mind_writer.perspective.encoding_weights = {
        "editorial_brief": 0.95,
        "user_interviews": 0.75,
        "market_report": 0.40,
        "competitive_analysis": 0.30,
    }
    mind_writer.perspective.priors = {
        "relevant_sources": ["editorial_brief", "user_interviews"],
        "source_trust": {"editorial_brief": 0.95, "user_interviews": 0.8},
    }

    print("--- eachmind agents created with role-specific perspectives ---")
    for mind in (mind_researcher, mind_analyst, mind_writer):
        print(f"  Mind agent: {mind.name!r}  role: {mind.role!r}")
    print()

    # -----------------------------------------------------------------------
    # 3. Connect all agents to shared memory (team knowledge base)
    # -----------------------------------------------------------------------
    shared = SharedMemory(team_id="research-team")
    mind_researcher.connect(shared)
    mind_analyst.connect(shared)
    mind_writer.connect(shared)

    # -----------------------------------------------------------------------
    # 4. Researcher observes findings — each agent encodes differently
    # -----------------------------------------------------------------------
    print("=== Step 1: Researcher produces findings ===")
    print(f"  (Simulated — in production: Runner.run(sdk_researcher, task))")
    print()

    for finding in RESEARCHER_FINDINGS:
        enc_r = mind_researcher.observe(finding)
        enc_a = mind_analyst.observe(finding)
        enc_w = mind_writer.observe(finding)

        print(f"  Finding: \"{finding.content[:60]}...\"")
        print(f"    Researcher salience: {enc_r.salience:.2f}")
        print(f"    Analyst salience:    {enc_a.salience:.2f}")
        print(f"    Writer salience:     {enc_w.salience:.2f}")
        print()

    # -----------------------------------------------------------------------
    # 5. Analyst derives insight and shares to team memory
    # -----------------------------------------------------------------------
    print("=== Step 2: Analyst shares key insight to shared memory ===")
    print(f"  (Simulated — in production: Runner.run(sdk_analyst, findings))")
    print()

    mind_analyst.share(
        content=ANALYST_INSIGHT,
        to=SharedMemory.TEAM,
        reason="Strategic insight ready for writer to incorporate",
    )
    print(f"  Shared: \"{ANALYST_INSIGHT[:70]}...\"")
    print()

    # -----------------------------------------------------------------------
    # 6. Writer observes the editorial brief, then recalls shared insights
    # -----------------------------------------------------------------------
    print("=== Step 3: Writer recalls shared insights ===")
    print(f"  (Simulated — in production: Runner.run(sdk_writer, brief))")
    print()

    mind_writer.observe(WRITER_DRAFT_PROMPT)

    shared_knowledge = mind_writer.recall(source=SharedMemory.TEAM)
    for entry in shared_knowledge:
        print(f"  From {entry.shared_by!r}: \"{entry.content[:70]}...\"")
    print()

    # -----------------------------------------------------------------------
    # 7. Measure perspective drift between agents
    # -----------------------------------------------------------------------
    print("=== Step 4: Measure perspective drift between agents ===")

    drift_tracker = Drift()

    m_ra = drift_tracker.measure(
        "researcher", mind_researcher.perspective,
        "analyst",    mind_analyst.perspective,
    )
    m_rw = drift_tracker.measure(
        "researcher", mind_researcher.perspective,
        "writer",     mind_writer.perspective,
    )
    m_aw = drift_tracker.measure(
        "analyst",    mind_analyst.perspective,
        "writer",     mind_writer.perspective,
    )

    print(f"  Researcher <-> Analyst drift: {m_ra.drift_value:.3f}")
    print(f"  Researcher <-> Writer drift:  {m_rw.drift_value:.3f}")
    print(f"  Analyst    <-> Writer drift:  {m_aw.drift_value:.3f}")
    print()

    # -----------------------------------------------------------------------
    # 8. Consolidate each agent's memories into beliefs
    # -----------------------------------------------------------------------
    print("=== Step 5: Consolidate memories into beliefs ===")

    for mind in (mind_researcher, mind_analyst, mind_writer):
        mind.consolidate()
        print(f"  {mind.name.capitalize()} beliefs:")
        for belief in mind.consolidation.beliefs:
            print(f"    [{belief.confidence:.2f}] {belief.content}")
    print()

    # -----------------------------------------------------------------------
    # 9. Summary stats
    # -----------------------------------------------------------------------
    print("=== Memory stats ===")
    print(f"  Researcher private memories: {mind_researcher.private_memory.size}")
    print(f"  Analyst    private memories: {mind_analyst.private_memory.size}")
    print(f"  Writer     private memories: {mind_writer.private_memory.size}")
    print(f"  Shared memory entries:       {shared.size}")
    print()

    print("=" * 60)
    print("Key takeaway:")
    print("  OpenAI Agents SDK  → orchestration, tool calls, model routing")
    print("  eachmind           → per-agent memory, perspective, drift")
    print("  Together           → agents that remember AND stay in role")
    print("=" * 60)


if __name__ == "__main__":
    main()
