"""
eachmind + CrewAI Integration Example

Demonstrates how eachmind provides per-agent memory alongside CrewAI.
Each crew member gets its own memory, perspective, and belief system
through eachmind, while CrewAI handles task orchestration.

Even though every crew member sees the same task outputs, eachmind
ensures each agent encodes them differently — preserving the role
specialisation that CrewAI defines at the instruction level.

Usage: python examples/crewai_integration.py
Requires: pip install crewai

Note: This example simulates crew task outputs so no LLM API key is needed.
      In production you would call crew.kickoff() and feed the resulting
      TaskOutput strings into eachmind as MemoryEvents.
"""

import os

# Allow CrewAI to instantiate agents without a real API key.
# In production this line is not needed — your key is already set.
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-key-for-demo")

from crewai import Agent as CrewAgent
from crewai import Crew, Task

from eachmind import Agent as MindAgent
from eachmind import MemoryEvent, SharedMemory
from eachmind.primitives.drift import Drift

# ---------------------------------------------------------------------------
# Simulated crew task outputs — no API calls needed
# ---------------------------------------------------------------------------

RESEARCHER_OUTPUTS = [
    MemoryEvent(
        content=(
            "Market analysis: AI agent frameworks grew 210% YoY;"
            " enterprises demand memory isolation per role"
        ),
        source="web_search",
    ),
    MemoryEvent(
        content=(
            "Survey of 150 engineering teams: 74% report agent 'role bleed'"
            " as a top pain point in multi-agent systems"
        ),
        source="survey_data",
    ),
    MemoryEvent(
        content=(
            "Three leading frameworks (LangGraph, AutoGen, CrewAI) offer shared context"
            " pools; none provide per-agent private memory"
        ),
        source="competitive_review",
    ),
]

ANALYST_OUTPUTS = [
    MemoryEvent(
        content=(
            "Role bleed stems from shared context: when agents read each other's"
            " scratchpad, they drift toward median behaviour"
        ),
        source="technical_analysis",
    ),
    MemoryEvent(
        content=(
            "Per-agent memory isolation reduces inter-agent confusion by an estimated"
            " 40% based on benchmark simulations"
        ),
        source="benchmark_results",
    ),
]

WRITER_OUTPUTS = [
    MemoryEvent(
        content=(
            "Draft intro: 'CrewAI orchestrates who does what; eachmind ensures each"
            " agent remembers it differently'"
        ),
        source="draft_copy",
    ),
    MemoryEvent(
        content=(
            "Target readers are senior engineers evaluating multi-agent stacks"
            " for production use"
        ),
        source="editorial_brief",
    ),
]

ANALYST_SHARED_INSIGHT = (
    "Role bleed + 74% pain-point survey + 40% isolation improvement = "
    "clear product-market fit for per-agent perspective-aware memory"
)


def main():
    print("=" * 60)
    print("eachmind + CrewAI Integration Demo")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # 1. Define CrewAI agents (orchestration layer)
    #    CrewAI owns roles, goals, backstories, and task routing.
    #    In production you call crew.kickoff() to run tasks with a real LLM.
    # -----------------------------------------------------------------------
    crew_researcher = CrewAgent(
        role="Market Researcher",
        goal="Gather data on AI agent frameworks and identify market gaps",
        backstory=(
            "You are a seasoned market researcher specialising in developer tooling. "
            "You surface hard numbers and cite your sources."
        ),
    )

    crew_analyst = CrewAgent(
        role="Data Analyst",
        goal="Interpret research findings and produce strategic insights",
        backstory=(
            "You are a rigorous data analyst. You find patterns in qualitative and "
            "quantitative research and translate them into actionable conclusions."
        ),
    )

    crew_writer = CrewAgent(
        role="Technical Writer",
        goal="Transform insights into clear narratives for a developer audience",
        backstory=(
            "You are a technical writer with a background in software engineering. "
            "You make complex ideas accessible without sacrificing precision."
        ),
    )

    task_research = Task(
        description=(
            "Research the current landscape of multi-agent AI frameworks and"
            " identify memory-related pain points."
        ),
        expected_output=(
            "A structured summary of market data, competitor gaps, and developer"
            " pain points."
        ),
        agent=crew_researcher,
    )
    task_analysis = Task(
        description=(
            "Analyse the research findings and derive strategic insights about"
            " per-agent memory isolation."
        ),
        expected_output="A concise analysis with quantified impact estimates.",
        agent=crew_analyst,
    )
    task_writing = Task(
        description=(
            "Write a compelling introduction for a developer-facing product page"
            " about per-agent memory."
        ),
        expected_output="A two-paragraph introduction draft.",
        agent=crew_writer,
    )

    crew = Crew(
        agents=[crew_researcher, crew_analyst, crew_writer],
        tasks=[task_research, task_analysis, task_writing],
    )

    print("--- CrewAI crew defined (orchestration layer) ---")
    for agent in crew.agents:
        print(f"  Crew agent: {agent.role!r}")
    print(f"  Tasks:      {len(crew.tasks)} tasks queued")
    print()
    print("  (In production: crew.kickoff() drives LLM calls.)")
    print("  (Here: task outputs are simulated — no API key needed.)")
    print()

    # -----------------------------------------------------------------------
    # 2. Create paired eachmind agents — one per crew member
    #    Each carries its own perspective encoding and private memory store.
    # -----------------------------------------------------------------------
    mind_researcher = MindAgent(name="researcher", role="market research")
    mind_researcher.perspective.encoding_weights = {
        "web_search":         0.95,
        "survey_data":        0.90,
        "competitive_review": 0.85,
        "technical_analysis": 0.40,
        "benchmark_results":  0.30,
        "draft_copy":         0.10,
        "editorial_brief":    0.10,
    }
    mind_researcher.perspective.priors = {
        "relevant_sources": ["web_search", "survey_data", "competitive_review"],
        "source_trust": {"web_search": 0.85, "survey_data": 0.90, "competitive_review": 0.80},
    }

    mind_analyst = MindAgent(name="analyst", role="data analysis")
    mind_analyst.perspective.encoding_weights = {
        "benchmark_results":  0.95,
        "technical_analysis": 0.90,
        "survey_data":        0.75,
        "competitive_review": 0.70,
        "web_search":         0.50,
        "draft_copy":         0.10,
        "editorial_brief":    0.05,
    }
    mind_analyst.perspective.priors = {
        "relevant_sources": ["benchmark_results", "technical_analysis"],
        "source_trust": {"benchmark_results": 0.95, "technical_analysis": 0.90},
    }

    mind_writer = MindAgent(name="writer", role="content creation")
    mind_writer.perspective.encoding_weights = {
        "editorial_brief":    0.95,
        "draft_copy":         0.90,
        "survey_data":        0.60,
        "web_search":         0.35,
        "technical_analysis": 0.20,
        "competitive_review": 0.15,
        "benchmark_results":  0.10,
    }
    mind_writer.perspective.priors = {
        "relevant_sources": ["editorial_brief", "draft_copy"],
        "source_trust": {"editorial_brief": 0.95, "draft_copy": 0.85},
    }

    print("--- eachmind agents created with role-specific perspectives ---")
    for mind in (mind_researcher, mind_analyst, mind_writer):
        print(f"  Mind agent: {mind.name!r}  role: {mind.role!r}")
    print()

    # -----------------------------------------------------------------------
    # 3. Connect all agents to a shared team memory
    # -----------------------------------------------------------------------
    shared = SharedMemory(team_id="crewai-demo-team")
    mind_researcher.connect(shared)
    mind_analyst.connect(shared)
    mind_writer.connect(shared)

    # -----------------------------------------------------------------------
    # 4. Step 1 — Researcher task output observed by all agents
    #    Each encodes the same findings through its own perspective lens.
    # -----------------------------------------------------------------------
    print("=== Step 1: Researcher task completes — all agents observe output ===")
    print()

    all_minds = (mind_researcher, mind_analyst, mind_writer)

    for event in RESEARCHER_OUTPUTS:
        enc_r = mind_researcher.observe(event)
        enc_a = mind_analyst.observe(event)
        enc_w = mind_writer.observe(event)

        print(f"  Finding: \"{event.content[:65]}...\"")
        print(f"    Researcher salience: {enc_r.salience:.2f}  (source: {event.source!r})")
        print(f"    Analyst salience:    {enc_a.salience:.2f}")
        print(f"    Writer salience:     {enc_w.salience:.2f}")
        print()

    # -----------------------------------------------------------------------
    # 5. Step 2 — Analyst task output observed by all agents
    # -----------------------------------------------------------------------
    print("=== Step 2: Analyst task completes — all agents observe output ===")
    print()

    for event in ANALYST_OUTPUTS:
        enc_r = mind_researcher.observe(event)
        enc_a = mind_analyst.observe(event)
        enc_w = mind_writer.observe(event)

        print(f"  Analysis: \"{event.content[:65]}...\"")
        print(f"    Researcher salience: {enc_r.salience:.2f}")
        print(f"    Analyst salience:    {enc_a.salience:.2f}  (source: {event.source!r})")
        print(f"    Writer salience:     {enc_w.salience:.2f}")
        print()

    # -----------------------------------------------------------------------
    # 6. Analyst shares strategic insight to shared memory
    # -----------------------------------------------------------------------
    print("=== Step 3: Analyst shares key insight to shared team memory ===")
    print()

    mind_analyst.share(
        content=ANALYST_SHARED_INSIGHT,
        to=SharedMemory.TEAM,
        reason="Strategic synthesis ready for writer to incorporate into draft",
    )
    print(f"  Shared: \"{ANALYST_SHARED_INSIGHT[:70]}...\"")
    print()

    # -----------------------------------------------------------------------
    # 7. Step 3 — Writer task output; writer recalls shared insights
    # -----------------------------------------------------------------------
    print("=== Step 4: Writer task completes — recalls shared insights ===")
    print()

    for event in WRITER_OUTPUTS:
        mind_writer.observe(event)

    shared_knowledge = mind_writer.recall(source=SharedMemory.TEAM)
    print("  Shared knowledge accessible to writer:")
    for entry in shared_knowledge:
        print(f"    From {entry.shared_by!r}: \"{entry.content[:70]}...\"")
    print()

    # -----------------------------------------------------------------------
    # 8. Measure perspective drift between crew members
    # -----------------------------------------------------------------------
    print("=== Step 5: Measure perspective drift between crew members ===")
    print()

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
    print("  Higher drift = more differentiated perspectives.")
    print("  Each agent has encoded the same crew outputs differently.")
    print()

    # -----------------------------------------------------------------------
    # 9. Consolidate each agent's memories into beliefs
    # -----------------------------------------------------------------------
    print("=== Step 6: Consolidate memories into per-agent beliefs ===")
    print()

    for mind in all_minds:
        mind.consolidate()
        print(f"  {mind.name.capitalize()} beliefs:")
        for belief in mind.consolidation.beliefs:
            print(f"    [{belief.confidence:.2f}] {belief.content}")
    print()

    # -----------------------------------------------------------------------
    # 10. Memory stats summary
    # -----------------------------------------------------------------------
    print("=== Memory stats ===")
    print(f"  Researcher private memories: {mind_researcher.private_memory.size}")
    print(f"  Analyst    private memories: {mind_analyst.private_memory.size}")
    print(f"  Writer     private memories: {mind_writer.private_memory.size}")
    print(f"  Shared memory entries:       {shared.size}")
    print()

    print("=" * 60)
    print("Key takeaway:")
    print("  CrewAI   → task orchestration, role definitions, LLM routing")
    print("  eachmind → per-agent memory, perspective encoding, drift")
    print("  Together → crew members that remember AND stay in role")
    print("=" * 60)


if __name__ == "__main__":
    main()
