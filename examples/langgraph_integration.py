"""
eachmind + LangGraph Integration Example

Demonstrates how eachmind provides per-agent memory alongside LangGraph.
Each graph node gets its own memory and perspective through eachmind,
while LangGraph handles state management and graph execution.

Usage: python examples/langgraph_integration.py
Requires: pip install langgraph
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eachmind import Agent as MindAgent
from eachmind import MemoryEvent, SharedMemory
from eachmind.primitives.drift import Drift


class WorkflowState(TypedDict):
    input: str
    research: str
    analysis: str
    report: str


# ---------------------------------------------------------------------------
# Create one eachmind agent per graph node — each has its own perspective
# ---------------------------------------------------------------------------

researcher = MindAgent(name="researcher", role="information gathering")
researcher.perspective.encoding_weights = {
    "factual_data": 0.9,
    "source_reliability": 0.85,
    "novelty": 0.7,
    "sentiment": 0.2,
}
researcher.perspective.priors = {
    "relevant_sources": ["academic", "industry_report"],
    "source_trust": {"academic": 0.95, "news": 0.55},
}

analyst = MindAgent(name="analyst", role="data analysis and synthesis")
analyst.perspective.encoding_weights = {
    "numerical_data": 0.95,
    "trends": 0.85,
    "causal_links": 0.8,
    "sentiment": 0.3,
}
analyst.perspective.priors = {
    "relevant_sources": ["quarterly_report", "industry_report"],
    "source_trust": {"quarterly_report": 0.9, "academic": 0.8},
}

writer = MindAgent(name="writer", role="report authoring")
writer.perspective.encoding_weights = {
    "narrative": 0.95,
    "sentiment": 0.8,
    "audience_fit": 0.85,
    "numerical_data": 0.4,
}
writer.perspective.priors = {
    "relevant_sources": ["analyst_brief", "executive_summary"],
    "source_trust": {"analyst_brief": 0.9, "academic": 0.6},
}

# Connect all agents to a shared memory space
shared = SharedMemory(team_id="langgraph-workflow")
researcher.connect(shared)
analyst.connect(shared)
writer.connect(shared)


# ---------------------------------------------------------------------------
# LangGraph nodes — each node is a plain Python function that:
#   1. Updates the LangGraph WorkflowState
#   2. Has its eachmind agent observe the relevant state via MemoryEvent
# ---------------------------------------------------------------------------

def research_node(state: WorkflowState) -> WorkflowState:
    """Gather information on the input topic (simulated)."""
    topic = state["input"]

    # Simulated research output — no API keys needed
    research_output = (
        f"Research on '{topic}': Market size estimated at $4.2B with 18% CAGR. "
        "Three major incumbents hold 60% share. Regulatory environment stabilising "
        "post-2024 framework update. Academic consensus supports adoption curve "
        "aligning with typical S-curve diffusion."
    )

    # eachmind: researcher observes both the task and its own output
    researcher.observe(MemoryEvent(content=f"Tasked to research: {topic}", source="workflow"))
    enc = researcher.observe(
        MemoryEvent(content=research_output, source="academic")
    )

    # Share key finding with the team
    researcher.share(
        content="Market size $4.2B, CAGR 18% — high-confidence from academic sources.",
        to=SharedMemory.TEAM,
        reason="Critical baseline metric for downstream analysis",
    )

    print(
        f"[research_node]  salience={enc.salience:.2f}"
        f"  memories={researcher.private_memory.size}"
    )
    return {**state, "research": research_output}


def analysis_node(state: WorkflowState) -> WorkflowState:
    """Analyse the research findings (simulated)."""
    research = state["research"]

    # Simulated analysis
    analysis_output = (
        "Analysis: 18% CAGR with incumbent concentration creates a classic "
        "fast-follower opportunity. Unit economics improve non-linearly beyond $500M ARR. "
        "Regulatory clarity reduces entry barrier — expect 3–5 new entrants in 24 months. "
        "Recommended focus: mid-market segment where incumbents are weakest."
    )

    # eachmind: analyst observes the incoming research and its own analysis
    enc_in = analyst.observe(
        MemoryEvent(content=research, source="quarterly_report")
    )
    enc_out = analyst.observe(
        MemoryEvent(content=analysis_output, source="quarterly_report")
    )

    # Retrieve shared knowledge from researcher before sharing own finding
    team_knowledge = analyst.recall(source=SharedMemory.TEAM)
    for entry in team_knowledge:
        analyst.observe(
            MemoryEvent(content=entry.content, source="analyst_brief")
        )

    analyst.share(
        content="Mid-market segment identified as primary opportunity — low incumbent coverage.",
        to=SharedMemory.TEAM,
        reason="Strategic recommendation for report framing",
    )

    avg_salience = (enc_in.salience + enc_out.salience) / 2
    print(f"[analysis_node]  salience={avg_salience:.2f}  memories={analyst.private_memory.size}")
    return {**state, "analysis": analysis_output}


def report_node(state: WorkflowState) -> WorkflowState:
    """Author the final report (simulated)."""
    analysis = state["analysis"]

    # Simulated report
    report_output = (
        "Executive Report: The market presents a compelling entry window driven by "
        "regulatory clarity and a 18% CAGR tailwind. Incumbent weakness in the mid-market "
        "creates differentiated positioning. We recommend a phased go-to-market targeting "
        "Series-B SaaS companies, with full expansion in 18 months pending"
        " product-market fit signals."
    )

    # eachmind: writer observes the analysis and team knowledge before writing
    team_knowledge = writer.recall(source=SharedMemory.TEAM)
    for entry in team_knowledge:
        writer.observe(
            MemoryEvent(content=entry.content, source="analyst_brief")
        )

    enc_analysis = writer.observe(
        MemoryEvent(content=analysis, source="analyst_brief")
    )
    enc_report = writer.observe(
        MemoryEvent(content=report_output, source="executive_summary")
    )

    avg_salience = (enc_analysis.salience + enc_report.salience) / 2
    print(f"[report_node]    salience={avg_salience:.2f}  memories={writer.private_memory.size}")
    return {**state, "report": report_output}


# ---------------------------------------------------------------------------
# Build the LangGraph state graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("research", research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("eachmind + LangGraph Integration Demo")
    print("=" * 60)
    print()

    app = build_graph()

    initial_state: WorkflowState = {
        "input": "AI infrastructure investment landscape 2025",
        "research": "",
        "analysis": "",
        "report": "",
    }

    print("--- Graph execution (node salience + memory growth) ---")
    final_state = app.invoke(initial_state)
    print()

    # --- Show how the same shared knowledge was encoded differently ---
    print("=== Same state data, different per-node encodings ===")
    shared_entries = shared.recall(agent_id="analyst", limit=10)
    print(f"Shared memory entries visible to all agents: {len(shared_entries)}")
    for entry in shared_entries:
        print(f"  [{entry.shared_by}] {entry.content[:80]}...")
    print()

    # --- Private memory sizes reflect each node's perspective breadth ---
    print("=== Per-node private memory (perspective-filtered) ===")
    print(f"  researcher : {researcher.private_memory.size} memories stored")
    print(f"  analyst    : {analyst.private_memory.size} memories stored")
    print(f"  writer     : {writer.private_memory.size} memories stored")
    print()

    # --- Measure perspective drift between nodes ---
    drift_tracker = Drift()

    drift_ra = drift_tracker.measure(
        "researcher", researcher.perspective,
        "analyst", analyst.perspective,
    )
    drift_aw = drift_tracker.measure(
        "analyst", analyst.perspective,
        "writer", writer.perspective,
    )
    drift_rw = drift_tracker.measure(
        "researcher", researcher.perspective,
        "writer", writer.perspective,
    )

    print("=== Perspective drift between nodes (after graph run) ===")
    print(f"  researcher <-> analyst : {drift_ra.drift_value:.3f}")
    print(f"  analyst    <-> writer  : {drift_aw.drift_value:.3f}")
    print(f"  researcher <-> writer  : {drift_rw.drift_value:.3f}")
    print()
    print("  (Higher drift = more divergent encoding of shared state)")
    print()

    # --- Consolidate each agent's memories into beliefs ---
    researcher.consolidate()
    analyst.consolidate()
    writer.consolidate()

    print("=== Beliefs formed per node after graph completion ===")
    for agent in [researcher, analyst, writer]:
        beliefs = agent.consolidation.beliefs
        print(f"  {agent.name} ({len(beliefs)} beliefs):")
        for b in beliefs[:2]:  # show top 2 per agent
            print(f"    [{b.confidence:.2f}] {b.content}")
    print()

    print("=== Final report excerpt ===")
    print(final_state["report"][:200] + "...")
    print()
    print("Done. LangGraph managed state flow; eachmind gave each node")
    print("its own memory, salience encoding, and belief formation.")


if __name__ == "__main__":
    main()
