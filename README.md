# eachmind

**Per-agent memory protocol for multi-agent systems**

---

## Problem Statement

Current agent swarms share a single memory system. Every agent draws from the same pool of context, producing the same perspective. This creates the illusion of collaboration — agents divide tasks but never truly think differently from each other. Shared memory = shared perspective = no real team cognition.

The opposite — fully isolated agents — breaks collaboration entirely. Agents cannot learn from each other, cannot build on shared experience, and cannot develop institutional knowledge over time.

**eachmind** solves this by giving each agent its own memory — privately encoded, individually shaped — while defining a protocol for what gets selectively shared, when, and how. Agents develop genuine perspectives. Teams develop genuine collective intelligence.

## What It Is

A standalone, framework-agnostic Python library that defines how memory is stored, differentiated, and selectively shared across agents in a multi-agent system. It is a memory protocol — not an agent framework, not a task runner. Any agent system can adopt it.

## Core Primitives

| Primitive | Description |
|---|---|
| **PrivateMemory** | Each agent's own store. Encoded from its perspective. Never automatically shared. |
| **SharedMemory** | What agents explicitly publish to the collective. Opt-in, not default. |
| **MemoryEvent** | A discrete experience. Same event, encoded differently per agent based on its context. |
| **Perspective** | The lens through which an agent encodes events — shaped by its history and role. |
| **Consolidation** | How repeated private experiences abstract into durable beliefs over time. |
| **Drift** | Agents in the same team naturally diverge in perspective over time. Measurable. |

## Design Principles

### Private by default
Memory is private unless explicitly shared. Sharing is a deliberate act, not an automatic sync.

### Same event, different encoding
When agents observe the same event, each encodes it through its own perspective. Divergence is the feature, not the bug.

### Framework agnostic
Works alongside OpenAI Swarm, CrewAI, LangGraph, or a hand-written agent loop. No lock-in.

### Institutional memory emerges
Over time, what agents repeatedly share consolidates into team-level knowledge — without forcing a single shared brain.

## Installation

```bash
pip install eachmind
```

## Quick Start

```python
from eachmind import Agent, MemoryEvent, SharedMemory

# Create agents with their own private memory
analyst = Agent(name="analyst", role="data analysis")
writer = Agent(name="writer", role="content creation")

# Both agents observe the same event
event = MemoryEvent(
    content="Q1 revenue grew 23% YoY",
    source="quarterly_report",
    timestamp="2026-04-10T09:00:00Z"
)

# Each encodes it through their own perspective
analyst.observe(event)  # Encodes: statistical significance, trend implications
writer.observe(event)   # Encodes: narrative angle, audience framing

# Analyst decides to share a finding
analyst.share(
    content="Revenue growth acceleration suggests market expansion",
    to=SharedMemory.TEAM
)

# Writer can now access shared knowledge
shared = writer.recall(source=SharedMemory.TEAM)

# Over time, perspectives naturally drift — and that's measurable
drift = analyst.perspective.drift_from(writer.perspective)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Agent Team                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Agent A   │  │ Agent B   │  │ Agent C   │      │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│      │
│  ││Private ││  ││Private ││  ││Private ││      │
│  ││Memory  ││  ││Memory  ││  ││Memory  ││      │
│  │└────────┘│  │└────────┘│  │└────────┘│      │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│      │
│  ││Perspect││  ││Perspect││  ││Perspect││      │
│  ││ive     ││  ││ive     ││  ││ive     ││      │
│  │└────────┘│  │└────────┘│  │└────────┘│      │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘      │
│        │  share()     │  share()    │            │
│        ▼              ▼             ▼            │
│  ┌──────────────────────────────────────┐       │
│  │          Shared Memory (opt-in)       │       │
│  │  ┌─────────────────────────────────┐ │       │
│  │  │    Consolidated Team Knowledge   │ │       │
│  │  └─────────────────────────────────┘ │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## What It Is NOT

- **Not a vector database or RAG system** — eachmind defines memory *behavior*, not storage engines.
- **Not an agent framework or task runner** — it doesn't orchestrate agents or assign tasks.
- **Not a replacement for mem0, Zep, or MemGPT** — those are memory backends; eachmind is a protocol layer above them.
- **Not a multi-agent orchestrator** — it doesn't manage agent coordination or communication routing.

## Project Status

> **Foundation complete.** Project 1 of 2. All core primitives, protocol specification, storage backends, visualizations, and integration examples are implemented. Project 2 is an agent architecture built on top of eachmind that demonstrates genuine team cognition — agents that challenge, review, disagree, and accumulate institutional knowledge over time.

## Roadmap

- [x] Core primitives implementation
- [x] Protocol specification
- [x] Storage backend adapters (in-memory, SQLite, Redis)
- [x] Integration examples (OpenAI Agents SDK, CrewAI, LangGraph)
- [x] Drift measurement and visualization
- [x] Consolidation algorithms
- [ ] Project 2: Team cognition demo

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
