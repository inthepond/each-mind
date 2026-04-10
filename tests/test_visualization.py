"""Smoke tests for drift visualization (matplotlib)."""
import pytest

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from eachmind import Agent, MemoryEvent
from eachmind.primitives.drift import Drift


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestDriftVisualization:
    def _build_scenario(self):
        agents = {
            "analyst": Agent(name="analyst", role="data analysis"),
            "writer": Agent(name="writer", role="content creation"),
            "reviewer": Agent(name="reviewer", role="code review"),
        }
        agents["analyst"].perspective.encoding_weights = {"numerical_data": 0.9, "trends": 0.8}
        agents["writer"].perspective.encoding_weights = {"narrative": 0.9, "sentiment": 0.8}
        agents["reviewer"].perspective.encoding_weights = {"correctness": 0.9, "clarity": 0.7}

        events = [
            MemoryEvent(content=f"data point {i}", source="report")
            for i in range(5)
        ]
        for event in events:
            for agent in agents.values():
                agent.observe(event)

        drift_tracker = Drift()
        perspectives = {name: a.perspective for name, a in agents.items()}
        # Take multiple measurements to build timeline
        for _ in range(3):
            drift_tracker.team_diversity(perspectives)
            # Observe more events to evolve perspectives
            for agent in agents.values():
                agent.observe(MemoryEvent(content="evolving data", source="stream"))

        return drift_tracker, agents

    def test_heatmap_returns_figure(self):
        from eachmind.visualization import plot_drift_heatmap
        _, agents = self._build_scenario()
        perspectives = {name: a.perspective for name, a in agents.items()}
        fig = plot_drift_heatmap(perspectives)
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_timeline_returns_figure(self):
        from eachmind.visualization import plot_drift_timeline
        drift_tracker, _ = self._build_scenario()
        fig = plot_drift_timeline(drift_tracker)
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_diversity_returns_figure(self):
        from eachmind.visualization import plot_team_diversity
        _, agents = self._build_scenario()
        perspectives = {name: a.perspective for name, a in agents.items()}
        fig = plot_team_diversity(perspectives)
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_heatmap_single_agent(self):
        """Heatmap with one agent should not crash."""
        from eachmind.visualization import plot_drift_heatmap
        agent = Agent(name="solo", role="general")
        fig = plot_drift_heatmap({"solo": agent.perspective})
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_timeline_empty_tracker(self):
        """Timeline with no measurements should produce a figure."""
        from eachmind.visualization import plot_drift_timeline
        fig = plot_drift_timeline(Drift())
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_diversity_two_agents(self):
        """Diversity with exactly two agents should work."""
        from eachmind.visualization import plot_team_diversity
        a = Agent(name="alpha", role="analysis")
        b = Agent(name="beta", role="writing")
        a.perspective.encoding_weights = {"data": 0.9}
        b.perspective.encoding_weights = {"prose": 0.9}
        for i in range(3):
            ev = MemoryEvent(content=f"event {i}", source="test")
            a.observe(ev)
            b.observe(ev)
        fig = plot_team_diversity({"alpha": a.perspective, "beta": b.perspective})
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)
