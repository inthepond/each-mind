"""Team-cognition demo: watch a cast of characters' perspectives diverge.

This is the roadmap's "Project 2" demo. It drives a small
`sandbox-anything <https://github.com/inthepond/sandbox-anything>`_ world —
a crew talking across a few overlapping channels — and uses each-mind to
measure how far each character's *perspective* drifts from the others as the
simulation runs. The payoff is visual: a heatmap of who has diverged from
whom, and a timeline of that divergence growing over sandbox time.

The integration point is ``sandbox_anything.memory.drift_bridge.DriftBridge``,
a ``MemoryEncoder`` that rides alongside sandbox-anything's own memory layer.
sandbox keeps writing its per-character memories exactly as before; the bridge
additionally feeds each observation into a per-character each-mind
``Perspective`` and a shared ``Drift`` tracker.

Run it::

    pip install -e ./each-mind
    pip install -e './sandbox-anything[eachmind]'
    python each-mind/examples/sandbox_drift_demo.py --ticks 200 --out ./drift_demo_out

It is free and fully deterministic. Rather than sandbox's ``random`` stub —
which gives every character the *same* canned phrases, so nobody truly
diverges — it uses a tiny "flavoured" generator that makes each character
speak from their own vocabulary. That stands in for what an LLM would
produce (characters talking about their own concerns); swap in a real LLM
generator + ``LLMMemoryEncoder`` and the drift machinery is identical.
"""

from __future__ import annotations

import argparse
import copy
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# sandbox-anything: the world engine.
from sandbox_anything.core import Sandbox
from sandbox_anything.entity import Entity
from sandbox_anything.generation.action import Action, ActionContext, NoAction, SpeakAction
from sandbox_anything.memory.drift_bridge import DriftBridge
from sandbox_anything.scheduling import Scheduler, SchedulerConfig
from sandbox_anything.timesys import clock as clock_module

# each-mind: the perspective/drift layer + charts.
from eachmind.visualization import (
    plot_drift_heatmap,
    plot_drift_timeline,
    plot_team_diversity,
)

# --------------------------------------------------------------------------- #
# Cast and channels — overlapping cliques so drift has structure to find.
# --------------------------------------------------------------------------- #

# (name, role, traits) — role/traits seed each character's encoding weights.
CAST = [
    ("Luffy", "captain", "reckless brave hungry"),
    ("Zoro", "swordsman", "stoic loyal fighter"),
    ("Nami", "navigator", "clever cautious money"),
    ("Sanji", "cook", "romantic fiery galley"),
    ("Usopp", "sniper", "anxious creative storyteller"),
    ("Robin", "archaeologist", "calm curious history"),
]

# Channels with partial overlap. Nobody is in every channel, so characters
# hear different sets of speakers — which is what makes perspectives diverge.
# Luffy / Nami / Usopp sit in two channels each (hubs); Zoro / Sanji / Robin
# in one each (leaves) — so the leaves should end up most diverged.
CHANNELS = {
    "galley": ["Sanji", "Nami", "Luffy"],
    "deck": ["Zoro", "Luffy", "Usopp"],
    "nav-room": ["Nami", "Usopp", "Robin"],
}

# Each character speaks from their own vocabulary — a stand-in for an LLM
# voicing their distinct concerns. Words are 5+ letters (so the bridge's
# topic extractor picks them up) and disjoint per character (so listeners in
# different channels accumulate genuinely different priors).
VOCAB = {
    "Luffy": ["treasure", "adventure", "freedom", "captain", "journey", "crewmate"],
    "Zoro": ["katana", "training", "discipline", "swords", "sparring", "honor"],
    "Nami": ["charts", "weather", "navigation", "berries", "climate", "compass"],
    "Sanji": ["recipe", "kitchen", "cooking", "dinner", "ingredient", "flavour"],
    "Usopp": ["pellets", "sniping", "inventions", "slingshot", "stories", "tinkering"],
    "Robin": ["history", "ruins", "poneglyph", "ancient", "scholar", "mystery"],
}


@dataclass
class FlavoredGenerator:
    """Deterministic generator where each character talks about their own things.

    A closer stand-in for an LLM than sandbox's ``random`` stub (which gives
    everyone identical canned lines). Swap an ``LLMActionGenerator`` in here
    and nothing else about the demo changes.
    """

    vocab: dict[str, list[str]]
    speak_probability: float = 0.7
    seed: int | None = None
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def generate(self, ctx: ActionContext) -> Action:
        if self._rng.random() >= self.speak_probability:
            return NoAction(reason="quiet this tick")
        words = self.vocab.get(ctx.actor.name, ["something"])
        word = self._rng.choice(words)
        return SpeakAction(content=f"{ctx.actor.name}: {word}")


def build_world(worlds_root: Path) -> tuple[Sandbox, DriftBridge]:
    """Create the sandbox, cast, channels, and a registered DriftBridge."""
    sb = Sandbox.create("Drift Demo", worlds_root=worlds_root)
    sb.start()

    by_name: dict[str, Entity] = {}
    for name, role, traits in CAST:
        ent = sb.add(
            Entity(
                name=name,
                type="character",
                is_actor=True,
                attributes={"role": role, "traits": traits},
            )
        )
        by_name[name] = ent

    for ch_name, members in CHANNELS.items():
        sb.create_channel(ch_name, members=[by_name[m].id for m in members])

    bridge = DriftBridge()
    for ent in by_name.values():
        bridge.register(ent)

    return sb, bridge


def run(
    ticks: int, snapshot_every: int, seed: int, out_dir: Path, merge_at: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic time: control the monotonic clock the sandbox reads, so
    # we can step exactly one tick per loop without real-time sleeps. This is
    # the same hook sandbox-anything's own scheduler tests use.
    fake = {"now": 0.0}
    clock_module._now_monotonic = lambda: fake["now"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as tmp:
        sb, bridge = build_world(Path(tmp))
        per_tick = sb.clock.real_seconds_per_tick

        scheduler = Scheduler(
            sandbox=sb,
            channel_store=sb.channels,
            generator=FlavoredGenerator(VOCAB, speak_probability=0.7, seed=seed),
            config=SchedulerConfig(
                channel_activity_probability=1.0,
                max_opportunities_per_channel=2,
                memory_decay_every_n_ticks=0,  # keep the demo deterministic
            ),
            seed=seed,
            memory_store=sb.memories,
            memory_encoder=bridge,
        )

        # Name-keyed view of the perspectives so charts/labels read nicely.
        def named() -> dict:
            return {bridge.names[eid]: p for eid, p in bridge.perspectives.items()}

        print(
            f"Phase 1 (ticks 0-{merge_at - 1}): crew talks in {len(CHANNELS)} "
            "isolated cliques — perspectives differentiate.\n"
        )
        # Perspective state captured at peak divergence (end of phase 1), so the
        # heatmap shows the clique structure before the all-hands merge blurs it.
        peak: dict = {}
        for i in range(ticks):
            if i == merge_at:
                # Regime change: the whole crew starts sharing one channel.
                # Now everyone hears everyone, and perspectives reconverge.
                peak = copy.deepcopy(named())
                all_ids = [e.id for e in sb.entities(is_actor=True)]
                sb.create_channel("all-hands", members=all_ids)
                print(
                    f"\nPhase 2 (ticks {merge_at}-{ticks - 1}): an all-hands "
                    "channel opens — perspectives start to reconverge.\n"
                )
            fake["now"] += per_tick
            scheduler.tick()
            if i % snapshot_every == 0:
                bridge.drift.snapshot(named())
                diversity = bridge.drift.team_diversity(named())
                print(f"  tick {i:>4}: team diversity = {diversity:.3f}")

        # Final snapshot so the timeline ends at the last tick.
        bridge.drift.snapshot(named())

        final = named()
        if not peak:  # merge never happened (merge_at >= ticks)
            peak = final
        total_memories = sb.memories.count()

        _print_summary(peak, final, bridge.drift, total_memories)
        _save_charts(bridge.drift, peak, out_dir)
        sb.close()

    print(f"\nCharts written to {out_dir}/")


def _print_summary(peak: dict, final: dict, drift, memories: int) -> None:
    names = sorted(peak)
    pairs = [
        (a, b, peak[a].drift_from(peak[b]))
        for i, a in enumerate(names)
        for b in names[i + 1:]
    ]
    pairs.sort(key=lambda t: t[2], reverse=True)
    print(f"\n  {memories} memories formed.")
    print(f"  Peak divergence (end of phase 1): {drift.team_diversity(peak):.3f}")
    print(f"    most diverged:  {pairs[0][0]} vs {pairs[0][1]}  ({pairs[0][2]:.3f})")
    print(f"    most aligned:   {pairs[-1][0]} vs {pairs[-1][1]}  ({pairs[-1][2]:.3f})")
    print(f"  After all-hands reconvergence:    {drift.team_diversity(final):.3f}")


def _save_charts(drift, peak: dict, out_dir: Path) -> None:
    # Heatmap + diversity overview use the peak (phase-1) structure; the
    # timeline uses the full recorded history (differentiation then reconvergence).
    heatmap = plot_drift_heatmap(peak)
    heatmap.savefig(out_dir / "drift_heatmap.png", dpi=150, bbox_inches="tight")

    timeline = plot_drift_timeline(drift)
    timeline.savefig(out_dir / "drift_timeline.png", dpi=150, bbox_inches="tight")

    diversity = plot_team_diversity(peak)
    diversity.savefig(out_dir / "team_diversity.png", dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=120, help="sandbox ticks to run")
    parser.add_argument("--snapshot-every", type=int, default=3)
    parser.add_argument(
        "--merge-at",
        type=int,
        default=None,
        help="tick at which the all-hands channel opens (default: ticks // 2)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("drift_demo_out"))
    args = parser.parse_args()
    merge_at = args.merge_at if args.merge_at is not None else args.ticks // 2
    run(args.ticks, args.snapshot_every, args.seed, args.out, merge_at)


if __name__ == "__main__":
    main()
