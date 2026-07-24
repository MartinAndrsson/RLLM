"""Core interfaces for RLLM — the general LLM-driven RL experimentation harness.

A problem is orchestrated purely through a `ProblemAdapter`. RLLM (dispatcher, registry, ladder,
reporting, and later the LLM controller) knows nothing problem-specific; all specifics live behind
this interface, in a small adapter generated into the problem repo. See DESIGN.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Fidelity:
    """One rung of the coarse-to-fine ladder. `overrides` make a run cheaper (screen) or
    costlier/full (confirm); `seeds` is how many training seeds to run at this rung."""
    name: str                       # "screen" | "refine" | "confirm"
    overrides: dict[str, Any]       # knob overrides applied on top of an experiment's config
    seeds: list[int]                # training seeds to run at this rung

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentSpec:
    """One experiment = a named hypothesis + a config (knob values). Runs at one fidelity at a time;
    promoted up the ladder by the search driver."""
    id: str
    hypothesis: str
    config: dict[str, Any]          # knob values (base config + this experiment's overrides)
    fidelity: str = "screen"
    parent: str | None = None       # lineage: which experiment this was promoted/derived from

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """Result of one (experiment, seed) run, in problem-agnostic form."""
    exp_id: str
    seed: int
    fidelity: str
    run_dir: str
    status: str                     # "done" | "failed" | "running"
    metrics: dict[str, Any] = field(default_factory=dict)   # standardized, from adapter.parse_result
    wall_seconds: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class ProblemAdapter(ABC):
    """Everything RLLM needs to know about a problem. Implemented per-problem (usually LLM-generated
    into the problem repo, then reviewer-checked)."""

    name: str                       # short problem id, e.g. "rx"
    success_metric: str             # key in metrics used to rank (see success_is_better)
    success_goal: str               # human-readable "solved" definition (from problem.md)

    @abstractmethod
    def fidelity_levels(self) -> list[Fidelity]:
        """Ordered cheap -> expensive rungs of the ladder."""

    @abstractmethod
    def build_command(self, spec: ExperimentSpec, seed: int, run_dir: str, device: str) -> list[str]:
        """The shell command (argv list) that runs ONE (experiment, seed) training run and writes
        its output under run_dir. RLLM executes this; the adapter owns all problem specifics."""

    @abstractmethod
    def parse_result(self, run_dir: str) -> dict[str, Any]:
        """Read a finished run_dir into a standardized metrics dict (must include success_metric)."""

    @abstractmethod
    def is_solved(self, metrics: dict[str, Any]) -> bool:
        """Has this run met the problem's 'solved' bar? (adapter encodes problem.md's definition)."""

    # Ranking: default lower-is-better=False; adapters override if their metric is minimised.
    def better(self, a: dict, b: dict) -> bool:
        """True if metrics `a` rank strictly better than `b` on success_metric."""
        return a.get(self.success_metric, float("-inf")) > b.get(self.success_metric, float("-inf"))
