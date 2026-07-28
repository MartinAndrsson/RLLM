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
    cost_multiplier: float = 1.0    # cost of ONE run here vs one run at the cheapest rung; used to
                                    # estimate durations before any run of this rung has been observed

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class KnobSpec:
    """Declaration of ONE configurable knob (implementations.md §3.3). The adapter's knob table is the
    whitelist: anything an LLM proposes that isn't declared here is rejected deterministically, before
    any command is built. Types and ranges are checked; `llm_may_change=False` knobs are harness-only;
    `category="task_definition"` knobs change what "solved" means and need explicit human permission
    even when both models agree."""
    name: str
    value_type: str                 # "int" | "float" | "bool" | "string" | "enum"
    default: Any
    category: str                   # algorithm|hyperparameter|reward|observation|action|
                                    # training_budget|evaluation|task_definition
    minimum: float | None = None
    maximum: float | None = None
    choices: list[Any] | None = None
    fidelity_safe: bool = False     # may a fidelity rung override it? (must not change realism)
    llm_may_change: bool = True     # may the actor propose it?

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

    def base_config(self) -> dict[str, Any]:
        """Knob values applied to EVERY run before an experiment's own config (the current base recipe).
        Shown to the actor so it doesn't propose a knob value the base already uses (a no-op run)."""
        return {}

    def declared_knobs(self) -> dict[str, "KnobSpec"]:
        """The knob whitelist, keyed by knob name (see KnobSpec). Default {} = no knob may be set by a
        proposal, which is the safe default: an adapter must opt in explicitly to LLM-tunable knobs."""
        return {}

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

    def sort_key(self, metrics: dict[str, Any]) -> tuple:
        """Ranking key, higher = better (the core sorts on this and knows nothing else about metrics).
        Default: the success metric alone, maximised; adapters override to minimise or to add
        tie-breakers. A missing/non-numeric metric must rank last, never best."""
        v = metrics.get(self.success_metric)
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool) else float("-inf"),)

    def better(self, a: dict, b: dict) -> bool:
        """True if metrics `a` rank strictly better than `b`."""
        return self.sort_key(a) > self.sort_key(b)
