"""ProblemBrief — the frozen contract for one problem, and the budget for one session.

This is what the onboarding interview (`scripts/rllm-onboard.sh`) writes and what every later command
reads. It answers, in a machine-checkable form, the questions a human must settle before any autonomous
compute is spent (implementations.md §3.1/§3.2/§10.1):

  * which repo, and what is the problem in one paragraph;
  * what "better" means (primary metric + direction) and what "solved" means (success criterion,
    including at which fidelity and over how many seeds it must hold);
  * what the DETERMINISTIC performance test is (the command, and where it writes its metrics) — the
    harness never asks a model whether something worked;
  * what must not change, because changing it would change the problem (realism constraints, and the
    explicit permitted/forbidden task-knob lists);
  * how long the models may explore before a human reviews, and the hard caps on runs/LLM calls.

The brief is FROZEN for a session: it is hashed on save, the hash goes into the handoff, and a model may
propose amendments only as text for a human to accept. Stdlib-only (JSON, dataclasses) on purpose — no
dependency needed to read a brief.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

OPERATORS = (">=", ">", "<=", "<", "==")
DIRECTIONS = ("maximize", "minimize")


@dataclass
class MetricConstraint:
    metric: str
    operator: str
    value: float

    def satisfied_by(self, metrics: dict) -> bool | None:
        """True/False, or None when the metric is absent or not finite (never guess a pass)."""
        v = metrics.get(self.metric)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:   # v != v -> NaN
            return None
        return {">=": v >= self.value, ">": v > self.value, "<=": v <= self.value,
                "<": v < self.value, "==": v == self.value}[self.operator]

    def describe(self) -> str:
        return f"{self.metric} {self.operator} {self.value:g}"


@dataclass
class SuccessCriterion:
    """What "solved" means. Deliberately finite: a criterion is a claim about N seeds at one fidelity,
    never "will never fail" (§3.2)."""
    primary: MetricConstraint
    additional: list[MetricConstraint] = field(default_factory=list)
    required_fidelity: str = "confirm"
    minimum_training_seeds: int = 3
    minimum_evaluation_samples: int = 3
    require_fresh_confirmation_seeds: bool = True
    confidence_level: float | None = 0.95

    def all_constraints(self) -> list[MetricConstraint]:
        return [self.primary, *self.additional]

    def describe(self) -> str:
        return (" and ".join(c.describe() for c in self.all_constraints())
                + f", at fidelity '{self.required_fidelity}' over >={self.minimum_training_seeds} seeds")


@dataclass
class SessionBudget:
    """Hard limits, enforced by deterministic code outside the LLM loop (§2.3). `explore_seconds` is the
    user's answer to "how long may the models explore before I review?"; the deadline is computed when a
    session actually starts, so a brief written today can be run tomorrow."""
    explore_seconds: float = 8 * 3600.0
    wind_down_seconds: float = 900.0
    maximum_gpu_hours: float | None = None       # None = bounded only by wall-clock
    maximum_runs: int = 200
    maximum_concurrent_runs: int = 1
    maximum_llm_calls: int = 200
    # Subscription limits are token-based, so a call cap alone does not protect them: keep these set to
    # what one session may spend of your allowance. None = bounded only by the call cap.
    maximum_llm_tokens: int | None = 2_000_000
    maximum_llm_cost_usd: float | None = None
    maximum_actor_reviewer_revisions: int = 2
    maximum_retry_runs: int = 2
    per_run_timeout_seconds: float | None = None  # None = derived from the fidelity estimate

    def deadline_from(self, start: datetime) -> datetime:
        return start + timedelta(seconds=self.explore_seconds)


@dataclass
class AdapterRef:
    """Which adapter drives the problem. "rx" is the built-in Rx adapter; "brief" is the generic
    command-driven adapter configured entirely from `test_command`/`metrics_file`/`knobs` below."""
    name: str = "rx"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemBrief:
    problem_id: str
    description: str                                   # what the problem is
    goal: str                                          # what winning looks like, in the user's words
    problem_repo: str
    primary_metric: str
    primary_direction: str                             # "maximize" | "minimize"
    success_criterion: SuccessCriterion
    # The deterministic performance test (§2.5): the harness's ONLY source of truth about performance.
    test_command: str = ""                             # how one evaluation/run is executed
    metrics_file: str = ""                             # path (relative to a run dir) holding its metrics
    tie_break_metric: str = ""                          # optional secondary ranking metric
    tie_break_direction: str = "minimize"
    constraints: list[MetricConstraint] = field(default_factory=list)
    realism_constraints: list[str] = field(default_factory=list)
    permitted_task_changes: list[str] = field(default_factory=list)
    forbidden_task_changes: list[str] = field(default_factory=list)
    notes: str = ""
    adapter: AdapterRef = field(default_factory=AdapterRef)
    session_budget: SessionBudget = field(default_factory=SessionBudget)
    created_at: str = ""
    schema_version: int = 1

    # ---------------- persistence ----------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProblemBrief":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemBrief":
        return _build(cls, data, "brief")

    def problems(self) -> list[str]:
        """Deterministic self-check of the brief itself; empty list = usable."""
        errs = []
        if not re.match(r"^[a-z][a-z0-9_-]{0,63}$", self.problem_id or ""):
            errs.append(f"problem_id {self.problem_id!r} must match ^[a-z][a-z0-9_-]{{0,63}}$")
        for name in ("description", "goal", "problem_repo", "primary_metric"):
            if not str(getattr(self, name)).strip():
                errs.append(f"{name} is required")
        if self.primary_direction not in DIRECTIONS:
            errs.append(f"primary_direction must be one of {DIRECTIONS}")
        if not Path(self.problem_repo).is_dir():
            errs.append(f"problem_repo {self.problem_repo!r} is not a directory")
        for c in [*self.success_criterion.all_constraints(), *self.constraints]:
            if c.operator not in OPERATORS:
                errs.append(f"constraint on {c.metric!r}: operator must be one of {OPERATORS}")
        if self.success_criterion.primary.metric != self.primary_metric:
            errs.append(f"success_criterion.primary must constrain the primary metric "
                        f"{self.primary_metric!r}, not {self.success_criterion.primary.metric!r}")
        if self.success_criterion.minimum_training_seeds < 1:
            errs.append("success_criterion.minimum_training_seeds must be >= 1")
        b = self.session_budget
        if b.explore_seconds <= 0:
            errs.append("session_budget.explore_seconds must be > 0")
        if b.wind_down_seconds < 0 or b.wind_down_seconds >= b.explore_seconds:
            errs.append("session_budget.wind_down_seconds must be >= 0 and < explore_seconds")
        for name in ("maximum_runs", "maximum_llm_calls", "maximum_concurrent_runs"):
            if getattr(b, name) < 1:
                errs.append(f"session_budget.{name} must be >= 1")
        for name in ("maximum_llm_tokens", "maximum_llm_cost_usd"):
            value = getattr(b, name)
            if value is not None and value <= 0:
                errs.append(f"session_budget.{name} must be > 0 or null")
        overlap = set(self.permitted_task_changes) & set(self.forbidden_task_changes)
        if overlap:
            errs.append(f"knob(s) both permitted and forbidden: {sorted(overlap)}")
        if self.adapter.name == "brief" and not self.metrics_file:
            errs.append("the generic 'brief' adapter needs metrics_file (where a run writes its numbers)")
        # Whether a `test_command` is required depends on the mode — variant mode replaces it with a
        # variant entry point — and only the adapter knows which variants exist, so that check lives in
        # `rllm.cli validate` where both the brief and the adapter are in hand.
        return errs

    def is_solved(self, metrics: dict) -> bool:
        """Every success constraint satisfied. Absent/NaN metrics never count as satisfied."""
        return all(c.satisfied_by(metrics) is True for c in self.success_criterion.all_constraints())

    def unmet(self, metrics: dict) -> list[str]:
        return [c.describe() for c in self.success_criterion.all_constraints()
                if c.satisfied_by(metrics) is not True]


def _build(cls, data: Any, where: str):
    """Recursive dataclass construction that REJECTS unknown fields (§3: no silent pass-through, so a
    typo in a hand-edited brief is an error rather than a silently ignored constraint)."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value, t = data[name], f.type
        if is_dataclass(t) and not isinstance(t, str):
            kwargs[name] = _build(t, value, f"{where}.{name}")
        elif name in ("additional", "constraints") and isinstance(value, list):
            kwargs[name] = [_build(MetricConstraint, v, f"{where}.{name}[{i}]")
                            for i, v in enumerate(value)]
        elif name in ("primary",):
            kwargs[name] = _build(MetricConstraint, value, f"{where}.{name}")
        elif name == "success_criterion":
            kwargs[name] = _build(SuccessCriterion, value, f"{where}.{name}")
        elif name == "session_budget":
            kwargs[name] = _build(SessionBudget, value, f"{where}.{name}")
        elif name == "adapter":
            kwargs[name] = _build(AdapterRef, value, f"{where}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


# Resolve the string annotations `from __future__ import annotations` leaves behind, so _build can see
# the real dataclass types for nested fields.
for _cls in (SuccessCriterion, ProblemBrief):
    _hints = {"primary": MetricConstraint, "success_criterion": SuccessCriterion,
              "session_budget": SessionBudget, "adapter": AdapterRef}
    for _f in fields(_cls):
        if _f.name in _hints:
            _f.type = _hints[_f.name]


def parse_duration(text: str) -> float:
    """'90m' / '8h' / '2d' / '3600' -> seconds. Used by the onboarding interview and the CLI."""
    text = str(text).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", text)
    if not m:
        raise ValueError(f"cannot parse duration {text!r} (try '45m', '8h', '2d')")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
