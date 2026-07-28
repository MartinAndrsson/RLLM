"""Budget ledger and runtime estimation — the part of the harness the models cannot talk their way past.

§2.3: hard budgets are enforced outside the LLM loop. Nothing here consults a model, and every limit is
checked before an action, not after. §10.2: durations start as a conservative user-supplied estimate and
are replaced by observation as runs complete, so the "does this fit before the deadline?" question gets
more accurate as a session progresses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median

from .brief import SessionBudget
from .interfaces import ProblemAdapter, RunResult
from .registry import Registry

STARTUP_OVERHEAD_SECONDS = 120.0     # env setup, imports, checkpoint IO before real work begins
SAFETY_FACTOR = 1.35                 # conservative padding on top of observed durations


@dataclass
class Ledger:
    """What this session has actually consumed. Persisted with the session state so a restart resumes
    against the same limits rather than starting the allowance over."""
    runs_launched: int = 0
    runs_failed: int = 0
    llm_calls: int = 0
    run_seconds: float = 0.0
    waves: int = 0
    stopped_because: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def gpu_hours(self) -> float:
        """Approximated by run wall-clock: one run occupies one device here (maximum_concurrent_runs=1
        per worker). Reported as an approximation, never as measured utilisation."""
        return self.run_seconds / 3600.0

    def record_run(self, result: RunResult) -> None:
        self.runs_launched += 1
        self.run_seconds += float(result.wall_seconds or 0.0)
        if result.status != "done":
            self.runs_failed += 1

    def record_llm_calls(self, n: int = 1) -> None:
        self.llm_calls += n

    # ---------------- limits (deterministic; no model input) ----------------

    def exhausted(self, budget: SessionBudget) -> str | None:
        """The reason further work must stop, or None. Checked before every launch and every LLM call."""
        if self.runs_launched >= budget.maximum_runs:
            return f"run cap reached ({self.runs_launched}/{budget.maximum_runs})"
        if self.llm_calls >= budget.maximum_llm_calls:
            return f"LLM-call cap reached ({self.llm_calls}/{budget.maximum_llm_calls})"
        if budget.maximum_gpu_hours is not None and self.gpu_hours >= budget.maximum_gpu_hours:
            return f"compute cap reached (~{self.gpu_hours:.1f}/{budget.maximum_gpu_hours:.1f} GPU-h)"
        return None

    def runs_remaining(self, budget: SessionBudget) -> int:
        left = budget.maximum_runs - self.runs_launched
        if budget.maximum_gpu_hours is not None:
            left = min(left, 10**6 if self.gpu_hours >= budget.maximum_gpu_hours else left)
        return max(0, left)


class Estimator:
    """Conservative per-fidelity duration estimates: observed median x safety factor + startup, falling
    back to the user's cheap-run estimate scaled by the rung's declared cost multiplier until at least
    one run of that rung has completed."""

    def __init__(self, adapter: ProblemAdapter, work_dir, cheap_run_seconds: float,
                 startup_overhead: float = STARTUP_OVERHEAD_SECONDS):
        self.adapter = adapter
        self.work_dir = work_dir
        self.cheap_run_seconds = float(cheap_run_seconds)
        # Per-run overhead the problem pays before real work starts (env setup, imports, checkpoint IO).
        # Configurable because it is problem-specific: a heavyweight trainer pays minutes, a cheap
        # evaluation pays milliseconds, and an overhead larger than the run itself would make every
        # fit-before-deadline check fail.
        self.startup_overhead = float(startup_overhead)
        self._multipliers = {f.name: max(1e-6, f.cost_multiplier) for f in adapter.fidelity_levels()}

    def observed(self, fidelity: str) -> list[float]:
        return [r.wall_seconds for r in Registry(self.work_dir).all_results()
                if r.fidelity == fidelity and r.wall_seconds and r.status == "done"]

    def seconds(self, fidelity: str) -> float:
        """Conservative UPPER estimate for one run at this rung (used for fit-before-deadline checks)."""
        obs = self.observed(fidelity)
        if obs:
            return median(obs) * SAFETY_FACTOR + self.startup_overhead
        return self.cheap_run_seconds * self._multipliers.get(fidelity, 1.0) + self.startup_overhead

    def hard_timeout(self, fidelity: str) -> float:
        """Wall-clock cap for one run: generous versus the estimate (a slow run is not a hung run) but
        finite, so nothing can outlive the session."""
        return max(600.0, self.seconds(fidelity) * 3.0)

    def is_observed(self, fidelity: str) -> bool:
        return bool(self.observed(fidelity))
