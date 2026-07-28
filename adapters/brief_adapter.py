"""BriefAdapter — a problem adapter configured entirely from a ProblemBrief.

The onboarding interview asks for a repo, a deterministic test command, the file that command writes its
metrics to, and the knobs that may be tuned. That is exactly enough to drive the harness, so a new
problem needs no Python: this adapter turns those answers into runs.

Contract with the problem repo (kept deliberately small so it is easy to satisfy):

  * the harness runs `test_command` with `cd <problem_repo>`, the knob values exported as environment
    variables, plus `RUN_DIR` (a fresh directory owned by the harness) and `SEED`;
  * the command must write `metrics_file` (a path relative to RUN_DIR) as flat JSON, e.g.
    `{"success_fraction": 0.8, "mean_shortage_days": 1.1}`;
  * that JSON is the ONLY thing the harness believes about performance (§2.5) — no model is asked
    whether a run went well.

`adapter.config` in the brief may carry:
  knobs:      [ {name, value_type, default, category, minimum, maximum, choices,
                 fidelity_safe, llm_may_change}, ... ]   -> the whitelist (see KnobSpec)
  fidelities: [ {name, overrides: {...}, seeds: [...]}, ... ]  -> the coarse-to-fine ladder
  base_config: {KNOB: value}                              -> applied to every run
  python:     interpreter to expose as $PY (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rllm.brief import ProblemBrief
from rllm.interfaces import Fidelity, KnobSpec, ProblemAdapter, ExperimentSpec

# A ladder that spends little first and more later, used when the brief declares no rungs of its own.
DEFAULT_FIDELITIES = [
    {"name": "screen", "overrides": {}, "seeds": [1]},
    {"name": "refine", "overrides": {}, "seeds": [1, 2]},
    {"name": "confirm", "overrides": {}, "seeds": [1, 2, 3]},
]


class BriefAdapter(ProblemAdapter):
    def __init__(self, brief: ProblemBrief):
        self.brief = brief
        self.name = brief.problem_id
        self.success_metric = brief.primary_metric
        self.success_goal = brief.goal
        cfg = brief.adapter.config or {}
        self._knobs = {k["name"]: KnobSpec(**k) for k in cfg.get("knobs", [])}
        self._base = dict(cfg.get("base_config", {}))
        self._fidelities = [Fidelity(f["name"], dict(f.get("overrides", {})), list(f["seeds"]))
                            for f in cfg.get("fidelities") or DEFAULT_FIDELITIES]
        self._python = cfg.get("python", "python")

    # ---------------- declarations ----------------

    def declared_knobs(self) -> dict[str, KnobSpec]:
        return self._knobs

    def base_config(self) -> dict[str, Any]:
        return dict(self._base)

    def fidelity_levels(self) -> list[Fidelity]:
        return self._fidelities

    # ---------------- running ----------------

    def build_command(self, spec: ExperimentSpec, seed: int, run_dir: str, device: str) -> list[str]:
        env = {**self._base, **spec.config}
        env.update({"RUN_DIR": run_dir, "SEED": seed, "CUDA_VISIBLE_DEVICES": device, "PY": self._python})
        envstr = " ".join(f"{k}={_sh(v)}" for k, v in env.items())
        # `set -euo pipefail` so a failing step inside the user's command fails the run (§4).
        script = (f"set -euo pipefail; cd {_sh(self.brief.problem_repo)}; "
                  f"export {envstr}; {self.brief.test_command}")
        return ["bash", "-c", script]

    def parse_result(self, run_dir: str) -> dict[str, Any]:
        path = Path(run_dir) / self.brief.metrics_file
        if not path.exists():
            return {"error": f"metrics file {self.brief.metrics_file!r} not written under {run_dir}"}
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"metrics file unreadable: {exc}"}
        if not isinstance(data, dict):
            return {"error": f"metrics file must hold a JSON object, got {type(data).__name__}"}
        metrics = {k: v for k, v in data.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if self.success_metric not in metrics:
            return {"error": f"metrics file has no finite {self.success_metric!r} "
                             f"(found: {', '.join(sorted(metrics)) or 'nothing numeric'})",
                    **metrics}
        return metrics

    # ---------------- ranking / success ----------------

    def is_solved(self, metrics: dict[str, Any]) -> bool:
        return self.brief.is_solved(metrics)

    def sort_key(self, metrics: dict[str, Any]) -> tuple:
        sign = 1.0 if self.brief.primary_direction == "maximize" else -1.0
        worst = float("-inf")
        primary = metrics.get(self.success_metric)
        primary = sign * primary if isinstance(primary, (int, float)) else worst
        if not self.brief.tie_break_metric:
            return (primary,)
        tsign = 1.0 if self.brief.tie_break_direction == "maximize" else -1.0
        tie = metrics.get(self.brief.tie_break_metric)
        return (primary, tsign * tie if isinstance(tie, (int, float)) else worst)


def _sh(v: Any) -> str:
    """Quote one value for the bash recipe; refuse anything we cannot reason about (validate.py has
    already type/range-checked every knob, so a rejection here means a harness bug)."""
    s = str(v)
    if s and all(c.isalnum() or c in "._-/" for c in s):
        return s
    if all(c.isalnum() or c in "._-/ +=:,@" for c in s):
        return f"'{s}'"
    raise ValueError(f"unsafe value for shell recipe: {s!r}")
