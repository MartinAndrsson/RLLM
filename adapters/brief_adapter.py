"""BriefAdapter — a problem adapter configured entirely from a ProblemBrief.

It runs experiments in one of two modes, and the difference matters:

**Variant mode** (the interesting one). The models design the observation, action space and reward
themselves, as code under `problems/<problem_id>/variants/<variant>/` IN THIS REPOSITORY, importing the
user's repo as a read-only simulator. Each variant directory is one design; `VARIANT` becomes a declared
enum knob whose choices are the variants on disk, so a design is just another dimension the search can
compare — and the comparison table gets a `--by VARIANT` view for free. The problem repo is never
edited, and `rllm/integrity.py` verifies that after every run.

**Command mode** (the simple one). The brief names a `test_command` that already exists in the problem
repo, and the harness just runs it with knobs as environment variables. Good for a problem that already
has a tunable entry point; no code is generated.

Contract for both (kept small so it is easy to satisfy):

  * the harness exports the knob values as environment variables, plus `RUN_DIR` (a fresh directory the
    harness owns), `SEED`, `PROBLEM_REPO` (the read-only simulator) and `PY`;
  * the command must write `metrics_file` (relative to `RUN_DIR`) as flat JSON, e.g.
    `{"success_fraction": 0.8, "mean_shortage_days": 1.1}`;
  * that JSON is the ONLY thing the harness believes about performance (§2.5) — no model is ever asked
    whether a run went well;
  * everything written goes under `RUN_DIR`. Nothing writes to the problem repo.

`adapter.config` in the brief may carry:
  knobs:       [ {name, value_type, default, category, ...}, ... ]  -> the whitelist (see KnobSpec)
  fidelities:  [ {name, overrides: {...}, seeds: [...]}, ... ]      -> the coarse-to-fine ladder
  base_config: {KNOB: value}                                        -> applied to every run
  variants_dir: path to the variants tree (default problems/<problem_id>/variants)
  variant_entry: entry-point filename inside a variant (default train.py)
  python:      interpreter to expose as $PY
"""
from __future__ import annotations

import json
import sys
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


RLLM_ROOT = Path(__file__).resolve().parent.parent
VARIANT_KNOB = "VARIANT"


class BriefAdapter(ProblemAdapter):
    def __init__(self, brief: ProblemBrief):
        self.brief = brief
        self.name = brief.problem_id
        self.success_metric = brief.primary_metric
        self.success_goal = brief.goal
        cfg = brief.adapter.config or {}
        self._knobs = {k["name"]: KnobSpec(**k) for k in cfg.get("knobs", [])}
        self._base = dict(cfg.get("base_config", {}))
        self._fidelities = [Fidelity(f["name"], dict(f.get("overrides", {})), list(f["seeds"]),
                                     float(f.get("cost_multiplier", 1.0)))
                            for f in cfg.get("fidelities") or DEFAULT_FIDELITIES]
        # `python` is frequently absent (python3 only), and a variant that cannot start looks
# exactly like a variant that performs badly — default to the interpreter we are running.
        self._python = cfg.get("python") or sys.executable
        self._variant_entry = cfg.get("variant_entry", "train.py")
        self.variants_dir = Path(cfg.get("variants_dir")
                                 or RLLM_ROOT / "problems" / brief.problem_id / "variants")

    # ---------------- variants: the designs the models wrote ----------------

    def variants(self) -> list[str]:
        """Variant ids available on disk: a directory holding the entry point. Discovered rather than
        declared, so a newly authored design is picked up without editing the brief."""
        if not self.variants_dir.is_dir():
            return []
        return sorted(p.name for p in self.variants_dir.iterdir()
                      if p.is_dir() and (p / self._variant_entry).is_file()
                      and not p.name.startswith((".", "_")))

    # ---------------- declarations ----------------

    def declared_knobs(self) -> dict[str, KnobSpec]:
        variants = self.variants()
        if not variants:
            return self._knobs
        # VARIANT is a real knob: the search compares designs exactly as it compares hyperparameters,
        # and every existing gate (whitelist, enum choices, duplicate detection, ranking) applies to it
        # unchanged. Its choices come from the filesystem, so it cannot name a design that does not exist.
        knob = KnobSpec(VARIANT_KNOB, "enum", self._base.get(VARIANT_KNOB, variants[0]),
                        "algorithm", choices=variants)
        return {**self._knobs, VARIANT_KNOB: knob}

    def base_config(self) -> dict[str, Any]:
        return dict(self._base)

    def fidelity_levels(self) -> list[Fidelity]:
        return self._fidelities

    # ---------------- running ----------------

    def readonly_paths(self) -> list[str]:
        """The problem repo is a read-only simulator; the harness verifies this after every run."""
        return [self.brief.problem_repo]

    def build_command(self, spec: ExperimentSpec, seed: int, run_dir: str, device: str) -> list[str]:
        env = {**self._base, **spec.config}
        env.update({"RUN_DIR": run_dir, "SEED": seed, "CUDA_VISIBLE_DEVICES": device,
                    "PY": self._python, "PROBLEM_REPO": self.brief.problem_repo})
        variants = self.variants()
        if variants:
            variant = str(env.get(VARIANT_KNOB, variants[0]))
            if variant not in variants:                      # validate.py enforces this; belt and braces
                raise ValueError(f"unknown variant {variant!r}; available: {variants}")
            entry = self.variants_dir / variant / self._variant_entry
            # Run from the VARIANT's directory with the problem repo importable but not writable-by-intent
            # — the variant owns observation/action/reward, the repo only provides the simulator.
            script = (f"set -euo pipefail; cd {_sh(str(entry.parent))}; export {_env(env)}; "
                      f"export PYTHONPATH={_sh(self.brief.problem_repo)}:${{PYTHONPATH:-}}; "
                      f'"$PY" {_sh(self._variant_entry)}')
            return ["bash", "-c", script]
        # Command mode: the problem repo already has a tunable entry point.
        if not self.brief.test_command.strip():
            raise ValueError(
                f"nothing to run: no variants found under {self.variants_dir} and the brief declares no "
                f"test_command. Either author a variant (see problems/README.md) or set test_command.")
        script = (f"set -euo pipefail; cd {_sh(self.brief.problem_repo)}; export {_env(env)}; "
                  f"{self.brief.test_command}")
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


def _env(env: dict) -> str:
    return " ".join(f"{k}={_sh(v)}" for k, v in env.items())


def _sh(v: Any) -> str:
    """Quote one value for the bash recipe; refuse anything we cannot reason about (validate.py has
    already type/range-checked every knob, so a rejection here means a harness bug)."""
    s = str(v)
    if s and all(c.isalnum() or c in "._-/" for c in s):
        return s
    if all(c.isalnum() or c in "._-/ +=:,@" for c in s):
        return f"'{s}'"
    raise ValueError(f"unsafe value for shell recipe: {s!r}")
