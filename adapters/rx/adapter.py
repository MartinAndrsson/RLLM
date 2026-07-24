"""RxAdapter — first RLLM problem adapter: the Rx rollbox redistribution sim (SB3/TQC).

Wraps the existing parameterised recipe `run_tqc_deep.sh` (env-var knobs) in Supply/Rx_dan/supply.
Fidelity is scaled via SAFE knobs that don't change the problem's realism (training length, #seeds,
eval frequency) — NOT via episode length (n_days stays 100; shortening it would change what 'solved'
means, per problem.md). Success = never runs out of Rx: success_fraction == 1.0.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

# Allow `from rllm.interfaces import ...` when RLLM/ is on sys.path.
from rllm.interfaces import ProblemAdapter, Fidelity, ExperimentSpec

RX_REPO = "/home/hep/maander/Supply/Rx_dan/supply"
PSIM_PY = "/home/hep/maander/miniconda3/envs/psim/bin/python"

# M1 base (the current winner): bootstrap fix + train==eval RFID/forecast + shortage-days selection.
BASE_CONFIG: dict[str, Any] = {
    "TRUNC": 1, "SOFT_SCALE": 0, "SHORTAGE_SCALE": 45000,
    "RFID_PROB": 0.1, "FORECAST_UNC": 0.02, "SELECT_METRIC": "shortage_days",
    "PLOT_EXTRA": 0, "EVAL_EPISODES": 3,
}


class RxAdapter(ProblemAdapter):
    name = "rx"
    success_metric = "success_fraction"          # PRIMARY: Normal-ops fraction of seeds with ZERO shortage
    # Tiered goal (see problem.md): (1) push Normal-ops success -> 1.0 (never run out); (2) then improve
    # robustness on the OTHER test scenarios that exist in the problem (Surge/lull, SP, Obstacle);
    # (3) if 1.0 is unreachable, the goal is the best achievable (a session-level plateau call, not is_solved).
    success_goal = ("Tiered: Normal-ops success_fraction -> 1.0 first, then lift the other scenarios; "
                    "if 1.0 is unreachable, maximise (best-achievable / plateau).")
    success_target = 1.0                          # primary bar for is_solved; best-possible handled at session level

    def __init__(self, rx_repo: str = RX_REPO, py: str = PSIM_PY):
        self.rx_repo = rx_repo
        self.py = py

    def fidelity_levels(self) -> list[Fidelity]:
        return [
            Fidelity("screen",  {"MAXGEN": 60,  "EVAL_EVERY": 30, "BC_EPOCHS": 0},   seeds=[1]),
            Fidelity("refine",  {"MAXGEN": 250, "EVAL_EVERY": 25},                   seeds=[1, 2]),
            Fidelity("confirm", {"MAXGEN": 500, "EVAL_EVERY": 25},                   seeds=[1, 2, 3]),
        ]

    def _prefix(self, run_dir: str) -> str:
        return "rllm_" + Path(run_dir).name          # deterministic; parse_result recomputes it

    def build_command(self, spec: ExperimentSpec, seed: int, run_dir: str, device: str) -> list[str]:
        env = {**BASE_CONFIG, **spec.config}         # base <- experiment overrides (fidelity already merged in)
        env.update({
            "CUDA_VISIBLE_DEVICES": device, "SEED": seed,
            "PREFIX": self._prefix(run_dir), "PY": self.py,
        })
        envstr = " ".join(f"{k}={_sh(v)}" for k, v in env.items())
        script = f"cd {self.rx_repo} && {envstr} bash run_tqc_deep.sh"
        return ["bash", "-c", script]

    def parse_result(self, run_dir: str) -> dict[str, Any]:
        prefix = self._prefix(run_dir)
        matches = sorted(glob.glob(os.path.join(self.rx_repo, "rx_sb3_runs", f"{prefix}_tqc_*")))
        if not matches:
            return {"error": f"no rx run dir for prefix {prefix}"}
        fe_path = os.path.join(matches[-1], "final_eval.json")
        if not os.path.exists(fe_path):
            return {"error": "final_eval.json missing", "rx_run_dir": matches[-1]}
        fe = json.loads(Path(fe_path).read_text())
        # Capture ALL scenarios (primary = Normal; the rest are the robustness tier).
        per_scenario = {}
        for s in fe.get("scenarios", []):
            per_scenario[str(s.get("scenario", "?"))] = {
                "success_fraction": float(s.get("success_fraction", 0.0)),
                "mean_shortage_days": float(s.get("mean_shortage_days", float("nan"))),
            }
        normal = per_scenario.get("Normal operation")
        if normal is None:
            return {"error": "no Normal scenario", "rx_run_dir": matches[-1], "scenarios": per_scenario}
        return {
            "success_fraction": normal["success_fraction"],          # PRIMARY metric
            "mean_shortage_days": normal["mean_shortage_days"],
            "scenarios": per_scenario,                                # robustness tier lives here
            "rx_run_dir": matches[-1],
        }

    def is_solved(self, metrics: dict[str, Any]) -> bool:
        # Primary bar only. "Best achievable if 1.0 is unreachable" is a plateau judgement made at the
        # session level (stall detector), not from a single run's metrics.
        return float(metrics.get("success_fraction", 0.0)) >= self.success_target

    def better(self, a: dict, b: dict) -> bool:
        # Higher success first; break ties by fewer shortage-days.
        ka = (a.get("success_fraction", -1.0), -a.get("mean_shortage_days", 1e9))
        kb = (b.get("success_fraction", -1.0), -b.get("mean_shortage_days", 1e9))
        return ka > kb


def _sh(v: Any) -> str:
    s = str(v)
    return s if all(c.isalnum() or c in "._-/" for c in s) else f"'{s}'"
