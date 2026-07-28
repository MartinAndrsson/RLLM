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
from rllm.interfaces import ProblemAdapter, Fidelity, ExperimentSpec, KnobSpec

RX_REPO = "/home/hep/maander/Supply/Rx_dan/supply"
PSIM_PY = "/home/hep/maander/miniconda3/envs/psim/bin/python"

# M1 base (the current winner): bootstrap fix + train==eval RFID/forecast + shortage-days selection.
BASE_CONFIG: dict[str, Any] = {
    "TRUNC": 1, "SOFT_SCALE": 0, "SHORTAGE_SCALE": 45000,
    "RFID_PROB": 0.1, "FORECAST_UNC": 0.02, "SELECT_METRIC": "shortage_days",
    "PLOT_EXTRA": 0, "EVAL_EPISODES": 3,
}


def _k(name, value_type, default, category, minimum=None, maximum=None, choices=None,
       fidelity_safe=False, llm_may_change=True) -> KnobSpec:
    return KnobSpec(name, value_type, default, category, minimum, maximum, choices,
                    fidelity_safe, llm_may_change)


# The knob WHITELIST: exactly the env-var knobs `run_tqc_deep.sh` reads, with their real defaults.
# Anything not listed here is rejected before a command is ever built (rllm/validate.py).
# Deliberate gates:
#   * training-budget/eval-cadence knobs are fidelity-owned (llm_may_change=False) — the ladder sets
#     them, an experiment must not buy itself more compute;
#   * RFID_PROB / FORECAST_UNC / SCENARIO_POOL / DR_* are `task_definition` — they change how much of
#     the world the agent can see, i.e. what "solved" means (problem.md realism), so they stay
#     human-gated even if both models agree;
#   * path-valued knobs (DEMOS, DR_CONFIG) are never model-settable (no model-controlled paths).
RX_KNOBS: dict[str, KnobSpec] = {k.name: k for k in [
    # --- algorithm ---
    _k("ALGO", "enum", "tqc", "algorithm", choices=["sac", "tqc", "tqc-double", "ppo"]),
    _k("TRUNC", "int", 0, "algorithm", 0, 1),                    # bootstrap value at the fixed horizon
    _k("TD_N_STEPS", "int", 1, "algorithm", 1, 10),              # n-step returns (idea backlog)
    _k("DEMOS", "string", "flux_demos_availfrac.npz", "algorithm", llm_may_change=False),
    # --- hyperparameters ---
    _k("GAMMA", "float", 0.99, "hyperparameter", 0.5, 0.9999),
    _k("LR", "float", 3e-4, "hyperparameter", 1e-6, 1e-2),
    _k("BATCH", "int", 256, "hyperparameter", 32, 4096),
    _k("GRAD_STEPS", "int", 8, "hyperparameter", 1, 32),         # UTD ratio
    _k("ENT_COEF", "float", 0.01, "hyperparameter", 0.0, 1.0),
    _k("PPO_N_STEPS", "int", 512, "hyperparameter", 64, 8192),
    _k("HIDDEN_SIZES", "enum", "256 256", "hyperparameter",
       choices=["256 256", "256 256 256", "400 300", "512 512", "512 512 512"]),
    # --- reward shaping ---
    _k("SHORTAGE_PENALTY", "float", 200, "reward", 0, 100000),
    _k("SHORTAGE_SCALE", "float", 45000, "reward", 0, 1e6),
    _k("SOFT_SCALE", "float", 0, "reward", 0, 1e5),              # smooth survival bonus
    _k("SURVIVAL_BONUS", "float", 1.0, "reward", 0, 1e4),
    _k("SHAPING_SCALE", "float", 1.0, "reward", 0, 100),
    _k("SAFETY_LEAD", "int", 2, "reward", 0, 10),                # base-stock shaping target
    _k("SAFETY_Z", "float", 1.0, "reward", 0, 5),
    # --- action space ---
    _k("MAX_SEND_FRAC", "float", 1.0, "action", 0.01, 1.0),      # per-route daily drain cap
    _k("SCALE_MODE", "enum", "available_fraction", "action", choices=["fixed", "available_fraction"]),
    _k("GRS", "float", 0.93, "action", 0.0, 1.0),
    # --- observation ---
    _k("FORECAST_H", "int", 10, "observation", 1, 30),
    _k("DR_OBSERVABLE_TRACE", "int", 0, "observation", 0, 1),
    # --- evaluation / selection ---
    _k("SELECT_METRIC", "enum", "return", "evaluation",
       choices=["return", "shortage_days", "shortage_total"]),
    _k("EVAL_EVERY", "int", 25, "evaluation", 1, 500, fidelity_safe=True, llm_may_change=False),
    _k("EVAL_EPISODES", "int", 3, "evaluation", 1, 50, fidelity_safe=True, llm_may_change=False),
    # --- training budget (owned by the fidelity ladder) ---
    _k("MAXGEN", "int", 500, "training_budget", 1, 5000, fidelity_safe=True, llm_may_change=False),
    _k("BC_EPOCHS", "int", 300, "training_budget", 0, 5000, fidelity_safe=True, llm_may_change=False),
    _k("EARLY_STOP_PATIENCE", "int", 0, "training_budget", 0, 1000, fidelity_safe=True),
    _k("EARLY_STOP_MIN_GENS", "int", 0, "training_budget", 0, 5000, fidelity_safe=True),
    # --- task definition (human-gated: changes what "solved" means) ---
    _k("RFID_PROB", "float", 0.0, "task_definition", 0.0, 1.0),
    _k("FORECAST_UNC", "float", 0.02, "task_definition", 0.0, 1.0),
    _k("SCENARIO_POOL", "string", "none", "task_definition"),
    _k("DR_CONFIG", "string", "", "task_definition", llm_may_change=False),
    # --- harness-only plumbing ---
    _k("PLOT_EXTRA", "int", 0, "evaluation", 0, 1, llm_may_change=False),
    _k("REQUIRE_CUDA", "int", 0, "evaluation", 0, 1, llm_may_change=False),
    _k("SEED", "int", 1729, "evaluation", 0, 10**9, llm_may_change=False),
]}


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

    def base_config(self) -> dict[str, Any]:
        return dict(BASE_CONFIG)

    def declared_knobs(self) -> dict[str, KnobSpec]:
        return RX_KNOBS

    def fidelity_levels(self) -> list[Fidelity]:
        # cost_multiplier = cost of ONE run at this rung relative to one screen run, from the MAXGEN
        # ratio (250/60, 500/60) rounded up because refine/confirm also pay for BC pretraining that
        # screen skips. Only used until real durations are observed, which then take over.
        return [
            Fidelity("screen",  {"MAXGEN": 60,  "EVAL_EVERY": 30, "BC_EPOCHS": 0},   seeds=[1],
                     cost_multiplier=1.0),
            Fidelity("refine",  {"MAXGEN": 250, "EVAL_EVERY": 25},                   seeds=[1, 2],
                     cost_multiplier=5.0),
            Fidelity("confirm", {"MAXGEN": 500, "EVAL_EVERY": 25},                   seeds=[1, 2, 3],
                     cost_multiplier=9.0),
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

    def sort_key(self, metrics: dict) -> tuple:
        # Higher Normal-ops success first; break ties by fewer shortage-days.
        return (metrics.get("success_fraction", -1.0), -metrics.get("mean_shortage_days", 1e9))


def _sh(v: Any) -> str:
    """Quote one env-var value for the `bash -c` recipe. Defense in depth: validate.py has already
    type/range-checked every knob, so anything exotic reaching here is a bug — fail loudly rather than
    emit a shell string we can't reason about."""
    s = str(v)
    if all(c.isalnum() or c in "._-/" for c in s):
        return s
    if all(c.isalnum() or c in "._-/ +=:," for c in s):
        return f"'{s}'"
    raise ValueError(f"unsafe value for shell recipe: {s!r}")
