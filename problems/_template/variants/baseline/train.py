#!/usr/bin/env python3
"""SKELETON variant — copy this directory, then make it real. It does not run as-is (the TODOs below
need the actual simulator API), and it is deliberately kept in `problems/_template/` where the harness
does not look for variants (leading underscore).

This is one DESIGN: the observation, the action space and the reward are defined here, in this repository,
against the problem repo imported as a read-only simulator. See ../../../README.md for the contract.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------- harness contract

RUN_DIR = Path(os.environ["RUN_DIR"])           # write EVERYTHING here, nothing anywhere else
SEED = int(os.environ.get("SEED", "1"))
PROBLEM_REPO = Path(os.environ["PROBLEM_REPO"])  # read-only simulator; also on PYTHONPATH


def knob(name: str, default, cast=float):
    """Read a declared knob from the environment. Never hard-code a value that might want tuning —
    declare it as a knob instead, and the search can explore it without a new variant."""
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else cast(raw)


# Knobs this design uses. Declare each of these in the brief's adapter.config.knobs.
SHORTAGE_SCALE = knob("SHORTAGE_SCALE", 45000.0)
GAMMA = knob("GAMMA", 0.99)
LR = knob("LR", 3e-4)
TRAIN_STEPS = knob("TRAIN_STEPS", 200_000, int)   # fidelity-owned: the ladder sets this per rung


def main() -> int:
    random.seed(SEED)

    # ------------------------------------------------------------ 1. the simulator (read-only)
    # TODO: import the user's simulator. It is on PYTHONPATH, e.g.
    #   from rx_gym_env import RxGymEnv
    #   sim = RxGymEnv(config_path=PROBLEM_REPO / "config_rx.json", seed=SEED)
    raise NotImplementedError("copy this template and wire up the real simulator")

    # ------------------------------------------------------------ 2. THE DESIGN — yours to choose
    #
    # observation: what the agent sees each step. Say it in the manifest too.
    # action:      what it may do, and its bounds/scaling.
    # reward:      what it is paid. This is where SHORTAGE_SCALE and friends are used.
    #
    # Wrap the simulator rather than editing it:
    #
    #   class Env(gym.Wrapper):
    #       def observation(self, raw): ...      # <- the state space
    #       def step(self, action): ...          # <- the action space + the reward
    #
    # ------------------------------------------------------------ 3. train
    #   model = SAC("MlpPolicy", env, gamma=GAMMA, learning_rate=LR, seed=SEED,
    #               tensorboard_log=str(RUN_DIR / "tb"))
    #   model.learn(total_timesteps=TRAIN_STEPS)
    #   model.save(RUN_DIR / "policy")
    #
    # ------------------------------------------------------------ 4. evaluate with the USER'S test
    # The numbers must come from the user's own evaluation, not from a metric this variant invented:
    #   sys.path.insert(0, str(PROBLEM_REPO))
    #   from final_eval_dev import evaluate            # whatever the real entry point is
    #   results = evaluate(policy=RUN_DIR / "policy", seed=SEED)
    #
    # ------------------------------------------------------------ 5. write the metrics
    metrics = {
        "success_fraction": 0.0,      # MUST include the brief's primary metric
        "mean_shortage_days": 0.0,
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    # Exit non-zero on failure: a crash that looks like a bad result is worse than a crash.
    sys.exit(main())
