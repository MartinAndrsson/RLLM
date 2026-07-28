# problems/ — the designs, kept out of the user's repo

```
problems/<problem_id>/
  variants/
    <variant_id>/
      manifest.json     what this design is and why (one paragraph, written when it is created)
      train.py          the entry point: builds the env, trains, evaluates, writes metrics
      *.py              whatever else the design needs (reward.py, obs.py, ...)
```

Each variant directory is **one design of the observation, the action space and the reward**. The
problem repository is imported as a read-only simulator and is never modified — `rllm/integrity.py`
fingerprints it around every run and fails the session if it changed. That separation is the whole point:
the user's simulator and the user's test stay exactly as they wrote them, so a good number cannot be an
artifact of the thing being measured having moved.

`VARIANT` is a declared enum knob whose choices are the directories found here, so a design is just
another dimension of the search: the same whitelist, duplicate detection, ladder and ranking apply, and
`python -m rllm.cli compare <work_dir> --by VARIANT` gives the design-against-design table.

## The contract

The harness runs `train.py` with the variant directory as the working directory, and provides:

| Environment variable | What it is |
|---|---|
| `RUN_DIR` | a fresh directory owned by the harness. **Write everything here** — metrics, checkpoints, logs |
| `SEED` | the training seed for this run |
| `PROBLEM_REPO` | absolute path to the read-only simulator (also on `PYTHONPATH`) |
| `PY` | the interpreter being used |
| `CUDA_VISIBLE_DEVICES` | the device assigned to this run |
| *every declared knob* | e.g. `SHORTAGE_SCALE`, `GAMMA`, `TRAIN_STEPS` — read them, do not hard-code |

`train.py` must write `$RUN_DIR/metrics.json` — flat JSON, numbers only:

```json
{"success_fraction": 0.8, "mean_shortage_days": 1.1}
```

That file is the only thing the harness believes about performance. It must include the brief's primary
metric, and those numbers must come from **the user's own test**, not from a metric the variant invented.
A run that writes no metrics file is a failed run, not a bad score.

Exit non-zero on failure. Do not catch an exception and write a zero score — a crash that looks like a
bad result is worse than a crash.

## Rules

1. **Never write outside `$RUN_DIR`.** Not into the problem repo, not into the variant directory.
2. **A variant that has produced a result is immutable.** Improving it means a new variant directory, so
   the comparison table keeps its meaning. `manifest.json` records `parent` when a design descends from
   another.
3. **Read knobs from the environment**, with the declared default as the fallback. A value worth tuning
   belongs in the knob table, not hard-coded — then the search can tune it without a new variant.
4. **Keep it readable.** A reviewer must be able to see what the observation is, what the action is, and
   how the reward is computed. If that takes more than a couple of short files, the design is probably
   doing too much at once.
5. **Deterministic given `SEED`.** Seed every source of randomness you use.

## manifest.json

```json
{
  "id": "baseline_shortage_penalty",
  "hypothesis": "Penalising shortage days directly, with per-hub stock and a demand forecast as the observation, is enough to keep every centre supplied.",
  "observation": "per-hub stock levels, 10-day demand forecast, in-transit counts",
  "action": "continuous per-route send fraction in [0, 1]",
  "reward": "-shortage_days * SHORTAGE_SCALE + survival bonus per day",
  "parent": null,
  "knobs_used": ["SHORTAGE_SCALE", "GAMMA", "LR"]
}
```

The manifest is documentation, not configuration — the harness only requires `train.py` to exist. It is
what lets the next session (and the reviewer) understand a design without reading its code, so write it
as if the reader has no context, because they do not.
