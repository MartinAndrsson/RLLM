# journal.md — Rx experiment log & idea backlog (RLLM per-problem memory)

Running log: what's been tried, outcomes, and the idea backlog. Never delete an idea — mark it tried
with its result. Mirrors the essentials of the Rx repo's `RL.md` (source of truth for full detail).

## Results so far (from prior manual work, pre-RLLM)
| id | change | Normal success | shortage days | verdict |
|----|--------|:--------------:|:-------------:|---------|
| p200 baseline | pre-fix | 0% | ~3.6-4.4 | never fully clean; value not bootstrapped at horizon |
| M1 | + truncation-at-horizon bootstrap; RFID 0.1 | ~45% (n=2, 30-60%) | ~1.0 | **big win, current base** |
| M2 | + shortage_scale 15000 | ~50% | ~1.15 | ~neutral vs M1 |
| M3 | + smooth survival 500 | ~40% | ~1.0 | ~neutral vs M1 |
| M4 | forecast_horizon 3 | ~20% | ~1.3 | worse — keep fh=10 |
| utd2 | gradient_steps 2 | 30% | 1.4 | worse + only ~18% faster — keep UTD=8 |

## Key learnings
- Bootstrapping at the fixed horizon was THE fix (0% -> ~45%).
- High seed variance (~15-30 pts) -> single seed = screen only; confirm winners with more seeds.
- Convergence is LATE (best near gen 500) -> conservative pruning; short screens can mislead.
- Eval was ~2/3 of wall-clock (serial); now eval_every=25, eval_episodes=3.
- Training step is ~3/4 env-collection, ~1/4 gradients -> speed lever is n_envs (192 cores, using 16),
  not UTD. Device: GPU only ~1.3x over CPU (tiny net); avoid the flaky GPU index.

## How to run this problem now (2026-07-28)
`brief.json` in this directory is the frozen contract: primary metric `success_fraction` (maximize),
solved = `success_fraction >= 1.0 and mean_shortage_days <= 0` at `confirm` over >=3 seeds, 8h window,
60-run cap, and `RFID_PROB`/`FORECAST_UNC`/`SCENARIO_POOL`/`DR_CONFIG` forbidden. One bounded session:

```bash
cd /home/hep/maander/Supply/RLLM
python -m rllm.cli solve adapters/rx/rllm_work --device 0 --until 8h
python -m rllm.cli handoff adapters/rx/rllm_work
```
It plans/runs/promotes waves until wind-down, then writes `sessions/<id>/handoff.md` and appends a
facts-only block here. Edit brief.json to change the window, the caps or the solved bar — the models may
request those changes in a handoff but cannot make them.

## RLLM wave 1 (2026-07-28) — first LLM-proposed wave, ENQUEUED, not yet run
Actor claude / reviewer codex, via `python -m rllm.cli propose adapters/rx/rllm_work`. Queued at
`screen` (MAXGEN=60, 1 seed, BC_EPOCHS=0) on top of the M1 base. Nothing has trained yet — run
`python -m rllm.cli work adapters/rx/rllm_work --device 0` to execute.

| id | delta vs M1 base | hypothesis (actor) | status |
|----|------------------|--------------------|--------|
| m1_nstep3 | `TD_N_STEPS=3` | n-step returns propagate the delayed shortage signal faster than 1-step TD | queued |
| m1_send_cap50 | `MAX_SEND_FRAC=0.5` | capping per-route daily drain prevents the over-draining behind the last ~1 shortage day | queued |
| m1_gamma995 | `GAMMA=0.995` | post-bootstrap, a longer effective horizon lets the critic value multi-day buffers | queued |

Reviewer (codex) rejected a 4th, `SELECT_METRIC=shortage_days` — correctly, since the M1 base already
sets it (it would have been a duplicate of the base). Its risk flags to carry forward: `proxy_transfer`
/ `short_screen_may_misrank`, `insufficient_seeds`, `late_convergence`, `normal_ops_only`, and
`send_cap_may_reduce_action_feasibility` (check MAX_SEND_FRAC=0.5 doesn't make demand unservable).

Harness lessons from this first live wave (both now fixed + regression-tested):
- The reviewer initially blocked the whole wave demanding more generations/seeds — knobs the actor
  structurally cannot set. Both prompts now state the fidelity ladder as harness-owned and off-limits.
- Neither model saw the base recipe, so all four proposals re-stated `TRUNC=1` and one duplicated the
  base. Both prompts now include the base config and ask for deltas only.

## Idea backlog (untried, prioritised) — pull these into screen experiments
- [high] Recurrent policy (POMDP from RFID 0.1) — integrate history to infer true stock.
- [high] n-step returns (`--td-n-steps 2/3`, in-framework) — faster propagation of the delayed signal.
- [high] LayerNorm / DroQ critic — cheap stability, may fix late-training collapse.
- [med] gamma sweep (now value is stationary post-bootstrap); MPC look-ahead (we have a forecast);
  AWAC (blend BC demos + RL); PER; per-hub shortage weighting; per_route_max_send_fraction cap.
- [robustness tier] scenario-pool domain randomization for Surge/SP/Obstacle.
