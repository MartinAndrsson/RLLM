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

## Idea backlog (untried, prioritised) — pull these into screen experiments
- [high] Recurrent policy (POMDP from RFID 0.1) — integrate history to infer true stock.
- [high] n-step returns (`--td-n-steps 2/3`, in-framework) — faster propagation of the delayed signal.
- [high] LayerNorm / DroQ critic — cheap stability, may fix late-training collapse.
- [med] gamma sweep (now value is stationary post-bootstrap); MPC look-ahead (we have a forecast);
  AWAC (blend BC demos + RL); PER; per-hub shortage weighting; per_route_max_send_fraction cap.
- [robustness tier] scenario-pool domain randomization for Surge/SP/Obstacle.
