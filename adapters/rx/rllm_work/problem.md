# problem.md — Rx rollbox redistribution (RLLM understanding, living doc)

READ FIRST on every launch. Seeded from the user's brief; maintained by the LLM as understanding grows.

## Goal (the "solved" definition — TIERED)
Running out of Rx rollboxes is unacceptable in reality, so:
1. **Primary:** Normal (training) operations — `success_fraction -> 1.0` (a hub never runs out over the
   episode), confirmed over multiple seeds.
2. **Then robustness:** lift the OTHER test scenarios that exist in the problem (Surge/lull, SP/
   swisspost-yearly, Obstacle run) — DR-style, without sacrificing the primary.
3. **Fallback:** if 1.0 is genuinely unreachable, the goal is the **best achievable** (recognised as a
   plateau — no further improvement over a window — at the session level, not from one run).

## What the agent controls / observes / is rewarded (current)
- **Action:** per-route `available_fraction` redistribution of rollboxes across 8 hubs (outbound <= stock).
- **State:** `history` layout — RFID-observed availability + forecast path + per-hub history features.
- **Reward:** `-shortage_penalty*shortage/scale - movement - paid - excess + survival_bonus` (survival =
  binary +1/day at zero shortage, or a smooth ramp). Algo = TQC (sb3_contrib).

## Modeling assumptions & realism (OWN these, revisit them)
- **Episode = 100 sim-days.** Open question: is 100 days representative of the real (yearly) operation?
  The SP scenario is 365 days — a different profile, so it's NOT direct evidence about horizon length.
  Decide what horizon actually matters before trusting 100-day "solved".
- **Train == eval conditions:** RFID inefficiency 0.1, forecast 2%. (RFID makes it a POMDP.)
- **Fidelity/speed knobs that are SAFE (don't change realism):** training length (`MAXGEN`), #seeds,
  eval frequency/episodes. Episode length / problem size are NOT safe fidelity knobs (they change the
  task) — used only if their effect on the ranking is validated.
- **How much data is enough:** convergence is LATE (best checkpoints near gen 500 in prior runs) — do
  not assume early plateau; short screens can mislead (proxy-validity guard).

## Known-good baseline (start here, don't rediscover)
- **M1** (bootstrap fix: report horizon as truncation so SB3 bootstraps the value) took Normal-ops
  from 0% -> ~45% (n=2, seeds vary 30-60%). This is the current winner and the base for experiments.
- Sharper scale / smooth survival: ~neutral. Shorter forecast (fh=3): worse. Lower UTD: worse + barely
  faster. (Full detail lives in the Rx repo's RL.md; mirror key results into journal.md.)

## Open questions for the user (surface at end of session)
- Is a 100-day horizon acceptable as the realism target, or should we match the yearly operation?
- Priority weighting of the robustness scenarios vs pushing Normal-ops to 1.0?
