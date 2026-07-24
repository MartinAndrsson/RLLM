# RLLM

LLM-driven RL experimentation harness. Given a problem repo, an LLM (Claude) reads the code, proposes
action/state/reward/algorithm choices, runs experiments **coarse-to-fine** across machines, prunes what's
bad, suggests what to try next, and produces an overview table + beamer slides. A second model (Codex)
reviews before anything commits/executes. First target problem: the Rx rollbox sim.

**See [DESIGN.md](DESIGN.md) for the architecture.** This is early scaffolding (Phase 1: the loop, no LLM yet).

## Layout
```
rllm/            core library (general, problem-agnostic)
  interfaces.py    ProblemAdapter ABC + Fidelity/ExperimentSpec/RunResult
  registry.py      durable JSON record of specs + results (shared-FS, resume-safe)
  dispatcher.py    NFS-queue multi-machine claim/run/report + worker loop
  ladder.py        coarse-to-fine (screen -> refine -> confirm) promotion
  cli.py           `python -m rllm.cli ...` (Phase-1 entrypoint)
adapters/rx/     first adapter: wraps Supply/Rx_dan/supply run_tqc_deep.sh knobs
  rllm_work/       per-problem MEMORY (read first): problem.md, journal.md, registry/, queue/, runs/
slides/          beamer template for generated reports
```
Per-problem persistent memory (problem.md = understanding + realism; journal.md = tried/ideas) is read
FIRST on every launch — resume, never restart.

## Phase-1 usage (MVP loop, no LLM)
```bash
WD=adapters/rx/rllm_work
python -m rllm.cli seed   $WD                 # enqueue the screen wave (Rx candidate experiments)
python -m rllm.cli work   $WD --device 0 --dry-run   # (per machine) claim+run jobs; --dry-run = no training
python -m rllm.cli status $WD                 # queue + ranking + SOLVED flags
python -m rllm.cli promote $WD --from screen --to refine --top-k 3
```
Real runs: drop `--dry-run` and run `work` on each machine (they share the NFS queue). Each job shells
out to `run_tqc_deep.sh` in the Rx repo via the psim env.

## Status / roadmap (DESIGN.md build order)
1. **[in progress] MVP loop on Rx, no LLM** — interfaces, Rx adapter, dispatcher, registry, ladder, memory.
2. LLM propose + report (`claude -p`).  3. Codex reviewer.  4. LLM onboarding of a new problem.
5. Adaptive pruning + proxy-validity guard.  6. Bounded autonomous `solve` session (schedule-aware).
