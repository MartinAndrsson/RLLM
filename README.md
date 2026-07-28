# RLLM

LLM-driven RL experimentation harness. Given a problem repo, an LLM (Claude) reads the code, proposes
action/state/reward/algorithm choices, runs experiments **coarse-to-fine** across machines, prunes what's
bad, suggests what to try next, and produces an overview table + beamer slides. A second model (Codex)
reviews before anything commits/executes. First target problem: the Rx rollbox sim.

**See [DESIGN.md](DESIGN.md) for the architecture** and [implementations.md](implementations.md) for the
hardening plan. The actor/reviewer loop is live end-to-end (claude proposes, codex reviews); training
itself is still launched by hand.

## Layout
```
rllm/            core library (general, problem-agnostic)
  interfaces.py    ProblemAdapter ABC + Fidelity/KnobSpec/ExperimentSpec/RunResult
  validate.py      THE authorization layer: knob whitelist, slug/type/range checks, task-change gate
  registry.py      durable JSON record of specs + results (shared-FS, resume-safe)
  dispatcher.py    NFS-queue multi-machine claim/run/report + worker loop
  ladder.py        coarse-to-fine (screen -> refine -> confirm) promotion
  llm/             actor/reviewer loop: backend.py (claude/codex CLIs), prompts.py, controller.py
  cli.py           `python -m rllm.cli ...`
adapters/rx/     first adapter: wraps Supply/Rx_dan/supply run_tqc_deep.sh knobs
  rllm_work/       per-problem MEMORY (read first): problem.md, journal.md, registry/, queue/, runs/
tests/           the gate tests (`python -m pytest tests -q`) — run these before any unattended session
slides/          beamer template for generated reports
```
Per-problem persistent memory (problem.md = understanding + realism; journal.md = tried/ideas) is read
FIRST on every launch — resume, never restart.

## Usage

**[START_HERE.md](START_HERE.md)** is the entry point. In short:

```bash
./scripts/rllm-onboard.sh                     # the interview -> <work_dir>/brief.json  (--rx to prefill)
python -m rllm.cli solve <work_dir> --device 0 --until 8h    # one bounded session -> handoff.md
python -m rllm.cli handoff <work_dir>         # what was learned, and what to do next
```

`solve` is the whole loop: plan a wave → run it → promote what survives → repeat, until the wind-down
point, then write the handoff. It escalates on purpose — while most of the window remains it prefers a
new cheap screening wave, and past the halfway mark it prefers promoting proven candidates, so early
wall-clock buys breadth and late wall-clock buys confidence. A promotion is only started when its
conservative duration estimate fits before the deadline.

The individual steps are still runnable by hand (this is what `solve` orchestrates):
```bash
WD=adapters/rx/rllm_work
python -m rllm.cli validate $WD                # brief + adapter self-check, knob whitelist
python -m rllm.cli ask --backend codex         # smoke-test a CLI backend (needs the logged-in CLI)
python -m rllm.cli propose $WD -n 4            # one LLM wave; enqueues, never trains
python -m rllm.cli work    $WD --device 0 --dry-run   # render commands, consume nothing
python -m rllm.cli work    $WD --device 0      # (per machine) claim + actually train
python -m rllm.cli status  $WD                 # brief, queue, rankings, sessions
python -m rllm.cli promote $WD --from screen --to refine --top-k 3
```
Multi-machine: run `work` (or `solve`) on each machine — they share the NFS queue. `propose` only ever
*enqueues*, so the review gate always precedes GPU time.

## Onboarding a new problem
No Python needed. The interview asks for a repo, a command that runs one evaluation, and the JSON file
that command writes; `adapters/brief_adapter.py` turns those answers into runs. The only fiddly part is
declaring which knobs may be tuned, with types and ranges — copy [docs/knobs.example.json](docs/knobs.example.json),
or use the prompt in START_HERE.md to have a model draft it from the repo. The built-in Rx adapter
(`adapters/rx/`) is what a hand-written adapter looks like when the problem needs one.

## What authorizes what
Models advise; deterministic code authorizes (implementations.md §2.1). Every proposal passes
`rllm/validate.py` before it can influence a command line: the id must be a strict slug, every config
key must be a knob the adapter DECLARED, values are type/range/choice-checked, duplicates (same id, or
the same effective config under a new name) are refused, budget and eval-cadence knobs are
fidelity-owned, and `task_definition` knobs (for Rx: RFID_PROB, FORECAST_UNC, SCENARIO_POOL — they
change what "solved" means) stay human-gated even when both models agree. The reviewer can only ever
shrink the approved set; an absent or empty approval list approves nothing.

The session's limits are enforced the same way — outside the models. The deadline, the run cap and the
LLM-call cap live in `rllm/session.py` and `rllm/budget.py`; a model can only *request* more time, in
the handoff. Runs are launched in their own process group with a hard timeout, so nothing outlives the
window. "Solved" is only ever claimed from the brief's required fidelity over its required seed count;
everything cheaper is reported as an unconfirmed lead.

Backends run with no ability to act: claude with `--tools ""`, codex with `--sandbox read-only
--ephemeral`, prompts on stdin. Verified against claude 2.1.220 / codex-cli 0.145.0.

## Status / roadmap (DESIGN.md build order)
1. **[done]** MVP loop on Rx — interfaces, Rx adapter, dispatcher, registry, ladder, memory.
2. **[done]** LLM propose + Codex review, gated by the deterministic knob whitelist + `tests/`.
3. **[done]** Frozen problem brief, onboarding interview, generic brief-driven adapter.
4. **[done]** Bounded budgeted sessions with escalation, handoff report, resume.
5. **[next]** Live progress monitoring + conservative mid-run cancellation; statistical confirmation
   (confidence intervals, fresh confirmation seeds); proxy-validity guard; beamer report generation.

Known gaps before leaving it unattended overnight (implementations.md §15): there is no progress-based
cancellation of a bad run mid-flight, `~device-hours` is approximated by run wall-clock rather than
measured, the external Rx runner's git revision is not yet hashed into each RunSpec, and concurrency is
one run per worker process.
