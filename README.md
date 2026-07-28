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
  brief.py         the frozen per-problem contract (metric, success criterion, budget, guard rails)
  validate.py      THE authorization layer: knob whitelist, slug/type/range checks, task-change gate
  integrity.py     proof the problem repo was not modified by a run
  session.py       bounded escalating sessions; budget.py enforces the caps
  report.py        the handoff; compare.py the deliverable comparison table
  registry.py      durable JSON record of specs + results (shared-FS, resume-safe)
  dispatcher.py    NFS-queue multi-machine claim/run/report + worker loop
  ladder.py        coarse-to-fine (screen -> refine -> confirm) promotion
  llm/             actor/reviewer loop: backend.py (claude/codex CLIs), prompts.py, controller.py
  cli.py           `python -m rllm.cli ...`
adapters/
  brief_adapter.py generic adapter: runs variants, or a brief-declared test command
  rx/              hand-written adapter for Supply/Rx_dan/supply run_tqc_deep.sh knobs
  rx/rllm_work/    per-problem MEMORY (read first): brief.json, problem.md, journal.md, sessions/
problems/        THE DESIGNS the models write: <problem_id>/variants/<variant>/train.py
                 (observation + action space + reward, importing the problem repo read-only)
METHOD.md        standing research instructions, loaded into every prompt, hashed into every decision
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

## The designs live here, not in your repo
The observation, the action space and the reward are the models' design work. Each design is a directory
under `problems/<problem_id>/variants/<name>/` **in this repo**, importing the problem repo as a read-only
simulator and evaluating with the user's own test — contract in [problems/README.md](problems/README.md),
skeleton in `problems/_template/`. `VARIANT` becomes a declared enum knob whose choices are the designs
on disk, so a design is searched, gated and ranked exactly like a hyperparameter, and
`rllm.cli compare --by VARIANT` gives the design-against-design table.

The problem repo is never modified, and that is checked rather than trusted: `rllm/integrity.py`
fingerprints it (git revision + working-tree state, or a file walk) around every run and fails the run
**and the session** if anything moved.

## Onboarding a new problem
No Python needed for the harness itself. The interview asks for the repo, the deterministic test, the
metric and the guard rails; `adapters/brief_adapter.py` turns those answers into runs — either by running
a variant, or (simplest case) by running a `test_command` the repo already has. Copy
[docs/knobs.example.json](docs/knobs.example.json) for the knob declarations, or use the prompt in
START_HERE.md to have a model draft them and a first variant from a paragraph of description. The
built-in Rx adapter (`adapters/rx/`) is what a hand-written adapter looks like when a problem needs one.

## What authorizes what
Models advise; deterministic code authorizes (implementations.md §2.1). Every proposal passes
`rllm/validate.py` before it can influence a command line: the id must be a strict slug, every config
key must be a knob the adapter DECLARED, values are type/range/choice-checked, duplicates (same id, or
the same effective config under a new name) are refused, budget and eval-cadence knobs are
fidelity-owned, and `task_definition` knobs (for Rx: RFID_PROB, FORECAST_UNC, SCENARIO_POOL — they
change what "solved" means) stay human-gated even when both models agree. The reviewer can only ever
shrink the approved set; an absent or empty approval list approves nothing.

The session's limits are enforced the same way — outside the models. The deadline, the run cap and the
LLM call/token/spend caps live in `rllm/session.py` and `rllm/budget.py`; a model can only *request* more,
in the handoff. Token and cost accounting is **measured, not estimated**: `claude -p --output-format json`
returns usage and price in the same envelope as the answer, and codex reports a token count, so
`maximum_llm_tokens` / `maximum_llm_cost_usd` in the brief protect the subscription allowance directly.
Headroom is reserved on every axis so the handoff always gets written. Where a backend reports no price
(codex), the spend is labelled a lower bound rather than silently treated as free.

**Model tiering** (default on): screening proposals run on the cheap tier, while review and the handoff
write-up stay on the strongest model — a proposal is a short structured object that a strong reviewer then
attacks, so that is where the cheap tier is safe. It is a policy, not the model's own judgement; override
with `--propose-model` or turn it off with `--same-tier`. Runs are launched in their own process group with a hard timeout, so nothing outlives the
window. "Solved" is only ever claimed from the brief's required fidelity over its required seed count;
everything cheaper is reported as an unconfirmed lead.

Backends run with no ability to act: claude with `--tools ""`, codex with `--sandbox read-only
--ephemeral`, prompts on stdin. Verified against claude 2.1.220 / codex-cli 0.145.0.

## Status / roadmap (DESIGN.md build order)
1. **[done]** MVP loop on Rx — interfaces, Rx adapter, dispatcher, registry, ladder, memory.
2. **[done]** LLM propose + Codex review, gated by the deterministic knob whitelist + `tests/`.
3. **[done]** Frozen problem brief, onboarding interview, generic brief-driven adapter.
4. **[done]** Bounded budgeted sessions with escalation, handoff report, resume.
5. **[done]** METHOD.md as standing instructions; variants (model-authored observation/action/reward)
   as a searchable dimension; read-only enforcement of the problem repo; comparison tables.
6. **[next]** The variant *authoring* loop — a model writing a new design mid-session, reviewed by the
   second model before it can run. Today a variant is authored out-of-band (START_HERE.md prompt B) and
   the session only *selects between* the designs already on disk.
7. Live progress monitoring + conservative mid-run cancellation; statistical confirmation (confidence
   intervals, fresh confirmation seeds); proxy-validity guard; beamer report generation.

Known gaps before leaving it unattended overnight (implementations.md §15): no progress-based
cancellation of a bad run mid-flight; `~device-hours` approximated by run wall-clock rather than
measured; one run per worker process; the integrity guard detects modification after a run rather than
preventing it (a container or read-only mount is the next step up); and it is disabled for the Rx adapter
specifically, because `run_tqc_deep.sh` writes its outputs inside the problem repo — that adapter needs
its outputs redirected under the harness `run_dir` before the guard can be turned on for it.
