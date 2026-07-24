# RLLM — LLM-driven RL experimentation harness

A general tool that, given a *problem repo*, has an LLM (Claude, via this subscription) read the code,
propose action/state/reward/algorithm choices, run experiments across machines **compute-cheaply
(coarse-to-fine)**, prune what's clearly bad, suggest what to try next, and produce an overview table
+ beamer slides. First target problem: the Rx rollbox redistribution sim (`Supply/Rx_dan/supply`).

Design principles:
- **RLLM stays general**; each problem gets a small generated **adapter** so it stays self-contained.
- The LLM is the *researcher*; a proven mechanism does the *bookkeeping*.
- **Resume, never restart.** Every problem carries persistent notes the LLM reads FIRST on every launch
  (see §0), so relaunching — or launching onto a problem with prior work — continues from where it left
  off, not from scratch.
- **The LLM owns the modeling choices, not just the hyperparameters** — episode length, how much data
  is enough, whether the sim horizon is faithful to reality — and records its reasoning (see §0).

---

## Components

### 0. Per-problem memory (read FIRST on every launch — resume, never restart)
Each problem keeps a small set of living docs (in the problem repo, e.g. `<problem>/rllm_work/`) that
the LLM **always reads before doing anything**:
- **`problem.md`** — the LLM's evolving *understanding*: the goal, the chosen action/state/reward, and
  crucially the **modeling assumptions and their realism** — e.g. "episodes are 100 sim-days; is that
  representative of the real yearly operation? what horizon actually matters?", "how much training data
  is enough vs wasteful", "which sim knobs trade fidelity for speed and are they safe". Seeded at
  kickoff by the user's problem description, then maintained by the LLM as understanding improves.
- **`journal.md`** — the running log of what's been tried, outcomes, and the idea backlog (the per-problem
  analog of this project's `RL.md`); never deletes an idea, marks it tried with its result.
- **the experiment registry** — machine-readable config+result+rationale for every run.
On launch the LLM reconstructs state from these (what's done, what worked, what's queued, open questions)
and continues. This is what makes relaunch / hand-off / weekend-continuation work without repeating work.

### 1. Problem onboarding (LLM scans code → writes an adapter)
`rllm onboard <problem_repo>`:
1. Claude (headless) reads the problem's code to identify: the **simulator/env**, the natural
   **action space** (what the agent controls), the **observation/state** (what's available), the
   **reward / success signal**, and the **tunable knobs** (episode length, problem size, etc.).
2. It generates a **ProblemAdapter** implementing RLLM's standard interface (below) and writes it
   INTO the problem repo (e.g. `<problem_repo>/rllm_adapter/`) along with any run/eval scripts needed.
   RLLM provides the base classes/helpers the adapter imports; the problem repo keeps the specifics.
3. It proposes **variants to search**, not one fixed choice: candidate action spaces, state layouts,
   reward functions, and algorithms — because choosing these IS part of the experiment.
   It also drafts the initial **`problem.md`** (§0): its understanding + the modeling-realism questions
   (episode horizon vs reality, data-sufficiency, fidelity/speed trade-offs) it intends to resolve.
4. A **reviewer model reviews the adapter before anything runs** (see §7a) — by default Codex, not a
   human. Onboarding is the one heavy step.

### 2. Standard interface (`rllm/interfaces.py`)
A `ProblemAdapter` exposes: `make_env(config)`, the declared action/state/reward **variants**, the
**fidelity knobs** (see §4), a `metrics()` contract (what each run emits), and a `success_metric`.
Anything speaking this interface can be orchestrated — Rx is just the first implementation.

### 3. Technique / knob catalog (`techniques.yaml`)
Machine-readable version of a matured `RL.md`: general RL techniques (bootstrap, n-step, PER,
recurrent-for-POMDP, DroQ/LayerNorm, MPC, AWAC…), hyperparameter axes, and — via the adapter —
problem-specific axes (reward shaping, action/state variants). Each entry carries a hypothesis,
a cost hint, and prerequisites. This is the search space the controller draws from, and results
feed back into it (the doc becomes self-updating — the "walk through RL.md" idea, closed-loop).

### 4. Coarse-to-fine multi-fidelity search (`rllm/search.py`) — the compute-smart core
NOT full-length Optuna trials. A **fidelity ladder** using problem-agnostic cheap→expensive knobs the
adapter exposes (for Rx: `n_endpoints`, `n_days`, `max_generations`, eval frequency/episodes, seeds):
- **Rung 0 — screen (seconds–minutes):** small size, short episodes, few gens, cheap eval, 1 seed.
  Rank MANY configs cheaply.
- **Rung 1 — refine:** medium size/length/gens, 2 seeds. Promote top-k from rung 0.
- **Rung 2 — confirm:** full fidelity, more seeds. Only the few survivors.
Successive-halving / Hyperband in spirit, BUT the **LLM picks what enters rung 0** (from the catalog +
code understanding), reads rung results **semantically** (convergence status, curve shape — not just a
scalar), decides promotions, and proposes new directions. Optuna/ASHA can drive sampling *within* a
rung; the LLM drives *across* rungs and directions.
- **Proxy-validity guard (critical):** cheap-fidelity rankings must correlate with full fidelity, or
  screening is worse than useless. Periodically run a full-fidelity confirmation of a screened winner;
  if the ranking doesn't hold, widen/adjust the proxy. (Real cautionary tales from Rx: a config that
  wins at short forecast horizon can lose at full; and best checkpoints appeared as late as gen 495 —
  so cheap/short proxies can systematically mislead.)

### 5. Dispatcher (`rllm/dispatcher.py`) — multi-machine
Queue on the shared filesystem: experiments are claim-able job files; each machine's agent atomically
claims the next, runs it via the adapter's runner, writes results back. No central server (you already
share NFS). Enforces a **global compute budget**.

### 6. Monitor / pruner (`rllm/monitor.py`)
Conservative early-kill: only stop runs that are **dominated or diverging**, never "slow/flat" ones
(the gen-495 late-bloomer would be wrongly killed otherwise). Reuses the convergence diagnostics we
already built (`analyze_convergence`-style: best@gen, tail-slope, IMPROVING/converged/past-peak).

### 7. LLM controller (`rllm/llm/controller.py`)
Claude Agent SDK / `claude -p`, subscription auth, headless. Roles: **onboard**, **propose** next batch,
**prune** (flag candidates; hard actions budget-gated), **report**. Runs **periodically** on a small
digest of run states (NOT continuous polling — respect rate limits). Every LLM decision is logged with
its rationale (reproducibility).

### 7a. Two-model actor / reviewer loop (Codex reviews Claude, no human in the inner loop)
Instead of a human gate, a **second, independent model reviews the first's output.** Default split:
- **Actor = Claude** (Agent SDK / `claude -p`): RL research/planning, generates adapters + experiment
  batches, reads results, proposes prune/promote decisions, writes the report.
- **Reviewer = Codex** (`codex exec`, that account's auth): independently reviews before commit/execute —
  sanity-checks generated adapter/script CODE, and adversarially critiques experiment/prune/promote
  decisions ("this kill looks premature", "this proxy won't transfer").
Protocol: actor emits an artifact + rationale → reviewer returns a structured verdict
(`approve | revise(reasons) | block(reasons)`) → actor revises or proceeds. Both auth via their own
CLIs/subscriptions. Why: cross-model review reduces (not eliminates) single-model blind spots — most
valuable exactly where autonomy is scariest (running generated code, killing runs, big launches).
Caveat: two models can still be jointly wrong or "agree" on a mistake, so hard budgets (§9) remain the
real safety net, not the review.

### 9. Autonomous session — the "kick it off overnight / over the weekend" run
`rllm solve <problem_repo> --until "09:00" --brief brief.md` (or `--budget 13h`): human starts it and
walks away; it reads §0 memory, (onboards if new), then loops actor↔reviewer over the coarse-to-fine
ladder, escalating fidelity on winners, until the **success criterion**, the **time window**, or a
budget is hit.

**Kickoff spec (from the user):** a short problem **brief** (goal, constraints, any realism notes —
folded into `problem.md`) and the **available wall-clock window**, expressed as a duration or a
finish-by time. Weeknight "leave 18:00 / back 09:00" → ~13–15h; "Fri eve → Mon morning" → a long run.

**Graceful wind-down (roughly-on-time, not hard-killed):** the session budgets against the finish time —
in the last slice it **stops launching new long runs**, lets in-flight runs finish or checkpoint,
updates §0 memory, and writes the handoff. So you come back to a clean, reviewable state, not a job
severed mid-thought.

**End-of-session handoff (for the human to read before relaunch):** a report of what was done + current
best + updated table/slides, AND an explicit **"requests for next session"** — e.g. "needs more time to
confirm cell X over more seeds", "clarify whether a 100-day horizon is acceptable for reality",
"decide between reward variant A vs B". The next launch (with the user's answers) resumes from §0 memory.

Requires, up front:
- **A concrete "solved" definition** — e.g. Normal-ops success ≥ target at FULL fidelity, confirmed over
  K seeds. Without it, "until solved" burns 50h chasing seed noise. Include a **plateau/stall detector**
  (no full-fidelity improvement over a window → stop and report).
- **Hard budgets (the real safety net):** wall-clock (50h), GPU-hours, max concurrent runs, LLM
  token/call caps, and a cap on actor↔reviewer back-and-forth (two bots can loop expensively).
- **Execute-gates:** generated code runs only after (a) reviewer approve AND (b) a dry-run/validation
  passes. Irreversible/expensive actions (writing to the problem repo, launches above a per-step
  GPU-hour cap) either need the reviewer's explicit sign-off or are refused.
- **Full decision log:** every actor proposal + reviewer verdict + rationale, timestamped, so the 50h is
  auditable and the final report explains *why* each path was taken/dropped.
Output at the end: the overview table + beamer deck + updated `techniques.yaml` (what worked/didn't).

### 8. Reporting (`rllm/report.py`, `slides/`)
Overview table (à la `compare_runs.py`) + beamer deck generated from `slides/template_beamer_slides.tex`:
what was tested, how it performed, what's next. LLM writes the narrative; code fills the numbers/plots.

---

## Repo skeleton (proposed)
```
RLLM/
  DESIGN.md                     # this file
  README.md                     # quickstart
  techniques.yaml               # machine-readable technique/knob catalog
  rllm/
    interfaces.py               # ProblemAdapter ABC + metrics contract
    registry.py                 # experiment registry (config+result+LLM rationale, logged)
    search.py                   # coarse-to-fine fidelity ladder + promotion logic
    dispatcher.py               # NFS-queue multi-machine claim/run/report
    monitor.py                  # conservative convergence-based pruning
    report.py                   # table + beamer generation
    llm/
      controller.py             # Claude loop: onboard / propose / prune / report
      prompts/                  # prompt templates
  adapters/
    rx/                         # first adapter (wraps the existing run_tqc_deep infra)
  slides/
    template_beamer_slides.tex  # (already here)
```
The generated per-problem adapter lives in the PROBLEM repo (`<problem>/rllm_adapter/`), not here.

---

## Build order (incremental; prove the loop before scaling)
1. **MVP loop on Rx (no LLM yet):** interfaces + Rx adapter (wrap `run_tqc_deep.sh` knobs) + NFS
   dispatcher + registry + the §0 memory files (`problem.md`/`journal.md`, hand-written for now) +
   reuse `analyze_convergence`/`compare_runs`. Run a hand-written rung-0→2 ladder. Goal: prove
   multi-fidelity + multi-machine + resume-from-memory end-to-end.
2. **LLM propose + report:** `claude -p` reads registry, proposes the next batch from techniques.yaml,
   generates the table + beamer. Human approves launches.
3. **Add the reviewer model (Codex):** actor↔reviewer verdict protocol on generated code + decisions;
   replace the human approve-gate with reviewer + hard budgets. Validate the review actually catches
   seeded-in bad adapters/decisions before trusting it.
4. **LLM onboarding:** point it at a fresh problem repo → it writes an adapter (reviewer-checked).
   Validate on a 2nd problem.
5. **Adaptive pruning + proxy-validity guard:** conservative auto-kill + periodic full-fidelity checks.
6. **Autonomous `solve` session (§9):** wire the ~50h bounded loop with the success criterion, hard
   budgets, execute-gates, and decision log. Do a SHORT dry run (e.g. 2h budget) before a full 50h one.

## Open risks (carry from the Rx experience)
- Cheap-proxy rankings may not transfer to full fidelity → proxy-validity guard is mandatory.
- LLM pruning can kill late-bloomers → conservative, dominated-only kills.
- Seed variance is large → confirmation needs multiple seeds; single-seed = screen only.
- LLM usage limits → periodic digest-based calls, budget caps.
- **Autonomy-specific (the 50h run):** runaway GPU/token cost (→ hard budgets are the real safety net,
  not the review); "solved" undefined (→ require a full-fidelity success criterion + stall detector);
  running LLM-generated code unattended (→ reviewer approve + dry-run gate before any script executes);
  two models jointly wrong / agreeing on a mistake (→ budgets + audit log, and a human skim of the final
  report). Recommend a short (~2h) dry run before committing a full 50h session.
