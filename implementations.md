# RLLM implementation plan

## 1. Purpose

RLLM is a bounded autonomous research system. Given:

- a problem repository;
- an initial problem description;
- a machine-checkable definition of "solved";
- realism and safety constraints;
- available machines and devices; and
- a wall-clock deadline and compute budget,

it uses two LLMs as research controllers and conventional RL implementations as the problem solvers.
One LLM acts as the researcher/actor and proposes experiments or control decisions. The other acts as
an independent reviewer. Deterministic code validates both models' output, runs approved experiments,
monitors them, and enforces all hard limits.

The LLMs are not themselves trained with RL in this design. They decide how RL should be used:
environment and observation choices, action and reward formulations, algorithms, hyperparameters,
fidelity, promotion, cancellation, and interpretation of results.

The target operating loop is:

```text
problem brief + memory + deadline
              |
              v
       actor proposal
              |
              v
 deterministic schema/policy validation
              |
              v
       reviewer verdict
              |
              v
 deterministic budget/execute gate
              |
              v
      immutable queued runs
              |
              v
 execution + heartbeats + metrics
              |
              v
 actor/reviewer promote, revise, or cancel
              |
              v
 full-fidelity holdout confirmation
              |
              v
 graceful wind-down + human handoff
```

This document defines the implementation required to make that loop safe, reproducible, general
across adapters, and usable for unattended overnight or weekend sessions.

## 2. Non-negotiable design rules

### 2.1 Models advise; deterministic code authorizes

Claude and Codex may propose and review actions. They may not:

- invent arbitrary environment-variable names;
- construct filesystem paths;
- construct shell fragments;
- exceed a budget;
- change the user's solved criterion;
- mark a low-fidelity result as solved;
- accept incomplete or invalid results; or
- override a deterministic safety failure.

Agreement between two models is not a security or correctness boundary. Every model-produced artifact
must pass a strict schema and deterministic policy checks.

### 2.2 All experiment and run records are immutable

An experiment describes a scientific hypothesis and base configuration. A run describes one exact
execution of that experiment at one fidelity and seed. Once written, neither is overwritten.

Changes create new records linked to their parents. Every result must remain attributable to:

- the exact problem-repository revision;
- the exact RLLM revision;
- the adapter revision;
- the effective configuration;
- the fidelity and seed;
- the command, environment, and working directory;
- the actor proposal and reviewer verdict; and
- the worker attempt that produced it.

### 2.3 Hard budgets are enforced outside the LLM loop

A session controller owns wall-clock, GPU-hour, run-count, concurrency, and LLM-call budgets. It
reserves budget before launching work and rejects actions that cannot fit. A separate deadline
watchdog remains effective even if both LLMs fail or hang.

### 2.4 Cancellation is conservative and recoverable

Mechanically invalid runs may be stopped automatically. Research-based cancellation normally requires
an actor proposal and reviewer approval. Slow or flat learning alone is not sufficient evidence,
especially for environments such as Rx where improvements can occur late.

Before cancellation, RLLM requests a checkpoint when the adapter supports it, records the reason and
evidence, and gives the process a termination grace period.

### 2.5 "Solved" requires independent full-fidelity evidence

Screen and refine results guide search; they cannot declare success. Success requires the fidelity,
number of fresh seeds, evaluation split, constraints, and statistical rule from the original problem
brief. Search seeds and final confirmation seeds must be disjoint.

## 3. Core data model

Add `rllm/models.py`. Use strict typed models with rejection of unknown fields. Pydantic v2 is
recommended because it provides strict validation and JSON Schema generation for LLM output. If the
project remains standard-library-only, implement equivalent explicit validators and committed JSON
Schema files.

### 3.1 ProblemBrief

```python
class ProblemBrief:
    problem_id: str
    description: str
    primary_metric: str
    success_criterion: SuccessCriterion
    constraints: list[MetricConstraint]
    realism_constraints: list[str]
    permitted_task_changes: list[str]
    forbidden_task_changes: list[str]
    session_budget: SessionBudget
```

The brief is created by the user or onboarding workflow and then frozen for a session. The LLM may
suggest amendments, but only a human can accept changes to the solved criterion or realism constraints.

### 3.2 MetricSpec and SuccessCriterion

```python
class MetricSpec:
    name: str
    direction: Literal["maximize", "minimize"]
    aggregation: Literal["mean", "median", "worst", "fraction"]
    finite_required: bool = True
    lower_bound: float | None = None
    upper_bound: float | None = None

class MetricConstraint:
    metric: str
    operator: Literal[">=", ">", "<=", "<", "=="]
    value: float

class SuccessCriterion:
    primary: MetricConstraint
    additional: list[MetricConstraint]
    required_fidelity: str
    minimum_training_seeds: int
    minimum_evaluation_samples: int
    require_fresh_confirmation_seeds: bool = True
    confidence_level: float | None = 0.95
```

For a finite Bernoulli success metric such as "no shortage occurred", report the observed success
fraction and a confidence interval. If the user defines success as zero failures across a fixed number
of trials, state that exact finite-trial claim; do not translate it into "will never fail".

### 3.3 KnobSpec

Every adapter declares all configurable knobs:

```python
class KnobSpec:
    name: str
    value_type: Literal["int", "float", "bool", "string", "enum"]
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] | None = None
    default: JsonValue
    category: Literal[
        "algorithm", "hyperparameter", "reward", "observation",
        "action", "training_budget", "evaluation", "task_definition"
    ]
    fidelity_safe: bool
    llm_may_change: bool
    requires: list[str] = []
    conflicts_with: list[str] = []
```

Unknown knobs are rejected. Values are type-checked and range-checked. A knob marked
`task_definition` requires explicit permission in `ProblemBrief.permitted_task_changes`, even if both
models approve it. Fidelity overrides may only use `fidelity_safe=True` knobs.

### 3.4 ExperimentSpec

The actor supplies a human-readable slug; RLLM generates the identifier.

```python
class ExperimentSpec:
    experiment_id: UUID
    slug: str
    hypothesis: str
    base_config: dict[str, JsonValue]
    parent_experiment_id: UUID | None
    proposal_decision_id: UUID
    created_at: datetime
```

Rules:

- `slug` must match `^[a-z][a-z0-9_-]{0,63}$`;
- IDs are generated by RLLM, never by an LLM;
- `base_config` never contains fidelity overrides;
- duplicate semantic configs are detected by a canonical configuration hash;
- an existing experiment record is never overwritten.

### 3.5 RunSpec

```python
class RunSpec:
    run_id: UUID
    experiment_id: UUID
    session_id: UUID
    fidelity: str
    seed: int
    effective_config: dict[str, JsonValue]
    effective_config_sha256: str
    problem_git_revision: str
    problem_dirty_diff_sha256: str | None
    rllm_git_revision: str
    adapter_revision: str
    estimated_wall_seconds: float
    estimated_gpu_hours: float
    created_at: datetime
```

`effective_config` is computed once as:

```python
effective = adapter.defaults | experiment.base_config | fidelity.overrides
```

The stored experiment is never mutated and old fidelity overrides are never carried to a new rung.
The queue references `run_id`; the worker loads the immutable `RunSpec`, verifies its hash, and never
looks up a mutable experiment configuration.

### 3.6 CommandSpec

Replace shell-string construction with:

```python
class CommandSpec:
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    output_dir: Path
    timeout_seconds: float
    checkpoint_signal: int | None = signal.SIGUSR1
    terminate_grace_seconds: float = 120
```

The dispatcher calls `subprocess.Popen()` with `shell=False`, `cwd=...`, `env=...`, and
`start_new_session=True`. It must not invoke `bash -c`. Environment keys must come from the adapter's
declared mapping, not directly from LLM strings.

### 3.7 ProgressReport and RunResult

```python
class ProgressReport:
    run_id: UUID
    attempt: int
    step: int
    maximum_step: int | None
    metrics: dict[str, float]
    checkpoint_path: str | None
    observed_at: datetime

class RunResult:
    run_id: UUID
    attempt: int
    status: Literal[
        "completed", "failed", "cancelled", "timed_out", "lost"
    ]
    metrics: dict[str, JsonValue]
    wall_seconds: float
    gpu_hours: float
    exit_code: int | None
    failure_kind: str | None
    failure_message: str | None
    completed_at: datetime
```

An adapter parse error is a failed run, never a completed run containing an `"error"` metric.
Non-finite required metrics fail validation.

## 4. ProblemAdapter v2

Update `rllm/interfaces.py` so the core remains problem-agnostic:

```python
class ProblemAdapter(ABC):
    name: str

    def knob_specs(self) -> dict[str, KnobSpec]: ...
    def metric_specs(self) -> dict[str, MetricSpec]: ...
    def fidelity_levels(self) -> list[Fidelity]: ...
    def validate_config(self, config: dict) -> ValidationReport: ...
    def build_command(self, run: RunSpec, run_dir: Path, device: str) -> CommandSpec: ...
    def read_progress(self, run_dir: Path) -> ProgressReport | None: ...
    def parse_result(self, run_dir: Path) -> RunResultPayload: ...
    def estimate_cost(self, run: RunSpec, history: CostHistory) -> CostEstimate: ...
    def summarize_for_research(self, results: list[RunResult]) -> dict: ...
```

Generic ranking uses `MetricSpec.direction` and the problem brief. Adapters may provide a structured
multi-objective score, but they may not silently replace the user's success criterion.

For Rx specifically:

- replace absolute constants for repository and Python paths with adapter configuration;
- map allowed knobs explicitly to environment keys;
- write all outputs below the RLLM `run_dir`, or store a unique direct link to the external output;
- never glob a global directory and choose the lexicographically latest match;
- record the Rx Git revision and dirty-diff hash;
- require the external runner to use `set -euo pipefail`;
- make its output directory an explicit argument;
- make progress available in a stable JSONL schema; and
- handle checkpoint/termination signals.

## 5. Durable registry and filesystem layout

Refactor `rllm/registry.py` around append-only records:

```text
rllm_work/
  problem.md
  journal.md
  brief.json
  sessions/
    <session_id>/
      session.json
      budget_ledger.jsonl
      events.jsonl
      handoff.md
  registry/
    experiments/<experiment_id>.json
    runs/<run_id>.json
    results/<run_id>__attempt_<n>.json
    decisions/<decision_id>.json
  queue/
    pending/<run_id>.json
    claimed/<run_id>.json
    running/<run_id>.json
    cancel_requested/<run_id>.json
    completed/<run_id>.json
    failed/<run_id>.json
    cancelled/<run_id>.json
  runs/
    <run_id>/
      run.json
      stdout.log
      stderr.log
      progress.jsonl
      result.json
      checkpoints/
```

Every write uses an atomic temporary file plus `os.replace`. Creation of immutable records uses
exclusive creation and fails if the target already exists. Add a schema version to all JSON records.

Do not use model-provided strings in paths. Only internal UUIDs may name registry, queue, and run files.
After resolving a path, verify that it remains below the expected root.

Every event includes an ISO-8601 UTC timestamp, session ID, run ID when relevant, host, process ID, and
event type.

## 6. Queue, leases, and worker execution

Refactor `rllm/dispatcher.py` into an explicit state machine:

```text
pending -> claimed -> running -> completed
                         |-----> failed
                         |-----> timed_out
                         |-----> cancel_requested -> cancelled
                         |-----> lost -> pending (new attempt)
```

### 6.1 Claiming

Claim a pending file with an atomic same-filesystem rename. The claim record contains:

- worker ID, host, PID, and device;
- attempt number and random attempt token;
- claimed timestamp;
- lease expiry timestamp; and
- immutable RunSpec hash.

The worker moves the job to `running/` only after loading and validating its RunSpec, reserving its
device, opening logs, and successfully starting the child process.

All preparation is inside exception handling. Any failure produces a structured failed result and
moves the queue record out of `claimed/`.

### 6.2 Heartbeats and stale claims

Workers update a heartbeat atomically, for example every 30 seconds. The coordinator may mark a claim
lost after at least three missed heartbeat intervals plus an NFS clock-skew margin.

A lost run may be requeued with `attempt + 1` up to a configured retry limit. Results are accepted only
from the current attempt token, preventing a late stale process from overwriting a newer result.

### 6.3 Execution

The worker:

1. validates the RunSpec and budget reservation;
2. obtains the structured CommandSpec;
3. records the exact command and environment with secrets redacted;
4. launches a new process group with `shell=False`;
5. streams stdout and stderr to per-run files;
6. publishes heartbeats and parsed progress;
7. watches for cancellation and deadline requests;
8. checks the process timeout;
9. parses and validates final metrics; and
10. writes exactly one terminal result for its attempt.

### 6.4 Retries

Classify failures:

- `configuration`: never retry automatically;
- `parse_or_schema`: never retry without adapter correction;
- `oom`: retry at most once only if an approved deterministic fallback exists;
- `worker_lost`: retry within budget;
- `transient_io`: retry with bounded exponential backoff;
- `training_failure`: return to actor/reviewer for a decision.

Retries consume budget and are fully logged.

## 7. Actor/reviewer protocol

Add `rllm/llm/protocol.py` and committed JSON Schemas under `rllm/schemas/`.

### 7.1 Proposal artifact

The actor returns one of:

- `propose_experiments`;
- `promote`;
- `cancel_runs`;
- `request_more_evidence`;
- `declare_stall`;
- `recommend_success`;
- `request_user_input`.

Every proposal includes:

```json
{
  "schema_version": 1,
  "action": "propose_experiments",
  "rationale": "Concise evidence-based explanation",
  "evidence_run_ids": ["..."],
  "estimated_value": "high|medium|low",
  "experiments": [
    {
      "slug": "nstep_3",
      "hypothesis": "Three-step returns propagate delayed shortage signals faster.",
      "base_config": {"TD_N_STEPS": 3}
    }
  ]
}
```

Reject unknown fields and references to unknown runs. Normalize and validate configs before asking the
reviewer, so the reviewer spends time on a scientifically plausible artifact rather than malformed
JSON.

### 7.2 Reviewer verdict

The reviewer returns:

```json
{
  "schema_version": 1,
  "verdict": "approve|revise|block",
  "approved_item_ids": ["..."],
  "reasons": ["..."],
  "required_changes": ["..."],
  "risk_flags": ["proxy_transfer", "insufficient_seeds"]
}
```

An empty approved list means approve nothing; it must never default to all items. The reviewer cannot
introduce new experiments inside a verdict. New ideas go back through an actor proposal.

### 7.3 Independence and context

The reviewer receives:

- the frozen problem brief;
- relevant problem memory;
- the adapter's declared knobs and metrics;
- the proposal artifact;
- deterministic validation output;
- compact raw evidence and uncertainty, not just rankings;
- remaining budget and deadline; and
- known proxy-validity and convergence warnings.

It does not receive hidden actor reasoning. It receives the actor's explicit rationale because that is
part of the auditable proposal.

Require different backend/model identities for unattended sessions unless the user explicitly permits
same-model review. Bound revision loops, initially to two actor revisions per decision.

### 7.4 Model execution safety

Invoke review agents in read-only and ephemeral modes. For Codex, use the installed CLI's equivalent of:

```text
codex exec --sandbox read-only --ephemeral --output-schema <schema> -C <problem_repo> <prompt>
```

Use stdin rather than a command-line argument for large prompts. Configure Claude equivalently with
write and execution tools disabled unless a later onboarding phase explicitly needs them.

Treat repository files and memory as untrusted prompt data. Model sandboxing and deterministic output
validation remain required even when both models agree.

### 7.5 Decision log

Each decision record contains:

- exact prompt-template revision;
- actor and reviewer backend/model/version;
- input artifact hashes;
- raw model outputs;
- parsed artifacts;
- validation reports;
- verdict and approved subset;
- revision count;
- budget before and after;
- resulting run IDs or cancellation requests; and
- timestamps and durations.

If a model call or parser fails, log that failure too. The log must cover unsuccessful decisions, not
only accepted proposals.

## 8. Generic ranking, aggregation, and promotion

Replace Rx-specific aggregation in `rllm/ladder.py`.

### 8.1 Rung completeness

An experiment is eligible for ranking only when:

- every required run for that fidelity has a terminal state;
- the required number of successful seeds is present;
- all required metrics validate;
- no result comes from a stale attempt; and
- the run configurations match the expected effective-config hashes.

Failed seeds make the experiment incomplete or failed according to the problem policy; they are not
silently dropped from the mean.

### 8.2 Aggregation

Aggregate according to `MetricSpec`. Store:

- per-seed values;
- center estimate;
- dispersion;
- confidence interval when meaningful;
- number of training seeds and evaluation samples;
- failures and missing values; and
- constraint satisfaction.

Ranking first respects hard metric constraints, then the primary metric direction, then declared
secondary metrics. Do not hardcode `success_fraction` or `mean_shortage_days`.

### 8.3 Seeds and holdout

Use disjoint groups:

- screen search seeds;
- refine search seeds;
- confirm holdout seeds; and
- optional final audit seeds that are not exposed during adaptive search.

The session may reuse checkpoints for computational continuation, but statistical reporting must label
which seeds influenced selection. Final claims use fresh evidence.

### 8.4 Promotion

Promotion creates new immutable RunSpecs from the original base experiment config plus the destination
fidelity overrides. It does not copy the source effective config.

Mechanical top-k promotion is allowed only when the proxy is trusted and the rung is complete.
Otherwise, the actor proposes promotion with uncertainty and proxy warnings, and the reviewer decides.

### 8.5 Proxy-validity guard

Track configurations evaluated at both cheap and full fidelity. Measure:

- rank correlation;
- top-k winner recall;
- constraint-agreement rate; and
- important ranking reversals.

Until enough paired observations exist, mark the screen proxy `uncalibrated`. While uncalibrated:

- do not cancel solely from screen performance;
- promote at least one exploration candidate outside top-k; and
- periodically run a full-fidelity sentinel.

If correlation or winner recall falls below configured thresholds, widen the screen budget or disable
mechanical screening and record the change in `problem.md` and the session report.

## 9. Monitoring and cancellation

Add `rllm/monitor.py` and `rllm/cancellation.py`.

### 9.1 Progress contract

Adapters emit monotonic progress steps and comparable metrics. A monitor stores the raw curve and a
summary containing:

- fraction of planned training completed;
- best metric and step;
- recent slope and uncertainty;
- distance to the current reference;
- non-finite values or divergence indicators;
- resource consumption rate;
- last checkpoint; and
- whether the run remains comparable to its peers.

The raw curve remains available to both models; semantic summaries do not replace evidence.

### 9.2 Cancellation classes

#### Class A: deterministic cancellation

No LLM approval is required for:

- non-finite loss or metrics repeated beyond an adapter threshold;
- an invalid action/output invariant;
- a dead child process;
- corrupt progress output;
- hard wall-clock or GPU-hour exhaustion;
- explicit user cancellation; or
- a process exceeding its approved timeout.

Record the exact rule and evidence.

#### Class B: research cancellation

Requires actor proposal and reviewer approval:

- statistically dominated by a comparable run;
- an ablation invalidated its own hypothesis;
- excessive projected cost for negligible attainable benefit; or
- a proxy check shows the run is solving the wrong task.

Initial conservative requirements:

- same problem, fidelity, metric contract, and comparable evaluation conditions;
- minimum adapter-defined progress fraction;
- at least three post-warmup evaluation points;
- candidate's optimistic bound is worse than the reference's pessimistic bound by a configured margin;
- no compensating constraint or secondary-metric advantage;
- cancellation fits the known convergence behavior; and
- a usable checkpoint is requested first when supported.

#### Class C: protected from research cancellation

Do not cancel merely because:

- the curve is flat or noisy;
- the run starts below the current leader;
- only one seed is available;
- a cheap proxy ranks it poorly while proxy validity is uncalibrated;
- the run tests a deliberately diverse exploration direction; or
- the environment is documented to have late improvements and the minimum progress threshold is unmet.

### 9.3 Cancellation mechanics

The coordinator writes an immutable cancellation decision and atomically creates a
`cancel_requested/<run_id>.json` request. The worker:

1. verifies the request targets its current attempt token;
2. asks for a checkpoint using the adapter's checkpoint signal or hook;
3. waits for checkpoint acknowledgement up to a bounded interval;
4. sends `SIGTERM` to the entire process group;
5. waits `terminate_grace_seconds`;
6. sends `SIGKILL` only if needed;
7. parses any partial metrics;
8. records `cancelled`, not `failed`; and
9. releases reserved resources.

The CLI exposes `rllm cancel <run_id> --reason ...` for human cancellation using the same mechanism.

## 10. Session controller and deadlines

Add `rllm/session.py`, `rllm/scheduler.py`, and `rllm/budget.py`.

### 10.1 SessionBudget

```python
class SessionBudget:
    finish_at: datetime
    wind_down_seconds: int
    maximum_gpu_hours: float
    maximum_cpu_hours: float | None
    maximum_runs: int
    maximum_concurrent_runs: int
    maximum_llm_calls: int
    maximum_actor_reviewer_revisions: int
    maximum_retry_runs: int
```

The budget ledger records reservations and actual consumption. Enqueueing reserves estimated cost.
Completion reconciles estimated and actual cost. Failed and cancelled runs still consume their actual
usage.

### 10.2 Runtime estimation

Each adapter supplies an initial conservative estimate per fidelity. Replace it over time with observed
durations from comparable completed runs. Use a conservative upper estimate, such as a configured
percentile plus startup overhead, when determining whether a run fits before wind-down.

Do not launch a run when:

```text
now + conservative_duration > finish_at - wind_down_seconds
```

unless it is explicitly classified as a short diagnostic that can checkpoint and still fits its own
hard timeout.

### 10.3 Session state machine

```text
initializing
  -> planning
  -> executing
  -> evaluating
  -> planning            (next research wave)
  -> confirming
  -> winding_down
  -> completed | solved | stalled | budget_exhausted | failed
```

The controller persists state after every transition and resumes from disk after restart.

### 10.4 Stop conditions

Stop launching new research when any of these is true:

- the full success criterion is satisfied;
- wall-clock or compute budget is exhausted;
- the session enters wind-down;
- a human requests stop;
- no safe, reviewed experiment fits the remaining budget;
- the configured full-fidelity stall rule is satisfied; or
- repeated infrastructure failures make further results unreliable.

A stall is based on full-fidelity results over a configured window, not on cheap screens or LLM
intuition alone.

### 10.5 Wind-down

At `finish_at - wind_down_seconds`:

1. stop proposing and enqueueing long jobs;
2. reject new budget reservations;
3. allow runs that safely finish before the deadline to continue;
4. checkpoint or cancel runs that cannot finish, according to the brief;
5. reconcile the budget ledger;
6. validate registry consistency;
7. update `problem.md` and `journal.md` through reviewed patches;
8. generate the report and handoff; and
9. terminate all model and worker processes by the hard deadline.

The watchdog enforces the hard deadline independently of the LLM controller.

## 11. Human handoff

Add `rllm/report.py`. Each session produces `sessions/<session_id>/handoff.md` containing:

- session start, finish, and terminal reason;
- the original solved criterion and whether it was met;
- best full-fidelity result with seeds, uncertainty, and constraints;
- best search-only result clearly marked unconfirmed;
- every experiment and its lineage;
- promotions and their evidence;
- cancelled runs, saved compute estimate, and exact reasons;
- failed or lost runs;
- proxy-validity status;
- consumed wall time, GPU hours, run count, and LLM calls;
- repository and adapter revisions;
- unresolved realism questions;
- recommended next experiments;
- requests requiring human authority; and
- exact commands to resume.

The report must distinguish facts, statistical estimates, and LLM interpretations.

## 12. CLI target

Keep existing commands during migration, then provide:

```text
rllm onboard <problem_repo> --brief <brief.yaml>
rllm solve <problem_repo> --brief <brief.yaml> --until <timestamp>
rllm worker <work_dir> --device <device>
rllm status <work_dir> [--watch]
rllm inspect <run_id>
rllm cancel <run_id> --reason <text>
rllm resume <session_id>
rllm report <session_id>
rllm validate <problem_repo>
```

`status` reports all queue states, active leases, progress, budget remaining, current leaders,
unconfirmed versus confirmed results, cancellations, and the next scheduled controller action.

`--dry-run` validates and renders RunSpecs and CommandSpecs but does not consume pending jobs. A dry-run
must never move jobs to `completed/`.

## 13. Implementation sequence

Each phase should be a reviewable pull request with tests.

### Phase 0: packaging and test foundation

Files:

- add `pyproject.toml`;
- add `.gitignore`;
- remove tracked `__pycache__` and `.pyc` files;
- add `tests/` with fake adapters and process fixtures; and
- document supported Python versions and dependencies.

Acceptance:

- clean installation in a fresh virtual environment;
- `pytest` discovers tests;
- lint/type/test commands are documented; and
- imports do not depend on running from the repository root.

### Phase 1: schemas and command safety

Files:

- add `rllm/models.py`;
- add `rllm/validation.py`;
- update `rllm/interfaces.py`;
- migrate `adapters/rx/adapter.py`.

Acceptance:

- path traversal IDs are rejected;
- unknown, wrong-type, and out-of-range knobs are rejected;
- task-changing knobs require brief permission;
- values containing quotes or shell metacharacters remain plain environment values;
- no adapter uses `bash -c`; and
- commands and outputs are tied to a unique run directory.

### Phase 2: immutable registry and correct fidelity

Files:

- refactor `rllm/registry.py`;
- refactor `rllm/ladder.py`;
- add schema migration for any existing registry.

Acceptance:

- existing specs/results cannot be overwritten;
- jobs reference immutable RunSpecs;
- canonical duplicate configurations are detected;
- destination fidelity is computed from base config, not source effective config;
- screen-only `BC_EPOCHS=0` cannot leak into refine/confirm;
- generic minimize and maximize metrics rank correctly; and
- incomplete rungs cannot be promoted.

### Phase 3: reliable queue and workers

Files:

- refactor `rllm/dispatcher.py`;
- add process and heartbeat helpers;
- add run-state CLI output.

Acceptance:

- multiple workers cannot execute the same current attempt;
- a worker crash leaves a recoverable lease;
- stale attempts cannot overwrite current results;
- build, launch, parse, and schema failures reach terminal states;
- stdout/stderr are retained;
- timeouts terminate the process group; and
- dry-run does not consume jobs.

### Phase 4: budgeted sessions

Files:

- add `rllm/budget.py`;
- add `rllm/scheduler.py`;
- add `rllm/session.py`;
- extend CLI with `solve`, `resume`, and session-aware `status`.

Acceptance:

- no launch can exceed reserved GPU hours, concurrency, run count, or deadline;
- restart resumes the same session;
- the hard deadline works without either model;
- wind-down stops launches and handles active jobs; and
- every terminal path produces a handoff.

### Phase 5: structured actor/reviewer decisions

Files:

- add `rllm/llm/protocol.py`;
- add JSON Schemas;
- update `rllm/llm/backend.py`, `controller.py`, and prompts.

Acceptance:

- malformed model output cannot enqueue or cancel anything;
- an empty reviewer approval remains empty;
- revision loops are bounded;
- actor and reviewer identities are logged;
- Codex/Claude run read-only for research decisions;
- model failures are logged and budgeted; and
- hard validation and budget failures cannot be overridden by model agreement.

### Phase 6: monitoring and conservative cancellation

Files:

- add `rllm/monitor.py`;
- add `rllm/cancellation.py`;
- add adapter progress/checkpoint support.

Acceptance:

- deterministic invalid runs are stopped and classified;
- research cancellation requires the configured two-model approval;
- protected late/flat runs are not cancelled prematurely;
- cancellation targets an exact attempt token;
- checkpoint, TERM, and KILL stages are logged; and
- cancelled runs remain available for later analysis or resumption.

### Phase 7: statistical confirmation and proxy guard

Files:

- add generic aggregation/confidence utilities;
- implement disjoint seed groups;
- add proxy-validation records and policy;
- update `status` and reporting.

Acceptance:

- screen/refine cannot report solved;
- confirm uses fresh holdout seeds;
- missing or failed seeds cannot disappear from aggregates;
- proxy status is visible and affects promotion/cancellation;
- success is evaluated exactly from the frozen brief; and
- the final report includes uncertainty and sample counts.

### Phase 8: onboarding and generality

Implement LLM-assisted adapter generation only after the execution substrate is reliable.

Acceptance:

- generated adapters are schema-validated and reviewed read-only;
- generated code runs first in a restricted test environment;
- writes to the problem repository require an explicit reviewed patch workflow;
- a second substantially different problem works without changing RLLM core ranking or dispatch code;
- adapter conformance tests pass for both Rx and the second problem.

## 14. Required tests

### 14.1 Unit tests

- reject `../`, absolute, separator-containing, and oversized slugs;
- reject unknown config keys and invalid numeric ranges;
- prove shell metacharacters remain inert environment data;
- validate model JSON with nested braces and escaped strings;
- preserve an explicitly empty reviewer keep-list;
- prevent duplicate immutable-record writes;
- prevent old fidelity overrides from leaking;
- rank minimize/maximize metrics correctly;
- reject NaN and infinity;
- require every configured seed before promotion;
- distinguish parse failure from completion;
- prevent low-fidelity solved status;
- enforce disjoint confirmation seeds;
- enforce every individual budget dimension.

### 14.2 Queue/concurrency tests

- several workers racing for one job;
- worker death immediately after claim;
- worker death after child launch;
- expired lease and requeue;
- stale attempt finishing after a replacement attempt;
- atomic cancellation request during progress update;
- graceful checkpoint and TERM;
- child ignoring TERM and receiving KILL;
- NFS-style delayed visibility simulation where practical.

### 14.3 LLM protocol tests with MockBackend

- actor returns invalid JSON;
- actor returns valid JSON with an invalid knob;
- reviewer references nonexistent items;
- reviewer approves an empty subset;
- repeated revise responses hit the revision cap;
- actor or reviewer times out;
- both models approve an over-budget action;
- cancellation lacks sufficient evidence;
- prompt/memory contains attempted instruction injection.

No case above may enqueue, cancel, or modify code unless all deterministic conditions pass.

### 14.4 Integration fake adapter

Create a fast fake RL adapter whose subprocess:

- emits a configurable learning curve;
- can improve late;
- can emit NaN;
- can hang;
- can crash;
- can checkpoint on signal; and
- can ignore TERM.

Use it to exercise a complete bounded session in seconds, including proposal, review, dispatch,
promotion, cancellation, restart, deadline, confirmation, and handoff.

### 14.5 Rx validation

Before a real long run:

1. validate all declared Rx knobs against the actual runner;
2. run command-only dry-runs;
3. run one minimal real smoke job;
4. deliberately terminate its worker and verify recovery;
5. deliberately request cancellation and verify checkpointing;
6. compare parsed metrics with the original `final_eval.json`;
7. verify full-fidelity BC and other defaults;
8. pin or record every external dependency revision; and
9. run a short two-hour autonomous session with a small hard budget.

## 15. Readiness gate for the first unattended run

Do not start an overnight/weekend autonomous session until:

- all Phase 1-6 acceptance tests pass;
- no model-controlled path or shell construction remains;
- queue recovery and stale-attempt rejection are tested;
- wall-clock and GPU budgets are enforced without LLM cooperation;
- cancellation checkpoints and terminates the entire process group;
- the actor/reviewer loop is structured, bounded, and fully logged;
- at least one full fake-adapter session survives a coordinator restart;
- Rx produces results exclusively attributable to immutable RunSpecs;
- the external Rx runner is pinned or its dirty state is hashed;
- the short autonomous smoke session produces a correct handoff; and
- a human reviews the smoke-session decisions before increasing the budget.

Initially keep the following human-gated:

- changes to the solved criterion;
- changes to task realism;
- generated code writes;
- dependency installation;
- Git commits or pushes;
- launches exceeding a configured per-action cost;
- deletion of results or checkpoints; and
- broad cancellation based only on LLM judgment.

After repeated successful bounded sessions, individual gates can be relaxed deliberately and recorded
in the problem brief. They should not disappear merely because the two models usually agree.

