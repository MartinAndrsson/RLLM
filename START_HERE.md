# START HERE

Two ways in. Both end at the same place: a validated `brief.json` and a bounded session that stops by
itself and writes you a handoff.

## A. Run the interview

```bash
cd /home/hep/maander/Supply/RLLM
./scripts/rllm-onboard.sh            # add --rx to pre-fill the Rx rollbox problem
```

It asks for: the repo, the problem and goal, the deterministic performance test (the command and the
JSON file it writes), what counts as solved and at what fidelity, what must never change, and how long
the models may explore before you review. Then it prints the command that starts the session:

```bash
python -m rllm.cli solve <work_dir> --device 0
```

That session plans a wave, runs it, promotes what survives, and repeats until the wind-down point —
then writes `<work_dir>/sessions/<id>/handoff.md` telling you whether to grant more time, answer a
question, or step in.

## B. Ask a model to prepare it for you

For a brand-new repo the fiddly part is declaring which knobs may be tuned, with types and ranges.
Paste this to Claude Code (or Codex) **inside the problem repo**:

```text
Read this repo and prepare it for the RLLM experimentation harness at
/home/hep/maander/Supply/RLLM. Do not change any training code.

1. Find the entry point that trains/evaluates one configuration, and the file it writes results to.
   Tell me the exact command to run ONE run, given $SEED and an output directory $RUN_DIR, and the
   JSON file (relative to $RUN_DIR) holding its metrics. If no such JSON exists, write the smallest
   possible patch that emits one, and show it to me before applying it.
2. List every knob that command reads (env vars, CLI flags, config keys), and for each give:
   name, value_type (int/float/bool/string/enum), default, min/max or choices, and a category
   (algorithm, hyperparameter, reward, observation, action, training_budget, evaluation,
   task_definition). Anything that changes what "solved" means -- episode length, observability,
   the scenario distribution -- must be category task_definition. Anything path-valued gets
   llm_may_change=false. Prefer enum with explicit choices over free-form strings.
3. Write that list as JSON in the shape of RLLM/docs/knobs.example.json, including a 3-rung
   screen/refine/confirm ladder whose ONLY differences are training budget and seed count -- never
   episode length or anything else that changes the problem.
4. Tell me: the primary metric and whether it is maximized or minimized, a sensible secondary metric
   for tie-breaks, roughly how long one short run takes, and which knobs I should forbid outright.
5. Print the answers as a numbered list I can paste into ./scripts/rllm-onboard.sh, and save the knob
   JSON to a file whose path you tell me.

Do not run any training. Do not commit anything.
```

Then run the interview with that file to hand.

## What you are agreeing to when you start a session

- The models **propose**; deterministic code **authorizes**. Every proposal passes a knob whitelist
  before it can reach a command line (`rllm/validate.py`, `tests/test_gates.py`).
- The deadline, the run cap and the LLM-call cap are enforced outside the models. A model can only
  *request* more time, in the handoff.
- Success is only ever claimed from the fidelity and seed count you specified — a good cheap-rung
  result is reported as an unconfirmed lead.
- Task-defining knobs stay yours. The models may ask; they cannot change them.
- Nothing is committed, and nothing in your problem repo is edited, by the harness.

## Reading the result

```bash
python -m rllm.cli status  <work_dir>     # rankings per rung, sessions, budget
python -m rllm.cli handoff <work_dir>     # the write-up: what was learned, what to do next
```

The handoff separates **facts** (from the run registry), **estimates** (seed means, durations), and
**interpretation** (what the models concluded, whether the reviewer agreed, and what the harness's own
model-free reading of the numbers was). When those disagree it says so.
