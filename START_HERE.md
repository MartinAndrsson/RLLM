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

## B. Describe the problem in a paragraph and have a model prepare it

Write what you actually want, the way you would say it to a colleague:

> I have a problem in this repo `/home/rxrepo` where an RL agent needs to learn to move empty rollboxes
> around such that mail can always be sent. There are tests in `final_eval_dev.py` where the performance
> of the RL agent is measured. The goal is to never run out of rollboxes in any sorting center, as that
> is disastrous.

Then paste that, plus the prompt below, to Claude Code **inside the problem repo**. It reads the repo and
produces the two things the interview needs: the knob declarations, and the answers to paste in.

```text
<your paragraph here>

Prepare this repo for the RLLM harness at /home/hep/maander/Supply/RLLM.

HARD RULE: this repository is a read-only simulator. Do not modify a single file in it, and do not
propose modifying it. Everything written for the harness lives in RLLM under
problems/<problem_id>/variants/ and imports this repo. Read RLLM/problems/README.md first.

1. Find (a) how one training run is launched, and (b) the deterministic test that measures
   performance, including the exact function/CLI entry point and the metric names it reports. Quote
   the signatures. If the test only prints numbers and cannot be called programmatically, say so --
   the variant will parse its output rather than us changing the test.
2. Draft ONE first variant in RLLM/problems/<problem_id>/variants/<name>/train.py that: builds the
   env from this repo, defines the observation, action space and reward explicitly, trains, evaluates
   using the test from step 1, and writes $RUN_DIR/metrics.json. Follow the contract in
   RLLM/problems/README.md exactly (RUN_DIR, SEED, PROBLEM_REPO, knobs from the environment, write
   nothing outside RUN_DIR). Keep the observation/action/reward visible in one file.
3. List the knobs that variant reads, and for each: name, value_type (int/float/bool/string/enum),
   default, min/max or choices, category (algorithm, hyperparameter, reward, observation, action,
   training_budget, evaluation, task_definition). Anything that changes what "solved" means --
   episode length, observability, the scenario distribution -- is task_definition. Path-valued knobs
   get llm_may_change=false. Prefer enum with explicit choices over free-form strings.
4. Write that as JSON in the shape of RLLM/docs/knobs.example.json, with a 3-rung
   screen/refine/confirm ladder whose ONLY differences are training budget and seed count.
5. Tell me: the primary metric and whether it is maximized or minimized, a secondary metric for
   tie-breaks, any hard bound that must also hold, roughly how long one short run takes, and which
   knobs I should forbid outright.
6. Print the interview answers as a numbered list I can paste into ./scripts/rllm-onboard.sh.

Do not run any training. Do not commit anything. Do not write outside RLLM/problems/.
```

Review the variant it wrote before starting a session — it is the design your results will be about.

## What the models write, and where

The observation, the action space and the reward are the models' design work, not fixed inputs. Each
design is a directory under `problems/<problem_id>/variants/<name>/` **in this repo**, importing your
repo as a read-only simulator and evaluating with your own test. See
[problems/README.md](problems/README.md) for the contract.

`VARIANT` is then just another declared knob whose choices are the designs on disk, so the search
compares designs exactly as it compares hyperparameters:

```bash
python -m rllm.cli compare <work_dir> --by VARIANT     # design against design
python -m rllm.cli compare <work_dir> --by ALGO        # or algorithm against algorithm
```

Your repo is never modified. The harness fingerprints it around every run (git revision plus
working-tree state, or a file walk if it is not a git repo) and **fails the run and stops the session**
if anything changed — because from that moment on, every number is suspect.

## What you are agreeing to when you start a session

- The models **propose**; deterministic code **authorizes**. Every proposal passes a knob whitelist
  before it can reach a command line (`rllm/validate.py`, `tests/test_gates.py`).
- **Your repo is read-only, and that is checked** — not merely requested (`rllm/integrity.py`,
  `tests/test_variants.py`).
- How the models work is set by [METHOD.md](METHOD.md), which is loaded into every prompt and whose hash
  is recorded with every decision. Edit it to change their methodology.
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
