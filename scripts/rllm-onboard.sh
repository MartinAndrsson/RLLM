#!/usr/bin/env bash
# rllm-onboard.sh — the interview that starts a problem.
#
# Asks the questions a human must answer before any autonomous compute is spent, writes them to
# <work_dir>/brief.json (validated by `python -m rllm.cli onboard`), and prints — or, with --start,
# runs — the command that begins the first bounded exploration session.
#
# Everything it asks maps to a field the harness actually enforces: the metric it ranks on, the
# criterion it must meet before claiming success, the deterministic test it believes, the knobs it may
# not touch, and how long it may explore before handing back to you. Nothing here is advisory.
#
#   ./scripts/rllm-onboard.sh                 # interview, then print the next command
#   ./scripts/rllm-onboard.sh --start         # interview, then start the session immediately
#   ./scripts/rllm-onboard.sh --rx            # pre-fill the answers for the Rx rollbox problem
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
START=0
PRESET=""

for arg in "$@"; do
  case "$arg" in
    --start) START=1 ;;
    --rx)    PRESET="rx" ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -t 0 ]]; then
  echo "ERROR: this is an interview; run it from a terminal (or write brief.json yourself and call" >&2
  echo "       'python -m rllm.cli onboard <work_dir> --answers answers.json')." >&2
  exit 2
fi

# ---------------------------------------------------------------- prompting helpers

BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'

ask() {                     # ask VAR "question" "default" ["hint"]
  local __var="$1" __q="$2" __default="${3:-}" __hint="${4:-}" __ans=""
  printf '\n%s%s%s\n' "$BOLD" "$__q" "$RESET"
  [[ -n "$__hint" ]] && printf '%s%s%s\n' "$DIM" "$__hint" "$RESET"
  if [[ -n "$__default" ]]; then
    read -r -p "  [$__default] > " __ans || true
    __ans="${__ans:-$__default}"
  else
    while [[ -z "$__ans" ]]; do read -r -p "  > " __ans || true; done
  fi
  printf -v "$__var" '%s' "$__ans"
}

ask_opt() {                 # ask_opt VAR "question" ["hint"]  — an empty answer is a valid answer
  local __var="$1" __q="$2" __hint="${3:-}" __ans=""
  printf '\n%s%s%s\n' "$BOLD" "$__q" "$RESET"
  [[ -n "$__hint" ]] && printf '%s%s%s\n' "$DIM" "$__hint" "$RESET"
  read -r -p "  > " __ans || true
  printf -v "$__var" '%s' "$__ans"
}

ask_multiline() {           # ask_multiline VAR "question" "hint"  (blank line ends input)
  local __var="$1" __q="$2" __hint="${3:-}" __line="" __acc=""
  printf '\n%s%s%s\n' "$BOLD" "$__q" "$RESET"
  [[ -n "$__hint" ]] && printf '%s%s%s\n' "$DIM" "$__hint" "$RESET"
  printf '%s  (end with an empty line)%s\n' "$DIM" "$RESET"
  while IFS= read -r __line; do
    [[ -z "$__line" ]] && break
    __acc+="${__acc:+ }$__line"
  done
  printf -v "$__var" '%s' "$__acc"
}

ask_list() {                # ask_list VAR "question" "hint"  -> newline-separated items
  local __var="$1" __q="$2" __hint="${3:-}" __line="" __acc=""
  printf '\n%s%s%s\n' "$BOLD" "$__q" "$RESET"
  [[ -n "$__hint" ]] && printf '%s%s%s\n' "$DIM" "$__hint" "$RESET"
  printf '%s  (one per line; empty line to finish)%s\n' "$DIM" "$RESET"
  while IFS= read -r __line; do
    [[ -z "$__line" ]] && break
    __acc+="${__acc:+$'\n'}$__line"
  done
  printf -v "$__var" '%s' "$__acc"
}

confirm() {                 # confirm "question" default(y|n)
  local __q="$1" __d="${2:-y}" __ans=""
  read -r -p "$(printf '\n%s%s%s [%s/%s] > ' "$BOLD" "$__q" "$RESET" \
      "$([[ $__d == y ]] && echo Y || echo y)" "$([[ $__d == n ]] && echo N || echo n)")" __ans || true
  __ans="${__ans:-$__d}"
  [[ "$__ans" =~ ^[Yy] ]]
}

cat <<BANNER

${BOLD}RLLM onboarding${RESET}
${DIM}A few questions, then the models start exploring your RL problem within the limits you set here.
Answers land in a brief.json that is frozen for each session: the models may request changes to it,
but only you can make them.${RESET}
BANNER

# ---------------------------------------------------------------- 1. the problem

if [[ "$PRESET" == "rx" ]]; then
  PROBLEM_ID="rx"
  PROBLEM_REPO="/home/hep/maander/Supply/Rx_dan/supply"
  WORK_DIR="$REPO_ROOT/adapters/rx/rllm_work"
  ADAPTER="rx"
else
  ask PROBLEM_REPO "Which repository holds the problem?" "" \
      "Absolute path. The harness only ever runs the test command you give below, inside this directory."
  PROBLEM_REPO="${PROBLEM_REPO/#\~/$HOME}"
  [[ -d "$PROBLEM_REPO" ]] || { echo "  not a directory: $PROBLEM_REPO" >&2; exit 2; }
  ask PROBLEM_ID "Short id for this problem?" "$(basename "$PROBLEM_REPO" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')" \
      "Lower-case slug; names the work directory and every experiment id."
  ask WORK_DIR "Where should RLLM keep this problem's memory, queue and results?" \
      "$REPO_ROOT/work/$PROBLEM_ID" \
      "Shared filesystem if several machines will run workers — the queue lives here."
  ADAPTER="brief"
fi

ask_multiline DESCRIPTION "Describe the problem." \
    "What is being controlled, what makes it hard, what has already been tried. The models read this first."
ask_multiline GOAL "What is the goal — what does winning look like?" \
    "In your words. This is quoted back in every proposal and in the handoff."

# ---------------------------------------------------------------- 2. the deterministic test

cat <<SECTION

${BOLD}The performance test${RESET}
${DIM}This is the only thing the harness believes about performance. No model is ever asked whether a run
went well: it runs your command and reads the numbers your command writes.${RESET}
SECTION

if [[ "$PRESET" == "rx" ]]; then
  TEST_COMMAND=""
  METRICS_FILE=""
  PRIMARY_METRIC="success_fraction"
  PRIMARY_DIRECTION="maximize"
  TIE_METRIC="mean_shortage_days"
  TIE_DIRECTION="minimize"
  PRIMARY_OP=">="
  PRIMARY_VALUE="1.0"
else
  ask TEST_COMMAND "What command runs ONE training/evaluation and writes its metrics?" "" \
      "Run from the repo root. The harness exports your knobs plus RUN_DIR and SEED; write outputs under \$RUN_DIR.
   e.g.  bash train.sh --seed \$SEED --out \$RUN_DIR"
  ask METRICS_FILE "Which file does it write the metrics to, relative to \$RUN_DIR?" "metrics.json" \
      "Flat JSON of numbers, e.g. {\"success_fraction\": 0.8, \"mean_shortage_days\": 1.1}"
  ask PRIMARY_METRIC "Which key in that file is the primary metric?" "" \
      "The one the search ranks on."
  ask PRIMARY_DIRECTION "Should it be maximized or minimized?" "maximize" "maximize | minimize"
  ask_opt TIE_METRIC "Secondary metric to break ties? (leave empty for none)"
  # Spelled as `if` rather than `[[ ... ]] && ask ...`: the one-liner form leaves a non-zero status
  # behind, which under `set -e` bites as soon as it is ever the last statement in a block.
  if [[ -n "$TIE_METRIC" ]]; then
    ask TIE_DIRECTION "...maximized or minimized?" "minimize" "maximize | minimize"
  fi
  ask PRIMARY_OP "What comparison makes it SOLVED? ($PRIMARY_METRIC ? value)" ">=" ">= | > | <= | < | =="
  ask PRIMARY_VALUE "...against what value?" "" "The bar you would accept as done."
fi

ask_list CONSTRAINTS "Any OTHER metric that must also hold for it to count as solved?" \
    "One per line as 'metric op value', e.g. 'mean_queue_days <= 0.5'. These are hard constraints, not
   tie-breaks: a candidate that violates one is not solved however good the primary metric is."

ask REQUIRED_FIDELITY "At which fidelity must that hold to count as solved?" "confirm" \
    "The most expensive rung: cheap screening results never end a session."
ask MIN_SEEDS "Over how many training seeds, minimum?" "3" \
    "Seed variance is usually large; 1 seed screens, it never confirms."

# ---------------------------------------------------------------- 3. what must not change

cat <<SECTION

${BOLD}Guard rails${RESET}
${DIM}What would make a result meaningless if the models changed it to look good.${RESET}
SECTION

ask_list REALISM "Realism constraints — things that must stay true about the problem." \
    "e.g. 'episode length stays 100 days', 'stock is only 10% observable (RFID)'."
ask_list FORBIDDEN "Knobs the models must NEVER change (even if both agree)?" \
    "Knob/env-var names. Anything that changes the task itself belongs here."
ask_list PERMITTED "Knobs that change the task but ARE allowed this session?" \
    "Usually empty. Listing one here is you explicitly authorizing it."

# ---------------------------------------------------------------- 4. time and compute

cat <<SECTION

${BOLD}Budget${RESET}
${DIM}Enforced by the harness, not by the models: they cannot grant themselves more.${RESET}
SECTION

ask EXPLORE "How long may the models explore before you review?" "8h" "e.g. 90m, 8h, 2d"
ask WIND_DOWN "How long before that deadline should it stop launching new runs and write the handoff?" \
    "15m" "Must be long enough to finish writing up; runs that cannot finish are terminated."
ask CHEAP_RUN "Roughly how long does ONE cheap/screening run take?" "30m" \
    "A starting estimate only — replaced by measured durations as runs complete."
ask MAX_RUNS "Cap on total runs this session?" "200" ""
ask MAX_LLM "Cap on LLM calls this session?" "200" ""
ask_opt GPU_HOURS "Cap on device-hours? (leave empty for no cap beyond the clock)"
ask DEVICE "Which device should the worker use?" "0" "CUDA_VISIBLE_DEVICES value."

if [[ "$PRESET" != "rx" ]]; then
  cat <<SECTION

${BOLD}Tunable knobs${RESET}
${DIM}The models may only set knobs that are declared, with declared types and ranges — anything else is
rejected before a command is built. Declaring these is the one fiddly part, so it is a separate file.${RESET}
SECTION
  ask_opt KNOBS_FILE "Path to a JSON file declaring the tunable knobs (leave empty to add them later)" \
      "See docs/knobs.example.json for the format."
fi

# ---------------------------------------------------------------- assemble

mkdir -p "$WORK_DIR"
ANSWERS="$(mktemp "${TMPDIR:-/tmp}/rllm-answers-XXXXXX.json")"
trap 'rm -f "$ANSWERS"' EXIT

KNOBS_JSON="[]"
if [[ "$PRESET" != "rx" && -n "${KNOBS_FILE:-}" ]]; then
  [[ -f "$KNOBS_FILE" ]] || { echo "  no such file: $KNOBS_FILE" >&2; exit 2; }
  KNOBS_JSON="$(cat "$KNOBS_FILE")"
fi

export REPO_ROOT
export PROBLEM_ID PROBLEM_REPO DESCRIPTION GOAL TEST_COMMAND METRICS_FILE PRIMARY_METRIC \
       PRIMARY_DIRECTION TIE_METRIC TIE_DIRECTION PRIMARY_OP PRIMARY_VALUE REQUIRED_FIDELITY \
       MIN_SEEDS REALISM FORBIDDEN PERMITTED EXPLORE WIND_DOWN CHEAP_RUN MAX_RUNS MAX_LLM \
       GPU_HOURS ADAPTER KNOBS_JSON CONSTRAINTS
TIE_DIRECTION="${TIE_DIRECTION:-minimize}"

"$PY" - > "$ANSWERS" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ.get("REPO_ROOT", "."))
from rllm.brief import parse_duration

def lines(name):
    return [l.strip() for l in os.environ.get(name, "").splitlines() if l.strip()]

OPS = (">=", "<=", "==", ">", "<")

def constraints(name):
    """'mean_queue_days <= 0.5' -> a MetricConstraint dict. Refuses anything it cannot parse rather
    than dropping a constraint the user believes is being enforced."""
    out = []
    for line in lines(name):
        for op in OPS:                      # longest operators first, so '>=' never parses as '>'
            metric, sep, value = line.partition(op)
            if sep:
                try:
                    out.append({"metric": metric.strip(), "operator": op, "value": float(value)})
                except ValueError:
                    sys.exit(f"cannot read a number from constraint {line!r} (use 'metric <= 0.5')")
                break
        else:
            sys.exit(f"cannot parse constraint {line!r} — expected 'metric op value', op one of {OPS}")
    return out

adapter_config = {}
if os.environ["ADAPTER"] == "brief":
    declared = json.loads(os.environ.get("KNOBS_JSON") or "[]")
    if isinstance(declared, list):                       # a bare list of knobs
        adapter_config["knobs"] = declared
    else:                                                # the full example-file shape
        adapter_config.update({k: v for k, v in declared.items() if not k.startswith("_")})
# The interview's own answer wins over anything in the knobs file.
adapter_config["cheap_run_seconds"] = parse_duration(os.environ["CHEAP_RUN"])

brief = {
    "problem_id": os.environ["PROBLEM_ID"],
    "description": os.environ["DESCRIPTION"],
    "goal": os.environ["GOAL"],
    "problem_repo": os.environ["PROBLEM_REPO"],
    "primary_metric": os.environ["PRIMARY_METRIC"],
    "primary_direction": os.environ["PRIMARY_DIRECTION"],
    "test_command": os.environ.get("TEST_COMMAND", ""),
    "metrics_file": os.environ.get("METRICS_FILE", ""),
    "tie_break_metric": os.environ.get("TIE_METRIC", ""),
    "tie_break_direction": os.environ.get("TIE_DIRECTION", "minimize"),
    "success_criterion": {
        "primary": {"metric": os.environ["PRIMARY_METRIC"],
                    "operator": os.environ["PRIMARY_OP"],
                    "value": float(os.environ["PRIMARY_VALUE"])},
        "additional": constraints("CONSTRAINTS"),
        "required_fidelity": os.environ["REQUIRED_FIDELITY"],
        "minimum_training_seeds": int(os.environ["MIN_SEEDS"]),
    },
    "realism_constraints": lines("REALISM"),
    "forbidden_task_changes": lines("FORBIDDEN"),
    "permitted_task_changes": lines("PERMITTED"),
    "adapter": {"name": os.environ["ADAPTER"], "config": adapter_config},
    "session_budget": {
        "explore_seconds": parse_duration(os.environ["EXPLORE"]),
        "wind_down_seconds": parse_duration(os.environ["WIND_DOWN"]),
        "maximum_runs": int(os.environ["MAX_RUNS"]),
        "maximum_llm_calls": int(os.environ["MAX_LLM"]),
        "maximum_gpu_hours": float(os.environ["GPU_HOURS"]) if os.environ.get("GPU_HOURS") else None,
    },
    "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
}
json.dump(brief, sys.stdout, indent=2)
PYEOF

cd "$REPO_ROOT"
"$PY" -m rllm.cli onboard "$WORK_DIR" --answers "$ANSWERS" --force

SOLVE="$PY -m rllm.cli solve $WORK_DIR --device ${DEVICE:-0}"
cat <<NEXT

${BOLD}Ready.${RESET}
  brief    : $WORK_DIR/brief.json
  memory   : $WORK_DIR/problem.md, $WORK_DIR/journal.md ${DIM}(write anything the models should know)${RESET}

Start the session (it stops by itself and writes a handoff):
  ${BOLD}cd $REPO_ROOT && $SOLVE${RESET}

While it runs / afterwards:
  $PY -m rllm.cli status  $WORK_DIR
  $PY -m rllm.cli handoff $WORK_DIR
NEXT

if [[ "$ADAPTER" == "brief" && "$KNOBS_JSON" == "[]" ]]; then
  cat <<WARN

${BOLD}One thing first:${RESET} no tunable knobs were declared, so every proposal will be rejected
(nothing is settable). Copy docs/knobs.example.json, declare your knobs, and add them under
adapter.config.knobs in the brief — or re-run this interview with the file ready.
WARN
  exit 0
fi

if [[ $START -eq 1 ]] || confirm "Start the session now?" n; then
  exec $SOLVE
fi
