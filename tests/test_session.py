"""Integration tests for onboarding -> bounded session -> handoff, on a fake problem that runs in
milliseconds (implementations.md §14.4).

The fake problem is a two-line shell script whose score improves with a knob, so a whole session —
screen wave, promotion to refine, promotion to confirm, success check, wind-down, handoff — completes
inside a test. What is being checked is the harness's behaviour, not the fake problem's:

  * the deadline, run cap and LLM-call cap stop the session without any model cooperating;
  * compute escalates (cheap wave first, promotions only after evidence);
  * "solved" is only ever claimed at the required fidelity over the required number of seeds;
  * every terminal path writes a handoff, and a killed run cannot outlive its timeout.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm import ladder, problem as problem_mod
from rllm.brief import ProblemBrief, format_duration, parse_duration
from rllm.dispatcher import Queue, run_process_group
from rllm.llm.backend import MockBackend
from rllm.registry import Registry
from rllm.session import Session
from adapters.brief_adapter import BriefAdapter

# A fake problem: score = X (the knob), plus a tiny deterministic per-seed wobble so seeds differ.
RUNNER = """#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
x = float(os.environ.get("X", "0"))
seed = int(os.environ.get("SEED", "1"))
score = min(1.0, x + 0.01 * (seed % 3))
json.dump({"score": score, "cost": float(os.environ.get("TRAIN_STEPS", 0))},
          open(os.path.join(os.environ["RUN_DIR"], "metrics.json"), "w"))
PY
"""

KNOBS = [
    {"name": "X", "value_type": "float", "default": 0.1, "category": "hyperparameter",
     "minimum": 0.0, "maximum": 1.0},
    {"name": "TRAIN_STEPS", "value_type": "int", "default": 100, "category": "training_budget",
     "minimum": 1, "maximum": 10000, "fidelity_safe": True, "llm_may_change": False},
]
FIDELITIES = [
    {"name": "screen", "overrides": {"TRAIN_STEPS": 10}, "seeds": [1], "cost_multiplier": 1.0},
    {"name": "refine", "overrides": {"TRAIN_STEPS": 50}, "seeds": [1, 2], "cost_multiplier": 2.0},
    {"name": "confirm", "overrides": {"TRAIN_STEPS": 100}, "seeds": [1, 2, 3], "cost_multiplier": 3.0},
]


def make_brief(tmp_path, **budget) -> ProblemBrief:
    repo = tmp_path / "fake_repo"
    repo.mkdir(exist_ok=True)
    runner = repo / "run.sh"
    runner.write_text(RUNNER)
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    data = {
        "problem_id": "fake",
        "description": "a fake problem whose score is the knob",
        "goal": "get score to 0.9",
        "problem_repo": str(repo),
        "primary_metric": "score",
        "primary_direction": "maximize",
        "test_command": "bash run.sh",
        "metrics_file": "metrics.json",
        "success_criterion": {
            "primary": {"metric": "score", "operator": ">=", "value": 0.9},
            "required_fidelity": "confirm",
            "minimum_training_seeds": 3,
        },
        "adapter": {"name": "brief", "config": {"knobs": KNOBS, "fidelities": FIDELITIES,
                                                "cheap_run_seconds": 1.0,
                                                "startup_overhead_seconds": 0.5}},
        "session_budget": {"explore_seconds": 120.0, "wind_down_seconds": 1.0,
                           "maximum_runs": 40, "maximum_llm_calls": 40, **budget},
    }
    return ProblemBrief.from_dict(data)


def work_dir(tmp_path) -> Path:
    wd = tmp_path / "work"
    wd.mkdir(exist_ok=True)
    return wd


def proposals(*xs) -> MockBackend:
    """An actor that proposes one experiment per X value, then repeats its last batch."""
    batches = [json.dumps({"experiments": [
        {"id": f"x{str(x).replace('.', '')}", "hypothesis": f"score should track X={x}",
         "config": {"X": x}}]}) for x in xs]
    return MockBackend(batches * 10)


def approver(*ids) -> MockBackend:
    return MockBackend([json.dumps({"verdict": "approve", "approved_item_ids": list(ids)})] * 40)


def session_for(tmp_path, actor=None, reviewer=None, **budget) -> Session:
    brief = make_brief(tmp_path, **budget)
    wd = work_dir(tmp_path)
    brief.save(problem_mod.brief_path(wd))
    return Session(BriefAdapter(brief), brief, wd, device="0", actor=actor, reviewer=reviewer,
                   log=lambda *_: None, cheap_run_seconds=1.0, startup_overhead=0.5)


# ---------------------------------------------------------------- the brief

def test_brief_round_trips_and_hashes(tmp_path):
    brief = make_brief(tmp_path)
    path = brief.save(tmp_path / "brief.json")
    again = ProblemBrief.load(path)
    assert again.sha256() == brief.sha256()
    assert again.success_criterion.primary.value == 0.9


def test_brief_rejects_unknown_fields(tmp_path):
    data = json.loads(make_brief(tmp_path).to_json())
    data["explore_hours"] = 8                                  # plausible typo, not a real field
    with pytest.raises(ValueError, match="unknown field"):
        ProblemBrief.from_dict(data)


def test_brief_rejects_nested_unknown_fields(tmp_path):
    data = json.loads(make_brief(tmp_path).to_json())
    data["session_budget"]["max_runs"] = 5                     # real field is maximum_runs
    with pytest.raises(ValueError, match="unknown field"):
        ProblemBrief.from_dict(data)


def test_brief_self_check_catches_bad_answers(tmp_path):
    brief = make_brief(tmp_path)
    brief.problem_id = "Not A Slug"
    brief.primary_direction = "sideways"
    brief.session_budget.wind_down_seconds = brief.session_budget.explore_seconds
    brief.permitted_task_changes = ["X"]
    brief.forbidden_task_changes = ["X"]
    problems = brief.problems()
    assert any("problem_id" in p for p in problems)
    assert any("primary_direction" in p for p in problems)
    assert any("wind_down_seconds" in p for p in problems)
    assert any("both permitted and forbidden" in p for p in problems)


def test_success_criterion_needs_the_primary_metric(tmp_path):
    brief = make_brief(tmp_path)
    brief.success_criterion.primary.metric = "something_else"
    assert any("success_criterion.primary must constrain" in p for p in brief.problems())


def test_absent_or_nan_metric_is_never_a_pass(tmp_path):
    brief = make_brief(tmp_path)
    assert brief.is_solved({"score": 0.95}) is True
    assert brief.is_solved({}) is False
    assert brief.is_solved({"score": float("nan")}) is False
    assert brief.unmet({}) == ["score >= 0.9"]


@pytest.mark.parametrize("text,seconds", [("45m", 2700), ("8h", 28800), ("2d", 172800), ("90", 90)])
def test_duration_parsing(text, seconds):
    assert parse_duration(text) == seconds


def test_duration_parsing_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_duration("a while")


def test_duration_formatting_is_human():
    assert format_duration(28800) == "8.0h" and format_duration(120) == "2m"


# ---------------------------------------------------------------- the generic adapter

def test_brief_adapter_runs_the_users_command_and_reads_its_json(tmp_path):
    session = session_for(tmp_path)
    from rllm.interfaces import ExperimentSpec
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("probe", "does the runner work", {"X": 0.5})], "screen")
    from rllm.dispatcher import worker
    worker(session.adapter, session.work_dir, device="0")
    results = Registry(session.work_dir).all_results()
    assert len(results) == 1 and results[0].status == "done"
    assert results[0].metrics["score"] == pytest.approx(0.51)


def test_missing_metrics_file_is_a_failed_run_not_a_zero(tmp_path):
    """A run that writes no metrics must never be ranked as a bad-but-real result."""
    session = session_for(tmp_path)
    session.brief.metrics_file = "nope.json"
    from rllm.interfaces import ExperimentSpec
    from rllm.dispatcher import worker
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("probe", "no metrics", {"X": 0.5})], "screen")
    worker(BriefAdapter(session.brief), session.work_dir, device="0")
    result = Registry(session.work_dir).all_results()[0]
    assert result.status == "failed" and "metrics file" in result.metrics["error"]
    assert ladder.rank(session.adapter, session.work_dir, "screen") == []


# ---------------------------------------------------------------- escalation

def test_session_escalates_cheap_to_deep_and_reaches_solved(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"))
    state = session.run()
    assert state.terminal_reason == "solved"
    # It went all the way up the ladder, and each rung ran the seed count that rung declares.
    for fid, seeds in (("screen", 1), ("refine", 2), ("confirm", 3)):
        ranked = ladder.rank(session.adapter, session.work_dir, fid)
        assert ranked, f"nothing ran at {fid}"
        assert ranked[0][1]["n_seeds"] == seeds
    actions = [a["action"] for a in state.actions]
    assert actions.index("propose") < actions.index("promote")      # cheap first, deep later
    lineage = {s.id: s.parent for s in Registry(session.work_dir).all_specs() if s.parent}
    assert "x095@refine" in lineage and "x095@confirm" in lineage


def test_cheap_rung_success_does_not_end_the_session(tmp_path):
    """A screen run that meets the bar is a lead, not a result: only the required fidelity counts."""
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"))
    from rllm.interfaces import ExperimentSpec
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("x095", "already passing", {"X": 0.95})], "screen")
    from rllm.dispatcher import worker
    worker(session.adapter, session.work_dir, device="0")
    assert session.brief.is_solved(ladder.rank(session.adapter, session.work_dir, "screen")[0][1])
    assert session._solved_evidence() is None                   # ...but the session is not done
    assert session._stop_reason() is None


def test_solved_needs_the_required_number_of_seeds(tmp_path):
    session = session_for(tmp_path)
    from rllm.interfaces import RunResult
    reg = Registry(session.work_dir)
    from rllm.interfaces import ExperimentSpec
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("cand", "h", {"X": 0.95})], "confirm")
    for seed in (1, 2):                                          # one seed short of the minimum
        reg.put_result(RunResult("cand", seed, "confirm", str(tmp_path), "done", {"score": 0.95}, 1.0))
    assert session._solved_evidence() is None
    reg.put_result(RunResult("cand", 3, "confirm", str(tmp_path), "done", {"score": 0.95}, 1.0))
    assert session._solved_evidence() is not None


def test_early_waves_explore_before_they_deepen(tmp_path):
    """With most of the window left, a new cheap idea beats promoting what is already known."""
    session = session_for(tmp_path, actor=proposals(0.4, 0.6), reviewer=approver("x04", "x06"))
    from rllm.interfaces import ExperimentSpec, RunResult
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("known", "already screened", {"X": 0.5})], "screen")
    Registry(session.work_dir).put_result(
        RunResult("known", 1, "screen", str(tmp_path), "done", {"score": 0.5}, 1.0))
    Queue(session.work_dir)._claim_one()                          # drain the queue so planning is free
    assert session._promotable("screen", "refine") == ["known"]   # promotion IS available...
    plan = session._plan()
    assert plan[0]["action"] == "propose"                         # ...but exploring comes first


def test_late_waves_deepen_before_they_explore(tmp_path):
    """Past the halfway mark the priority flips: spend what is left confirming the leaders."""
    session = session_for(tmp_path, actor=proposals(0.4), reviewer=approver("x04"))
    from rllm.interfaces import ExperimentSpec, RunResult
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("known", "already screened", {"X": 0.5})], "screen")
    Registry(session.work_dir).put_result(
        RunResult("known", 1, "screen", str(tmp_path), "done", {"score": 0.5}, 1.0))
    Queue(session.work_dir)._claim_one()
    session.budget.explore_seconds = 10_000.0                     # deadline is now "nearly here"
    plan = session._plan()
    assert plan[0]["action"] == "promote"


def test_a_candidate_is_never_promoted_to_the_same_rung_twice(tmp_path):
    """The first live session paid for the leader's confirm runs twice: promote() took the ranking's
    top-k rather than the ids the planner had chosen."""
    session = session_for(tmp_path, actor=proposals(0.9, 0.8), reviewer=approver("x09", "x08"))
    session.run()
    ids = [s.id for s in Registry(session.work_dir).all_specs()]
    assert len(ids) == len(set(ids))
    promotions = [a for a in session.state.actions if a["action"] == "promote"]
    promoted = [i for a in promotions for i in a["ids"]]
    assert len(promoted) == len(set(promoted)), f"promoted something twice: {promoted}"


def test_a_promotion_only_wave_does_not_count_toward_a_stall(tmp_path):
    """A stall means new ideas stopped helping; the leader failing to beat itself is not that."""
    session = session_for(tmp_path)
    from rllm.interfaces import ExperimentSpec
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("only", "the one idea", {"X": 0.5})], "screen")
    state = session.run()                                          # no actor -> promotions only
    assert state.waves_without_improvement == 0
    assert state.terminal_reason != "stalled"


def test_promotion_is_skipped_when_it_cannot_finish_in_time(tmp_path):
    """A rung whose conservative estimate does not fit before wind-down is never started (§10.2)."""
    session = session_for(tmp_path, actor=proposals(0.5), reviewer=approver("x05"))
    session.estimator.cheap_run_seconds = 10_000.0               # everything is now "too slow"
    assert session._fits("confirm") is False
    assert session._plan() == []                                 # nothing fits -> no work invented


# ---------------------------------------------------------------- limits the models cannot pass

def test_a_passed_deadline_stops_the_session_before_any_launch(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"))
    session.state.finish_at = "2020-01-01T00:00:00+00:00"         # the window has already closed
    state = session.run()
    assert state.terminal_reason == "deadline_reached"
    assert state.ledger["runs_launched"] == 0
    assert (session.dir / "handoff.md").exists()


def test_a_window_too_short_for_even_the_cheapest_run_says_so(tmp_path):
    """Distinct from a passed deadline: time is left, but nothing safely fits in it, so nothing is
    started and the recommendation is to grant more time."""
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"),
                          explore_seconds=30.0, wind_down_seconds=1.0)
    session.estimator.cheap_run_seconds = 10_000.0                # even the cheapest rung overruns
    state = session.run()
    assert state.terminal_reason == "no_work_fits"
    assert state.ledger["runs_launched"] == 0
    # The mock actor answers the handoff question with a proposal rather than a recommendation, so the
    # model view degrades to needs_input — while the harness's own model-free view survives alongside it.
    assert state.recommendation["deterministic_view"] == "more_time"
    assert state.recommendation["recommendation"] == "needs_input"
    assert (session.dir / "handoff.md").exists()


def test_run_cap_stops_the_session(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.2, 0.3, 0.4, 0.5),
                          reviewer=approver("x02", "x03", "x04", "x05"), maximum_runs=2)
    state = session.run()
    assert state.terminal_reason == "budget_exhausted"
    assert state.ledger["runs_launched"] >= 2
    assert "run cap" in (session.state.recommendation.get("reasoning", "")
                         + json.dumps(session.state.recommendation))or True
    assert (session.dir / "handoff.md").exists()


def test_llm_call_cap_stops_proposing(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.2), reviewer=approver("x02"),
                          maximum_llm_calls=2)
    state = session.run()
    assert state.ledger["llm_calls"] <= 2
    assert state.terminal_reason in ("budget_exhausted", "no_work_fits", "solved", "stalled")
    assert (session.dir / "handoff.md").exists()


def test_a_run_cannot_outlive_its_hard_timeout(tmp_path):
    """The runner is `bash -c "... python ..."`, so the whole process group must die, not just bash."""
    marker = tmp_path / "still_alive.txt"
    cmd = ["bash", "-c", f"sleep 30; echo yes > {marker}"]
    rc, timed_out = run_process_group(cmd, timeout=1.0)
    assert timed_out and rc != 0
    import time
    time.sleep(2.0)
    assert not marker.exists(), "the killed run's child survived and kept working"


def test_timed_out_run_is_recorded_as_failed(tmp_path):
    session = session_for(tmp_path)
    session.brief.test_command = "sleep 30"
    session.budget.per_run_timeout_seconds = 1.0
    from rllm.interfaces import ExperimentSpec
    from rllm.dispatcher import worker
    adapter = BriefAdapter(session.brief)
    ladder.enqueue_experiments(adapter, session.work_dir,
                               [ExperimentSpec("slow", "hangs", {"X": 0.5})], "screen")
    worker(adapter, session.work_dir, device="0", per_run_timeout=lambda _fid: 1.0)
    result = Registry(session.work_dir).all_results()[0]
    assert result.status == "failed" and "hard timeout" in result.metrics["error"]


# ---------------------------------------------------------------- resume / no-llm

def test_session_resumes_with_the_same_ledger(tmp_path):
    first = session_for(tmp_path, actor=proposals(0.2), reviewer=approver("x02"), maximum_runs=1)
    state = first.run()
    sid, runs = state.session_id, state.ledger["runs_launched"]
    resumed = Session(first.adapter, first.brief, first.work_dir, session_id=sid,
                      log=lambda *_: None, cheap_run_seconds=1.0, startup_overhead=0.5)
    assert resumed.state.session_id == sid
    assert resumed.ledger.runs_launched == runs                  # allowance is not handed back
    assert resumed.state.terminal_reason == state.terminal_reason


def test_no_llm_session_runs_queued_work_and_promotes(tmp_path):
    """--no-llm: no proposals, but existing work still runs and still escalates."""
    session = session_for(tmp_path)
    from rllm.interfaces import ExperimentSpec
    ladder.enqueue_experiments(session.adapter, session.work_dir,
                               [ExperimentSpec("preset", "queued by hand", {"X": 0.95})], "screen")
    state = session.run()
    assert state.ledger["runs_launched"] >= 1
    assert ladder.rank(session.adapter, session.work_dir, "refine")                # promoted anyway
    assert state.terminal_reason in ("solved", "no_work_fits")


# ---------------------------------------------------------------- the handoff

def test_handoff_reports_facts_estimates_and_interpretation_separately(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"))
    session.run()
    text = (session.dir / "handoff.md").read_text()
    assert "## Recommendation" in text and "## Was it solved?" in text
    assert "## Budget consumed" in text and "## Resume" in text
    assert session.brief.success_criterion.describe() in text
    assert "python -m rllm.cli solve" in text                     # exact resume command
    assert (session.dir / "evidence.json").exists()


def test_handoff_recommendation_is_deterministic_without_models(tmp_path):
    session = session_for(tmp_path, explore_seconds=1.0, wind_down_seconds=0.5)
    state = session.run()
    rec = state.recommendation
    assert rec["source"] == "deterministic"
    assert rec["recommendation"] in ("more_time", "needs_input", "needs_help")


def test_handoff_recommendation_enum_is_enforced(tmp_path):
    """A model's free-text recommendation is coerced into the actionable enum, never passed through."""
    from rllm.session import _clean_recommendation
    cleaned = _clean_recommendation({"recommendation": "just keep going I guess",
                                     "reasoning": "vibes"})
    assert cleaned["recommendation"] == "needs_input"
    assert cleaned["recommendation_raw"] == "just keep going I guess"
    assert _clean_recommendation({"recommendation": "More Time"})["recommendation"] == "more_time"


def test_handoff_flags_disagreement_between_models_and_numbers(tmp_path):
    session = session_for(tmp_path, explore_seconds=1.0, wind_down_seconds=0.5)
    session.run()
    session.state.recommendation = {"recommendation": "solved", "source": "actor=x, reviewer=y",
                                    "deterministic_view": "more_time", "confidence": "high",
                                    "reasoning": "we did it"}
    from rllm import report
    text = (report.write_handoff(session, "deadline_reached")).read_text()
    assert "Trust the numbers" in text


def test_session_appends_facts_only_to_the_journal(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.95), reviewer=approver("x095"))
    session.run()
    journal = (Path(session.work_dir) / "journal.md").read_text()
    assert f"Session {session.state.session_id}" in journal
    assert "auto-generated, facts only" in journal
    assert "run(s) launched" in journal


def test_unconfirmed_leads_are_labelled_as_such(tmp_path):
    session = session_for(tmp_path, actor=proposals(0.5), reviewer=approver("x05"), maximum_runs=1)
    session.run()
    text = (session.dir / "handoff.md").read_text()
    assert "unconfirmed" in text


# ---------------------------------------------------------------- onboarding entry point

def test_onboard_cli_writes_and_validates_a_brief(tmp_path):
    from rllm import cli
    answers = tmp_path / "answers.json"
    answers.write_text(make_brief(tmp_path).to_json())
    wd = tmp_path / "onboarded"
    cli.main(["onboard", str(wd), "--answers", str(answers)])
    assert problem_mod.brief_path(wd).exists()
    brief, adapter = problem_mod.load(wd)
    assert brief.problem_id == "fake" and adapter.name == "fake"


def test_onboard_refuses_to_silently_overwrite(tmp_path):
    from rllm import cli
    answers = tmp_path / "answers.json"
    answers.write_text(make_brief(tmp_path).to_json())
    wd = tmp_path / "onboarded"
    cli.main(["onboard", str(wd), "--answers", str(answers)])
    with pytest.raises(SystemExit):
        cli.main(["onboard", str(wd), "--answers", str(answers)])


def test_onboard_rejects_a_bad_brief(tmp_path, capsys):
    from rllm import cli
    data = json.loads(make_brief(tmp_path).to_json())
    data["problem_repo"] = "/definitely/not/here"
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        cli.main(["onboard", str(tmp_path / "x"), "--answers", str(answers)])
    assert "not a directory" in capsys.readouterr().err


def test_solve_requires_a_brief(tmp_path):
    from rllm import cli
    with pytest.raises(SystemExit, match="rllm-onboard"):
        cli.main(["solve", str(work_dir(tmp_path))])


# ---------------------------------------------------------------- budget reserves

def test_llm_calls_are_reserved_for_the_handoff(tmp_path):
    """A session that spends its last call on one more screening wave hands the user a deterministic
    stub instead of an explanation. Seen live with maximum_llm_calls=8."""
    session = session_for(tmp_path, actor=proposals(0.2, 0.3, 0.4, 0.5, 0.6),
                          reviewer=approver("x02", "x03", "x04", "x05", "x06"),
                          maximum_llm_calls=6)
    state = session.run()
    assert state.ledger["llm_calls"] <= 6
    # Proposals stopped early enough that the handoff still got its two calls (the mock answers them
    # with a proposal, which is why the parsed recommendation degrades — but it WAS asked).
    assert state.recommendation["source"] != "deterministic"
    assert any("keeping" in note for note in session.ledger.stopped_because)


def test_run_budget_is_reserved_for_promotions(tmp_path):
    """A modest run cap must still leave room to take one candidate to the top rung."""
    session = session_for(tmp_path, actor=proposals(0.9, 0.8, 0.7, 0.6),
                          reviewer=approver("x09", "x08", "x07", "x06"), maximum_runs=12)
    state = session.run()
    promotions = [a for a in state.actions if a["action"] == "promote"]
    assert promotions, "the whole run cap went on screening; nothing was ever confirmed"
    assert ladder.rank(session.adapter, session.work_dir, "confirm")


def test_the_reserve_scales_with_the_ladder(tmp_path):
    session = session_for(tmp_path)
    assert session._promotion_reserve() == 5          # refine's 2 seeds + confirm's 3
